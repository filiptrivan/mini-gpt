/*
 * DataLoader implementation — see data/dataloader.h for the contract and
 * docs/dataloader.md for a full walked-by-hand example.
 *
 * Design notes for beginners:
 *   - The entire token file is slurped into one malloc'd buffer at init.
 *     Our training corpus is a textbook, a few MB at most; loading once is
 *     dramatically simpler than streaming with fseek/fread on each batch.
 *   - Batches are produced by copying from the buffer into caller-provided
 *     output arrays. We never hand back a pointer into our internal buffer
 *     — that would expose us to use-after-free bugs and break encapsulation.
 *   - All memory ownership is explicit: dataloader_init returns memory the
 *     caller must free with dataloader_free; the caller owns the inputs/
 *     targets arrays passed to dataloader_next_batch.
 */

#include "data/dataloader.h"

#include <assert.h>  /* assert() — runtime check in debug builds, no-op under -DNDEBUG */
#include <stddef.h>  /* NULL */
#include <stdio.h>   /* fopen, fclose, fread, fseek, ftell, SEEK_END, SEEK_SET */
#include <stdlib.h>  /* malloc, free */

/*
 * dataloader_init — see header for full contract.
 *
 * Steps:
 *   1. Open file in binary mode.
 *   2. Determine its size by seeking to the end and asking the position.
 *   3. Compute how many ints that is and reject files too small for one batch.
 *   4. Allocate the in-memory buffer and the DataLoader struct.
 *   5. Read all tokens in one fread call, then close the file.
 *   6. Fill in the struct fields and return.
 *
 * Any failure along the way frees what we've allocated so far and returns NULL.
 */
DataLoader *dataloader_init(const char *filepath,
                            int batch_size, int seq_len, int vocab_size) {
    /* Programmer-contract preconditions. Bad arguments here mean the caller
     * has a bug (e.g. forgot to pass vocab_size); a graceful NULL return
     * would just hide that. Asserts compile out under -DNDEBUG. */
    assert(filepath   != NULL);
    assert(batch_size  > 0);
    assert(seq_len     > 0);
    assert(vocab_size  > 0);

    /* "rb" = read, binary mode. On Unix the 'b' is redundant; we keep it
     * for portability and to make intent explicit. */
    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        /* Missing file, permission denied, etc. — fopen returns NULL and
         * sets errno. We don't propagate errno; the contract is just
         * "returns NULL on failure". */
        return NULL;
    }

    /* Measure the file's size by seeking to the end and reading the
     * position. fseek/ftell is the classic way to do this in portable C.
     * SEEK_END = "offset is from end of file"; offset 0 puts us exactly
     * at the end. ftell then returns that absolute byte offset = file size. */
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long file_bytes = ftell(f);
    if (file_bytes < 0) {  /* ftell returns -1 on error */
        fclose(f);
        return NULL;
    }
    /* Rewind to the start so fread below starts at byte 0. */
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    /* How many ints are in the file. Each token is one int (sizeof(int)
     * bytes, typically 4). If the file size isn't an exact multiple of
     * sizeof(int), we silently truncate to the next-lower whole token —
     * that's a corrupt file, but tokenize.c always writes whole ints so
     * in practice it's exact. */
    size_t num_tokens = (size_t)file_bytes / sizeof(int);

    /* Reject files too small for even one batch. We need batch_size*seq_len
     * tokens for inputs PLUS one more for the rightmost target. Wrapping
     * doesn't rescue a too-small file because even a full wrap-around
     * window doesn't fit. */
    size_t need = (size_t)batch_size * (size_t)seq_len + 1;
    if (num_tokens < need) {
        fclose(f);
        return NULL;
    }

    /* Allocate the token buffer. malloc returns NULL on out-of-memory. */
    int *tokens = (int *)malloc(num_tokens * sizeof(int));
    if (tokens == NULL) {
        fclose(f);
        return NULL;
    }

    /* Read every token in a single call. fread returns the number of items
     * (not bytes) it successfully read. We require all of them or we treat
     * it as a failure. */
    size_t read = fread(tokens, sizeof(int), num_tokens, f);
    fclose(f);  /* done with the file regardless of success */
    if (read != num_tokens) {
        free(tokens);
        return NULL;
    }

    /* Token-range validation — the ONE boundary check that protects every
     * downstream layer (embed_forward, the loss, etc.) from out-of-range
     * ids. We scan the whole file once; that's O(num_tokens), runs once
     * per training run, and is cheap next to the file I/O above. After
     * this loop, every consumer can index wte / wpe by id without
     * re-checking. If even one id is bad, the whole file is rejected:
     * a corrupt tokenizer output isn't something we can salvage. */
    for (size_t i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= vocab_size) {
            free(tokens);
            return NULL;
        }
    }

    /* Allocate the struct itself and populate it. We could put it on the
     * stack at the call site, but heap-allocating + returning a pointer
     * makes ownership crystal clear: the caller has exactly one thing to
     * free, the same pattern as fopen returning FILE*. */
    DataLoader *loader = (DataLoader *)malloc(sizeof(DataLoader));
    if (loader == NULL) {
        free(tokens);
        return NULL;
    }
    loader->tokens     = tokens;
    loader->num_tokens = num_tokens;
    loader->batch_size = batch_size;
    loader->seq_len    = seq_len;
    loader->cursor     = 0;
    return loader;
}

/*
 * dataloader_next_batch — see header for full contract.
 *
 * Two-step logic:
 *   1. If a full batch (B*T + 1 tokens) won't fit starting at cursor,
 *      reset cursor to 0. The +1 accounts for the borrowed target token.
 *   2. Copy B*T tokens starting at cursor into inputs_out, and B*T tokens
 *      starting at cursor+1 into targets_out. The two windows overlap by
 *      B*T - 1 tokens — that's the shift in action.
 *   3. Advance cursor by B*T (NOT by B*T+1: the borrowed last target token
 *      will be re-used as the next batch's first input).
 */
void dataloader_next_batch(DataLoader *loader, int *inputs_out, int *targets_out) {
    size_t batch_tokens = (size_t)loader->batch_size * (size_t)loader->seq_len;

    /* Wrap if we can't fit a full batch from the current cursor. The check
     * is BEFORE the read, so we never index past the end of `tokens`. */
    if (loader->cursor + batch_tokens + 1 > loader->num_tokens) {
        loader->cursor = 0;
    }

    /* Fill the two output buffers in a single pass. Each iteration writes
     * one input token and the corresponding target (one position to the
     * right in the source buffer). */
    for (size_t i = 0; i < batch_tokens; i++) {
        inputs_out[i]  = loader->tokens[loader->cursor + i];
        targets_out[i] = loader->tokens[loader->cursor + i + 1];
    }

    /* Advance the cursor for next time. */
    loader->cursor += batch_tokens;
}

/*
 * dataloader_free — release everything dataloader_init allocated.
 *
 * Two malloc'd things: the tokens buffer (allocated first, inside init)
 * and the DataLoader struct itself. Free both, in any order.
 *
 * NULL guard matches libc free's behavior: free(NULL) is a no-op.
 */
void dataloader_free(DataLoader *loader) {
    if (loader == NULL) {
        return;
    }
    free(loader->tokens);
    free(loader);
}
