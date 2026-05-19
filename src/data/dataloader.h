#ifndef DATALOADER_H
#define DATALOADER_H

/* Include guard above (#ifndef / #define / #endif) — without it, including
 * this header twice in the same .c file would redefine `DataLoader` and the
 * compiler would error. Standard C boilerplate. */

#include <stddef.h>  /* size_t */

/*
 * DataLoader — owns the in-memory copy of a tokenized training file and
 * hands out fixed-size training batches one at a time.
 *
 * Mental model: the file on disk is one long sequence of integer token ids.
 * The loader reads the whole file into a malloc'd buffer at init, then
 * maintains a `cursor` (read position) that walks forward through the
 * buffer. Each call to dataloader_next_batch consumes `batch_size * seq_len`
 * positions (plus one extra borrowed for the shifted targets). When the
 * cursor can't fit another batch, it wraps back to 0 — the corpus is
 * treated as an infinite cyclic stream during training.
 *
 * Fields are public so tests can poke at them; production callers should
 * only touch them through the functions below.
 *
 *   tokens     : pointer to a heap-allocated int array (the whole file).
 *                Owned by this struct — freed by dataloader_free.
 *   num_tokens : number of ints in `tokens`.
 *   batch_size : rows per batch (commonly called B in transformer code).
 *   seq_len    : tokens per row (commonly called T).
 *   cursor     : index into `tokens` where the next batch will start.
 */
typedef struct {
    int *tokens;
    size_t num_tokens;
    int batch_size;
    int seq_len;
    size_t cursor;
} DataLoader;

/*
 * dataloader_init — open a .bin file of token ids, load it fully into RAM,
 *                   and return a ready-to-use DataLoader.
 *
 *   filepath   : NUL-terminated path to a binary file written as a flat
 *                sequence of ints (the format produced by tools/tokenize.c).
 *   batch_size : how many sequences per batch (>= 1).
 *   seq_len    : how many tokens per sequence (>= 1).
 *
 * Returns: a freshly malloc'd DataLoader on success, or NULL if:
 *            - the file cannot be opened,
 *            - memory allocation fails,
 *            - or the file contains fewer than batch_size*seq_len + 1
 *              tokens (not enough for even one batch — wrapping cannot
 *              rescue this; the +1 is needed for the shifted targets).
 *
 * Ownership: caller must release with dataloader_free.
 */
DataLoader *dataloader_init(const char *filepath, int batch_size, int seq_len);

/*
 * dataloader_next_batch — fill caller-provided buffers with the next batch.
 *
 *   loader      : a DataLoader from dataloader_init (must be non-NULL).
 *   inputs_out  : caller-owned buffer of size batch_size*seq_len ints.
 *                 Filled with the tokens the model will see.
 *   targets_out : caller-owned buffer of size batch_size*seq_len ints.
 *                 Filled with inputs_out shifted right by one position
 *                 (the next-token labels for next-token prediction).
 *
 * The loader advances its internal cursor by batch_size*seq_len after the
 * read. If the cursor cannot fit another full batch in what's left of the
 * buffer, it wraps to 0 BEFORE this read happens — so this call always
 * returns a complete, valid batch.
 *
 * No allocation: the loader writes into the caller's buffers. The caller
 * is expected to allocate them once and reuse them across many batches
 * (training does millions of batches; per-batch malloc would be wasteful).
 */
void dataloader_next_batch(DataLoader *loader, int *inputs_out, int *targets_out);

/*
 * dataloader_free — release the in-memory token buffer and the DataLoader
 *                   struct itself.
 *
 * Safe to call with NULL (matches stdlib free's behavior), so callers don't
 * need to null-check before cleanup.
 */
void dataloader_free(DataLoader *loader);

#endif /* DATALOADER_H */
