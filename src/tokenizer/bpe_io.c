#include "tokenizer/bpe_io.h"

#include <stdio.h>   /* fopen, fread, fwrite, fclose */
#include <stdlib.h>  /* malloc, realloc */
#include <string.h>  /* memcpy */

/* Magic number identifying a BPE tokenizer file: the four bytes 'B','P','E',0
 * read as a little-endian int32 = 0x00455042. Stored on disk in the file's
 * native byte order. We only target little-endian platforms (Mac M2, Linux
 * x86_64) so this stays consistent. */
#define BPE_MAGIC   0x00455042
#define BPE_VERSION 1

/*
 * bpe_save — write a tokenizer to a binary file.
 *
 * File layout:
 *   header  : three int32 fields (magic, version, num_merges) = 12 bytes
 *   merges  : num_merges Merge structs (each = two int32 = 8 bytes)
 *
 * We track success in `rc` and reach a single fclose at the end on every
 * exit path — keeps the cleanup logic simple without using goto.
 */
int bpe_save(const BPETokenizer *tok, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (f == NULL) return -1;

    int rc = 0;

    /* Build the 12-byte header in a local array, then write it in one call.
     * The order of fields matches the file format documented in bpe_io.h. */
    int header[3] = { BPE_MAGIC, BPE_VERSION, tok->num_merges };
    if (fwrite(header, sizeof(int), 3, f) != 3) rc = -1;

    /* Write the merges array. The Merge struct is exactly two ints with
     * no padding (compilers always pack two ints tightly), so we can dump
     * the whole array as `num_merges` records of sizeof(Merge) bytes. */
    if (rc == 0 && tok->num_merges > 0) {
        size_t n = (size_t)tok->num_merges;
        if (fwrite(tok->merges, sizeof(Merge), n, f) != n) rc = -1;
    }

    fclose(f);
    return rc;
}

/*
 * bpe_load — read a tokenizer file produced by bpe_save.
 *
 * Steps:
 *   1. Open the file and read+validate the 12-byte header.
 *   2. Create a fresh tokenizer (256 base byte tokens).
 *   3. If num_merges > 0, allocate the merges array, grow vocab, and read
 *      the merges from disk.
 *   4. Reconstruct each merged vocab entry by replaying the merge:
 *      vocab[256+i].bytes = vocab[a].bytes ++ vocab[b].bytes.
 *   5. Update num_merges and vocab_size, return.
 *
 * On any error we close the file, free anything we've allocated so far via
 * bpe_free, and return NULL — leaving no leaks behind.
 */
BPETokenizer *bpe_load(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (f == NULL) return NULL;

    /* Read the 12-byte header. */
    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) {
        fclose(f);
        return NULL;
    }

    /* Validate magic + version + sanity. */
    if (header[0] != BPE_MAGIC || header[1] != BPE_VERSION || header[2] < 0) {
        fclose(f);
        return NULL;
    }
    int n = header[2];

    /* Start with a fresh tokenizer (256 base bytes, no merges yet). */
    BPETokenizer *tok = bpe_create();

    /* Untrained tokenizer special case: nothing more to read. */
    if (n == 0) {
        fclose(f);
        return tok;
    }

    /* Allocate space for merges and grow vocab. Mirrors what bpe_train does. */
    tok->merges = malloc((size_t)n * sizeof(Merge));
    tok->vocab = realloc(tok->vocab, (size_t)(256 + n) * sizeof(VocabEntry));

    /* Read the merges array straight off disk. */
    if (fread(tok->merges, sizeof(Merge), (size_t)n, f) != (size_t)n) {
        bpe_free(tok);
        fclose(f);
        return NULL;
    }
    fclose(f);

    /* Reconstruct each merged vocab entry by replaying the merge: token
     * (256 + i) is the concatenation of vocab[a].bytes and vocab[b].bytes.
     * Because merges reference earlier ids only (the priority order
     * established during training), iterating i from 0 upwards always
     * has both parents already present in vocab. */
    for (int i = 0; i < n; i++) {
        int new_id = 256 + i;
        int a = tok->merges[i].a;
        int b = tok->merges[i].b;
        size_t la = tok->vocab[a].length;
        size_t lb = tok->vocab[b].length;
        tok->vocab[new_id].length = la + lb;
        tok->vocab[new_id].bytes = malloc(la + lb);
        memcpy(tok->vocab[new_id].bytes, tok->vocab[a].bytes, la);
        memcpy(tok->vocab[new_id].bytes + la, tok->vocab[b].bytes, lb);
    }

    /* Finalize bookkeeping only after the vocab loop fully succeeds, so
     * a failure earlier leaves bpe_free's loop bound (vocab_size) at 256
     * — which avoids touching uninitialized vocab entries. */
    tok->num_merges = n;
    tok->vocab_size = 256 + n;

    return tok;
}
