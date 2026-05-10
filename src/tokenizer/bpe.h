#ifndef BPE_H
#define BPE_H

/* Include guard above (#ifndef / #define / #endif) prevents this header from
 * being included twice in the same translation unit. Without it, the typedefs
 * below would be redefined and the compiler would error. */

#include <stddef.h>  /* for size_t */

/* ---------------------------------------------------------------------------
 * NOTE: This header is a FIRST-DRAFT sketch of the BPE public API. Function
 * signatures, struct shape, and ownership rules are likely to change after
 * we discuss the design. Treat anything here as provisional until we agree
 * and implement the real logic via TDD.
 * ------------------------------------------------------------------------- */

/*
 * VocabEntry — one entry in the tokenizer's vocabulary.
 *
 *   bytes  : pointer to a heap-allocated buffer of `length` raw bytes.
 *            NOT NUL-terminated — BPE works on arbitrary byte sequences,
 *            including the byte 0x00, so we use an explicit length instead.
 *   length : number of bytes in `bytes`.
 *
 * The vocab has one VocabEntry per token id:
 *   - ids 0..255 are single-byte tokens (length=1, bytes={i})
 *   - ids 256..255+num_merges are merged tokens, each formed by
 *     concatenating the bytes of its two parent tokens.
 *
 * The `bytes` buffer is owned by the parent BPETokenizer and is freed by
 * bpe_free().
 */
typedef struct {
    unsigned char *bytes;
    size_t length;
} VocabEntry;

/*
 * Merge — one merge rule learned during training.
 *
 *   a, b : the two existing token ids that get merged together.
 *
 * The position of a Merge in the `merges` array implicitly encodes its
 * priority and its resulting token id:
 *   - merges[0] is the highest-priority merge; its result has token id 256
 *   - merges[i] has result token id (256 + i)
 *
 * During encode, we apply merges in priority order: scan for any
 * adjacent pair (a, b), replace with (256 + i), repeat.
 */
typedef struct {
    int a;
    int b;
} Merge;

/*
 * BPETokenizer — the trained tokenizer state.
 *
 *   vocab_size  : total number of tokens (always 256 + num_merges).
 *   num_merges  : number of merge rules learned (0 until bpe_train is called).
 *   vocab       : array of `vocab_size` VocabEntry structs, indexed by token id.
 *   merges      : array of `num_merges` Merge structs, in priority order.
 *
 * Owned by the caller. Free with bpe_free() exactly once per bpe_create().
 */
typedef struct {
    int vocab_size;
    int num_merges;
    VocabEntry *vocab;
    Merge *merges;
} BPETokenizer;

/*
 * bpe_create — allocate a fresh tokenizer with the 256 base byte tokens.
 *
 * After this returns:
 *   - vocab_size  == 256
 *   - num_merges  == 0
 *   - vocab[i]    == single byte equal to i, for i in 0..255
 *   - merges      == NULL (no merges yet)
 *
 * Returns: a malloc'd BPETokenizer*, or NULL on allocation failure.
 *
 * Ownership: caller must call bpe_free() exactly once.
 */
BPETokenizer *bpe_create(void);

/*
 * bpe_free — release a tokenizer and ALL memory it owns.
 *
 * Frees: each vocab[i].bytes, the vocab array, the merges array, and the
 * struct itself. Safe to call with NULL (does nothing), mirroring free().
 */
void bpe_free(BPETokenizer *tok);

/*
 * bpe_train — learn `num_merges` merge rules from the given byte buffer.
 *
 *   tok        : tokenizer to train (must come from bpe_create()).
 *   text       : raw input bytes (no NUL-termination needed; we use length).
 *   length     : number of bytes in `text`.
 *   num_merges : how many merge rules to learn. After training,
 *                tok->vocab_size == 256 + num_merges.
 *
 * Algorithm sketch (to be implemented):
 *   1. Convert input bytes to a sequence of token ids (each byte is its own id).
 *   2. Repeat num_merges times:
 *      a. Count adjacent pairs.
 *      b. Pick the most frequent pair (a, b). Ties broken deterministically.
 *      c. Add (a, b) -> (256 + merge_index) to the merge table and vocab.
 *      d. Replace every (a, b) in the sequence with the new id.
 */
void bpe_train(BPETokenizer *tok,
               const unsigned char *text,
               size_t length,
               int num_merges);

/*
 * bpe_encode — convert a byte buffer into a sequence of token ids.
 *
 *   tok       : trained tokenizer (training optional — encodes as raw bytes
 *               if no merges have been learned).
 *   text      : input bytes.
 *   length    : number of bytes in `text`.
 *   out_count : OUT — set to the number of token ids returned.
 *
 * Returns: a malloc'd int* array of length *out_count, or NULL on failure.
 *
 * Edge case: length == 0  → returns a malloc'd zero-length buffer (or NULL
 * sentinel; we'll pin this down during TDD) and *out_count == 0.
 *
 * Ownership: caller must free() the returned pointer.
 */
int *bpe_encode(const BPETokenizer *tok,
                const unsigned char *text,
                size_t length,
                size_t *out_count);

/*
 * bpe_decode — convert a sequence of token ids back into raw bytes.
 *
 *   tok        : tokenizer used during encoding.
 *   tokens     : input token ids.
 *   count      : number of token ids.
 *   out_length : OUT — set to the number of bytes returned.
 *
 * Returns: a malloc'd unsigned char* buffer of length *out_length, or NULL.
 *          The buffer is NOT NUL-terminated — the contents may include 0x00
 *          bytes, so always use *out_length to know where it ends.
 *
 * Guarantee: for any text, decode(encode(text)) reproduces the original bytes
 * exactly, regardless of which merges were learned (lossless roundtrip).
 *
 * Ownership: caller must free() the returned pointer.
 */
unsigned char *bpe_decode(const BPETokenizer *tok,
                          const int *tokens,
                          size_t count,
                          size_t *out_length);

#endif /* BPE_H */
