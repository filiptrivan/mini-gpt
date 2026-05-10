#ifndef BPE_H
#define BPE_H

/* Include guard above (#ifndef / #define / #endif) prevents this header from
 * being included twice in the same translation unit. Without it, the typedefs
 * below would be redefined and the compiler would error. */

#include <stddef.h>  /* for size_t */

/* Number of base byte tokens — token id i (for i in 0..255) represents the
 * single byte i. All learned merges use ids >= BPE_BASE_VOCAB. */
#define BPE_BASE_VOCAB 256

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
 *                tok->vocab_size == BPE_BASE_VOCAB + num_merges. If the corpus
 *                is too small to produce all requested merges, training stops
 *                early and num_merges reflects the actual count achieved.
 *
 * See docs/bpe-training.md for a worked example of the algorithm.
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
 * Returns: a malloc'd int* array of length *out_count, or NULL on failure
 * or when length == 0 (in which case *out_count is set to 0).
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

/*
 * bpe_build_merged_vocab_entry — internal helper shared by bpe_train and
 * bpe_load. Sets vocab[new_id] to the byte concatenation of vocab[a] and
 * vocab[b], malloc'ing the bytes buffer.
 *
 * Caller must have grown the vocab array to include index new_id.
 */
void bpe_build_merged_vocab_entry(VocabEntry *vocab, int new_id, int a, int b);

#endif /* BPE_H */
