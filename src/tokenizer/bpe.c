#include "tokenizer/bpe.h"

#include <stdlib.h>  /* malloc, free */
#include <string.h>  /* memcpy */

/*
 * bpe_create — allocate a fresh tokenizer with the 256 base byte tokens.
 *
 * The vocab is laid out so that token id i (for i in 0..255) represents
 * the single raw byte i. No merges have been learned yet.
 *
 * Memory model: we make 1 + 1 + 256 = 258 separate malloc calls — one for
 * the struct, one for the vocab array, and one per VocabEntry's bytes buffer.
 * bpe_free undoes them all in reverse.
 *
 * We deliberately do NOT handle malloc failure: on a desktop OS asking for
 * a few hundred bytes essentially never fails, and the cleanup logic would
 * dwarf the actual code. This matches the project's "learning over robustness"
 * stance for now.
 */
BPETokenizer *bpe_create(void) {
    /* Step 1: allocate the struct itself.
     * sizeof(BPETokenizer) is "however many bytes one BPETokenizer occupies".
     * The cast `(BPETokenizer *)` is implicit in C — malloc returns void*
     * (a pointer to anything) and C auto-converts it to whatever type we
     * assign it to. */
    BPETokenizer *tok = malloc(sizeof(BPETokenizer));

    /* Step 2: initialize the top-level fields.
     * `tok->field` is shorthand for `(*tok).field` — follow the pointer,
     * then read/write the field. */
    tok->vocab_size = 256;
    tok->num_merges = 0;
    tok->merges = NULL;  /* no merges learned yet; bpe_train fills this in */

    /* Step 3: allocate the vocab array — 256 VocabEntry structs back-to-back
     * in one contiguous block. tok->vocab[i] then accesses the i-th entry. */
    tok->vocab = malloc(256 * sizeof(VocabEntry));

    /* Step 4: fill in each base byte token. Token id i stores the single
     * byte whose value is also i, so decoding token 65 produces 'A', token
     * 97 produces 'a', and so on for the full 0..255 range. */
    for (int i = 0; i < 256; i++) {
        tok->vocab[i].length = 1;
        tok->vocab[i].bytes = malloc(1);  /* exactly one byte */
        tok->vocab[i].bytes[0] = (unsigned char)i;
    }

    return tok;
}

/*
 * bpe_free — release a tokenizer and all the memory it owns.
 *
 * Order matters: we free the leaves first (each VocabEntry's bytes buffer),
 * then the arrays that held them, then the struct. Freeing in the other
 * order would leave us holding a dangling pointer to the array and unable
 * to reach the leaves.
 */
void bpe_free(BPETokenizer *tok) {
    /* free(NULL) is a no-op in C. We mirror that here so callers can
     * unconditionally call bpe_free(tok) without an explicit NULL check. */
    if (tok == NULL) return;

    /* Free each vocab entry's owned bytes buffer. */
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->vocab[i].bytes);
    }
    /* Free the vocab array itself. */
    free(tok->vocab);
    /* Free the merges array. May be NULL if bpe_train was never called —
     * that's fine, free(NULL) is safe. */
    free(tok->merges);
    /* Finally, free the struct. */
    free(tok);
}

void bpe_train(BPETokenizer *tok,
               const unsigned char *text,
               size_t length,
               int num_merges) {
    (void)tok;
    (void)text;
    (void)length;
    (void)num_merges;
}

/*
 * bpe_encode — convert a byte buffer into a sequence of token ids.
 *
 * For an untrained tokenizer (no merges learned yet), token id == byte value:
 * the byte 0x61 ('a') becomes token 97, byte 0xC4 becomes token 196, etc.
 * Since vocab[i].bytes == {i} for i in 0..255, we can shortcut the lookup
 * and emit the byte values directly.
 *
 * Once bpe_train fills in merges, this function will need to scan for
 * adjacent (a, b) pairs and replace them with merged token ids — that's
 * the next TDD step. For now, the no-merges path is enough.
 */
int *bpe_encode(const BPETokenizer *tok,
                const unsigned char *text,
                size_t length,
                size_t *out_count) {
    (void)tok;  /* unused on the no-merges path; merges will use it */

    if (length == 0) {
        *out_count = 0;
        return NULL;
    }

    /* Allocate the output array: one int per input byte. sizeof(int) is
     * usually 4 on a 64-bit system, so we get length*4 bytes back. */
    int *toks = malloc(length * sizeof(int));

    /* Copy each byte's value into the corresponding token slot. The cast
     * `(int)text[i]` is implicit (unsigned char promotes to int) but written
     * explicitly here for clarity. */
    for (size_t i = 0; i < length; i++) {
        toks[i] = (int)text[i];
    }

    *out_count = length;
    return toks;
}

/*
 * bpe_decode — convert a sequence of token ids back into raw bytes.
 *
 * Each token id i refers to vocab[i], a byte sequence of length
 * vocab[i].length. The output is the concatenation of all those byte
 * sequences. For an untrained tokenizer each entry is 1 byte, so output
 * length == count. For a trained tokenizer, merged tokens contribute
 * multiple bytes each, so output length >= count.
 *
 * Implemented as two passes: first sum the lengths to know how big the
 * output buffer must be, then allocate and copy.
 */
unsigned char *bpe_decode(const BPETokenizer *tok,
                          const int *tokens,
                          size_t count,
                          size_t *out_length) {
    if (count == 0) {
        *out_length = 0;
        return NULL;
    }

    /* Pass 1: compute total output size by summing each token's vocab length. */
    size_t total = 0;
    for (size_t i = 0; i < count; i++) {
        total += tok->vocab[tokens[i]].length;
    }

    /* Allocate the output buffer. NOT NUL-terminated — caller uses *out_length. */
    unsigned char *out = malloc(total);

    /* Pass 2: copy each token's bytes into the buffer in order.
     * `cursor` is a "running pointer" that starts at the beginning and
     * advances after each memcpy by the number of bytes just written.
     * `cursor += len` advances by `len` elements of unsigned char (= len bytes). */
    unsigned char *cursor = out;
    for (size_t i = 0; i < count; i++) {
        size_t len = tok->vocab[tokens[i]].length;
        memcpy(cursor, tok->vocab[tokens[i]].bytes, len);
        cursor += len;
    }

    *out_length = total;
    return out;
}
