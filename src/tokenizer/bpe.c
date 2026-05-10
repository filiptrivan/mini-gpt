#include "tokenizer/bpe.h"

#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * STUB IMPLEMENTATIONS.
 *
 * Every public function has an empty body that returns a "do nothing" value.
 * This lets the library link cleanly so the build stays green while we
 * iterate on the API design. Real implementations land in the TDD step.
 *
 * The (void) casts on unused parameters silence -Wunused-parameter warnings
 * without changing behavior — they explicitly tell the compiler "yes, I know
 * I'm not using this; that's intentional".
 * ------------------------------------------------------------------------- */

BPETokenizer *bpe_create(void) {
    return NULL;
}

void bpe_free(BPETokenizer *tok) {
    (void)tok;
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

int *bpe_encode(const BPETokenizer *tok,
                const unsigned char *text,
                size_t length,
                size_t *out_count) {
    (void)tok;
    (void)text;
    (void)length;
    if (out_count) *out_count = 0;
    return NULL;
}

unsigned char *bpe_decode(const BPETokenizer *tok,
                          const int *tokens,
                          size_t count,
                          size_t *out_length) {
    (void)tok;
    (void)tokens;
    (void)count;
    if (out_length) *out_length = 0;
    return NULL;
}
