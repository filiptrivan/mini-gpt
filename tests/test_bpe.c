/* Test executable for the BPE tokenizer.
 *
 * Tests are added one at a time, TDD-style:
 *   - write the test,
 *   - run it and watch it fail,
 *   - implement just enough in src/tokenizer/bpe.c to make it pass. */

#include <stdarg.h>  /* required by cmocka.h before its own includes */
#include <stddef.h>  /* size_t */
#include <setjmp.h>  /* required by cmocka.h */
#include <stdlib.h>  /* free — for releasing buffers returned by encode/decode */
#include <cmocka.h>

#include "tokenizer/bpe.h"

/*
 * test_create_has_256_base_byte_tokens — the very first test.
 *
 * After bpe_create() the tokenizer should be in a clean "untrained" state:
 * vocab_size == 256, no merges, and vocab[i] holds the single byte i.
 * This is the foundation every later test builds on.
 */
static void test_create_has_256_base_byte_tokens(void **state) {
    (void)state;  /* per-test state — we don't use it */

    /* Allocate the tokenizer. assert_non_null prints a clear message and
     * stops the test if bpe_create returned NULL. */
    BPETokenizer *tok = bpe_create();
    assert_non_null(tok);

    /* Top-level invariants of a freshly-created tokenizer. */
    assert_int_equal(tok->vocab_size, 256);
    assert_int_equal(tok->num_merges, 0);
    assert_null(tok->merges);   /* no merges learned yet */
    assert_non_null(tok->vocab); /* but vocab itself must exist */

    /* Walk the vocab and confirm each entry is exactly the single byte
     * equal to its index. `unsigned char` here matches the type in
     * VocabEntry.bytes — keeps the comparison unambiguous for byte 0xFF. */
    for (int i = 0; i < 256; i++) {
        assert_int_equal(tok->vocab[i].length, 1);
        assert_non_null(tok->vocab[i].bytes);
        assert_int_equal(tok->vocab[i].bytes[0], (unsigned char)i);
    }

    /* Always pair create with free, even in tests, to catch leaks early. */
    bpe_free(tok);
}

/*
 * test_encode_decode_empty_input — encoding/decoding nothing should
 * return nothing, set the out-counts to zero, and not crash.
 *
 * This locks in the agreed edge-case behavior:
 *   - empty input + length == 0 → return NULL, *out_count = 0
 *   - NULL token list + count == 0 → return NULL, *out_length = 0
 *
 * The current stub implementations already match this contract, so the
 * test passes immediately. Its real value is as a regression guard once
 * the real encode/decode logic lands.
 */
static void test_encode_decode_empty_input(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();
    assert_non_null(tok);

    /* We pre-set the OUT parameters to a sentinel value (999) so we can
     * tell the difference between "the function wrote 0" and "the value
     * happened to be 0 already." A real call must overwrite the sentinel. */
    size_t n = 999;
    int *toks = bpe_encode(tok, (const unsigned char *)"", 0, &n);
    assert_null(toks);
    assert_int_equal(n, 0);

    size_t m = 999;
    unsigned char *back = bpe_decode(tok, NULL, 0, &m);
    assert_null(back);
    assert_int_equal(m, 0);

    bpe_free(tok);
}

/*
 * test_encode_decode_single_ascii_char — simplest non-trivial roundtrip.
 *
 * On an untrained tokenizer (no merges yet), encoding the single byte 'a'
 * should produce one token with id 97 (the ASCII value of 'a'). Decoding
 * that token list should reproduce the original byte exactly.
 *
 * This is the first test that forces real implementation work: the stub
 * returns NULL, but here we expect a non-NULL malloc'd buffer with the
 * correct contents.
 */
static void test_encode_decode_single_ascii_char(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();
    assert_non_null(tok);

    /* Encode the single ASCII byte 'a' (= 0x61 = 97 in decimal).
     * The cast to const unsigned char * matches the API's parameter type;
     * the bytes themselves are unchanged. */
    size_t n = 0;
    int *toks = bpe_encode(tok, (const unsigned char *)"a", 1, &n);
    assert_non_null(toks);
    assert_int_equal(n, 1);
    assert_int_equal(toks[0], 97);

    /* Decode the token list back to bytes. */
    size_t m = 0;
    unsigned char *back = bpe_decode(tok, toks, n, &m);
    assert_non_null(back);
    assert_int_equal(m, 1);
    assert_int_equal(back[0], (unsigned char)'a');

    /* Three independent allocations -> three frees. */
    free(back);
    free(toks);
    bpe_free(tok);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_create_has_256_base_byte_tokens),
        cmocka_unit_test(test_encode_decode_empty_input),
        cmocka_unit_test(test_encode_decode_single_ascii_char),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
