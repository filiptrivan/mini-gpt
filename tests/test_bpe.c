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
#include <stdio.h>   /* remove — clean up the temp file the IO test writes */
#include <string.h>  /* strlen */
#include <cmocka.h>

#include "tokenizer/bpe.h"
#include "tokenizer/bpe_io.h"

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

/*
 * test_encode_decode_longer_ascii_roundtrip — regression guard: verifies
 * that encode/decode are correct on multi-byte input, not just one byte.
 *
 * Untrained tokenizer means one token per input byte, so we expect
 * n == input_len, and decoding must return the exact same bytes.
 */
static void test_encode_decode_longer_ascii_roundtrip(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();

    const char *input = "Hello, World! This is a longer test string.";
    size_t input_len = strlen(input);

    size_t n = 0;
    int *toks = bpe_encode(tok, (const unsigned char *)input, input_len, &n);
    assert_non_null(toks);
    assert_int_equal(n, input_len);  /* untrained: one token per byte */

    size_t m = 0;
    unsigned char *back = bpe_decode(tok, toks, n, &m);
    assert_non_null(back);
    assert_int_equal(m, input_len);
    /* assert_memory_equal compares input_len bytes starting at back and input;
     * fails if any byte differs. */
    assert_memory_equal(back, input, input_len);

    free(back);
    free(toks);
    bpe_free(tok);
}

/*
 * test_encode_decode_serbian_utf8_roundtrip — proves the byte-level
 * tokenizer handles multi-byte UTF-8 characters correctly, with no
 * special Unicode logic on our side.
 *
 * Each Serbian Latin diacritic (č ž š đ ć) is 2 bytes in UTF-8. So an
 * untrained tokenizer encoding "čžšđć" (5 chars, 10 bytes) produces a
 * 10-token list. Decoding gives the same 10 bytes back, which any
 * UTF-8 reader (including the test source string itself) interprets
 * as the original 5 characters.
 */
static void test_encode_decode_serbian_utf8_roundtrip(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();

    /* The source file is UTF-8, so this literal is the byte sequence
     * 0xC4 0x8D 0xC5 0xBE 0xC5 0xA1 0xC4 0x91 0xC4 0x87 — 10 bytes. */
    const char *input = "čžšđć";
    size_t input_len = strlen(input);
    assert_int_equal(input_len, 10);  /* sanity-check our UTF-8 assumption */

    size_t n = 0;
    int *toks = bpe_encode(tok, (const unsigned char *)input, input_len, &n);
    assert_non_null(toks);
    assert_int_equal(n, 10);

    size_t m = 0;
    unsigned char *back = bpe_decode(tok, toks, n, &m);
    assert_non_null(back);
    assert_int_equal(m, 10);
    assert_memory_equal(back, input, 10);

    free(back);
    free(toks);
    bpe_free(tok);
}

/*
 * test_encode_is_deterministic — same input must produce identical tokens.
 *
 * Tokenizers MUST be deterministic: a model trained on one tokenization
 * is incompatible with a different one for the same text. This test
 * locks the property in as a regression guard.
 *
 * `assert_memory_equal` compares bytes, so we pass the array length
 * times sizeof(int) to compare full int values.
 */
static void test_encode_is_deterministic(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();
    const char *input = "Same input twice produces same tokens.";
    size_t input_len = strlen(input);

    size_t n1 = 0, n2 = 0;
    int *toks1 = bpe_encode(tok, (const unsigned char *)input, input_len, &n1);
    int *toks2 = bpe_encode(tok, (const unsigned char *)input, input_len, &n2);

    assert_int_equal(n1, n2);
    assert_memory_equal(toks1, toks2, n1 * sizeof(int));

    free(toks2);
    free(toks1);
    bpe_free(tok);
}

/*
 * test_train_classic_example — verifies bpe_train against the by-hand
 * walkthrough from docs/bpe-training.md.
 *
 * Training on "aaabdaaabac" with 3 merges should produce:
 *   merges[0] = (97, 97)   -> token 256, vocab "aa"   (2 bytes)
 *   merges[1] = (256, 97)  -> token 257, vocab "aaa"  (3 bytes)
 *   merges[2] = (257, 98)  -> token 258, vocab "aaab" (4 bytes)
 *
 * This is the smallest test that exercises the full training algorithm:
 * pair counting, tie-breaking, vocab growth via merged-byte concatenation,
 * and use of a previously-merged token as input to a later merge.
 */
static void test_train_classic_example(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();

    const unsigned char *input = (const unsigned char *)"aaabdaaabac";
    bpe_train(tok, input, 11, 3);

    /* Top-level state. */
    assert_int_equal(tok->num_merges, 3);
    assert_int_equal(tok->vocab_size, 259);

    /* Each merge rule is exactly the (a, b) pair we worked out by hand. */
    assert_int_equal(tok->merges[0].a, 97);
    assert_int_equal(tok->merges[0].b, 97);
    assert_int_equal(tok->merges[1].a, 256);
    assert_int_equal(tok->merges[1].b, 97);
    assert_int_equal(tok->merges[2].a, 257);
    assert_int_equal(tok->merges[2].b, 98);

    /* New vocab entries hold the concatenated bytes. */
    assert_int_equal(tok->vocab[256].length, 2);
    assert_memory_equal(tok->vocab[256].bytes, "aa", 2);
    assert_int_equal(tok->vocab[257].length, 3);
    assert_memory_equal(tok->vocab[257].bytes, "aaa", 3);
    assert_int_equal(tok->vocab[258].length, 4);
    assert_memory_equal(tok->vocab[258].bytes, "aaab", 4);

    bpe_free(tok);
}

/*
 * test_encode_applies_learned_merges — verifies that bpe_encode actually
 * uses the merges learned during training, not just emits raw bytes.
 *
 * After training on "aaabdaaabac" with 3 merges (see docs/bpe-training.md),
 * encoding "aaabd" should produce [258, 100] — only 2 tokens for 5 bytes,
 * because the substring "aaab" was learned as a single token (id 258).
 *
 * We also verify the roundtrip property still holds: decoding [258, 100]
 * must give back the original 5 bytes "aaabd".
 */
static void test_encode_applies_learned_merges(void **state) {
    (void)state;

    BPETokenizer *tok = bpe_create();
    bpe_train(tok, (const unsigned char *)"aaabdaaabac", 11, 3);

    size_t n = 0;
    int *toks = bpe_encode(tok, (const unsigned char *)"aaabd", 5, &n);
    assert_non_null(toks);
    assert_int_equal(n, 2);
    assert_int_equal(toks[0], 258);  /* token "aaab" */
    assert_int_equal(toks[1], 100);  /* byte 'd' */

    /* Roundtrip must still hold. */
    size_t m = 0;
    unsigned char *back = bpe_decode(tok, toks, n, &m);
    assert_non_null(back);
    assert_int_equal(m, 5);
    assert_memory_equal(back, "aaabd", 5);

    free(back);
    free(toks);
    bpe_free(tok);
}

/*
 * test_save_load_roundtrip — train a tokenizer, save it, load it back,
 * verify every piece of state matches and that encoding produces the
 * same tokens with the loaded tokenizer as with the original.
 *
 * Uses /tmp for the binary file (works on Mac M2 + Linux/Colab; fine
 * for our targets). Cleans up afterwards with remove().
 */
static void test_save_load_roundtrip(void **state) {
    (void)state;

    /* Step 1: train a tokenizer (the canonical example). */
    BPETokenizer *tok = bpe_create();
    bpe_train(tok, (const unsigned char *)"aaabdaaabac", 11, 3);

    /* Step 2: save it. */
    const char *path = "/tmp/test_bpe_io.bin";
    int rc = bpe_save(tok, path);
    assert_int_equal(rc, 0);

    /* Step 3: load it into a new tokenizer. */
    BPETokenizer *loaded = bpe_load(path);
    assert_non_null(loaded);

    /* Step 4: top-level state matches. */
    assert_int_equal(loaded->num_merges, tok->num_merges);
    assert_int_equal(loaded->vocab_size, tok->vocab_size);

    /* Step 5: every merge rule matches. */
    for (int i = 0; i < tok->num_merges; i++) {
        assert_int_equal(loaded->merges[i].a, tok->merges[i].a);
        assert_int_equal(loaded->merges[i].b, tok->merges[i].b);
    }

    /* Step 6: every vocab entry matches (including reconstructed ones). */
    for (int i = 0; i < tok->vocab_size; i++) {
        assert_int_equal(loaded->vocab[i].length, tok->vocab[i].length);
        assert_memory_equal(loaded->vocab[i].bytes, tok->vocab[i].bytes,
                            tok->vocab[i].length);
    }

    /* Step 7: encoding with the loaded tokenizer must produce identical
     * tokens — the strongest possible behavioral check. */
    size_t n1 = 0, n2 = 0;
    int *toks1 = bpe_encode(tok, (const unsigned char *)"aaabd", 5, &n1);
    int *toks2 = bpe_encode(loaded, (const unsigned char *)"aaabd", 5, &n2);
    assert_int_equal(n1, n2);
    assert_memory_equal(toks1, toks2, n1 * sizeof(int));

    free(toks2);
    free(toks1);
    bpe_free(loaded);
    bpe_free(tok);

    /* Clean up the temp file so reruns don't see stale state. */
    remove(path);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_create_has_256_base_byte_tokens),
        cmocka_unit_test(test_encode_decode_empty_input),
        cmocka_unit_test(test_encode_decode_single_ascii_char),
        cmocka_unit_test(test_encode_decode_longer_ascii_roundtrip),
        cmocka_unit_test(test_encode_decode_serbian_utf8_roundtrip),
        cmocka_unit_test(test_encode_is_deterministic),
        cmocka_unit_test(test_train_classic_example),
        cmocka_unit_test(test_encode_applies_learned_merges),
        cmocka_unit_test(test_save_load_roundtrip),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
