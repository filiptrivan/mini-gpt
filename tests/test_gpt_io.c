/*
 * Tests for src/model/gpt_io.c — saving and loading a model checkpoint.
 *
 * Checkpointing is the bridge from training (train.c) to inference
 * (tools/generate.c): the weights have to survive a round trip through a file.
 * We check that three ways:
 *
 *   1. ROUNDTRIP IS LOSSLESS (test_gpt_save_load_roundtrip)
 *      Init a model, save it, load it into a second model, and assert the config
 *      and every parameter come back byte-for-byte identical. A checkpoint that
 *      silently changed a weight would quietly degrade the model.
 *
 *   2. A LOADED MODEL COMPUTES THE SAME THING (test_loaded_model_forward_matches)
 *      Identical weights must produce an identical forward loss. This proves the
 *      load didn't just copy bytes but rebuilt a *usable* model (right layout,
 *      right tensor views) — the property generation actually depends on.
 *
 *   3. FAILURES ARE REPORTED, NOT CRASHES (test_gpt_load_missing_file_fails)
 *      Loading a path that doesn't exist returns -1 instead of crashing.
 *
 * Tiny config throughout so the whole file runs in milliseconds.
 */

/* CMocka's required four includes, in this exact order (see test_layers.c for
 * the full explanation). */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdio.h>   /* remove() — delete the temp checkpoint after each test */

#include "model/gpt.h"
#include "model/gpt_io.h"

/* A small but structurally complete model: every tensor exists, sizes stay
 * readable. Same shape the other model tests use. */
static GPTConfig tiny_config(void) {
    GPTConfig c = {.max_seq_len = 8, .vocab_size = 32, .n_embd = 8,
                   .n_head = 2, .n_layer = 1, .ff_dim = 16};
    return c;
}

/*
 * Save a freshly-initialized model, load it back, and assert the config and
 * every weight survived the round trip exactly. Tolerance 0.0 because this is a
 * pure byte copy — the floats must come back bit-identical, not merely close.
 */
static void test_gpt_save_load_roundtrip(void **state) {
    (void)state;
    const char *path = "/tmp/test_gpt_io_roundtrip.ckpt";

    GPTConfig cfg = tiny_config();
    GPTModel saved;
    gpt_init(&saved, cfg, /*seed=*/42u);

    assert_int_equal(gpt_save(&saved, path), 0);

    GPTModel loaded;
    assert_int_equal(gpt_load(&loaded, path), 0);

    /* config recovered field by field */
    assert_int_equal(loaded.config.max_seq_len, cfg.max_seq_len);
    assert_int_equal(loaded.config.vocab_size,  cfg.vocab_size);
    assert_int_equal(loaded.config.n_embd,      cfg.n_embd);
    assert_int_equal(loaded.config.n_head,      cfg.n_head);
    assert_int_equal(loaded.config.n_layer,     cfg.n_layer);
    assert_int_equal(loaded.config.ff_dim,      cfg.ff_dim);

    /* every parameter identical */
    assert_int_equal(loaded.num_params, saved.num_params);
    for (int i = 0; i < saved.num_params; i++) {
        assert_float_equal(loaded.params[i], saved.params[i], 0.0f);
    }

    gpt_free(&saved);
    gpt_free(&loaded);
    remove(path);  /* clean up the temp file (no-op-ish if it's gone) */
}

/*
 * The semantic check: a loaded model must compute the same forward loss as the
 * original on the same input. Byte-equal weights guarantee it, but running an
 * actual forward also proves gpt_load rebuilt the layout/tensor views correctly,
 * which is what generation relies on.
 */
static void test_loaded_model_forward_matches(void **state) {
    (void)state;
    const char *path = "/tmp/test_gpt_io_forward.ckpt";

    GPTConfig cfg = tiny_config();
    /* tokens/targets in [0, vocab_size); T == max_seq_len is allowed */
    const int tokens[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    const int targets[8] = {2, 3, 4, 5, 6, 7, 8, 9};

    GPTModel original;
    gpt_init(&original, cfg, /*seed=*/7u);
    gpt_forward(&original, tokens, targets, /*B=*/1, /*T=*/8);
    float loss_original = original.loss;

    assert_int_equal(gpt_save(&original, path), 0);

    GPTModel restored;
    assert_int_equal(gpt_load(&restored, path), 0);
    gpt_forward(&restored, tokens, targets, 1, 8);

    /* identical weights => identical loss, exactly */
    assert_float_equal(restored.loss, loss_original, 0.0f);

    gpt_free(&original);
    gpt_free(&restored);
    remove(path);
}

/*
 * A missing file must be reported as -1, not crash. gpt_load on a path that
 * cannot be opened returns the error and allocates nothing.
 */
static void test_gpt_load_missing_file_fails(void **state) {
    (void)state;
    GPTModel m;
    assert_int_equal(gpt_load(&m, "/tmp/definitely_not_a_real_checkpoint_42.ckpt"), -1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_gpt_save_load_roundtrip),
        cmocka_unit_test(test_loaded_model_forward_matches),
        cmocka_unit_test(test_gpt_load_missing_file_fails),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
