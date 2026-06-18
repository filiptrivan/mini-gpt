/*
 * Tests for the CPU training loop — the project's "acid test".
 *
 * Everything built in Tasks 5-8 only matters if, wired together, it actually
 * LEARNS. This test composes the full pipeline by hand on a tiny model:
 *
 *     gpt_zero_grad -> gpt_forward -> gpt_backward -> adamw_step
 *
 * and trains it repeatedly on ONE fixed batch (the classic "can it overfit a
 * single batch?" sanity check). A correct pipeline drives the loss steadily
 * down; if any piece were wrong — a sign flip in a gradient, a bad AdamW
 * update, a stale activation — the loss would stall, rise, or go NaN instead.
 *
 * This is the same loop src/train.c runs; here we run it on an in-memory batch
 * so the check is deterministic and needs no data file. The tiny dimensions
 * (n_embd=8, vocab=16, T=8) keep it to a few milliseconds.
 */

/* CMocka's required four includes, in this exact order (see test_layers.c
 * for the full explanation). */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <math.h>     /* isfinite */
#include <stdlib.h>   /* malloc, free */

#include "model/gpt.h"
#include "optimizer/adamw.h"
#include "test_utils.h"  /* fill_random_ids */

/*
 * The acid test: train a tiny model on one fixed batch and require the loss to
 * fall substantially. loss_first is measured before any update; loss_last is
 * measured after 100 AdamW steps with the fully-updated weights.
 */
static void test_train_overfits_single_batch(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 2, .ff_dim = 16};
    GPTModel model;
    gpt_init(&model, cfg, 1234u);

    /* lr deliberately a bit high (1e-2) and weight_decay 0: we WANT this run
     * to overfit the single batch as fast as possible — that's the signal. */
    AdamW opt;
    adamw_init(&opt, model.num_params,
               /*lr=*/1e-2f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.0f);

    int B = 2, T = 8, N = B * T;
    int *tokens  = malloc((size_t)N * sizeof(int));
    int *targets = malloc((size_t)N * sizeof(int));
    unsigned int s = 42u;
    fill_random_ids(tokens,  N, cfg.vocab_size, &s);
    fill_random_ids(targets, N, cfg.vocab_size, &s);

    /* loss BEFORE any optimization */
    gpt_forward(&model, tokens, targets, B, T);
    float loss_first = model.loss;
    assert_true(isfinite(loss_first));
    assert_true(loss_first > 0.0f);

    /* train on the SAME batch every step */
    for (int step = 0; step < 100; step++) {
        gpt_zero_grad(&model);
        gpt_forward(&model, tokens, targets, B, T);
        gpt_backward(&model, tokens, targets, B, T);
        adamw_step(&opt, model.params, model.grads);
        assert_true(isfinite(model.loss));   /* never blow up mid-training */
    }

    /* loss AFTER training, measured with the updated weights */
    gpt_forward(&model, tokens, targets, B, T);
    float loss_last = model.loss;

    assert_true(isfinite(loss_last));
    assert_true(loss_last < loss_first);          /* the headline guarantee */
    assert_true(loss_last < 0.5f * loss_first);   /* and a real, big drop */

    free(tokens);
    free(targets);
    adamw_free(&opt);
    gpt_free(&model);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_train_overfits_single_batch),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
