/*
 * Tests for src/model/gpt.c — the full GPT model (forward + backward).
 *
 * The headline test is test_gpt_numerical_gradient: it checks the ENTIRE
 * backward pass at once by comparing every analytical parameter gradient
 * against a finite-difference ("numerical") estimate. If any single layer's
 * backward is wrong — a missing term, a sign flip, a forgotten 1/sqrt scale
 * in attention — the two gradient vectors diverge and the test fails loudly.
 * This is the same gold-standard check used for the individual layers in
 * test_layers.c, now applied to the whole model.
 *
 * Everything runs on TINY models (n_embd=8, a handful of layers/heads,
 * vocab 16-32, T=4) so the numerical sweep — which re-runs the forward pass
 * twice per parameter — finishes in well under a second.
 */

/* CMocka's required four includes, in this exact order (see test_layers.c
 * for the full explanation). */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <math.h>     /* isfinite, fabsf, sqrtf */
#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset */

#include "model/gpt.h"
#include "test_utils.h"  /* numerical_gradient */

/* ------------------------------------------------------------------ */
/* small helpers                                                      */
/* ------------------------------------------------------------------ */

/*
 * fill_random_ids (deterministic pseudo-random token ids) now lives in the
 * shared tests/helpers/test_utils.{h,c} — test_train.c needs the same helper,
 * so it was extracted there rather than copied.
 *
 * randn_test — one standard-normal sample (Box-Muller over the same LCG as
 * fill_random_ids). Used to give the gradient-check model LARGER weights than
 * gpt_init's training default. See amplify_weights below for why that matters.
 */
static float randn_test(unsigned int *s) {
    *s = (*s) * 1664525u + 1013904223u;
    float u1 = ((float)*s + 1.0f) / 4294967296.0f;  /* (0, 1] */
    *s = (*s) * 1664525u + 1013904223u;
    float u2 = ((float)*s) / 4294967296.0f;         /* [0, 1) */
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318531f * u2);
}

/*
 * amplify_weights — replace the model's weights with LARGER random values.
 *
 * Why: gpt_init uses the realistic training scale (std 0.02). At that scale
 * the attention dot-products are nearly zero, so softmax comes out almost
 * uniform and the attention math sits in a flat, near-linear regime. A
 * gradient check there is weak — it literally cannot feel a bug like a
 * missing 1/sqrt(head_dim) scale, because scaling near-zero scores barely
 * changes a near-uniform softmax.
 *
 * Pushing the weights to std ~0.5 makes the attention scores O(1), so softmax
 * is genuinely non-uniform and EVERY term in the forward/backward — the scale,
 * the causal mask, the head split — actually influences the loss. That is the
 * regime where a numerical gradient check has teeth.
 *
 * Layer-norm gamma/beta are reset to the identity (1 / 0) afterward so the
 * normalization stays well-behaved and activations don't explode across the
 * stacked layers.
 */
static void amplify_weights(GPTModel *model, unsigned int seed) {
    unsigned int s = seed ^ 0x9E3779B9u;
    for (int i = 0; i < model->num_params; i++) {
        model->params[i] = 0.5f * randn_test(&s);
    }
    int C = model->config.n_embd, L = model->config.n_layer;
    for (int i = 0; i < L * C; i++) {
        model->w.ln1_g[i] = 1.0f; model->w.ln1_b[i] = 0.0f;
        model->w.ln2_g[i] = 1.0f; model->w.ln2_b[i] = 0.0f;
    }
    for (int i = 0; i < C; i++) { model->w.lnf_g[i] = 1.0f; model->w.lnf_b[i] = 0.0f; }
}

/*
 * gpt_loss_ctx_t — closure state for the numerical gradient sweep.
 *
 * numerical_gradient calls our loss function many times, each time having
 * perturbed one element of the parameter vector. The loss function needs to
 * know which model/tokens/targets to run — that extra state rides along in
 * this struct, passed through the opaque `void *ctx` (the standard C stand-in
 * for a closure; see test_utils.h).
 */
typedef struct {
    GPTModel *model;
    const int *tokens;
    const int *targets;
    int B;
    int T;
} gpt_loss_ctx_t;

/*
 * gpt_loss — scalar loss as a function of the parameter vector.
 *
 * numerical_gradient hands us `x`, which IS model->params (it perturbs that
 * buffer in place). The model's tensor views all point into model->params, so
 * simply running the forward pass picks up the perturbed weights — we never
 * have to copy `x` anywhere. We return the resulting scalar loss.
 */
static float gpt_loss(const float *x, void *ctx_) {
    (void)x;  /* unused: x aliases model->params, which forward reads directly */
    gpt_loss_ctx_t *ctx = (gpt_loss_ctx_t *)ctx_;
    gpt_forward(ctx->model, ctx->tokens, ctx->targets, ctx->B, ctx->T);
    return ctx->model->loss;
}

/*
 * run_gradcheck — the shared body of the numerical-gradient tests.
 *
 * Builds a model of the given shape, makes up a reproducible batch of
 * tokens/targets, computes the analytical gradients with one
 * forward+backward, then estimates every gradient numerically and reports
 * how close the two are.
 *
 * We summarize the agreement with the RELATIVE L2 ERROR of the whole
 * gradient vector:
 *
 *     ||analytical - numerical||_2  /  (||analytical||_2 + ||numerical||_2)
 *
 * Using the whole-vector norm (instead of checking each element on its own)
 * is deliberately robust: float32 finite differences are noisy element by
 * element, but that noise averages out in the norm, while any REAL bug — a
 * whole tensor computed wrong — moves the norm well outside the noise floor.
 * A correct implementation lands around 1e-3; a broken one lands near 1.
 *
 * Returns that relative error; the caller asserts it is small.
 */
static float run_gradcheck(GPTConfig cfg, int B, unsigned int seed) {
    GPTModel model;
    gpt_init(&model, cfg, seed);
    amplify_weights(&model, seed);  /* push attention into a non-trivial regime */

    int T = cfg.max_seq_len;
    int N = B * T;

    int *tokens  = malloc((size_t)N * sizeof(int));
    int *targets = malloc((size_t)N * sizeof(int));
    unsigned int s = seed ^ 0xABCDEF01u;
    fill_random_ids(tokens,  N, cfg.vocab_size, &s);
    fill_random_ids(targets, N, cfg.vocab_size, &s);

    /* analytical gradients: one clean forward + backward */
    gpt_zero_grad(&model);
    gpt_forward(&model, tokens, targets, B, T);
    gpt_backward(&model, tokens, targets, B, T);

    /* numerical gradients: central differences over every parameter */
    int n = model.num_params;
    float *num = malloc((size_t)n * sizeof(float));
    gpt_loss_ctx_t ctx = {.model = &model, .tokens = tokens,
                          .targets = targets, .B = B, .T = T};
    numerical_gradient(num, gpt_loss, model.params, n, &ctx, 1e-3f);

    /* relative L2 error between the analytical grads and the numerical ones */
    double diff_sq = 0.0, ana_sq = 0.0, num_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)model.grads[i] - (double)num[i];
        diff_sq += d * d;
        ana_sq  += (double)model.grads[i] * (double)model.grads[i];
        num_sq  += (double)num[i] * (double)num[i];
    }
    float rel = (float)(sqrt(diff_sq) / (sqrt(ana_sq) + sqrt(num_sq) + 1e-12));

    free(tokens);
    free(targets);
    free(num);
    gpt_free(&model);
    return rel;
}

/* ------------------------------------------------------------------ */
/* tests                                                              */
/* ------------------------------------------------------------------ */

/*
 * gpt_num_params must equal a hand-derived parameter count.
 *
 * We re-derive the count with a SECOND, independent formula here so a bug in
 * gpt_num_params can't hide (testing a function against a copy of itself
 * proves nothing). We also pin one concrete tiny config to an exact number
 * a human can check by hand.
 */
static void test_gpt_num_params_formula(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 4, .vocab_size = 32, .n_embd = 8,
                     .n_head = 1, .n_layer = 1, .ff_dim = 32};

    int V = cfg.vocab_size, C = cfg.n_embd, L = cfg.n_layer,
        FF = cfg.ff_dim, maxT = cfg.max_seq_len;

    /* independent re-derivation, tensor by tensor */
    int expected =
        V * C            /* wte         */
      + maxT * C         /* wpe         */
      + L * ( 2 * C      /* ln1 g+b     */
            + C * (3*C)  /* qkv_w       */
            + C * C      /* attn_proj_w */
            + 2 * C      /* ln2 g+b     */
            + C * FF     /* fc_w        */
            + FF * C )   /* fc_proj_w   */
      + 2 * C            /* lnf g+b     */
      + C * V;           /* lm_head_w   */

    assert_int_equal(gpt_num_params(&cfg), expected);
    assert_int_equal(gpt_num_params(&cfg), 1360);  /* concrete hand-check */
}

/*
 * After gpt_init, every buffer and every tensor view must be non-NULL, and
 * num_params must match gpt_num_params. (The plan's "no null pointers"
 * check.)
 */
static void test_gpt_init_views_non_null(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 2, .ff_dim = 16};
    GPTModel model;
    gpt_init(&model, cfg, 7u);

    assert_int_equal(model.num_params, gpt_num_params(&cfg));
    assert_true(model.num_params > 0);
    assert_non_null(model.params);
    assert_non_null(model.grads);

    /* every parameter view points somewhere inside the block */
    assert_non_null(model.w.wte);
    assert_non_null(model.w.wpe);
    assert_non_null(model.w.ln1_g);
    assert_non_null(model.w.ln1_b);
    assert_non_null(model.w.qkv_w);
    assert_non_null(model.w.attn_proj_w);
    assert_non_null(model.w.ln2_g);
    assert_non_null(model.w.ln2_b);
    assert_non_null(model.w.fc_w);
    assert_non_null(model.w.fc_proj_w);
    assert_non_null(model.w.lnf_g);
    assert_non_null(model.w.lnf_b);
    assert_non_null(model.w.lm_head_w);

    /* and the gradient views mirror them */
    assert_non_null(model.g.wte);
    assert_non_null(model.g.lm_head_w);

    gpt_free(&model);

    /* after free the owned buffers are released and nulled */
    assert_null(model.params);
    assert_null(model.grads);
}

/*
 * A forward pass with targets must produce a finite, positive loss.
 *
 * Positive because cross-entropy = -log(probability) and a probability is
 * at most 1, so its negative log is at least 0 (and strictly positive unless
 * the model is perfectly certain, which a freshly-initialized one never is).
 * Finite because any NaN/Inf here means a numerical bug (e.g. a layernorm
 * dividing by zero, or a softmax overflow) that would destroy training.
 */
static void test_gpt_forward_loss_is_finite(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 2, .ff_dim = 16};
    GPTModel model;
    gpt_init(&model, cfg, 42u);

    int B = 2, T = 8, N = B * T;
    int *tokens  = malloc((size_t)N * sizeof(int));
    int *targets = malloc((size_t)N * sizeof(int));
    unsigned int s = 5u;
    fill_random_ids(tokens,  N, cfg.vocab_size, &s);
    fill_random_ids(targets, N, cfg.vocab_size, &s);

    gpt_forward(&model, tokens, targets, B, T);

    assert_true(isfinite(model.loss));
    assert_true(model.loss > 0.0f);

    free(tokens);
    free(targets);
    gpt_free(&model);
}

/*
 * Two forward passes with the same weights and the same input must produce
 * exactly the same loss. Catches accidental use of uninitialized scratch or
 * a hidden dependence on call order. (Bit-for-bit equality is fair here: the
 * same float ops in the same order give the same result.)
 */
static void test_gpt_forward_deterministic(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 1, .ff_dim = 16};
    GPTModel model;
    gpt_init(&model, cfg, 3u);

    int B = 2, T = 5, N = B * T;
    int *tokens  = malloc((size_t)N * sizeof(int));
    int *targets = malloc((size_t)N * sizeof(int));
    unsigned int s = 11u;
    fill_random_ids(tokens,  N, cfg.vocab_size, &s);
    fill_random_ids(targets, N, cfg.vocab_size, &s);

    gpt_forward(&model, tokens, targets, B, T);
    float loss1 = model.loss;
    gpt_forward(&model, tokens, targets, B, T);
    float loss2 = model.loss;

    assert_float_equal(loss1, loss2, 0.0);  /* exactly equal */

    free(tokens);
    free(targets);
    gpt_free(&model);
}

/*
 * After a backward pass the gradients must not be all zero — if they were,
 * the optimizer would have nothing to act on and training could never make
 * progress. We require the vast majority of parameters to have a nonzero
 * gradient. (A handful of exact zeros are legitimate: an embedding row for a
 * token that never appears in the batch gets no gradient at all.)
 */
static void test_gpt_backward_grads_not_all_zero(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 2, .ff_dim = 16};
    GPTModel model;
    gpt_init(&model, cfg, 123u);

    int B = 2, T = 8, N = B * T;
    int *tokens  = malloc((size_t)N * sizeof(int));
    int *targets = malloc((size_t)N * sizeof(int));
    unsigned int s = 77u;
    fill_random_ids(tokens,  N, cfg.vocab_size, &s);
    fill_random_ids(targets, N, cfg.vocab_size, &s);

    gpt_zero_grad(&model);
    gpt_forward(&model, tokens, targets, B, T);
    gpt_backward(&model, tokens, targets, B, T);

    int nonzero = 0;
    for (int i = 0; i < model.num_params; i++) {
        if (model.grads[i] != 0.0f) nonzero++;
    }
    /* expect almost everything to receive gradient signal */
    assert_true(nonzero > (model.num_params * 9) / 10);

    free(tokens);
    free(targets);
    gpt_free(&model);
}

/*
 * THE big one: analytical gradients vs numerical gradients, single layer,
 * single head. Small enough to read, exercises the full forward/backward
 * chain (embed -> ln -> attention -> ln -> ffn -> ln -> output -> loss).
 */
static void test_gpt_numerical_gradient_single(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 4, .vocab_size = 32, .n_embd = 8,
                     .n_head = 1, .n_layer = 1, .ff_dim = 32};
    float rel = run_gradcheck(cfg, /*B=*/2, /*seed=*/1234u);
    assert_true(rel < 2e-2f);
}

/*
 * Same check with MULTIPLE heads and MULTIPLE layers. This is the part that
 * proves the trickier wiring is right: splitting Q/K/V across heads, and the
 * way a layer's residual stream feeds the next layer through BOTH the skip
 * connection and the layer-norm (so its gradient must accumulate from two
 * paths — exactly the `+=` convention from layers.h).
 */
static void test_gpt_numerical_gradient_multihead(void **state) {
    (void)state;

    GPTConfig cfg = {.max_seq_len = 4, .vocab_size = 16, .n_embd = 8,
                     .n_head = 2, .n_layer = 2, .ff_dim = 16};
    float rel = run_gradcheck(cfg, /*B=*/2, /*seed=*/2024u);
    assert_true(rel < 2e-2f);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_gpt_num_params_formula),
        cmocka_unit_test(test_gpt_init_views_non_null),
        cmocka_unit_test(test_gpt_forward_loss_is_finite),
        cmocka_unit_test(test_gpt_forward_deterministic),
        cmocka_unit_test(test_gpt_backward_grads_not_all_zero),
        cmocka_unit_test(test_gpt_numerical_gradient_single),
        cmocka_unit_test(test_gpt_numerical_gradient_multihead),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
