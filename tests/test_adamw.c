/*
 * Tests for src/optimizer/adamw.c — the AdamW optimizer.
 *
 * The optimizer is the piece that turns gradients into weight updates, so we
 * test it in three complementary ways (the three checks called for in PLAN.md):
 *
 *   1. MOMENTS ACCUMULATE CORRECTLY (test_adamw_moments_accumulate)
 *      Feed a KNOWN gradient and check m and v against values computed by hand.
 *      This pins the exact recurrence — beta1/beta2 mixing — with no guessing.
 *
 *   2. A STEP REDUCES LOSS (test_adamw_step_reduces_loss)
 *      Set up a trivial convex problem (a parabola bowl whose minimum we know)
 *      and run many steps. If the optimizer is correct, the loss must fall and
 *      the parameters must converge to the known minimum. This is the optimizer
 *      analogue of the model's "acid test".
 *
 *   3. WEIGHT DECAY SHRINKS WEIGHTS (test_adamw_weight_decay_shrinks)
 *      With a ZERO gradient and weight_decay > 0, the only thing acting on a
 *      weight is the decay term, so it must shrink toward zero by exactly the
 *      factor (1 - lr*weight_decay). An exact arithmetic check.
 *
 * Everything uses TINY parameter counts (1 to 4) so the hand-computed
 * expectations stay readable and the whole file runs in milliseconds.
 */

/* CMocka's required four includes, in this exact order (see test_layers.c
 * for the full explanation). */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <math.h>     /* sqrtf, fabsf, isfinite */

#include "optimizer/adamw.h"

/* ------------------------------------------------------------------ */
/* tests                                                              */
/* ------------------------------------------------------------------ */

/*
 * After adamw_init the moment buffers must exist and start at zero, t must be
 * 0, and the hyperparameters must be stored verbatim. Adam's bias correction
 * is only valid if m and v genuinely start from zero, so this is a real
 * correctness precondition, not just a smoke test.
 */
static void test_adamw_init_state(void **state) {
    (void)state;

    AdamW opt;
    adamw_init(&opt, /*num_params=*/4,
               /*lr=*/1e-3f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.01f);

    assert_int_equal(opt.num_params, 4);
    assert_int_equal(opt.t, 0);
    assert_non_null(opt.m);
    assert_non_null(opt.v);

    /* hyperparameters copied in unchanged */
    assert_float_equal(opt.lr, 1e-3f, 0.0);
    assert_float_equal(opt.beta1, 0.9f, 0.0);
    assert_float_equal(opt.beta2, 0.999f, 0.0);
    assert_float_equal(opt.eps, 1e-8f, 0.0);
    assert_float_equal(opt.weight_decay, 0.01f, 0.0);

    /* moments start at zero */
    for (int i = 0; i < 4; i++) {
        assert_float_equal(opt.m[i], 0.0f, 0.0);
        assert_float_equal(opt.v[i], 0.0f, 0.0);
    }

    adamw_free(&opt);

    /* after free the owned buffers are released and nulled */
    assert_null(opt.m);
    assert_null(opt.v);
}

/*
 * The exact-value test: with one parameter and a constant gradient of 2.0, the
 * first two updates of m and v are short enough to compute by hand.
 *
 *   g = 2.0, beta1 = 0.9, beta2 = 0.999  (m, v start at 0)
 *
 *   step 1:  m = 0.9*0   + 0.1*2.0       = 0.2
 *            v = 0.999*0 + 0.001*(2.0^2) = 0.004
 *   step 2:  m = 0.9*0.2   + 0.1*2.0          = 0.38
 *            v = 0.999*0.004 + 0.001*(2.0^2)  = 0.007996
 *
 * If beta1 and beta2 were swapped, or a (1-beta) factor were dropped, these
 * numbers would move far outside the tolerance.
 */
static void test_adamw_moments_accumulate(void **state) {
    (void)state;

    AdamW opt;
    adamw_init(&opt, /*num_params=*/1,
               /*lr=*/1e-3f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.0f);

    float params[1] = {0.0f};
    const float grads[1] = {2.0f};

    adamw_step(&opt, params, grads);
    assert_int_equal(opt.t, 1);
    assert_float_equal(opt.m[0], 0.2f,   1e-6f);
    assert_float_equal(opt.v[0], 0.004f, 1e-6f);

    adamw_step(&opt, params, grads);
    assert_int_equal(opt.t, 2);
    assert_float_equal(opt.m[0], 0.38f,     1e-6f);
    assert_float_equal(opt.v[0], 0.007996f, 1e-6f);

    adamw_free(&opt);
}

/*
 * On the very FIRST step, AdamW's bias correction makes the update magnitude
 * (ignoring the tiny eps and weight decay) equal to lr, regardless of how big
 * the gradient is. Reason:
 *
 *   m_hat = m/(1-beta1) = [(1-beta1)*g]/(1-beta1) = g
 *   v_hat = v/(1-beta2) = [(1-beta2)*g^2]/(1-beta2) = g^2
 *   update = lr * m_hat/sqrt(v_hat) = lr * g/|g| = lr * sign(g)
 *
 * So starting from p = 0 with g = 5.0 and lr = 0.1, the parameter must land at
 * about -0.1. This is a sharp, well-known property — a good catch-all that the
 * bias correction and the sqrt(v) scaling are both wired correctly.
 */
static void test_adamw_first_step_is_lr_sized(void **state) {
    (void)state;

    AdamW opt;
    adamw_init(&opt, /*num_params=*/1,
               /*lr=*/0.1f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.0f);

    float params[1] = {0.0f};
    const float grads[1] = {5.0f};

    adamw_step(&opt, params, grads);

    /* moved one lr-sized step downhill (against the positive gradient) */
    assert_float_equal(params[0], -0.1f, 1e-4f);

    adamw_free(&opt);
}

/*
 * The optimizer's job, demonstrated on a problem we can check exactly: a
 * quadratic "bowl"
 *
 *     L(p) = sum_i (p_i - target_i)^2,   so   dL/dp_i = 2*(p_i - target_i)
 *
 * whose unique minimum is p = target. Starting from all-zeros and running many
 * AdamW steps, the loss must drop sharply and every parameter must approach its
 * target. If the update direction or scaling were wrong, the loss would stall
 * or diverge instead.
 */
static void test_adamw_step_reduces_loss(void **state) {
    (void)state;

    enum { N = 4 };
    const float target[N] = {1.0f, -2.0f, 0.5f, 3.0f};

    AdamW opt;
    adamw_init(&opt, /*num_params=*/N,
               /*lr=*/0.1f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.0f);

    float params[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grads[N];

    /* loss BEFORE any optimization */
    float loss_before = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = params[i] - target[i];
        loss_before += d * d;
    }

    /* gradient descent with AdamW: recompute grads, take a step, repeat */
    for (int step = 0; step < 400; step++) {
        for (int i = 0; i < N; i++) {
            grads[i] = 2.0f * (params[i] - target[i]);
        }
        adamw_step(&opt, params, grads);
    }

    /* loss AFTER optimization */
    float loss_after = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = params[i] - target[i];
        loss_after += d * d;
    }

    assert_true(isfinite(loss_after));
    assert_true(loss_after < loss_before);   /* the headline guarantee */
    assert_true(loss_after < 1e-4f);          /* and it actually converged */

    /* every parameter reached (close to) its target */
    for (int i = 0; i < N; i++) {
        assert_float_equal(params[i], target[i], 1e-2f);
    }

    adamw_free(&opt);
}

/*
 * Decoupled weight decay, isolated: with a ZERO gradient the moments stay zero,
 * so m_hat/(sqrt(v_hat)+eps) = 0 and the ONLY term left is the decay pull:
 *
 *     params[i] -= lr * weight_decay * params[i]
 *     => params[i] *= (1 - lr*weight_decay)
 *
 * With lr = 0.1 and weight_decay = 0.5 the factor is 0.95, so a weight of 10.0
 * must become exactly 9.5 after one step and shrink geometrically after that.
 * This both proves decay is applied and proves it is applied the AdamW way
 * (straight to the weight), not folded into the gradient.
 */
static void test_adamw_weight_decay_shrinks(void **state) {
    (void)state;

    AdamW opt;
    adamw_init(&opt, /*num_params=*/1,
               /*lr=*/0.1f, /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.5f);

    float params[1] = {10.0f};
    const float grads[1] = {0.0f};   /* no gradient signal at all */

    adamw_step(&opt, params, grads);
    assert_float_equal(params[0], 9.5f, 1e-4f);   /* 10 * (1 - 0.1*0.5) */

    adamw_step(&opt, params, grads);
    assert_float_equal(params[0], 9.025f, 1e-4f); /* 9.5 * 0.95 */

    /* moments untouched by a zero gradient */
    assert_float_equal(opt.m[0], 0.0f, 0.0);
    assert_float_equal(opt.v[0], 0.0f, 0.0);

    adamw_free(&opt);
}

/*
 * Weight decay should pull weights SMALLER than the same run without it. Two
 * identical optimizers, identical (nonzero, noisy) gradients, differing only in
 * weight_decay: after many steps the decayed parameters must have a strictly
 * smaller magnitude. This is the practical effect decay exists for.
 */
static void test_adamw_weight_decay_vs_none(void **state) {
    (void)state;

    enum { N = 3 };

    AdamW with_decay, no_decay;
    adamw_init(&with_decay, N, 0.05f, 0.9f, 0.999f, 1e-8f, /*wd=*/0.1f);
    adamw_init(&no_decay,   N, 0.05f, 0.9f, 0.999f, 1e-8f, /*wd=*/0.0f);

    float p_decay[N] = {0.0f, 0.0f, 0.0f};
    float p_plain[N] = {0.0f, 0.0f, 0.0f};

    /* drive both toward the same nonzero targets with the same gradients */
    const float target[N] = {2.0f, -1.5f, 4.0f};
    for (int step = 0; step < 300; step++) {
        float g_decay[N], g_plain[N];
        for (int i = 0; i < N; i++) {
            g_decay[i] = 2.0f * (p_decay[i] - target[i]);
            g_plain[i] = 2.0f * (p_plain[i] - target[i]);
        }
        adamw_step(&with_decay, p_decay, g_decay);
        adamw_step(&no_decay,   p_plain, g_plain);
    }

    /* compare overall magnitude (L2 norm) of the two parameter vectors */
    float norm_decay = 0.0f, norm_plain = 0.0f;
    for (int i = 0; i < N; i++) {
        norm_decay += p_decay[i] * p_decay[i];
        norm_plain += p_plain[i] * p_plain[i];
    }
    assert_true(norm_decay < norm_plain);

    adamw_free(&with_decay);
    adamw_free(&no_decay);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_adamw_init_state),
        cmocka_unit_test(test_adamw_moments_accumulate),
        cmocka_unit_test(test_adamw_first_step_is_lr_sized),
        cmocka_unit_test(test_adamw_step_reduces_loss),
        cmocka_unit_test(test_adamw_weight_decay_shrinks),
        cmocka_unit_test(test_adamw_weight_decay_vs_none),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
