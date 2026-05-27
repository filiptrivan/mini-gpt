/*
 * Tests for src/model/layers.c — CPU forward pass operations.
 *
 * Each test follows the CMocka pattern:
 *   1. Set up small, hand-computed inputs.
 *   2. Call the layer function.
 *   3. Assert the output matches the expected values.
 *
 * Test dimensions are intentionally tiny (length 4, 6, etc.) so we can
 * eyeball the expected values and so any failure prints a manageable diff.
 */

/* These four #includes are required by every CMocka test file, in this order:
 *   stdarg.h, stddef.h, setjmp.h — CMocka uses variadic args / jmp_buf
 *                                  internally; cmocka.h won't compile
 *                                  without these declarations being visible.
 *   cmocka.h                     — the test framework itself.
 * Always include them as a block; reordering or omitting any one breaks
 * the build on at least one platform.
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <math.h>   /* expf, isfinite — used in softmax tests */

#include "model/layers.h"

/*
 * residual_forward: out[i] = a[i] + b[i] for every i in [0, N).
 *
 * We use length 4 with hand-picked values so the expected output is obvious:
 *   a = [1, 2, 3, 4], b = [10, 20, 30, 40]  →  out = [11, 22, 33, 44].
 *
 * `static` makes this function private to this .c file — the test runner
 * registers it by pointer (see main() below), so no other file needs to
 * see its name, and `static` prevents accidental name clashes when more
 * test files are added.
 *
 * The `void **state` parameter is CMocka boilerplate: it lets tests share
 * setup/teardown state. We don't use it, so `(void)state;` silences the
 * unused-parameter warning, same trick as in layers.c.
 */
static void test_residual_simple(void **state) {
    (void)state;

    float a[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4]   = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {0};  /* {0} zero-initializes the whole array */

    residual_forward(out, a, b, 4);

    /* assert_float_equal(actual, expected, epsilon)
     * Floats can't be compared with `==` because rounding (e.g. 0.1 + 0.2
     * is not exactly 0.3 in IEEE 754). Epsilon = how close is "close enough."
     * 1e-6 is tight enough to catch real bugs but loose enough to ignore
     * harmless rounding noise. For exact integer-valued floats like ours,
     * any positive epsilon would work; we keep 1e-6 as the project default. */
    assert_float_equal(out[0], 11.0f, 1e-6);
    assert_float_equal(out[1], 22.0f, 1e-6);
    assert_float_equal(out[2], 33.0f, 1e-6);
    assert_float_equal(out[3], 44.0f, 1e-6);
}

/*
 * gelu_forward: out[i] = 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3))).
 *
 * We test four characteristic inputs:
 *   x = 0    → exactly 0   (leading x term zeros the whole expression)
 *   x = 1    → ~0.841192   (precomputed from the formula above)
 *   x = -1   → ~-0.158808  (asymmetric: small negative output, not 0)
 *   x = 2    → ~1.954598   (large positive: output approaches x)
 *
 * Tolerance is 1e-4 (not 1e-6) because the expected values are themselves
 * rounded to six decimals — a tighter epsilon would fail on the rounding
 * we did when writing the test, not on any real bug.
 */
static void test_gelu_known_values(void **state) {
    (void)state;

    float in[4]  = {0.0f, 1.0f, -1.0f, 2.0f};
    float out[4] = {0};

    gelu_forward(out, in, 4);

    assert_float_equal(out[0],  0.0f,       1e-4);
    assert_float_equal(out[1],  0.841192f,  1e-4);
    assert_float_equal(out[2], -0.158808f,  1e-4);
    assert_float_equal(out[3],  1.954598f,  1e-4);
}

/*
 * matmul_forward, square 2×2 @ 2×2 case (the textbook example).
 *
 *   a = | 1 2 |    b = | 5 6 |
 *       | 3 4 |        | 7 8 |
 *
 *   a @ b = | 1*5 + 2*7    1*6 + 2*8 |   = | 19 22 |
 *           | 3*5 + 4*7    3*6 + 4*8 |     | 43 50 |
 *
 * Stored flat (row-major): a = [1,2,3,4], b = [5,6,7,8], out = [19,22,43,50].
 */
static void test_matmul_square_2x2(void **state) {
    (void)state;

    float a[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4]   = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {0};

    /* M=2 (rows of a), K=2 (cols of a = rows of b), N=2 (cols of b) */
    matmul_forward(out, a, b, 2, 2, 2);

    assert_float_equal(out[0], 19.0f, 1e-6);
    assert_float_equal(out[1], 22.0f, 1e-6);
    assert_float_equal(out[2], 43.0f, 1e-6);
    assert_float_equal(out[3], 50.0f, 1e-6);
}

/*
 * matmul_forward, non-square 2×3 @ 3×2 = 2×2 case.
 *
 * This catches a class of bugs that the square case can't: code that
 * accidentally hardcodes M, K, or N to the same value (e.g. uses N where
 * K was meant). With distinct M=2, K=3, N=2 such a mistake would show up
 * immediately as an index-out-of-bounds or wrong-value failure.
 *
 *   a = | 1 2 3 |    b = |  7  8 |    a @ b = |  58  64 |
 *       | 4 5 6 |        |  9 10 |            | 139 154 |
 *                        | 11 12 |
 *
 *   out[0][0] = 1*7  + 2*9  + 3*11 = 7  + 18 + 33 =  58
 *   out[0][1] = 1*8  + 2*10 + 3*12 = 8  + 20 + 36 =  64
 *   out[1][0] = 4*7  + 5*9  + 6*11 = 28 + 45 + 66 = 139
 *   out[1][1] = 4*8  + 5*10 + 6*12 = 32 + 50 + 72 = 154
 */
static void test_matmul_nonsquare_2x3_3x2(void **state) {
    (void)state;

    float a[6]   = {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f};
    float b[6]   = {7.0f,  8.0f,
                    9.0f, 10.0f,
                   11.0f, 12.0f};
    float out[4] = {0};

    matmul_forward(out, a, b, 2, 3, 2);  /* M=2, K=3, N=2 */

    assert_float_equal(out[0],  58.0f, 1e-6);
    assert_float_equal(out[1],  64.0f, 1e-6);
    assert_float_equal(out[2], 139.0f, 1e-6);
    assert_float_equal(out[3], 154.0f, 1e-6);
}

/*
 * softmax_forward — single-row known-values test.
 *
 * Input  = [1, 2, 3, 4]
 * exp(in)= [2.71828, 7.38906, 20.0855, 54.5982]
 * sum    = 84.79100
 * probs  = [0.0320586, 0.0871443, 0.2368828, 0.6439142]
 *
 * Tolerance 1e-5 leaves plenty of room for accumulated float rounding
 * while still catching genuine formula bugs (off-by-one in indexing,
 * wrong axis, missing division, etc.).
 */
static void test_softmax_known_values(void **state) {
    (void)state;

    float in[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};

    softmax_forward(out, in, 1, 4);  /* N=1 row, V=4 columns */

    assert_float_equal(out[0], 0.0320586f, 1e-5);
    assert_float_equal(out[1], 0.0871443f, 1e-5);
    assert_float_equal(out[2], 0.2368828f, 1e-5);
    assert_float_equal(out[3], 0.6439142f, 1e-5);
}

/*
 * softmax_forward — the output of every row must sum to exactly 1.
 *
 * This is a fundamental invariant: softmax produces a probability
 * distribution. We test it with mixed-sign inputs (including negatives)
 * to confirm the formula isn't accidentally relying on positivity.
 *
 * Tolerance 1e-6 for the sum because we're summing only 5 elements;
 * cumulative rounding error stays tiny.
 */
static void test_softmax_sums_to_one(void **state) {
    (void)state;

    float in[5]  = {-2.0f, -0.5f, 0.1f, 1.7f, 3.0f};
    float out[5] = {0};

    softmax_forward(out, in, 1, 5);

    float sum = 0.0f;
    for (int i = 0; i < 5; i++) {
        sum += out[i];
        /* Each probability must also be in (0, 1] — catches NaN, inf,
         * and negative values produced by a broken implementation. */
        assert_true(out[i] > 0.0f);
        assert_true(out[i] <= 1.0f);
    }
    assert_float_equal(sum, 1.0f, 1e-6);
}

/*
 * softmax_forward — the numerical-stability test (this is THE test
 * that distinguishes a correct implementation from a naive one).
 *
 * All four inputs are 1000. A naive implementation computes expf(1000),
 * which overflows to +inf in IEEE 754 single-precision; the result is
 * inf/inf = NaN and every assertion below would fail.
 *
 * A correct implementation subtracts max(row) = 1000 first, so it
 * actually computes expf(0) = 1 four times. Sum = 4, output = 0.25 each.
 *
 * isfinite() returns false for NaN AND infinity — both are catastrophic
 * here, so we check both at once.
 */
static void test_softmax_numerical_stability(void **state) {
    (void)state;

    float in[4]  = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    float out[4] = {0};

    softmax_forward(out, in, 1, 4);

    for (int i = 0; i < 4; i++) {
        assert_true(isfinite(out[i]));   /* fails on NaN or inf */
        assert_float_equal(out[i], 0.25f, 1e-6);
    }
}

/*
 * softmax_forward — multi-row independence.
 *
 * Each row must be softmaxed independently of the others. This catches
 * a bug where the implementation accidentally takes the max across the
 * whole matrix instead of per-row, or accumulates the denominator across
 * rows. Two rows with very different scales (one all 0s, one all 1000s)
 * would both yield uniform [0.5, 0.5] independently — easy to verify.
 */
static void test_softmax_multi_row(void **state) {
    (void)state;

    float in[4]  = { 0.0f,    0.0f,       /* row 0: uniform → uniform */
                     1000.0f, 1000.0f };  /* row 1: uniform → uniform, despite huge scale */
    float out[4] = {0};

    softmax_forward(out, in, 2, 2);  /* N=2 rows, V=2 cols */

    assert_float_equal(out[0], 0.5f, 1e-6);
    assert_float_equal(out[1], 0.5f, 1e-6);
    assert_float_equal(out[2], 0.5f, 1e-6);
    assert_float_equal(out[3], 0.5f, 1e-6);
}

/*
 * layernorm_forward — the core normalization invariant.
 *
 * After layernorm with gamma=[1,1,1,1] and beta=[0,0,0,0] (the "identity"
 * affine), every row's output should have mean ≈ 0 and variance ≈ 1
 * regardless of the input row's mean and variance. This is THE thing
 * layernorm is supposed to do.
 *
 * Input [1, 2, 3, 4] has mean 2.5 and variance 1.25. After normalization
 * the output mean should drop to 0 and variance should rise to 1.
 *
 * Tolerance 1e-4 accommodates the ε in the denominator (which makes
 * variance 1.00001-ish, not exactly 1) and accumulated float rounding.
 */
static void test_layernorm_normalizes_unit_variance(void **state) {
    (void)state;

    float in[4]    = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4]   = {0};

    layernorm_forward(out, in, gamma, beta, 1, 4);  /* N=1 row, C=4 features */

    /* Compute mean of output */
    float mean = 0.0f;
    for (int i = 0; i < 4; i++) mean += out[i];
    mean /= 4.0f;

    /* Compute variance of output (uses the mean we just computed) */
    float var = 0.0f;
    for (int i = 0; i < 4; i++) {
        float d = out[i] - mean;
        var += d * d;
    }
    var /= 4.0f;

    assert_float_equal(mean, 0.0f, 1e-4);
    assert_float_equal(var,  1.0f, 1e-4);
}

/*
 * layernorm_forward — per-element gamma and beta indexing.
 *
 * Input [-1, 1, -1, 1] has mean 0 and variance 1 already. So x_hat
 * ≈ [-1, 1, -1, 1] (the ε in the denominator makes it 0.999995-ish,
 * not exactly ±1, but within our tolerance).
 *
 * With gamma = [1, 2, 3, 4] and beta = [0.5, 1.5, 2.5, 3.5]:
 *   out[0] = 1 * (-1) + 0.5 = -0.5
 *   out[1] = 2 * ( 1) + 1.5 =  3.5
 *   out[2] = 3 * (-1) + 2.5 = -0.5
 *   out[3] = 4 * ( 1) + 3.5 =  7.5
 *
 * If the implementation used gamma[0] for all elements (a common bug),
 * out[1..3] would be very different and the test would catch it.
 */
static void test_layernorm_per_element_gamma_beta(void **state) {
    (void)state;

    float in[4]    = {-1.0f, 1.0f, -1.0f, 1.0f};
    float gamma[4] = { 1.0f, 2.0f,  3.0f, 4.0f};
    float beta[4]  = { 0.5f, 1.5f,  2.5f, 3.5f};
    float out[4]   = {0};

    layernorm_forward(out, in, gamma, beta, 1, 4);

    assert_float_equal(out[0], -0.5f, 1e-3);
    assert_float_equal(out[1],  3.5f, 1e-3);
    assert_float_equal(out[2], -0.5f, 1e-3);
    assert_float_equal(out[3],  7.5f, 1e-3);
}

/*
 * layernorm_forward — constant-row safety (the epsilon test).
 *
 * Input [3, 3, 3, 3] has variance exactly 0. Without ε in the
 * denominator we'd divide by zero and produce NaN/inf — destroying
 * training the moment any layer's output happens to flatten out.
 *
 * With ε = 1e-5 the denominator becomes sqrt(1e-5) ≈ 0.003162; every
 * numerator is (3 - 3) = 0, so x_hat = 0/anything = 0, and out = gamma*0 + beta
 * = beta. With beta = [0, 0, 0, 0] we expect all zeros and no NaN/inf.
 */
static void test_layernorm_constant_row_safety(void **state) {
    (void)state;

    float in[4]    = {3.0f, 3.0f, 3.0f, 3.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4]   = {0};

    layernorm_forward(out, in, gamma, beta, 1, 4);

    for (int i = 0; i < 4; i++) {
        assert_true(isfinite(out[i]));     /* fails on NaN or inf */
        assert_float_equal(out[i], 0.0f, 1e-6);
    }
}

/*
 * embed_forward — single batch, hand-computed.
 *
 * Tiny example with vocab_size=3, max_seq_len=2, C=2, B=1, T=2:
 *
 *   wte (token embeddings, vocab_size × C):
 *     row 0 = [10, 20]
 *     row 1 = [30, 40]
 *     row 2 = [50, 60]
 *
 *   wpe (position embeddings, max_seq_len × C):
 *     row 0 = [1, 2]
 *     row 1 = [3, 4]
 *
 *   tokens (B × T) = [[2, 0]]
 *
 *   out[0, 0, :] = wte[2] + wpe[0] = [50,60] + [1,2] = [51, 62]
 *   out[0, 1, :] = wte[0] + wpe[1] = [10,20] + [3,4] = [13, 24]
 *
 *   out flat = [51, 62, 13, 24]
 */
static void test_embed_single_batch(void **state) {
    (void)state;

    /* Token IDs are integers, not floats — the embedding table is what
     * holds the floats. Each token at position (b, t) just selects which
     * row of the table to read. */
    int tokens[2] = {2, 0};

    float wte[6] = {10.0f, 20.0f,   /* row 0 */
                    30.0f, 40.0f,   /* row 1 */
                    50.0f, 60.0f};  /* row 2 */

    float wpe[4] = {1.0f, 2.0f,     /* position 0 */
                    3.0f, 4.0f};    /* position 1 */

    float out[4] = {0};

    embed_forward(out, tokens, wte, wpe, 1, 2, 2);  /* B=1, T=2, C=2 */

    assert_float_equal(out[0], 51.0f, 1e-6);
    assert_float_equal(out[1], 62.0f, 1e-6);
    assert_float_equal(out[2], 13.0f, 1e-6);
    assert_float_equal(out[3], 24.0f, 1e-6);
}

/*
 * embed_forward — multi-batch, with the same token appearing at
 * different positions.
 *
 * Same wte and wpe as the single-batch test. Now B=2:
 *
 *   tokens = [[2, 0],     ← batch 0
 *             [1, 1]]     ← batch 1: same token (1) at both positions
 *
 *   batch 0:  out[0,0]=wte[2]+wpe[0]=[51,62], out[0,1]=wte[0]+wpe[1]=[13,24]
 *   batch 1:  out[1,0]=wte[1]+wpe[0]=[31,42], out[1,1]=wte[1]+wpe[1]=[33,44]
 *
 * Critical assertion: out[1,0] != out[1,1] even though both are token 1.
 * Why? Because position 0 and position 1 add different wpe rows. If a
 * broken implementation skipped wpe, both outputs would be [30,40] and
 * the test would fail.
 */
static void test_embed_multi_batch(void **state) {
    (void)state;

    int tokens[4] = {2, 0,    /* batch 0 */
                     1, 1};   /* batch 1 */

    float wte[6] = {10.0f, 20.0f,
                    30.0f, 40.0f,
                    50.0f, 60.0f};

    float wpe[4] = {1.0f, 2.0f,
                    3.0f, 4.0f};

    float out[8] = {0};

    embed_forward(out, tokens, wte, wpe, 2, 2, 2);  /* B=2, T=2, C=2 */

    /* Batch 0 — same as the single-batch test */
    assert_float_equal(out[0], 51.0f, 1e-6);
    assert_float_equal(out[1], 62.0f, 1e-6);
    assert_float_equal(out[2], 13.0f, 1e-6);
    assert_float_equal(out[3], 24.0f, 1e-6);

    /* Batch 1 — token 1 at both positions, position embedding makes them differ */
    assert_float_equal(out[4], 31.0f, 1e-6);
    assert_float_equal(out[5], 42.0f, 1e-6);
    assert_float_equal(out[6], 33.0f, 1e-6);
    assert_float_equal(out[7], 44.0f, 1e-6);
}

/*
 * main — the test runner entry point.
 *
 * cmocka_unit_test() wraps each test function into a "CMUnitTest" struct
 * with its name and a pointer to the function. cmocka_run_group_tests()
 * runs every test in the array, prints pass/fail summary, and returns
 * the count of failed tests (0 means success — matches the Unix convention
 * for process exit codes, where 0 = OK).
 */
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_residual_simple),
        cmocka_unit_test(test_gelu_known_values),
        cmocka_unit_test(test_matmul_square_2x2),
        cmocka_unit_test(test_matmul_nonsquare_2x3_3x2),
        cmocka_unit_test(test_softmax_known_values),
        cmocka_unit_test(test_softmax_sums_to_one),
        cmocka_unit_test(test_softmax_numerical_stability),
        cmocka_unit_test(test_softmax_multi_row),
        cmocka_unit_test(test_layernorm_normalizes_unit_variance),
        cmocka_unit_test(test_layernorm_per_element_gamma_beta),
        cmocka_unit_test(test_layernorm_constant_row_safety),
        cmocka_unit_test(test_embed_single_batch),
        cmocka_unit_test(test_embed_multi_batch),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
