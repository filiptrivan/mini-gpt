/*
 * Tests for the CUDA layer kernels (Task 9) — matmul and layernorm.
 *
 * THE PATTERN for every CUDA test in this project:
 *   1. Build small host inputs.
 *   2. Run the CPU function from src/model/layers.c — this is the ORACLE.
 *      It is already proven correct by the Task 5/6 numerical-gradient tests,
 *      so we trust its output as the reference "right answer".
 *   3. Run the CUDA wrapper on the SAME host inputs.
 *   4. assert_float_equal every element: CUDA must match CPU within tolerance.
 *
 * We do NOT redo numerical gradient checks here — the math was already
 * validated on the CPU. These tests only ask one question: "does the GPU
 * compute the same thing the CPU does?"
 *
 * This file is compiled by nvcc (as C++) and runs only on a machine with a
 * CUDA toolchain + GPU (Google Colab). It is excluded from the Mac CPU build
 * by an `if(ENABLE_CUDA)` guard in tests/CMakeLists.txt.
 */

/* CMocka's required four includes, same block/order as every test file.
 *
 * cmocka.h is wrapped in extern "C" here (the other, plain-C test files don't
 * need this). Reason: this file is compiled by nvcc as C++, and the cmocka.h
 * that ships in Colab's libcmocka-dev does NOT carry its own extern "C"
 * guards. Without the wrapper, C++ name-mangles cmocka's functions
 * (assert_float_equal, cmocka_run_group_tests, ...) and they fail to link
 * against the C libcmocka with "undefined reference". Same fix we apply to
 * model/layers.h just below. The stdarg/stddef/setjmp headers must come first
 * (cmocka.h uses va_list/size_t/jmp_buf) and stay as normal includes. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
extern "C" {
#include <cmocka.h>
}

#include <stdlib.h>   /* malloc, calloc, free */
#include <math.h>     /* logf — inline cross-entropy oracle */

/* The CPU oracles plus the shared test helpers. layers.h, gpt.h and
 * test_utils.h all carry their own extern "C" guards, so a plain include gives
 * the correct C linkage to match the C-built libraries — no need to wrap them
 * here (unlike cmocka.h above, which we can't edit). gpt.h is needed for
 * attention_forward/attention_backward, the oracles for the attention kernels
 * (that math lives in gpt.c, not layers.c); test_utils.h supplies
 * fill_random_ids for reproducible token batches. */
#include "model/layers.h"
#include "model/gpt.h"
#include "test_utils.h"

/* The CUDA wrappers under test. This header already declares them extern "C". */
#include "cuda/cuda_layers.cuh"

/* Tolerance for CPU-vs-GPU comparisons.
 *
 * The two implementations sum the same products in a DIFFERENT order (the CPU
 * accumulates a row at a time; the tiled GPU kernel accumulates tile by tile),
 * and float addition is not associative, so the results differ by a few units
 * in the last place. 1e-4 absolute comfortably covers that round-off for our
 * small test magnitudes while still catching any real algorithmic bug (which
 * would be off by far more than 1e-4). PLAN.md targets ~1e-5; 1e-4 is the
 * honest bound once different summation orders are in play. */
#define CMP_TOL 1e-4

/*
 * fill_pseudo_random — fill a float array with deterministic values in
 * roughly [-1, 1). It rolls a tiny linear-congruential generator (the same
 * Numerical Recipes multiplier/increment used elsewhere in the project) so the
 * output is fixed on every machine and every run. Note it maps the bits to a
 * float, so the sequence is NOT the same as test_utils's integer
 * fill_random_ids — reproducibility, not cross-helper equality, is the point:
 * when a remote (Colab) test fails we can reason about exactly what was fed in.
 *
 *   buf   — destination, length n (caller-allocated)
 *   n     — element count
 *   state — pointer to the generator state, advanced in place
 */
static void fill_pseudo_random(float *buf, int n, unsigned int *state) {
    for (int i = 0; i < n; i++) {
        /* Numerical Recipes LCG step. unsigned overflow is defined to wrap. */
        *state = (*state) * 1664525u + 1013904223u;
        /* Take the top bits (better distributed than the low bits) and map the
         * 24-bit mantissa range into [0,1), then shift to about [-1, 1). */
        float u = (float)((*state) >> 8) / (float)(1u << 24);
        buf[i] = 2.0f * u - 1.0f;
    }
}

/* Integer token ids for the embedding / cross-entropy tests come from
 * fill_random_ids (tests/helpers/test_utils.h) — the same reproducible LCG the
 * CPU tests use, so there is no need to roll a local id generator here. */

/* ====================================================================== */
/*                               MATMUL                                   */
/* ====================================================================== */

/*
 * Direct sanity check independent of the CPU oracle: the textbook 2x2 case.
 *   a = [[1,2],[3,4]], b = [[5,6],[7,8]]  ->  a@b = [[19,22],[43,50]].
 * If even this fails, the kernel or the launch config is fundamentally wrong.
 */
static void test_matmul_cuda_known_2x2(void **state) {
    (void)state;

    float a[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4]   = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {0};

    matmul_forward_cuda(out, a, b, 2, 2, 2);

    assert_float_equal(out[0], 19.0f, CMP_TOL);
    assert_float_equal(out[1], 22.0f, CMP_TOL);
    assert_float_equal(out[2], 43.0f, CMP_TOL);
    assert_float_equal(out[3], 50.0f, CMP_TOL);
}

/*
 * Helper: run both implementations on random M x K @ K x N inputs and assert
 * every output element matches. Sizes are passed in so one helper drives
 * several shape cases.
 */
static void check_matmul_against_cpu(int M, int K, int N, unsigned int seed) {
    float *a       = (float *)malloc((size_t)M * K * sizeof(float));
    float *b       = (float *)malloc((size_t)K * N * sizeof(float));
    float *out_cpu = (float *)malloc((size_t)M * N * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)M * N * sizeof(float));
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out_cpu);
    assert_non_null(out_gpu);

    unsigned int s = seed;
    fill_pseudo_random(a, M * K, &s);
    fill_pseudo_random(b, K * N, &s);

    matmul_forward(out_cpu, a, b, M, K, N);        /* oracle  */
    matmul_forward_cuda(out_gpu, a, b, M, K, N);   /* under test */

    for (int i = 0; i < M * N; i++) {
        assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);
    }

    free(a);
    free(b);
    free(out_cpu);
    free(out_gpu);
}

/* Square sizes that are an exact multiple of TILE (16): the "easy" case with
 * no partial tiles at the boundary. */
static void test_matmul_cuda_tiled_aligned(void **state) {
    (void)state;
    check_matmul_against_cpu(32, 32, 32, 12345u);
}

/* Sizes that are NOT multiples of TILE in every dimension: forces the kernel's
 * boundary handling (partial tiles, zero-padded loads, guarded writes). This
 * is where a naive tiled matmul most often has bugs. */
static void test_matmul_cuda_unaligned(void **state) {
    (void)state;
    check_matmul_against_cpu(33, 17, 40, 6789u);
}

/* A "tall-skinny @ skinny-wide" shape (M != N, small K) — also non-aligned. */
static void test_matmul_cuda_nonsquare(void **state) {
    (void)state;
    check_matmul_against_cpu(7, 3, 5, 555u);
}

/* ====================================================================== */
/*                             LAYERNORM                                  */
/* ====================================================================== */

/*
 * Helper: run CPU and CUDA layernorm on random N x C input (with random
 * gamma/beta) and assert every output element matches.
 */
static void check_layernorm_against_cpu(int N, int C, unsigned int seed) {
    float *in      = (float *)malloc((size_t)N * C * sizeof(float));
    float *gamma   = (float *)malloc((size_t)C * sizeof(float));
    float *beta    = (float *)malloc((size_t)C * sizeof(float));
    float *out_cpu = (float *)malloc((size_t)N * C * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)N * C * sizeof(float));
    assert_non_null(in);
    assert_non_null(gamma);
    assert_non_null(beta);
    assert_non_null(out_cpu);
    assert_non_null(out_gpu);

    unsigned int s = seed;
    fill_pseudo_random(in,    N * C, &s);
    fill_pseudo_random(gamma, C,     &s);
    fill_pseudo_random(beta,  C,     &s);

    layernorm_forward(out_cpu, in, gamma, beta, N, C);        /* oracle */
    layernorm_forward_cuda(out_gpu, in, gamma, beta, N, C);   /* under test */

    for (int i = 0; i < N * C; i++) {
        assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);
    }

    free(in);
    free(gamma);
    free(beta);
    free(out_cpu);
    free(out_gpu);
}

/* C is a power of two (128 — the model's real embed_dim) and bigger than one
 * warp, exercising a full multi-step tree reduction. */
static void test_layernorm_cuda_pow2_C(void **state) {
    (void)state;
    check_layernorm_against_cpu(8, 128, 24680u);
}

/* C is NOT a power of two and NOT a multiple of the block size: forces the
 * strided per-thread loop (C > blockDim) AND the power-of-two reduction over a
 * non-power-of-two row length. The block will be 32 threads (largest pow2 <=
 * min(50,256)) each striding over the 50 columns. */
static void test_layernorm_cuda_odd_C(void **state) {
    (void)state;
    check_layernorm_against_cpu(5, 50, 13579u);
}

/* Tiny C (smaller than a warp) — block becomes 4 threads. Confirms the
 * reduction and strided loop are correct at the small end too. */
static void test_layernorm_cuda_small_C(void **state) {
    (void)state;
    check_layernorm_against_cpu(3, 6, 42u);
}

/* ====================================================================== */
/*                          MATMUL BACKWARD                               */
/* ====================================================================== */

/*
 * Run CPU and CUDA matmul_backward on random inputs and assert both gradient
 * outputs match. Both grad buffers are zeroed first so the += accumulation in
 * each implementation starts from the same state.
 */
static void check_matmul_backward_against_cpu(int M, int K, int N,
                                              unsigned int seed) {
    float *a     = (float *)malloc((size_t)M * K * sizeof(float));
    float *b     = (float *)malloc((size_t)K * N * sizeof(float));
    float *d_out = (float *)malloc((size_t)M * N * sizeof(float));
    float *d_a_cpu = (float *)calloc((size_t)M * K, sizeof(float));
    float *d_b_cpu = (float *)calloc((size_t)K * N, sizeof(float));
    float *d_a_gpu = (float *)calloc((size_t)M * K, sizeof(float));
    float *d_b_gpu = (float *)calloc((size_t)K * N, sizeof(float));

    unsigned int s = seed;
    fill_pseudo_random(a,     M * K, &s);
    fill_pseudo_random(b,     K * N, &s);
    fill_pseudo_random(d_out, M * N, &s);

    matmul_backward(d_a_cpu, d_b_cpu, d_out, a, b, M, K, N);          /* oracle */
    matmul_backward_cuda(d_a_gpu, d_b_gpu, d_out, a, b, M, K, N);     /* test   */

    for (int i = 0; i < M * K; i++) assert_float_equal(d_a_gpu[i], d_a_cpu[i], CMP_TOL);
    for (int i = 0; i < K * N; i++) assert_float_equal(d_b_gpu[i], d_b_cpu[i], CMP_TOL);

    free(a); free(b); free(d_out);
    free(d_a_cpu); free(d_b_cpu); free(d_a_gpu); free(d_b_gpu);
}

static void test_matmul_backward_aligned(void **state) {
    (void)state;
    check_matmul_backward_against_cpu(32, 32, 32, 9001u);
}
static void test_matmul_backward_unaligned(void **state) {
    (void)state;
    check_matmul_backward_against_cpu(33, 17, 40, 9002u);
}

/* ====================================================================== */
/*                         LAYERNORM BACKWARD                             */
/* ====================================================================== */

static void check_layernorm_backward_against_cpu(int N, int C,
                                                 unsigned int seed) {
    float *in    = (float *)malloc((size_t)N * C * sizeof(float));
    float *gamma = (float *)malloc((size_t)C * sizeof(float));
    float *d_out = (float *)malloc((size_t)N * C * sizeof(float));
    float *d_in_cpu = (float *)calloc((size_t)N * C, sizeof(float));
    float *d_in_gpu = (float *)calloc((size_t)N * C, sizeof(float));
    float *d_g_cpu = (float *)calloc((size_t)C, sizeof(float));
    float *d_g_gpu = (float *)calloc((size_t)C, sizeof(float));
    float *d_b_cpu = (float *)calloc((size_t)C, sizeof(float));
    float *d_b_gpu = (float *)calloc((size_t)C, sizeof(float));

    unsigned int s = seed;
    fill_pseudo_random(in,    N * C, &s);
    fill_pseudo_random(gamma, C,     &s);
    fill_pseudo_random(d_out, N * C, &s);

    layernorm_backward(d_in_cpu, d_g_cpu, d_b_cpu, d_out, in, gamma, N, C);
    layernorm_backward_cuda(d_in_gpu, d_g_gpu, d_b_gpu, d_out, in, gamma, N, C);

    for (int i = 0; i < N * C; i++) assert_float_equal(d_in_gpu[i], d_in_cpu[i], CMP_TOL);
    for (int i = 0; i < C; i++) {
        assert_float_equal(d_g_gpu[i], d_g_cpu[i], CMP_TOL);
        assert_float_equal(d_b_gpu[i], d_b_cpu[i], CMP_TOL);
    }

    free(in); free(gamma); free(d_out);
    free(d_in_cpu); free(d_in_gpu);
    free(d_g_cpu); free(d_g_gpu); free(d_b_cpu); free(d_b_gpu);
}

static void test_layernorm_backward_pow2_C(void **state) {
    (void)state;
    check_layernorm_backward_against_cpu(8, 128, 1111u);
}
static void test_layernorm_backward_odd_C(void **state) {
    (void)state;
    check_layernorm_backward_against_cpu(5, 50, 2222u);
}

/* ====================================================================== */
/*                       RESIDUAL (forward + backward)                    */
/* ====================================================================== */

static void test_residual_forward(void **state) {
    (void)state;
    int N = 1000;
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *out_cpu = (float *)malloc(N * sizeof(float));
    float *out_gpu = (float *)malloc(N * sizeof(float));
    unsigned int s = 31u;
    fill_pseudo_random(a, N, &s);
    fill_pseudo_random(b, N, &s);

    residual_forward(out_cpu, a, b, N);
    residual_forward_cuda(out_gpu, a, b, N);
    for (int i = 0; i < N; i++) assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);

    free(a); free(b); free(out_cpu); free(out_gpu);
}

static void test_residual_backward(void **state) {
    (void)state;
    int N = 1000;
    float *d_out = (float *)malloc(N * sizeof(float));
    float *d_a_cpu = (float *)calloc(N, sizeof(float));
    float *d_b_cpu = (float *)calloc(N, sizeof(float));
    float *d_a_gpu = (float *)calloc(N, sizeof(float));
    float *d_b_gpu = (float *)calloc(N, sizeof(float));
    unsigned int s = 32u;
    fill_pseudo_random(d_out, N, &s);

    residual_backward(d_a_cpu, d_b_cpu, d_out, N);
    residual_backward_cuda(d_a_gpu, d_b_gpu, d_out, N);
    for (int i = 0; i < N; i++) {
        assert_float_equal(d_a_gpu[i], d_a_cpu[i], CMP_TOL);
        assert_float_equal(d_b_gpu[i], d_b_cpu[i], CMP_TOL);
    }

    free(d_out); free(d_a_cpu); free(d_b_cpu); free(d_a_gpu); free(d_b_gpu);
}

/* ====================================================================== */
/*                         GELU (forward + backward)                      */
/* ====================================================================== */

static void test_gelu_forward(void **state) {
    (void)state;
    int N = 777;
    float *in = (float *)malloc(N * sizeof(float));
    float *out_cpu = (float *)malloc(N * sizeof(float));
    float *out_gpu = (float *)malloc(N * sizeof(float));
    unsigned int s = 41u;
    fill_pseudo_random(in, N, &s);

    gelu_forward(out_cpu, in, N);
    gelu_forward_cuda(out_gpu, in, N);
    for (int i = 0; i < N; i++) assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);

    free(in); free(out_cpu); free(out_gpu);
}

static void test_gelu_backward(void **state) {
    (void)state;
    int N = 777;
    float *in    = (float *)malloc(N * sizeof(float));
    float *d_out = (float *)malloc(N * sizeof(float));
    float *d_in_cpu = (float *)calloc(N, sizeof(float));
    float *d_in_gpu = (float *)calloc(N, sizeof(float));
    unsigned int s = 42u;
    fill_pseudo_random(in,    N, &s);
    fill_pseudo_random(d_out, N, &s);

    gelu_backward(d_in_cpu, d_out, in, N);
    gelu_backward_cuda(d_in_gpu, d_out, in, N);
    for (int i = 0; i < N; i++) assert_float_equal(d_in_gpu[i], d_in_cpu[i], CMP_TOL);

    free(in); free(d_out); free(d_in_cpu); free(d_in_gpu);
}

/* ====================================================================== */
/*                        SOFTMAX (forward + backward)                    */
/* ====================================================================== */

static void check_softmax_forward(int N, int V, unsigned int seed) {
    float *in = (float *)malloc((size_t)N * V * sizeof(float));
    float *out_cpu = (float *)malloc((size_t)N * V * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)N * V * sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(in, N * V, &s);

    softmax_forward(out_cpu, in, N, V);
    softmax_forward_cuda(out_gpu, in, N, V);
    for (int i = 0; i < N * V; i++) assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);

    free(in); free(out_cpu); free(out_gpu);
}

static void test_softmax_forward_pow2(void **state) {
    (void)state;
    check_softmax_forward(4, 64, 5551u);
}
static void test_softmax_forward_odd(void **state) {
    (void)state;
    check_softmax_forward(7, 50, 5552u);   /* V not a power of two */
}

static void check_softmax_backward(int N, int V, unsigned int seed) {
    float *in    = (float *)malloc((size_t)N * V * sizeof(float));
    float *out   = (float *)malloc((size_t)N * V * sizeof(float));
    float *d_out = (float *)malloc((size_t)N * V * sizeof(float));
    float *d_in_cpu = (float *)calloc((size_t)N * V, sizeof(float));
    float *d_in_gpu = (float *)calloc((size_t)N * V, sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(in,    N * V, &s);
    fill_pseudo_random(d_out, N * V, &s);
    softmax_forward(out, in, N, V);   /* the cached forward output y */

    softmax_backward(d_in_cpu, d_out, out, N, V);
    softmax_backward_cuda(d_in_gpu, d_out, out, N, V);
    for (int i = 0; i < N * V; i++) assert_float_equal(d_in_gpu[i], d_in_cpu[i], CMP_TOL);

    free(in); free(out); free(d_out); free(d_in_cpu); free(d_in_gpu);
}

static void test_softmax_backward_pow2(void **state) {
    (void)state;
    check_softmax_backward(4, 64, 5553u);
}
static void test_softmax_backward_odd(void **state) {
    (void)state;
    check_softmax_backward(7, 50, 5554u);
}

/* ====================================================================== */
/*                       EMBEDDING (forward + backward)                   */
/* ====================================================================== */

static void check_embed_forward(int B, int T, int C, int V, unsigned int seed) {
    int *tokens = (int *)malloc((size_t)B * T * sizeof(int));
    float *wte = (float *)malloc((size_t)V * C * sizeof(float));
    float *wpe = (float *)malloc((size_t)T * C * sizeof(float));  /* rows 0..T-1 */
    float *out_cpu = (float *)malloc((size_t)B * T * C * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)B * T * C * sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(wte, V * C, &s);
    fill_pseudo_random(wpe, T * C, &s);
    fill_random_ids(tokens, B * T, V, &s);

    embed_forward(out_cpu, tokens, wte, wpe, B, T, C);
    embed_forward_cuda(out_gpu, tokens, wte, wpe, B, T, C, V);
    for (int i = 0; i < B * T * C; i++) assert_float_equal(out_gpu[i], out_cpu[i], CMP_TOL);

    free(tokens); free(wte); free(wpe); free(out_cpu); free(out_gpu);
}

static void test_embed_forward(void **state) {
    (void)state;
    check_embed_forward(2, 5, 8, 16, 6661u);  /* repeated ids likely, exercises gather */
}

static void check_embed_backward(int B, int T, int C, int V, unsigned int seed) {
    int *tokens = (int *)malloc((size_t)B * T * sizeof(int));
    float *d_out = (float *)malloc((size_t)B * T * C * sizeof(float));
    float *d_wte_cpu = (float *)calloc((size_t)V * C, sizeof(float));
    float *d_wte_gpu = (float *)calloc((size_t)V * C, sizeof(float));
    float *d_wpe_cpu = (float *)calloc((size_t)T * C, sizeof(float));
    float *d_wpe_gpu = (float *)calloc((size_t)T * C, sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(d_out, B * T * C, &s);
    /* small vocab so the SAME id repeats across positions -> exercises the
     * atomicAdd accumulation into one d_wte row (the bug atomicAdd prevents). */
    fill_random_ids(tokens, B * T, V, &s);

    embed_backward(d_wte_cpu, d_wpe_cpu, d_out, tokens, B, T, C);
    embed_backward_cuda(d_wte_gpu, d_wpe_gpu, d_out, tokens, B, T, C, V);
    for (int i = 0; i < V * C; i++) assert_float_equal(d_wte_gpu[i], d_wte_cpu[i], CMP_TOL);
    for (int i = 0; i < T * C; i++) assert_float_equal(d_wpe_gpu[i], d_wpe_cpu[i], CMP_TOL);

    free(tokens); free(d_out);
    free(d_wte_cpu); free(d_wte_gpu); free(d_wpe_cpu); free(d_wpe_gpu);
}

static void test_embed_backward(void **state) {
    (void)state;
    check_embed_backward(3, 6, 8, 5, 6662u);  /* V=5 < B*T=18 forces id reuse */
}

/* ====================================================================== */
/*                      CROSS-ENTROPY (forward + backward)                */
/* ====================================================================== */

/*
 * The cross-entropy oracle is computed INLINE here (the math lives in gpt.c,
 * fused with the rest of the forward/backward, not as a standalone CPU
 * function). We softmax random logits into valid probabilities, then check the
 * CE kernels against the textbook formulas:
 *   forward:  loss[n]      = -log(probs[n, target[n]])
 *   backward: d_logits[n,j] = (probs[n,j] - 1{j==target}) / N
 */
static void check_cross_entropy(int N, int V, unsigned int seed) {
    float *logits = (float *)malloc((size_t)N * V * sizeof(float));
    float *probs  = (float *)malloc((size_t)N * V * sizeof(float));
    int   *targets = (int *)malloc((size_t)N * sizeof(int));
    float *loss_cpu = (float *)malloc((size_t)N * sizeof(float));
    float *loss_gpu = (float *)malloc((size_t)N * sizeof(float));
    float *dlog_cpu = (float *)malloc((size_t)N * V * sizeof(float));
    float *dlog_gpu = (float *)malloc((size_t)N * V * sizeof(float));

    unsigned int s = seed;
    fill_pseudo_random(logits, N * V, &s);
    fill_random_ids(targets, N, V, &s);
    softmax_forward(probs, logits, N, V);   /* valid probability rows */

    /* forward: inline oracle vs kernel */
    for (int n = 0; n < N; n++) loss_cpu[n] = -logf(probs[(size_t)n * V + targets[n]]);
    cross_entropy_forward_cuda(loss_gpu, probs, targets, N, V);
    for (int n = 0; n < N; n++) assert_float_equal(loss_gpu[n], loss_cpu[n], CMP_TOL);

    /* backward: inline oracle vs kernel */
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < V; j++) {
            float indicator = (j == targets[n]) ? 1.0f : 0.0f;
            dlog_cpu[(size_t)n * V + j] =
                (probs[(size_t)n * V + j] - indicator) / (float)N;
        }
    }
    cross_entropy_backward_cuda(dlog_gpu, probs, targets, N, V);
    for (int i = 0; i < N * V; i++) assert_float_equal(dlog_gpu[i], dlog_cpu[i], CMP_TOL);

    free(logits); free(probs); free(targets);
    free(loss_cpu); free(loss_gpu); free(dlog_cpu); free(dlog_gpu);
}

static void test_cross_entropy(void **state) {
    (void)state;
    check_cross_entropy(6, 32, 7771u);
}

/* ====================================================================== */
/*                      ATTENTION (forward + backward)                    */
/* ====================================================================== */

static void check_attention_forward(int B, int T, int C, int NH,
                                    unsigned int seed) {
    float *qkv  = (float *)malloc((size_t)B * T * (3 * C) * sizeof(float));
    float *atty_cpu = (float *)malloc((size_t)B * T * C * sizeof(float));
    float *atty_gpu = (float *)malloc((size_t)B * T * C * sizeof(float));
    float *att_cpu  = (float *)malloc((size_t)B * NH * T * T * sizeof(float));
    float *att_gpu  = (float *)malloc((size_t)B * NH * T * T * sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(qkv, B * T * (3 * C), &s);

    attention_forward(atty_cpu, att_cpu, qkv, B, T, C, NH);
    attention_forward_cuda(atty_gpu, att_gpu, qkv, B, T, C, NH);
    for (int i = 0; i < B * T * C; i++) assert_float_equal(atty_gpu[i], atty_cpu[i], CMP_TOL);
    for (int i = 0; i < B * NH * T * T; i++) assert_float_equal(att_gpu[i], att_cpu[i], CMP_TOL);

    free(qkv); free(atty_cpu); free(atty_gpu); free(att_cpu); free(att_gpu);
}

static void test_attention_forward(void **state) {
    (void)state;
    check_attention_forward(2, 6, 8, 2, 8881u);  /* head_dim = 4 */
}

static void check_attention_backward(int B, int T, int C, int NH,
                                     unsigned int seed) {
    float *qkv   = (float *)malloc((size_t)B * T * (3 * C) * sizeof(float));
    float *att   = (float *)malloc((size_t)B * NH * T * T * sizeof(float));
    float *atty  = (float *)malloc((size_t)B * T * C * sizeof(float));  /* unused output */
    float *d_atty = (float *)malloc((size_t)B * T * C * sizeof(float));
    float *d_qkv_cpu = (float *)calloc((size_t)B * T * (3 * C), sizeof(float));
    float *d_qkv_gpu = (float *)calloc((size_t)B * T * (3 * C), sizeof(float));
    unsigned int s = seed;
    fill_pseudo_random(qkv,    B * T * (3 * C), &s);
    fill_pseudo_random(d_atty, B * T * C,       &s);

    /* the cached softmax weights `att` come from the forward (the oracle) */
    attention_forward(atty, att, qkv, B, T, C, NH);

    attention_backward(d_qkv_cpu, d_atty, qkv, att, B, T, C, NH);
    attention_backward_cuda(d_qkv_gpu, d_atty, qkv, att, B, T, C, NH);
    for (int i = 0; i < B * T * (3 * C); i++) assert_float_equal(d_qkv_gpu[i], d_qkv_cpu[i], CMP_TOL);

    free(qkv); free(att); free(atty); free(d_atty); free(d_qkv_cpu); free(d_qkv_gpu);
}

static void test_attention_backward(void **state) {
    (void)state;
    check_attention_backward(2, 6, 8, 2, 8882u);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        /* matmul (Task 9 forward + Task 10 backward) */
        cmocka_unit_test(test_matmul_cuda_known_2x2),
        cmocka_unit_test(test_matmul_cuda_tiled_aligned),
        cmocka_unit_test(test_matmul_cuda_unaligned),
        cmocka_unit_test(test_matmul_cuda_nonsquare),
        cmocka_unit_test(test_matmul_backward_aligned),
        cmocka_unit_test(test_matmul_backward_unaligned),
        /* layernorm (Task 9 forward + Task 10 backward) */
        cmocka_unit_test(test_layernorm_cuda_pow2_C),
        cmocka_unit_test(test_layernorm_cuda_odd_C),
        cmocka_unit_test(test_layernorm_cuda_small_C),
        cmocka_unit_test(test_layernorm_backward_pow2_C),
        cmocka_unit_test(test_layernorm_backward_odd_C),
        /* residual */
        cmocka_unit_test(test_residual_forward),
        cmocka_unit_test(test_residual_backward),
        /* gelu */
        cmocka_unit_test(test_gelu_forward),
        cmocka_unit_test(test_gelu_backward),
        /* softmax */
        cmocka_unit_test(test_softmax_forward_pow2),
        cmocka_unit_test(test_softmax_forward_odd),
        cmocka_unit_test(test_softmax_backward_pow2),
        cmocka_unit_test(test_softmax_backward_odd),
        /* embedding */
        cmocka_unit_test(test_embed_forward),
        cmocka_unit_test(test_embed_backward),
        /* cross-entropy */
        cmocka_unit_test(test_cross_entropy),
        /* attention */
        cmocka_unit_test(test_attention_forward),
        cmocka_unit_test(test_attention_backward),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
