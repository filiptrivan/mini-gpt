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

/* CMocka's required four includes, same block/order as every test file. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>   /* malloc, free */

/* The CPU oracle. layers.h is a plain C header with no extern "C" guards of
 * its own, so we add them here: without this, nvcc (C++) would mangle the
 * names and fail to link against the C-compiled `layers` static library. */
extern "C" {
#include "model/layers.h"
}

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
 * roughly [-1, 1). We use the same linear-congruential generator constants as
 * tests/helpers/test_utils.c so the sequence is identical on every machine and
 * every run — reproducibility matters when a remote (Colab) test fails and we
 * need to reason about exactly what was fed in.
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

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_matmul_cuda_known_2x2),
        cmocka_unit_test(test_matmul_cuda_tiled_aligned),
        cmocka_unit_test(test_matmul_cuda_unaligned),
        cmocka_unit_test(test_matmul_cuda_nonsquare),
        cmocka_unit_test(test_layernorm_cuda_pow2_C),
        cmocka_unit_test(test_layernorm_cuda_odd_C),
        cmocka_unit_test(test_layernorm_cuda_small_C),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
