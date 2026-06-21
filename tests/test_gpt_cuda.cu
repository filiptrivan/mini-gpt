/*
 * Full-model parity test (Task 10): the GPU forward/backward pass vs the CPU one.
 *
 * The per-kernel tests in test_cuda_layers.cu prove each CUDA kernel matches its
 * CPU oracle in isolation. This test proves the COMPOSITION in src/cuda/gpt_cuda.cu
 * is wired correctly: it runs the whole transformer two ways and asserts they
 * agree.
 *
 * THE PATTERN:
 *   1. Init TWO models with the SAME seed and config -> byte-identical weights.
 *   2. Run gpt_forward + gpt_backward (CPU) on model A.
 *   3. Run gpt_forward_cuda + gpt_backward_cuda (GPU) on model B.
 *   4. Assert: same loss, same logits/probs (forward), same gradients (backward).
 * Using two models (instead of one, run twice) keeps each side's cached
 * activations intact for its own backward, and makes the comparison obvious.
 *
 * Compiled by nvcc (C++) and run only where a CUDA toolchain + GPU exist
 * (Colab); excluded from the Mac CPU build by the ENABLE_CUDA guard in
 * tests/CMakeLists.txt.
 */

/* cmocka's required four includes; cmocka.h wrapped in extern "C" because this
 * is a C++ (nvcc) TU and Colab's cmocka.h lacks its own guards — same fix as
 * test_cuda_layers.cu. gpt.h / gpt_cuda.cuh carry their own extern "C". */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
extern "C" {
#include <cmocka.h>
}

#include <stdlib.h>   /* malloc, free */

#include "model/gpt.h"
#include "cuda/gpt_cuda.cuh"
#include "test_utils.h"   /* fill_random_ids — reproducible token batches */

/*
 * Tolerance for the WHOLE-MODEL CPU-vs-GPU comparison.
 *
 * Looser than the 1e-4 used per-kernel: a full forward chains ~30 kernels, each
 * summing floats in a different order from the CPU (tiled matmul, tree
 * reductions, atomicAdd accumulation), and those tiny last-place differences
 * compound layer over layer. 2e-3 absolute comfortably covers the accumulated
 * round-off at our small magnitudes while still being far tighter than any real
 * wiring bug (a mis-ordered op or wrong slice would diverge by >> 2e-3).
 */
#define GPT_TOL 2e-3

/* A tiny but structurally complete config: 2 layers, 2 heads (head_dim=4),
 * exercising every op the real model uses, at sizes a failure is readable at. */
static GPTConfig tiny_config(void) {
    GPTConfig c;
    c.max_seq_len = 8;
    c.vocab_size  = 16;
    c.n_embd      = 8;
    c.n_head      = 2;
    c.n_layer     = 2;
    c.ff_dim      = 32;
    return c;
}

/*
 * Forward parity: identical loss, logits and probs between CPU and GPU.
 */
static void test_gpt_forward_cuda_matches_cpu(void **state) {
    (void)state;
    GPTConfig cfg = tiny_config();
    int B = 2, T = 6;
    int V = cfg.vocab_size;

    int *tokens  = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    unsigned int s = 12345u;
    fill_random_ids(tokens,  B * T, V, &s);
    fill_random_ids(targets, B * T, V, &s);

    GPTModel cpu, gpu;
    gpt_init(&cpu, cfg, 7u);   /* same seed + config => identical weights */
    gpt_init(&gpu, cfg, 7u);

    gpt_forward(&cpu, tokens, targets, B, T);
    gpt_forward_cuda(&gpu, tokens, targets, B, T);

    /* the loss (a single averaged scalar) */
    assert_float_equal(gpu.loss, cpu.loss, GPT_TOL);

    /* every logit and probability */
    int BTV = B * T * V;
    for (int i = 0; i < BTV; i++) {
        assert_float_equal(gpu.a.logits[i], cpu.a.logits[i], GPT_TOL);
        assert_float_equal(gpu.a.probs[i],  cpu.a.probs[i],  GPT_TOL);
    }

    gpt_free(&cpu);
    gpt_free(&gpu);
    free(tokens);
    free(targets);
}

/*
 * Backward parity: identical parameter gradients between CPU and GPU.
 * (gpt_backward_cuda zeroes the device grads itself, so the GPU side needs no
 * separate gpt_zero_grad; the CPU side does.)
 */
static void test_gpt_backward_cuda_matches_cpu(void **state) {
    (void)state;
    GPTConfig cfg = tiny_config();
    int B = 2, T = 6;
    int V = cfg.vocab_size;

    int *tokens  = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    unsigned int s = 9999u;
    fill_random_ids(tokens,  B * T, V, &s);
    fill_random_ids(targets, B * T, V, &s);

    GPTModel cpu, gpu;
    gpt_init(&cpu, cfg, 3u);
    gpt_init(&gpu, cfg, 3u);

    gpt_forward(&cpu, tokens, targets, B, T);
    gpt_zero_grad(&cpu);
    gpt_backward(&cpu, tokens, targets, B, T);

    gpt_forward_cuda(&gpu, tokens, targets, B, T);
    gpt_backward_cuda(&gpu, tokens, targets, B, T);

    /* every parameter gradient (one contiguous block) */
    for (int i = 0; i < cpu.num_params; i++) {
        assert_float_equal(gpu.grads[i], cpu.grads[i], GPT_TOL);
    }

    gpt_free(&cpu);
    gpt_free(&gpu);
    free(tokens);
    free(targets);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_gpt_forward_cuda_matches_cpu),
        cmocka_unit_test(test_gpt_backward_cuda_matches_cpu),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
