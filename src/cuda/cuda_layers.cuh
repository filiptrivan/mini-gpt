#ifndef CUDA_LAYERS_CUH
#define CUDA_LAYERS_CUH

/*
 * CUDA layer operations — GPU forward and backward passes.
 *
 * Every neural-network op in this project has TWO GPU entry points here, plus
 * the kernel itself (the `__global__` function, which stays private to each
 * .cu file). Understanding the three layers is the key to reading this header:
 *
 *   1. <op>_<dir>_kernel   (in the .cu, file-local)
 *        The actual GPU code: runs on the device, one thread per work item.
 *
 *   2. <op>_<dir>_device   (declared here, "device-pointer launcher")
 *        A plain host function whose pointer arguments are ALREADY device
 *        memory. It only computes the grid/block configuration and launches
 *        the kernel — no cudaMalloc, no cudaMemcpy. This is what the resident
 *        forward/backward pass in src/cuda/gpt_cuda.cu calls: tensors are
 *        uploaded once, then dozens of these launchers run back-to-back on the
 *        SAME device buffers, so data never bounces over the PCIe bus between
 *        kernels. (CUDA serializes launches on the default stream, so each
 *        kernel finishes before the next begins — no explicit sync needed.)
 *
 *   3. <op>_<dir>_cuda     (declared here, "host wrapper")
 *        Same signature as the CPU twin in layers.h / gpt.h, taking HOST
 *        pointers. It does the full round trip: cudaMalloc, copy host->device,
 *        call the matching _device launcher, copy device->host, free. These
 *        are the Task 9 oracle-pattern wrappers: a test runs the CPU function
 *        (the proven oracle) and the _cuda wrapper on the same host inputs and
 *        asserts they match. Convenient and self-contained, but the per-call
 *        alloc/copy is exactly the waste that the resident pass (layer 2)
 *        avoids — which is the whole point of Task 10.
 *
 * <dir> is `forward` or `backward`. Backward ops ACCUMULATE into their gradient
 * outputs with += (the project-wide convention from layers.h): the caller must
 * zero those buffers first. The _cuda wrappers preserve this by copying the
 * caller's current gradient buffer up to the device before launching, so the
 * kernel's += adds onto it — letting tests check the accumulation property too.
 *
 * extern "C": .cu files are compiled by nvcc as C++, but these wrappers must
 * also be callable from plain C (the training loop, MPI glue). Wrapping the
 * declarations in extern "C" forces C linkage (no name mangling) so the
 * symbols link the same way regardless of which compiler emits the call.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/*       FORWARD — device-pointer launchers (all pointers are DEVICE)      */
/* ====================================================================== */
/* Contracts (shapes, math) mirror the CPU twins in layers.h / gpt.h exactly;
 * see there for the per-op details. The only difference is that every pointer
 * argument points at device (GPU) memory the caller has already allocated and
 * filled. None of these copy or allocate; they just launch a kernel. */

void matmul_forward_device(float *d_out, const float *d_a, const float *d_b,
                           int M, int K, int N);
void layernorm_forward_device(float *d_out, const float *d_in,
                              const float *d_gamma, const float *d_beta,
                              int N, int C);
void residual_forward_device(float *d_out, const float *d_a, const float *d_b,
                             int N);
void gelu_forward_device(float *d_out, const float *d_in, int N);
void softmax_forward_device(float *d_out, const float *d_in, int N, int V);
void embed_forward_device(float *d_out, const int *d_tokens,
                          const float *d_wte, const float *d_wpe,
                          int B, int T, int C);
void attention_forward_device(float *d_atty, float *d_att, const float *d_qkv,
                              int B, int T, int C, int NH);
void cross_entropy_forward_device(float *d_losses, const float *d_probs,
                                  const int *d_targets, int N, int V);

/* ====================================================================== */
/*      BACKWARD — device-pointer launchers (all pointers are DEVICE)      */
/* ====================================================================== */
/* Gradient outputs are accumulated with += and must be pre-zeroed on the
 * device by the caller. */

void matmul_backward_device(float *d_da, float *d_db, const float *d_dout,
                            const float *d_a, const float *d_b,
                            int M, int K, int N);
void layernorm_backward_device(float *d_din, float *d_dgamma, float *d_dbeta,
                               const float *d_dout, const float *d_in,
                               const float *d_gamma, int N, int C);
void residual_backward_device(float *d_da, float *d_db, const float *d_dout,
                              int N);
void gelu_backward_device(float *d_din, const float *d_dout, const float *d_in,
                          int N);
void softmax_backward_device(float *d_din, const float *d_dout,
                             const float *d_out, int N, int V);
void embed_backward_device(float *d_dwte, float *d_dwpe, const float *d_dout,
                           const int *d_tokens, int B, int T, int C);
void attention_backward_device(float *d_dqkv, const float *d_datty,
                               const float *d_qkv, const float *d_att,
                               int B, int T, int C, int NH);
void cross_entropy_backward_device(float *d_dlogits, const float *d_probs,
                                   const int *d_targets, int N, int V);

/* ====================================================================== */
/*   FORWARD — host wrappers (HOST pointers; do the full alloc/copy/free)  */
/* ====================================================================== */
/* Same signatures as the CPU functions in layers.h / gpt.h. Used by the CUDA
 * parity tests: run the CPU oracle and the _cuda wrapper on identical host
 * inputs, assert every element matches within tolerance. */

void matmul_forward_cuda(float *out, const float *a, const float *b,
                         int M, int K, int N);
void layernorm_forward_cuda(float *out, const float *in,
                            const float *gamma, const float *beta,
                            int N, int C);
void residual_forward_cuda(float *out, const float *a, const float *b, int N);
void gelu_forward_cuda(float *out, const float *in, int N);
void softmax_forward_cuda(float *out, const float *in, int N, int V);
/* embed wrappers take an extra vocab_size the CPU twin doesn't need: the CPU
 * only INDEXES into the caller-owned wte table, but the wrapper has to allocate
 * and copy the whole table to the device, so it must know its row count. (The
 * wpe table only needs its first T rows — positions 0..T-1 — so T suffices for
 * it.) The _device launchers need neither size: they just launch the kernel,
 * which indexes resident buffers exactly like the CPU does. */
void embed_forward_cuda(float *out, const int *tokens,
                        const float *wte, const float *wpe,
                        int B, int T, int C, int vocab_size);
void attention_forward_cuda(float *atty, float *att, const float *qkv,
                            int B, int T, int C, int NH);
void cross_entropy_forward_cuda(float *losses, const float *probs,
                                const int *targets, int N, int V);

/* ====================================================================== */
/*  BACKWARD — host wrappers (HOST pointers; preserve the += convention)   */
/* ====================================================================== */
/* The caller zeros (or pre-fills) the gradient outputs before calling; the
 * wrapper copies them up so the kernel's += accumulates onto the caller's
 * contents, then copies the result back. */

void matmul_backward_cuda(float *d_a, float *d_b, const float *d_out,
                          const float *a, const float *b, int M, int K, int N);
void layernorm_backward_cuda(float *d_in, float *d_gamma, float *d_beta,
                             const float *d_out, const float *in,
                             const float *gamma, int N, int C);
void residual_backward_cuda(float *d_a, float *d_b, const float *d_out, int N);
void gelu_backward_cuda(float *d_in, const float *d_out, const float *in,
                        int N);
void softmax_backward_cuda(float *d_in, const float *d_out, const float *out,
                           int N, int V);
void embed_backward_cuda(float *d_wte, float *d_wpe, const float *d_out,
                         const int *tokens, int B, int T, int C,
                         int vocab_size);
void attention_backward_cuda(float *d_qkv, const float *d_atty,
                             const float *qkv, const float *att,
                             int B, int T, int C, int NH);
void cross_entropy_backward_cuda(float *d_logits, const float *probs,
                                 const int *targets, int N, int V);

#ifdef __cplusplus
}
#endif

/*
 * CUDA_CHECK — error-checking macro shared by the .cu implementations.
 *
 * CUDA runtime calls return a cudaError_t instead of throwing; it is very
 * easy to ignore a failed cudaMalloc/cudaMemcpy/launch and then chase
 * garbage numbers for an hour. This macro wraps any runtime call, and on
 * failure prints the file/line and the human-readable error, then aborts.
 *
 * The do { ... } while (0) wrapper is the standard C idiom that lets a
 * multi-statement macro be used like a single statement (so it works
 * correctly after an `if` with no braces, and demands a trailing semicolon).
 *
 * Guarded by __CUDACC__ (defined only when nvcc is compiling the file) so a
 * plain C translation unit that includes this header just for the wrapper
 * declarations above does not try to pull in the CUDA runtime headers.
 */
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                  \
                    cudaGetErrorName(err__), __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

/*
 * cuda_largest_pow2_leq — largest power of two <= cap, clamped to >= 1.
 *
 * Shared by the reduction kernels (layernorm, softmax) to pick a block size:
 * their tree reductions repeatedly halve the active thread count, which only
 * sums correctly when the count is a power of two. Callers pass
 * cap = min(row_length, 256) and a row longer than the block is handled by
 * the per-thread strided loop. Examples: 128 -> 128, 200 -> 128, 5 -> 4, 1 -> 1.
 *
 * `static inline` in this header means each .cu gets its own private copy with
 * no link-time duplicate-symbol clash — the standard way to share a tiny
 * helper across translation units without a separate .cu just for it. Guarded
 * by __CUDACC__ alongside CUDA_CHECK since only the kernels use it.
 */
static inline int cuda_largest_pow2_leq(int cap) {
    int p = 1;
    while (p * 2 <= cap) p *= 2;
    return p;
}
#endif /* __CUDACC__ */

#endif /* CUDA_LAYERS_CUH */
