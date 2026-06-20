#ifndef CUDA_LAYERS_CUH
#define CUDA_LAYERS_CUH

/*
 * CUDA layer operations — GPU forward passes (Task 9 onward).
 *
 * This header declares the HOST WRAPPERS that the rest of the program calls.
 * Each wrapper has the SAME signature as its CPU twin in src/model/layers.h
 * and takes plain HOST (CPU) pointers. Internally the wrapper does the full
 * round trip: cudaMalloc device buffers, copy the inputs host->device, launch
 * the kernel, copy the result device->host, and free the device buffers.
 *
 * Why mirror the CPU signature exactly?
 *   - The Task 9 tests can call matmul_forward (CPU) and matmul_forward_cuda
 *     (GPU) on the SAME host inputs and assert the outputs match. The CPU
 *     version is the "oracle" — it's already proven correct by the Task 5/6
 *     tests, so any mismatch points at the new CUDA code.
 *   - Later (Task 10) the real training loop keeps tensors resident on the
 *     GPU across many kernels; these per-call alloc/copy wrappers are
 *     deliberately simple and a little wasteful, but they're all we need to
 *     prove the kernels themselves are correct.
 *
 * extern "C": this header is included from C++ translation units (the .cu
 * files are compiled by nvcc as C++), but the wrappers must also be callable
 * from plain C code (the training loop, future MPI glue). Wrapping the
 * declarations in extern "C" forces C linkage (no C++ name mangling) so the
 * symbols link the same way regardless of which compiler emits the call.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * matmul_forward_cuda — GPU tiled matrix multiply: out = a @ b.
 *
 * Identical contract to matmul_forward() in layers.h (row-major flat arrays):
 *   a   : M x K        b   : K x N        out : M x N
 *   out[i*N + j] = sum over k of a[i*K + k] * b[k*N + j]
 *
 * All three pointers are HOST memory; the wrapper handles the device side.
 * out must not alias a or b. Preconditions: non-NULL pointers, M,K,N > 0.
 */
void matmul_forward_cuda(float *out, const float *a, const float *b,
                         int M, int K, int N);

/*
 * layernorm_forward_cuda — GPU row-wise layer normalization with affine.
 *
 * Identical contract to layernorm_forward() in layers.h:
 *   in, out : N rows x C cols      gamma, beta : C floats (shared across rows)
 *   out = gamma * (in - mean) / sqrt(var + 1e-5) + beta   (per row)
 *
 * All pointers are HOST memory. out must not alias in. Epsilon is the same
 * 1e-5 the CPU uses, so the two implementations agree within float round-off.
 * Preconditions: non-NULL pointers, N > 0, C > 0.
 */
void layernorm_forward_cuda(float *out, const float *in,
                            const float *gamma, const float *beta,
                            int N, int C);

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
#endif /* __CUDACC__ */

#endif /* CUDA_LAYERS_CUH */
