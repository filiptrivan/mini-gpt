/*
 * Element-wise GPU kernels — GELU activation and residual (skip) add.
 *
 * These are the simplest kernels in the project: every output element depends
 * only on the SAME-index input element(s), with no reduction and no
 * cross-element dependency. That makes the parallelization trivial — one GPU
 * thread per element, all N computed simultaneously. The CPU twins live in
 * src/model/layers.c (gelu_forward/backward, residual_forward/backward); the
 * math here is a line-for-line mirror, using the same constants and the same
 * single-precision functions (tanhf, not tanh) so the results match within
 * float round-off.
 */

#include "cuda/cuda_layers.cuh"
#include <math.h>   /* tanhf used in the GELU kernels */

/* Threads per block for the flat 1-D element kernels. 256 is the conventional
 * default: a multiple of the 32-thread warp, enough to hide memory latency,
 * and small enough to keep many blocks resident per SM. */
#define EW_BLOCK 256

/* GELU constants — MUST match src/model/layers.c exactly, or the forward and
 * backward would disagree with the CPU oracle (and with each other). */
#define GELU_SQRT_2_OVER_PI 0.7978845608028654f   /* sqrt(2/pi)            */
#define GELU_COEF           0.044715f             /* GPT-2 paper constant */

/* ====================================================================== */
/*                              RESIDUAL ADD                               */
/* ====================================================================== */

/*
 * residual_forward_kernel — out[i] = a[i] + b[i].
 * One thread per element; threads past N-1 (the last block is usually partial)
 * simply return. No shared memory, no sync.
 */
__global__ void residual_forward_kernel(float *out, const float *a,
                                        const float *b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

/*
 * residual_backward_kernel — d_a[i] += d_out[i]; d_b[i] += d_out[i].
 *
 * The derivative of an element-wise sum is 1 into each input, so the upstream
 * gradient flows into both unchanged. Each thread owns one i and touches two
 * distinct destination addresses, so no atomics are needed; the += accumulates
 * onto whatever the caller already had.
 */
__global__ void residual_backward_kernel(float *d_a, float *d_b,
                                         const float *d_out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_a[i] += d_out[i];
        d_b[i] += d_out[i];
    }
}

/* ====================================================================== */
/*                                 GELU                                    */
/* ====================================================================== */

/*
 * gelu_forward_kernel — out[i] = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*(x+COEF*x^3))).
 * The GPT-2 tanh approximation of GELU, identical to gelu_forward in layers.c.
 */
__global__ void gelu_forward_kernel(float *out, const float *in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x       = in[i];
    float x_cubed = x * x * x;
    float inner   = GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
}

/*
 * gelu_backward_kernel — d_in[i] += d_out[i] * gelu'(in[i]).
 *
 * Derivative of the tanh-approx GELU (product + chain rule), exactly as in
 * gelu_backward in layers.c:
 *   s        = SQRT_2_OVER_PI * (x + COEF*x^3)
 *   gelu'(x) = 0.5*(1 + tanh(s)) + 0.5*x*(1 - tanh^2(s))*SQRT_2_OVER_PI*(1 + 3*COEF*x^2)
 * We cache tanh(s) so it is computed once. += accumulates onto the caller's
 * current d_in.
 */
__global__ void gelu_backward_kernel(float *d_in, const float *d_out,
                                     const float *in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x       = in[i];
    float x_cubed = x * x * x;
    float s       = GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
    float tanh_s  = tanhf(s);
    float sech2_s = 1.0f - tanh_s * tanh_s;     /* 1 - tanh^2 = sech^2 */
    float s_prime = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x * x);

    float dgelu_dx = 0.5f * (1.0f + tanh_s)
                   + 0.5f * x * sech2_s * s_prime;

    d_in[i] += d_out[i] * dgelu_dx;
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

extern "C" void residual_forward_device(float *d_out, const float *d_a,
                                        const float *d_b, int N) {
    int grid = (N + EW_BLOCK - 1) / EW_BLOCK;
    residual_forward_kernel<<<grid, EW_BLOCK>>>(d_out, d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void residual_backward_device(float *d_da, float *d_db,
                                         const float *d_dout, int N) {
    int grid = (N + EW_BLOCK - 1) / EW_BLOCK;
    residual_backward_kernel<<<grid, EW_BLOCK>>>(d_da, d_db, d_dout, N);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gelu_forward_device(float *d_out, const float *d_in, int N) {
    int grid = (N + EW_BLOCK - 1) / EW_BLOCK;
    gelu_forward_kernel<<<grid, EW_BLOCK>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gelu_backward_device(float *d_din, const float *d_dout,
                                     const float *d_in, int N) {
    int grid = (N + EW_BLOCK - 1) / EW_BLOCK;
    gelu_backward_kernel<<<grid, EW_BLOCK>>>(d_din, d_dout, d_in, N);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

extern "C" void residual_forward_cuda(float *out, const float *a,
                                      const float *b, int N) {
    size_t bytes = (size_t)N * sizeof(float);
    float *d_out = NULL, *d_a = NULL, *d_b = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_a,   bytes));
    CUDA_CHECK(cudaMalloc(&d_b,   bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    residual_forward_device(d_out, d_a, d_b, N);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

extern "C" void residual_backward_cuda(float *d_a, float *d_b,
                                       const float *d_out, int N) {
    size_t bytes = (size_t)N * sizeof(float);
    float *dd_a = NULL, *dd_b = NULL, *dd_out = NULL;
    CUDA_CHECK(cudaMalloc(&dd_a,   bytes));
    CUDA_CHECK(cudaMalloc(&dd_b,   bytes));
    CUDA_CHECK(cudaMalloc(&dd_out, bytes));

    /* grad accumulators copied up so the kernel's += starts from caller state */
    CUDA_CHECK(cudaMemcpy(dd_a,   d_a,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_b,   d_b,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_out, d_out, bytes, cudaMemcpyHostToDevice));

    residual_backward_device(dd_a, dd_b, dd_out, N);

    CUDA_CHECK(cudaMemcpy(d_a, dd_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_b, dd_b, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_a));
    CUDA_CHECK(cudaFree(dd_b));
    CUDA_CHECK(cudaFree(dd_out));
}

extern "C" void gelu_forward_cuda(float *out, const float *in, int N) {
    size_t bytes = (size_t)N * sizeof(float);
    float *d_out = NULL, *d_in = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));

    CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    gelu_forward_device(d_out, d_in, N);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
}

extern "C" void gelu_backward_cuda(float *d_in, const float *d_out,
                                   const float *in, int N) {
    size_t bytes = (size_t)N * sizeof(float);
    float *dd_in = NULL, *dd_out = NULL, *d_in_fwd = NULL;
    CUDA_CHECK(cudaMalloc(&dd_in,    bytes));
    CUDA_CHECK(cudaMalloc(&dd_out,   bytes));
    CUDA_CHECK(cudaMalloc(&d_in_fwd, bytes));

    CUDA_CHECK(cudaMemcpy(dd_in,    d_in,  bytes, cudaMemcpyHostToDevice)); /* += base */
    CUDA_CHECK(cudaMemcpy(dd_out,   d_out, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_fwd, in,    bytes, cudaMemcpyHostToDevice));

    gelu_backward_device(dd_in, dd_out, d_in_fwd, N);

    CUDA_CHECK(cudaMemcpy(d_in, dd_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_in));
    CUDA_CHECK(cudaFree(dd_out));
    CUDA_CHECK(cudaFree(d_in_fwd));
}
