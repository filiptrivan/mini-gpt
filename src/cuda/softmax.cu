/*
 * Row-wise softmax on the GPU — forward and backward.
 *
 * Treats the input as N independent rows of length V (in GPT this is the
 * logits -> probabilities step at the output, with V = vocab_size) and applies
 * softmax to each row. The CPU twins are softmax_forward/softmax_backward in
 * src/model/layers.c; the math is mirrored exactly, including the
 * numerical-stability max-subtraction.
 *
 * PARALLELIZATION: one thread BLOCK per row, like layernorm. A softmax needs
 * two reductions over the row — the row max (for stability) and the sum of
 * exponentials (the normalizer) — so the threads in a block cooperate via a
 * shared-memory tree reduction, then each thread writes a strided slice of the
 * row. Block size is a power of two <= min(V, 256) so the tree reduction folds
 * cleanly; rows longer than the block are handled by the strided loop.
 */

#include "cuda/cuda_layers.cuh"
#include <math.h>   /* expf, fmaxf, and the INFINITY macro used in the kernels */

#define SM_MAX_THREADS 256

/*
 * softmax_forward_kernel — one block softmaxes one row.
 *
 * Three reduction-driven passes, mirroring softmax_forward in layers.c:
 *   1. row max         (tree reduction with fmaxf)
 *   2. out[v] = exp(in[v] - max), and sum those (tree reduction with +)
 *   3. out[v] *= 1/sum
 * Subtracting the max before exp keeps expf in range (largest arg becomes 0,
 * so the largest exponential is 1) — without it large logits overflow to inf.
 */
__global__ void softmax_forward_kernel(float *out, const float *in,
                                       int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float *in_row  = in  + (size_t)row * V;
    float       *out_row = out + (size_t)row * V;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ float sdata[];

    /* ---- Pass 1: row max -------------------------------------------------- */
    float m = -INFINITY;
    for (int v = tid; v < V; v += nthreads) m = fmaxf(m, in_row[v]);
    sdata[tid] = m;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float maxv = sdata[0];
    __syncthreads();

    /* ---- Pass 2: shifted exp into out, and sum -> denom ------------------- */
    float partial = 0.0f;
    for (int v = tid; v < V; v += nthreads) {
        float e = expf(in_row[v] - maxv);
        out_row[v] = e;        /* stash; pass 3 normalizes in place */
        partial += e;
    }
    sdata[tid] = partial;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_denom = 1.0f / sdata[0];
    __syncthreads();

    /* ---- Pass 3: normalize ----------------------------------------------- */
    for (int v = tid; v < V; v += nthreads) out_row[v] *= inv_denom;
}

/*
 * softmax_backward_kernel — one block back-propagates one row.
 *
 * Factored Jacobian form from softmax_backward in layers.c:
 *   d_in[v] += y[v] * (d_out[v] - s),   s = sum_j d_out[j] * y[j]
 * where y = out is the cached forward output. One reduction computes the
 * row-scalar s; then each thread writes its strided slice of d_in with +=.
 */
__global__ void softmax_backward_kernel(float *d_in, const float *d_out,
                                        const float *out, int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float *out_row   = out   + (size_t)row * V;
    const float *d_out_row = d_out + (size_t)row * V;
    float       *d_in_row  = d_in  + (size_t)row * V;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ float sdata[];

    /* row-scalar s = sum_j d_out[j] * y[j] */
    float partial = 0.0f;
    for (int v = tid; v < V; v += nthreads) partial += d_out_row[v] * out_row[v];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float s_scalar = sdata[0];
    __syncthreads();

    for (int v = tid; v < V; v += nthreads) {
        d_in_row[v] += out_row[v] * (d_out_row[v] - s_scalar);
    }
}

/*
 * pick_block_threads — largest power of two <= min(V, SM_MAX_THREADS), >= 1.
 * The reductions need a power-of-two block size; rows longer than the block are
 * handled by the strided loops. Mirrors the same-named helper in layernorm.cu
 * (both delegate the pow2 logic to cuda_largest_pow2_leq in cuda_layers.cuh).
 */
static int pick_block_threads(int V) {
    return cuda_largest_pow2_leq(V < SM_MAX_THREADS ? V : SM_MAX_THREADS);
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

extern "C" void softmax_forward_device(float *d_out, const float *d_in,
                                       int N, int V) {
    int threads = pick_block_threads(V);
    dim3 grid(N), block(threads);
    size_t shmem = (size_t)threads * sizeof(float);
    softmax_forward_kernel<<<grid, block, shmem>>>(d_out, d_in, N, V);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void softmax_backward_device(float *d_din, const float *d_dout,
                                        const float *d_out, int N, int V) {
    int threads = pick_block_threads(V);
    dim3 grid(N), block(threads);
    size_t shmem = (size_t)threads * sizeof(float);
    softmax_backward_kernel<<<grid, block, shmem>>>(d_din, d_dout, d_out, N, V);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

extern "C" void softmax_forward_cuda(float *out, const float *in, int N, int V) {
    size_t bytes = (size_t)N * V * sizeof(float);
    float *d_out = NULL, *d_in = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));

    CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    softmax_forward_device(d_out, d_in, N, V);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
}

extern "C" void softmax_backward_cuda(float *d_in, const float *d_out,
                                      const float *out, int N, int V) {
    size_t bytes = (size_t)N * V * sizeof(float);
    float *dd_in = NULL, *dd_out = NULL, *d_out_fwd = NULL;
    CUDA_CHECK(cudaMalloc(&dd_in,     bytes));
    CUDA_CHECK(cudaMalloc(&dd_out,    bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fwd, bytes));

    CUDA_CHECK(cudaMemcpy(dd_in,     d_in,  bytes, cudaMemcpyHostToDevice)); /* += base */
    CUDA_CHECK(cudaMemcpy(dd_out,    d_out, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_fwd, out,   bytes, cudaMemcpyHostToDevice));

    softmax_backward_device(dd_in, dd_out, d_out_fwd, N, V);

    CUDA_CHECK(cudaMemcpy(d_in, dd_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_in));
    CUDA_CHECK(cudaFree(dd_out));
    CUDA_CHECK(cudaFree(d_out_fwd));
}
