/*
 * Layer normalization on the GPU — row-wise normalize + learned affine.
 *
 * Same math as layernorm_forward() in src/model/layers.c, per row of length C:
 *   mean  = (1/C) * sum(in)
 *   var   = (1/C) * sum((in - mean)^2)
 *   out   = gamma * (in - mean) / sqrt(var + eps) + beta
 *
 * PARALLELIZATION STRATEGY (from PLAN.md's kernel table):
 *   one thread BLOCK per row (there are N = B*T rows), block size ~min(C,256).
 *   The threads in a block cooperate to reduce that row down to a single mean
 *   and a single variance using shared memory, then each thread normalizes a
 *   strided slice of the row's columns.
 *
 * Why a whole block per row instead of one thread per row? Because computing a
 * mean/variance is a REDUCTION (sum over C elements). Spreading the row across
 * many threads and combining their partial sums in fast shared memory is far
 * quicker than making a single thread loop over all C columns alone — and it
 * mirrors how the real GPU training loop will want to use the hardware.
 */

#include "cuda/cuda_layers.cuh"

/* Must match LAYERNORM_EPS in src/model/layers.c so GPU and CPU agree. */
#define LAYERNORM_EPS 1e-5f

/* Upper bound on threads per block we will launch for one row. The wrapper
 * picks min(C, this) rounded DOWN to a power of two (see below). */
#define LN_MAX_THREADS 256

/*
 * layernorm_forward_kernel — one block normalizes one row.
 *
 * Launch contract (set up by the wrapper):
 *   gridDim.x  = N            (one block per row)
 *   blockDim.x = a POWER OF TWO, <= min(C, LN_MAX_THREADS)
 *   shared mem = blockDim.x * sizeof(float)
 *
 * The power-of-two block size matters: the tree reduction below repeatedly
 * halves the active thread count (s = nthreads/2, then /2, ...). That pattern
 * only sums every element correctly when nthreads is a power of two, so the
 * wrapper guarantees it. A row longer than the block (C > blockDim.x) is fine
 * — each thread simply strides over several columns (the `c += nthreads` step).
 */
__global__ void layernorm_forward_kernel(float *out, const float *in,
                                         const float *gamma, const float *beta,
                                         int N, int C) {
    int row = blockIdx.x;                 /* this block's row, 0..N-1 */
    if (row >= N) return;                 /* defensive: grid is sized to N anyway */

    const float *in_row  = in  + (size_t)row * C;
    float       *out_row = out + (size_t)row * C;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    /* Dynamically-sized shared scratch for the reductions. Its size (in bytes)
     * is the third <<<>>> launch argument in the wrapper. We reuse the same
     * buffer for the sum reduction and then the sum-of-squares reduction. */
    extern __shared__ float sdata[];

    /* ---- Pass 1: row sum -> mean ------------------------------------------
     * Each thread sums its strided slice of the row into a private partial,
     * then we tree-reduce the partials in shared memory down to sdata[0]. */
    float partial = 0.0f;
    for (int c = tid; c < C; c += nthreads) {
        partial += in_row[c];
    }
    sdata[tid] = partial;
    __syncthreads();

    /* Tree reduction: each step folds the upper half of the active range into
     * the lower half. After log2(nthreads) steps, sdata[0] holds the total. */
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();   /* every thread must reach this each step (uniform barrier) */
    }
    float mean = sdata[0] / (float)C;
    __syncthreads();       /* finish reading sdata[0] before Pass 2 overwrites it */

    /* ---- Pass 2: sum of squared deviations -> variance -> rstd ------------- */
    partial = 0.0f;
    for (int c = tid; c < C; c += nthreads) {
        float d = in_row[c] - mean;
        partial += d * d;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float var = sdata[0] / (float)C;
    /* 1/sqrt(var+eps): match the CPU's `1.0f / sqrtf(...)` exactly (rather than
     * rsqrtf) so the two implementations stay within tight float tolerance. */
    float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);

    /* ---- Pass 3: normalize, then apply learned per-feature scale/shift ----- */
    for (int c = tid; c < C; c += nthreads) {
        float x_hat = (in_row[c] - mean) * rstd;
        out_row[c]  = gamma[c] * x_hat + beta[c];
    }
}

/*
 * pick_block_threads — largest power of two <= min(C, LN_MAX_THREADS), >= 1.
 *
 * The reduction in the kernel requires a power-of-two block size. We round
 * DOWN (not up) so the count never exceeds C unnecessarily and never exceeds
 * LN_MAX_THREADS. Examples: C=128 -> 128, C=200 -> 128, C=5 -> 4, C=1 -> 1.
 * (Threads beyond C just contribute 0 to the sums via the strided loop.)
 */
static int pick_block_threads(int C) {
    int cap = (C < LN_MAX_THREADS) ? C : LN_MAX_THREADS;
    int threads = 1;
    while (threads * 2 <= cap) {
        threads *= 2;
    }
    return threads;
}

/*
 * layernorm_forward_cuda — host wrapper. See cuda_layers.cuh for the contract.
 *
 * Copies in/gamma/beta to the device, launches one block per row with a
 * power-of-two thread count and matching dynamic shared memory, copies the
 * result back, and frees everything. Every CUDA call is CUDA_CHECK-wrapped.
 */
extern "C" void layernorm_forward_cuda(float *out, const float *in,
                                       const float *gamma, const float *beta,
                                       int N, int C) {
    size_t bytes_io = (size_t)N * C * sizeof(float);   /* in and out */
    size_t bytes_gb = (size_t)C * sizeof(float);       /* gamma and beta */

    float *d_in = NULL, *d_out = NULL, *d_gamma = NULL, *d_beta = NULL;
    CUDA_CHECK(cudaMalloc(&d_in,    bytes_io));
    CUDA_CHECK(cudaMalloc(&d_out,   bytes_io));
    CUDA_CHECK(cudaMalloc(&d_gamma, bytes_gb));
    CUDA_CHECK(cudaMalloc(&d_beta,  bytes_gb));

    CUDA_CHECK(cudaMemcpy(d_in,    in,    bytes_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma, bytes_gb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  beta,  bytes_gb, cudaMemcpyHostToDevice));

    int threads = pick_block_threads(C);
    dim3 grid(N);                 /* one block per row */
    dim3 block(threads);
    size_t shmem = (size_t)threads * sizeof(float);   /* scratch for the reductions */

    layernorm_forward_kernel<<<grid, block, shmem>>>(d_out, d_in, d_gamma, d_beta, N, C);
    CUDA_CHECK(cudaGetLastError());   /* catch an invalid launch configuration */

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes_io, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}
