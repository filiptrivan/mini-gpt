/*
 * Layer normalization on the GPU — forward and backward.
 *
 * Same math as layernorm_forward/layernorm_backward in src/model/layers.c.
 * Forward, per row of length C:
 *   mean  = (1/C) * sum(in)
 *   var   = (1/C) * sum((in - mean)^2)
 *   rstd  = 1 / sqrt(var + eps)
 *   out   = gamma * (in - mean) * rstd + beta
 *
 * PARALLELIZATION STRATEGY (from PLAN.md's kernel table):
 *   one thread BLOCK per row (there are N = B*T rows), block size ~min(C,256).
 *   The threads in a block cooperate to reduce that row down to a single mean
 *   and variance (and, in backward, two more row-scalars) using shared memory,
 *   then each thread normalizes a strided slice of the row's columns.
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

/* Upper bound on threads per block we will launch for one row. The launcher
 * picks min(C, this) rounded DOWN to a power of two (see below). */
#define LN_MAX_THREADS 256

/*
 * layernorm_forward_kernel — one block normalizes one row.
 *
 * Launch contract (set up by the launcher):
 *   gridDim.x  = N            (one block per row)
 *   blockDim.x = a POWER OF TWO, <= min(C, LN_MAX_THREADS)
 *   shared mem = blockDim.x * sizeof(float)
 *
 * The power-of-two block size matters: the tree reduction below repeatedly
 * halves the active thread count (s = nthreads/2, then /2, ...). That pattern
 * only sums every element correctly when nthreads is a power of two, so the
 * launcher guarantees it. A row longer than the block (C > blockDim.x) is fine
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
     * is the third <<<>>> launch argument in the launcher. We reuse the same
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
 * layernorm_backward_kernel — one block back-propagates one row.
 *
 * Mirrors layernorm_backward in layers.c. Per row we need mean and rstd
 * (recomputed, two reductions) and then two more row-scalars:
 *   sum1 = (1/C) * sum_j d_x_hat[j]
 *   sum2 = (1/C) * sum_j d_x_hat[j] * x_hat[j]
 * where d_x_hat[c] = d_out[c] * gamma[c] and x_hat[c] = (in[c]-mean)*rstd.
 * Then:  d_in[c] += rstd * (d_x_hat[c] - sum1 - x_hat[c] * sum2).
 *
 * The gamma/beta gradients are SHARED across all N rows:
 *   d_gamma[c] += d_out[c] * x_hat[c]      d_beta[c] += d_out[c]
 * Every block (row) adds into the SAME d_gamma[c] / d_beta[c], so concurrent
 * blocks would race — hence atomicAdd. (Within one block each column c is owned
 * by exactly one thread via the strided loop, so the only conflict is between
 * blocks, which is exactly what atomicAdd serializes.)
 *
 * Shared memory: 2 * blockDim.x floats — s1 for the sum1 reduction (also reused
 * for the mean and variance reductions) and s2 for the sum2 reduction, folded
 * together in one tree-reduction loop.
 */
__global__ void layernorm_backward_kernel(float *d_in, float *d_gamma,
                                          float *d_beta, const float *d_out,
                                          const float *in, const float *gamma,
                                          int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float *in_row    = in    + (size_t)row * C;
    const float *d_out_row = d_out + (size_t)row * C;
    float       *d_in_row  = d_in  + (size_t)row * C;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    float inv_C  = 1.0f / (float)C;

    extern __shared__ float sm[];      /* size 2*nthreads (see launcher) */
    float *s1 = sm;                    /* first  half: sum reductions    */
    float *s2 = sm + nthreads;         /* second half: sum2 reduction    */

    /* ---- mean (reuse s1) -------------------------------------------------- */
    float partial = 0.0f;
    for (int c = tid; c < C; c += nthreads) partial += in_row[c];
    s1[tid] = partial;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) s1[tid] += s1[tid + s];
        __syncthreads();
    }
    float mean = s1[0] * inv_C;
    __syncthreads();

    /* ---- variance -> rstd (reuse s1) ------------------------------------- */
    partial = 0.0f;
    for (int c = tid; c < C; c += nthreads) {
        float d = in_row[c] - mean;
        partial += d * d;
    }
    s1[tid] = partial;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) s1[tid] += s1[tid + s];
        __syncthreads();
    }
    float var  = s1[0] * inv_C;
    float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);
    __syncthreads();

    /* ---- sum1, sum2, and the gamma/beta scatter -------------------------- */
    float p1 = 0.0f, p2 = 0.0f;
    for (int c = tid; c < C; c += nthreads) {
        float x_hat   = (in_row[c] - mean) * rstd;
        float d_x_hat = d_out_row[c] * gamma[c];
        p1 += d_x_hat;
        p2 += d_x_hat * x_hat;
        /* gamma/beta are shared across rows -> accumulate across blocks. */
        atomicAdd(&d_gamma[c], d_out_row[c] * x_hat);
        atomicAdd(&d_beta[c],  d_out_row[c]);
    }
    s1[tid] = p1;
    s2[tid] = p2;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s1[tid] += s1[tid + s];
            s2[tid] += s2[tid + s];
        }
        __syncthreads();
    }
    float sum1 = s1[0] * inv_C;
    float sum2 = s2[0] * inv_C;
    __syncthreads();

    /* ---- d_in (each (row,c) written by exactly one thread; += onto skip) -- */
    for (int c = tid; c < C; c += nthreads) {
        float x_hat   = (in_row[c] - mean) * rstd;
        float d_x_hat = d_out_row[c] * gamma[c];
        d_in_row[c] += rstd * (d_x_hat - sum1 - x_hat * sum2);
    }
}

/*
 * pick_block_threads — largest power of two <= min(C, LN_MAX_THREADS), >= 1.
 *
 * The reductions in the kernels require a power-of-two block size; we round
 * DOWN so the count never exceeds C unnecessarily nor LN_MAX_THREADS. (Threads
 * beyond C just contribute 0 to the sums via the strided loop.) The pow2 logic
 * itself is the shared cuda_largest_pow2_leq helper in cuda_layers.cuh.
 */
static int pick_block_threads(int C) {
    return cuda_largest_pow2_leq(C < LN_MAX_THREADS ? C : LN_MAX_THREADS);
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

/*
 * layernorm_forward_device — launch the forward kernel on DEVICE pointers.
 * One block per row, power-of-two thread count, matching dynamic shared mem.
 */
extern "C" void layernorm_forward_device(float *d_out, const float *d_in,
                                         const float *d_gamma,
                                         const float *d_beta, int N, int C) {
    int threads = pick_block_threads(C);
    dim3 grid(N);                 /* one block per row */
    dim3 block(threads);
    size_t shmem = (size_t)threads * sizeof(float);
    layernorm_forward_kernel<<<grid, block, shmem>>>(d_out, d_in, d_gamma,
                                                     d_beta, N, C);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * layernorm_backward_device — launch the backward kernel on DEVICE pointers.
 * Same one-block-per-row layout; shared memory is DOUBLED (s1 + s2).
 * d_in / d_gamma / d_beta accumulate with += onto their current device values.
 */
extern "C" void layernorm_backward_device(float *d_din, float *d_dgamma,
                                          float *d_dbeta, const float *d_dout,
                                          const float *d_in, const float *d_gamma,
                                          int N, int C) {
    int threads = pick_block_threads(C);
    dim3 grid(N);
    dim3 block(threads);
    size_t shmem = (size_t)2 * threads * sizeof(float);
    layernorm_backward_kernel<<<grid, block, shmem>>>(d_din, d_dgamma, d_dbeta,
                                                      d_dout, d_in, d_gamma, N, C);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

/*
 * layernorm_forward_cuda — host wrapper. See cuda_layers.cuh for the contract.
 * Copies in/gamma/beta to the device, launches, copies the result back, frees.
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

    layernorm_forward_device(d_out, d_in, d_gamma, d_beta, N, C);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes_io, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

/*
 * layernorm_backward_cuda — host wrapper. d_in/d_gamma/d_beta accumulate with
 * +=, so we copy the caller's current contents up before launching, then copy
 * the results back. d_out, in, gamma are read-only forward caches.
 */
extern "C" void layernorm_backward_cuda(float *d_in, float *d_gamma,
                                        float *d_beta, const float *d_out,
                                        const float *in, const float *gamma,
                                        int N, int C) {
    size_t bytes_io = (size_t)N * C * sizeof(float);   /* d_in, in, d_out */
    size_t bytes_gb = (size_t)C * sizeof(float);       /* d_gamma, d_beta, gamma */

    float *dd_in = NULL, *dd_gamma = NULL, *dd_beta = NULL;
    float *dd_out = NULL, *d_in_fwd = NULL, *d_gamma_fwd = NULL;
    CUDA_CHECK(cudaMalloc(&dd_in,       bytes_io));
    CUDA_CHECK(cudaMalloc(&dd_gamma,    bytes_gb));
    CUDA_CHECK(cudaMalloc(&dd_beta,     bytes_gb));
    CUDA_CHECK(cudaMalloc(&dd_out,      bytes_io));
    CUDA_CHECK(cudaMalloc(&d_in_fwd,    bytes_io));
    CUDA_CHECK(cudaMalloc(&d_gamma_fwd, bytes_gb));

    /* grad accumulators copied up so the kernel's += starts from caller state */
    CUDA_CHECK(cudaMemcpy(dd_in,       d_in,    bytes_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_gamma,    d_gamma, bytes_gb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_beta,     d_beta,  bytes_gb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_out,      d_out,   bytes_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_fwd,    in,      bytes_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma_fwd, gamma,   bytes_gb, cudaMemcpyHostToDevice));

    layernorm_backward_device(dd_in, dd_gamma, dd_beta, dd_out,
                              d_in_fwd, d_gamma_fwd, N, C);

    CUDA_CHECK(cudaMemcpy(d_in,    dd_in,    bytes_io, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_gamma, dd_gamma, bytes_gb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_beta,  dd_beta,  bytes_gb, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dd_in));
    CUDA_CHECK(cudaFree(dd_gamma));
    CUDA_CHECK(cudaFree(dd_beta));
    CUDA_CHECK(cudaFree(dd_out));
    CUDA_CHECK(cudaFree(d_in_fwd));
    CUDA_CHECK(cudaFree(d_gamma_fwd));
}
