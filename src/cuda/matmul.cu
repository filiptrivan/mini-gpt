/*
 * Tiled matrix multiply on the GPU — out = a @ b (forward), plus the two
 * gradient matmuls of the backward pass.
 *
 * This is the most performance-critical kernel in the project (every linear
 * layer and the attention projections are matmuls). The CPU version in
 * src/model/layers.c is a simple triple loop; the FORWARD here uses the classic
 * "shared-memory tiling" technique that every CUDA textbook teaches, because
 * a naive GPU matmul is bottlenecked on global-memory bandwidth.
 *
 * THE IDEA, in plain terms:
 *   Computing out[i][j] needs row i of `a` and column j of `b`. A naive
 *   kernel re-reads those from slow global memory for every output cell, so
 *   each element of `a` and `b` gets pulled from DRAM many times over. Tiling
 *   fixes that: the threads in a 16x16 block cooperate to copy one 16x16 tile
 *   of `a` and one 16x16 tile of `b` into fast on-chip SHARED memory, then
 *   every thread reuses those cached values. We slide the pair of tiles along
 *   the shared K dimension, accumulating partial dot products, until the whole
 *   row x column dot product is done. Each global value is now read once per
 *   tile instead of once per output cell — far fewer slow memory trips.
 *
 * The BACKWARD pass (matmul_backward) is deliberately NOT tiled — it uses the
 * simple "one thread per output element, loop over the contracted dimension"
 * design. Correctness over peak speed: the tiling idea is demonstrated once in
 * the forward kernel; duplicating it for the two transposed-operand backward
 * matmuls would triple the surface area where a subtle indexing bug could hide.
 * See the comment on matmul_backward_kernel_da below.
 */

#include "cuda/cuda_layers.cuh"

/*
 * TILE — side length of the square tile each thread block processes.
 * 16 means a block is 16x16 = 256 threads, each thread owning exactly one
 * output element of the tile. 16 is the conventional sweet spot: two TILE x
 * TILE float tiles (As + Bs) cost 16*16*4*2 = 2 KB of shared memory, well
 * within every GPU's per-block budget, and 256 threads/block keeps the SM
 * occupancy high. The PLAN.md kernel table pins this at TILE=16. The backward
 * kernels reuse it purely as their 2D block side length (no shared memory).
 */
#define TILE 16

/*
 * matmul_forward_kernel — one thread computes one out[row][col] element.
 *
 * Grid/block layout (set up by the launcher below):
 *   blockDim  = (TILE, TILE)                    -> 256 threads per block
 *   gridDim.x = ceil(N / TILE)  (covers columns of out / b)
 *   gridDim.y = ceil(M / TILE)  (covers rows of out / a)
 *
 * threadIdx.(x,y) is the thread's position WITHIN its tile; combined with the
 * block index it gives the global (row, col) this thread is responsible for.
 *
 * Boundary handling: M, K, N need NOT be multiples of TILE. Threads whose
 * (row, col) falls outside the matrix still participate in the cooperative
 * tile loads (so __syncthreads() stays uniform across the block — every
 * thread must hit every barrier or the kernel hangs), but they load 0.0f and
 * skip the final write. The zero padding is harmless: 0 * anything = 0 adds
 * nothing to the real dot products.
 */
__global__ void matmul_forward_kernel(float *out, const float *a, const float *b,
                                      int M, int K, int N) {
    /* Shared-memory tiles: one block-local copy of a TILE x TILE chunk of
     * `a` and of `b`. __shared__ memory is on-chip and visible to every
     * thread in the block; it is the cache we fill cooperatively below. */
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    /* Global output coordinate this thread owns. */
    int row = blockIdx.y * TILE + threadIdx.y;   /* row index into out and a */
    int col = blockIdx.x * TILE + threadIdx.x;   /* col index into out and b */

    /* Running dot-product accumulator for out[row][col], kept in a register. */
    float acc = 0.0f;

    /* Number of TILE-wide steps needed to sweep the shared dimension K.
     * ceil(K / TILE): the last step may be a partial tile, handled by the
     * bounds checks during the load. */
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        /* --- Cooperative load: each thread fetches ONE element of each tile. ---
         *
         * As[ty][tx] should hold a[row][ t*TILE + tx ]  (this block's slice of
         * row `row`, advancing across columns of a as t grows).
         * Bs[ty][tx] should hold b[ t*TILE + ty ][col]  (this block's slice of
         * column `col`, advancing down rows of b as t grows). */
        int a_col = t * TILE + threadIdx.x;      /* column of a this thread loads */
        int b_row = t * TILE + threadIdx.y;      /* row of b this thread loads */

        /* Guard every global read. Out-of-range -> store 0 so the multiply
         * contributes nothing (and we never touch memory outside the arrays). */
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? a[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? b[b_row * N + col] : 0.0f;

        /* Barrier: wait until ALL threads in the block have finished their
         * loads before anyone reads the tiles. Without this a thread could
         * read a tile slot its neighbor hasn't written yet. */
        __syncthreads();

        /* Multiply the two cached tiles together, accumulating into acc.
         * This inner loop reads only fast shared memory, TILE times. */
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        /* Second barrier: make sure every thread is done reading the current
         * tiles before the next iteration overwrites them with the next slice. */
        __syncthreads();
    }

    /* Only threads mapped to a real output cell write back. */
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

/*
 * matmul_backward_kernel_da — d_a[i,k] += sum_j d_out[i,j] * b[k,j].
 *
 * This is the gradient w.r.t. the first matmul input, mathematically
 * d_a = d_out @ b^T (shape M x K). One thread owns one output cell d_a[i,k]
 * and walks the contracted dimension j (length N) itself:
 *   for j in [0,N): sum += d_out[i*N + j] * b[k*N + j]
 * Both d_out's row i and b's row k are contiguous, so the inner loop streams
 * two cache-friendly arrays — the same access pattern as the CPU version's
 * pass 1. The result is ADDED (+=) so it accumulates onto whatever the caller
 * already had (the project-wide backward convention).
 *
 * Grid: blockDim = (TILE, TILE); gridDim.x covers k (the K axis),
 * gridDim.y covers i (the M axis). Out-of-range threads just return.
 */
__global__ void matmul_backward_kernel_da(float *d_a, const float *d_out,
                                          const float *b, int M, int K, int N) {
    int i = blockIdx.y * TILE + threadIdx.y;   /* row of d_a / d_out */
    int k = blockIdx.x * TILE + threadIdx.x;   /* col of d_a / row of b */
    if (i >= M || k >= K) return;

    const float *d_out_row = d_out + (size_t)i * N;   /* d_out[i, 0..N-1] */
    const float *b_row     = b     + (size_t)k * N;   /* b[k, 0..N-1]     */
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += d_out_row[j] * b_row[j];
    }
    d_a[(size_t)i * K + k] += sum;
}

/*
 * matmul_backward_kernel_db — d_b[k,j] += sum_i a[i,k] * d_out[i,j].
 *
 * Gradient w.r.t. the second matmul input, mathematically d_b = a^T @ d_out
 * (shape K x N). One thread owns one cell d_b[k,j] and walks the contracted
 * dimension i (length M). Note this stride is NOT contiguous (a[i*K+k] and
 * d_out[i*N+j] both jump a whole row per i), but correctness is the goal here;
 * the value is summed in a register and written once. += accumulates.
 *
 * Grid: blockDim = (TILE, TILE); gridDim.x covers j (the N axis),
 * gridDim.y covers k (the K axis).
 */
__global__ void matmul_backward_kernel_db(float *d_b, const float *d_out,
                                          const float *a, int M, int K, int N) {
    int k = blockIdx.y * TILE + threadIdx.y;   /* row of d_b / col of a */
    int j = blockIdx.x * TILE + threadIdx.x;   /* col of d_b / d_out    */
    if (k >= K || j >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += a[(size_t)i * K + k] * d_out[(size_t)i * N + j];
    }
    d_b[(size_t)k * N + j] += sum;
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

/*
 * matmul_forward_device — launch the tiled forward kernel on DEVICE pointers.
 * All three pointers are already device memory; this only sizes the 2D grid
 * (one block per TILE x TILE output tile) and launches. See cuda_layers.cuh.
 */
extern "C" void matmul_forward_device(float *d_out, const float *d_a,
                                      const float *d_b, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_forward_kernel<<<grid, block>>>(d_out, d_a, d_b, M, K, N);
    CUDA_CHECK(cudaGetLastError());   /* catch an invalid launch configuration */
}

/*
 * matmul_backward_device — launch both gradient kernels on DEVICE pointers.
 * d_da and d_db are independent buffers written by two separate kernels; both
 * accumulate with += onto their current device contents.
 */
extern "C" void matmul_backward_device(float *d_da, float *d_db,
                                       const float *d_dout, const float *d_a,
                                       const float *d_b, int M, int K, int N) {
    dim3 block(TILE, TILE);

    /* d_a is M x K: grid.x covers K, grid.y covers M. */
    dim3 grid_da((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_backward_kernel_da<<<grid_da, block>>>(d_da, d_dout, d_b, M, K, N);
    CUDA_CHECK(cudaGetLastError());

    /* d_b is K x N: grid.x covers N, grid.y covers K. */
    dim3 grid_db((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
    matmul_backward_kernel_db<<<grid_db, block>>>(d_db, d_dout, d_a, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

/*
 * matmul_forward_cuda — host wrapper. See cuda_layers.cuh for the contract.
 *
 * Allocates device buffers, copies a and b up, calls the device launcher,
 * copies out back, frees. size_t math (not int) so large matrices don't
 * overflow the byte count. Every CUDA call is CUDA_CHECK-wrapped; the final
 * device->host copy blocks until the kernel has actually finished.
 */
extern "C" void matmul_forward_cuda(float *out, const float *a, const float *b,
                                    int M, int K, int N) {
    size_t bytes_a   = (size_t)M * K * sizeof(float);
    size_t bytes_b   = (size_t)K * N * sizeof(float);
    size_t bytes_out = (size_t)M * N * sizeof(float);

    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_a,   bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b,   bytes_b));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice));

    matmul_forward_device(d_out, d_a, d_b, M, K, N);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

/*
 * matmul_backward_cuda — host wrapper for the two gradient matmuls.
 *
 * d_a and d_b are accumulated with += (the convention), so we copy the
 * caller's CURRENT d_a / d_b up to the device before launching, let the kernels
 * add onto them, and copy the results back. The forward inputs a and b and the
 * upstream gradient d_out are read-only.
 */
extern "C" void matmul_backward_cuda(float *d_a, float *d_b, const float *d_out,
                                     const float *a, const float *b,
                                     int M, int K, int N) {
    size_t bytes_a   = (size_t)M * K * sizeof(float);   /* a and d_a */
    size_t bytes_b   = (size_t)K * N * sizeof(float);   /* b and d_b */
    size_t bytes_out = (size_t)M * N * sizeof(float);   /* d_out     */

    float *dd_a = NULL, *dd_b = NULL, *dd_out = NULL, *d_a_in = NULL, *d_b_in = NULL;
    CUDA_CHECK(cudaMalloc(&dd_a,   bytes_a));   /* device copy of d_a (grad)  */
    CUDA_CHECK(cudaMalloc(&dd_b,   bytes_b));   /* device copy of d_b (grad)  */
    CUDA_CHECK(cudaMalloc(&dd_out, bytes_out)); /* device copy of d_out       */
    CUDA_CHECK(cudaMalloc(&d_a_in, bytes_a));   /* device copy of forward a   */
    CUDA_CHECK(cudaMalloc(&d_b_in, bytes_b));   /* device copy of forward b   */

    /* copy the grad accumulators UP too, so the kernel's += starts from the
     * caller's current contents (lets tests verify accumulation). */
    CUDA_CHECK(cudaMemcpy(dd_a,   d_a,   bytes_a,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_b,   d_b,   bytes_b,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_out, d_out, bytes_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_in, a,     bytes_a,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_in, b,     bytes_b,   cudaMemcpyHostToDevice));

    matmul_backward_device(dd_a, dd_b, dd_out, d_a_in, d_b_in, M, K, N);

    CUDA_CHECK(cudaMemcpy(d_a, dd_a, bytes_a, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_b, dd_b, bytes_b, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dd_a));
    CUDA_CHECK(cudaFree(dd_b));
    CUDA_CHECK(cudaFree(dd_out));
    CUDA_CHECK(cudaFree(d_a_in));
    CUDA_CHECK(cudaFree(d_b_in));
}
