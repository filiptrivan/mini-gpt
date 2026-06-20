/*
 * Tiled matrix multiply on the GPU — out = a @ b.
 *
 * This is the most performance-critical kernel in the project (every linear
 * layer and the attention projections are matmuls). The CPU version in
 * src/model/layers.c is a simple triple loop; here we use the classic
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
 */

#include "cuda/cuda_layers.cuh"

/*
 * TILE — side length of the square tile each thread block processes.
 * 16 means a block is 16x16 = 256 threads, each thread owning exactly one
 * output element of the tile. 16 is the conventional sweet spot: two TILE x
 * TILE float tiles (As + Bs) cost 16*16*4*2 = 2 KB of shared memory, well
 * within every GPU's per-block budget, and 256 threads/block keeps the SM
 * occupancy high. The PLAN.md kernel table pins this at TILE=16.
 */
#define TILE 16

/*
 * matmul_forward_kernel — one thread computes one out[row][col] element.
 *
 * Grid/block layout (set up by the wrapper below):
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
 * matmul_forward_cuda — host wrapper. See cuda_layers.cuh for the contract.
 *
 * Steps:
 *   1. Allocate device buffers for a, b, out.
 *   2. Copy a and b host->device.
 *   3. Launch the tiled kernel over a 2D grid that covers all of out.
 *   4. Copy out device->host.
 *   5. Free the device buffers.
 *
 * Every CUDA call is wrapped in CUDA_CHECK so a failure aborts loudly with a
 * file/line rather than silently returning wrong numbers. After the launch we
 * check cudaGetLastError() (catches bad launch configs, which the launch
 * syntax itself cannot report) and then cudaDeviceSynchronize() via the
 * device->host copy, which blocks until the kernel has actually finished.
 */
extern "C" void matmul_forward_cuda(float *out, const float *a, const float *b,
                                    int M, int K, int N) {
    /* size_t math (not int) so large matrices don't overflow the byte count. */
    size_t bytes_a   = (size_t)M * K * sizeof(float);
    size_t bytes_b   = (size_t)K * N * sizeof(float);
    size_t bytes_out = (size_t)M * N * sizeof(float);

    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_a,   bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b,   bytes_b));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice));

    /* One thread per output element; one block per TILE x TILE output tile.
     * grid.x covers the N (column) axis, grid.y covers the M (row) axis. */
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_forward_kernel<<<grid, block>>>(d_out, d_a, d_b, M, K, N);
    CUDA_CHECK(cudaGetLastError());   /* catch an invalid launch configuration */

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
