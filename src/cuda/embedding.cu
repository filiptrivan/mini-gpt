/*
 * Token + position embedding on the GPU — forward (gather) and backward
 * (scatter). The CPU twins are embed_forward/embed_backward in layers.c.
 *
 * Forward:  out[b,t,c] = wte[tokens[b,t], c] + wpe[t, c]
 * Backward: d_wte[tokens[b,t], c] += d_out[b,t,c]
 *           d_wpe[t,           c] += d_out[b,t,c]
 *
 * PARALLELIZATION: one thread per output element (B*T*C of them), a flat 1-D
 * grid. From the linear index we recover c, the (b,t) position, the position t
 * and the token id. Forward is a pure gather (each thread writes its own
 * out[idx], no conflicts). Backward is a scatter and DOES conflict:
 *   - the same token id appears at many (b,t) positions -> many threads add to
 *     the same row of d_wte;
 *   - every batch element shares position t -> B threads add to the same row of
 *     d_wpe.
 * Both collisions are resolved with atomicAdd (the PLAN.md note: "Backward for
 * embedding uses atomicAdd"). Without it, concurrent += would lose updates.
 */

#include "cuda/cuda_layers.cuh"

#define EMB_BLOCK 256

/*
 * embed_forward_kernel — one thread fills one out[idx] (idx over B*T*C).
 * Recover c (channel), bt (flattened batch*time), t (position), and the token
 * id, then add the two embedding rows. Pure gather — no atomics.
 */
__global__ void embed_forward_kernel(float *out, const int *tokens,
                                     const float *wte, const float *wpe,
                                     int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;

    int c     = idx % C;            /* channel within the embedding vector */
    int bt    = idx / C;            /* which (b,t) position, flattened     */
    int t     = bt % T;            /* position index, for wpe lookup      */
    int token = tokens[bt];         /* token id, for wte lookup            */

    out[idx] = wte[(size_t)token * C + c] + wpe[(size_t)t * C + c];
}

/*
 * embed_backward_kernel — scatter d_out into the two embedding tables.
 * Each thread owns one d_out[idx] and adds it into BOTH tables via atomicAdd
 * (see the file header for why both collide). Accumulates onto current values.
 */
__global__ void embed_backward_kernel(float *d_wte, float *d_wpe,
                                      const float *d_out, const int *tokens,
                                      int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;

    int c     = idx % C;
    int bt    = idx / C;
    int t     = bt % T;
    int token = tokens[bt];
    float g   = d_out[idx];

    atomicAdd(&d_wte[(size_t)token * C + c], g);
    atomicAdd(&d_wpe[(size_t)t     * C + c], g);
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

extern "C" void embed_forward_device(float *d_out, const int *d_tokens,
                                     const float *d_wte, const float *d_wpe,
                                     int B, int T, int C) {
    int total = B * T * C;
    int grid = (total + EMB_BLOCK - 1) / EMB_BLOCK;
    embed_forward_kernel<<<grid, EMB_BLOCK>>>(d_out, d_tokens, d_wte, d_wpe,
                                              B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void embed_backward_device(float *d_dwte, float *d_dwpe,
                                      const float *d_dout, const int *d_tokens,
                                      int B, int T, int C) {
    int total = B * T * C;
    int grid = (total + EMB_BLOCK - 1) / EMB_BLOCK;
    embed_backward_kernel<<<grid, EMB_BLOCK>>>(d_dwte, d_dwpe, d_dout, d_tokens,
                                               B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

/*
 * embed_forward_cuda — host wrapper. Needs vocab_size to size the wte table on
 * the device (the kernel only indexes it). wpe needs just its first T rows
 * (positions 0..T-1 are all that get read), so T*C floats suffice for it.
 */
extern "C" void embed_forward_cuda(float *out, const int *tokens,
                                   const float *wte, const float *wpe,
                                   int B, int T, int C, int vocab_size) {
    size_t bytes_out = (size_t)B * T * C * sizeof(float);
    size_t bytes_tok = (size_t)B * T * sizeof(int);
    size_t bytes_wte = (size_t)vocab_size * C * sizeof(float);
    size_t bytes_wpe = (size_t)T * C * sizeof(float);   /* only rows 0..T-1 */

    float *d_out = NULL, *d_wte = NULL, *d_wpe = NULL;
    int   *d_tok = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
    CUDA_CHECK(cudaMalloc(&d_wte, bytes_wte));
    CUDA_CHECK(cudaMalloc(&d_wpe, bytes_wpe));
    CUDA_CHECK(cudaMalloc(&d_tok, bytes_tok));

    CUDA_CHECK(cudaMemcpy(d_wte, wte,    bytes_wte, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wpe, wpe,    bytes_wpe, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tok, tokens, bytes_tok, cudaMemcpyHostToDevice));

    embed_forward_device(d_out, d_tok, d_wte, d_wpe, B, T, C);

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_wte));
    CUDA_CHECK(cudaFree(d_wpe));
    CUDA_CHECK(cudaFree(d_tok));
}

/*
 * embed_backward_cuda — host wrapper. d_wte/d_wpe accumulate with += so we copy
 * the caller's current contents up first. As in forward, the wpe table only
 * needs its first T rows (positions beyond T-1 receive no gradient this call).
 */
extern "C" void embed_backward_cuda(float *d_wte, float *d_wpe,
                                    const float *d_out, const int *tokens,
                                    int B, int T, int C, int vocab_size) {
    size_t bytes_out = (size_t)B * T * C * sizeof(float);
    size_t bytes_tok = (size_t)B * T * sizeof(int);
    size_t bytes_wte = (size_t)vocab_size * C * sizeof(float);
    size_t bytes_wpe = (size_t)T * C * sizeof(float);   /* only rows 0..T-1 */

    float *dd_wte = NULL, *dd_wpe = NULL, *dd_out = NULL;
    int   *d_tok = NULL;
    CUDA_CHECK(cudaMalloc(&dd_wte, bytes_wte));
    CUDA_CHECK(cudaMalloc(&dd_wpe, bytes_wpe));
    CUDA_CHECK(cudaMalloc(&dd_out, bytes_out));
    CUDA_CHECK(cudaMalloc(&d_tok,  bytes_tok));

    /* grad accumulators copied up so atomicAdd starts from caller state */
    CUDA_CHECK(cudaMemcpy(dd_wte, d_wte,  bytes_wte, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_wpe, d_wpe,  bytes_wpe, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_out, d_out,  bytes_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tok,  tokens, bytes_tok, cudaMemcpyHostToDevice));

    embed_backward_device(dd_wte, dd_wpe, dd_out, d_tok, B, T, C);

    CUDA_CHECK(cudaMemcpy(d_wte, dd_wte, bytes_wte, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_wpe, dd_wpe, bytes_wpe, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_wte));
    CUDA_CHECK(cudaFree(dd_wpe));
    CUDA_CHECK(cudaFree(dd_out));
    CUDA_CHECK(cudaFree(d_tok));
}
