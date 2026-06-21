/*
 * Cross-entropy loss on the GPU — forward (per-position loss) and the fused
 * softmax+cross-entropy gradient (backward).
 *
 * These mirror the loss math that lives INLINE in gpt.c (not in layers.c),
 * because it is specific to the GPT output head. There the forward computes,
 * per position, the negative log-probability of the correct next token, and
 * the backward uses the classic fused result for softmax-then-cross-entropy:
 *
 *   forward (per position n):   loss[n] = -log( probs[n, target[n]] )
 *   backward (per logit):       d_logits[n, j] = (probs[n, j] - 1{j==target[n]}) / N
 *
 * N is the number of positions (B*T) — the loss is averaged over all of them,
 * so the gradient carries the same 1/N factor. The fused gradient is why we do
 * NOT need a separate softmax_backward on the output logits: differentiating
 * softmax and -log together collapses to "probabilities minus the one-hot
 * target," which is both simpler and more numerically stable.
 *
 * PARALLELIZATION (PLAN.md): one thread per position (B*T threads). Forward
 * writes one loss[n]; backward writes that position's whole length-V gradient
 * row. The per-position mean is finished on the host (sum the loss array and
 * divide by N) — a trivial amount of work not worth a second reduction kernel.
 */

#include "cuda/cuda_layers.cuh"
#include <math.h>   /* logf used in the forward kernel */

#define CE_BLOCK 256

/*
 * cross_entropy_forward_kernel — loss[n] = -log(probs[n, target[n]]).
 * One thread per position. probs is the softmax output row (length V).
 */
__global__ void cross_entropy_forward_kernel(float *losses, const float *probs,
                                             const int *targets, int N, int V) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int tgt = targets[n];
    float p = probs[(size_t)n * V + tgt];
    losses[n] = -logf(p);
}

/*
 * cross_entropy_backward_kernel — d_logits[n, j] = (probs[n,j] - onehot) / N.
 *
 * One thread per position writes its whole gradient row. This is an ASSIGNMENT
 * (=, not +=) because d_logits is a leaf of the backward graph — nothing else
 * contributes to it — exactly as gpt.c writes it. (In the resident pass the
 * grads_acts block is zeroed first anyway, so = and += would coincide.)
 */
__global__ void cross_entropy_backward_kernel(float *d_logits, const float *probs,
                                              const int *targets, int N, int V) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int tgt = targets[n];
    float invN = 1.0f / (float)N;
    const float *p_row  = probs    + (size_t)n * V;
    float       *dl_row = d_logits + (size_t)n * V;
    for (int j = 0; j < V; j++) {
        float indicator = (j == tgt) ? 1.0f : 0.0f;
        dl_row[j] = (p_row[j] - indicator) * invN;
    }
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

extern "C" void cross_entropy_forward_device(float *d_losses, const float *d_probs,
                                             const int *d_targets, int N, int V) {
    int grid = (N + CE_BLOCK - 1) / CE_BLOCK;
    cross_entropy_forward_kernel<<<grid, CE_BLOCK>>>(d_losses, d_probs,
                                                     d_targets, N, V);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void cross_entropy_backward_device(float *d_dlogits, const float *d_probs,
                                              const int *d_targets, int N, int V) {
    int grid = (N + CE_BLOCK - 1) / CE_BLOCK;
    cross_entropy_backward_kernel<<<grid, CE_BLOCK>>>(d_dlogits, d_probs,
                                                      d_targets, N, V);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

extern "C" void cross_entropy_forward_cuda(float *losses, const float *probs,
                                           const int *targets, int N, int V) {
    size_t bytes_probs  = (size_t)N * V * sizeof(float);
    size_t bytes_losses = (size_t)N * sizeof(float);
    size_t bytes_tgt    = (size_t)N * sizeof(int);

    float *d_losses = NULL, *d_probs = NULL;
    int   *d_tgt = NULL;
    CUDA_CHECK(cudaMalloc(&d_losses, bytes_losses));
    CUDA_CHECK(cudaMalloc(&d_probs,  bytes_probs));
    CUDA_CHECK(cudaMalloc(&d_tgt,    bytes_tgt));

    CUDA_CHECK(cudaMemcpy(d_probs, probs,   bytes_probs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,   targets, bytes_tgt,   cudaMemcpyHostToDevice));

    cross_entropy_forward_device(d_losses, d_probs, d_tgt, N, V);

    CUDA_CHECK(cudaMemcpy(losses, d_losses, bytes_losses, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_losses));
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_tgt));
}

/*
 * cross_entropy_backward_cuda — host wrapper. d_logits is fully overwritten
 * (= semantics), so unlike the other backward wrappers we do NOT copy it up
 * first; we just compute it and copy the result down.
 */
extern "C" void cross_entropy_backward_cuda(float *d_logits, const float *probs,
                                            const int *targets, int N, int V) {
    size_t bytes_logits = (size_t)N * V * sizeof(float);
    size_t bytes_tgt    = (size_t)N * sizeof(int);

    float *dd_logits = NULL, *d_probs = NULL;
    int   *d_tgt = NULL;
    CUDA_CHECK(cudaMalloc(&dd_logits, bytes_logits));
    CUDA_CHECK(cudaMalloc(&d_probs,   bytes_logits));
    CUDA_CHECK(cudaMalloc(&d_tgt,     bytes_tgt));

    CUDA_CHECK(cudaMemcpy(d_probs, probs,   bytes_logits, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,   targets, bytes_tgt,    cudaMemcpyHostToDevice));

    cross_entropy_backward_device(dd_logits, d_probs, d_tgt, N, V);

    CUDA_CHECK(cudaMemcpy(d_logits, dd_logits, bytes_logits, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_logits));
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_tgt));
}
