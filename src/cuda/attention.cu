/*
 * Causal multi-head self-attention on the GPU — forward and backward.
 *
 * The CPU twins are attention_forward/attention_backward in gpt.c (exposed via
 * gpt.h so the parity test can use them as oracles). The math is mirrored
 * exactly; read the gpt.c block comments for the full derivation. Quick recap,
 * per head and per query position t (head_dim = C / NH, scale = 1/sqrt(head_dim)):
 *   forward:  score[t2] = scale * (Q_t . K_t2)  for t2 <= t  (causal mask)
 *             att[t2]   = softmax(score)         (stabilized by subtracting max)
 *             out_t     = sum_t2 att[t2] * V_t2
 *
 * PARALLELIZATION: one thread per (b, h, t) query position — B*NH*T threads in
 * a flat 1-D grid. Each thread does the whole length-(t+1) score / softmax /
 * weighted-sum for its query row by itself. This is simpler and obviously
 * correct (a direct transcription of the CPU loop body); it is not the most
 * memory-efficient layout, but T <= 64 here so a query row is tiny, and
 * correctness on a device we can only test remotely matters more than peak
 * throughput.
 *
 * THE BACKWARD RACE — why some writes need atomicAdd and one does not:
 *   Differentiating attention, query position t (with t >= t2) contributes to
 *   the gradients of KEY and VALUE position t2. So as we parallelize over the
 *   query t, many threads scatter into the same d_K_t2 / d_V_t2 rows -> those
 *   writes MUST be atomicAdd. The gradient of the QUERY itself, d_Q_t, is owned
 *   by exactly one thread (this t), so it uses a plain += . d_K and d_V of
 *   different heads/batches live in disjoint slices, so the only collision is
 *   across query positions within the same (b, h) — exactly what atomicAdd
 *   serializes.
 */

#include "cuda/cuda_layers.cuh"
#include <math.h>   /* expf, sqrtf, and the INFINITY macro used in the kernels */

#define ATT_BLOCK 256

/*
 * attention_forward_kernel — one thread handles one (b, h, t) query position.
 * Writes this query's softmax weights into att (length T; future positions
 * t2 > t are zeroed so the cache is clean) and its output vector into atty.
 */
__global__ void attention_forward_kernel(float *atty, float *att,
                                         const float *qkv,
                                         int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T;
    if (idx >= total) return;

    /* decode idx = (b*NH + h)*T + t  -> the same flattening att uses */
    int t   = idx % T;
    int tmp = idx / T;
    int h   = tmp % NH;
    int b   = tmp / NH;

    int head_dim = C / NH;
    float scale  = 1.0f / sqrtf((float)head_dim);
    int C3 = 3 * C;   /* row stride of qkv: Q (C) + K (C) + V (C) */

    /* Q for this (b, t, head); att row for this query (length T) */
    const float *q = qkv + (b * T + t) * C3 + 0 * C + h * head_dim;
    float *att_row = att + ((b * NH + h) * T + t) * T;

    /* 1) raw scores against allowed keys (t2 <= t), tracking the max */
    float maxval = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *k = qkv + (b * T + t2) * C3 + 1 * C + h * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++) dot += q[i] * k[i];
        dot *= scale;
        att_row[t2] = dot;
        if (dot > maxval) maxval = dot;
    }

    /* 2) softmax over the allowed keys (stabilized by subtracting maxval) */
    float sum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float e = expf(att_row[t2] - maxval);
        att_row[t2] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int t2 = 0; t2 <= t; t2++) att_row[t2] *= inv;
    for (int t2 = t + 1; t2 < T; t2++) att_row[t2] = 0.0f;  /* clean cache */

    /* 3) output = weighted sum of Values */
    float *out = atty + (b * T + t) * C + h * head_dim;
    for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v = qkv + (b * T + t2) * C3 + 2 * C + h * head_dim;
        float a_ = att_row[t2];
        for (int i = 0; i < head_dim; i++) out[i] += a_ * v[i];
    }
}

/*
 * attention_backward_kernel — one thread back-propagates one (b, h, t) query.
 *
 * Mirrors attention_backward in gpt.c, but to avoid a per-thread scratch array
 * for d_att (its length varies with t) we compute d_att[t2] = d_out . V_t2 in
 * two passes: first to form the row-scalar dsum = sum att[t2]*d_att[t2] (and
 * scatter d_V), then again to turn each into a d_score and scatter d_Q / d_K.
 * Recomputing the dot product is cheap (head_dim is small) and keeps the kernel
 * allocation-free.
 *
 *   d_V_t2[i] += att[t2] * d_out[i]                  (atomicAdd: shared across t)
 *   d_score   = att[t2] * (d_att[t2] - dsum) * scale
 *   d_Q_t[i]  += d_score * K_t2[i]                   (plain +=: unique to this t)
 *   d_K_t2[i] += d_score * Q_t[i]                    (atomicAdd: shared across t)
 */
__global__ void attention_backward_kernel(float *d_qkv, const float *d_atty,
                                          const float *qkv, const float *att,
                                          int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T;
    if (idx >= total) return;

    int t   = idx % T;
    int tmp = idx / T;
    int h   = tmp % NH;
    int b   = tmp / NH;

    int head_dim = C / NH;
    float scale  = 1.0f / sqrtf((float)head_dim);
    int C3 = 3 * C;

    const float *d_out   = d_atty + (b * T + t) * C + h * head_dim;
    const float *att_row = att + ((b * NH + h) * T + t) * T;

    /* Pass A: dsum and the d_V scatter. */
    float dsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v  = qkv   + (b * T + t2) * C3 + 2 * C + h * head_dim;
        float       *dv = d_qkv + (b * T + t2) * C3 + 2 * C + h * head_dim;
        float datt = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            datt += d_out[i] * v[i];                  /* d_att[t2] */
            atomicAdd(&dv[i], att_row[t2] * d_out[i]); /* d_V_t2 (shared) */
        }
        dsum += att_row[t2] * datt;
    }

    /* Pass B: scores back into d_Q (this thread's, plain +=) and d_K (shared). */
    const float *q  = qkv   + (b * T + t) * C3 + 0 * C + h * head_dim;
    float       *dq = d_qkv + (b * T + t) * C3 + 0 * C + h * head_dim;
    for (int t2 = 0; t2 <= t; t2++) {
        const float *v = qkv + (b * T + t2) * C3 + 2 * C + h * head_dim;
        float datt = 0.0f;
        for (int i = 0; i < head_dim; i++) datt += d_out[i] * v[i];  /* recompute */

        float dscore = att_row[t2] * (datt - dsum) * scale;
        const float *k  = qkv   + (b * T + t2) * C3 + 1 * C + h * head_dim;
        float       *dk = d_qkv + (b * T + t2) * C3 + 1 * C + h * head_dim;
        for (int i = 0; i < head_dim; i++) {
            dq[i] += dscore * k[i];                  /* d_Q_t  (unique)  */
            atomicAdd(&dk[i], dscore * q[i]);        /* d_K_t2 (shared)  */
        }
    }
}

/* ---------------------------------------------------------------------- */
/*                         device-pointer launchers                       */
/* ---------------------------------------------------------------------- */

extern "C" void attention_forward_device(float *d_atty, float *d_att,
                                         const float *d_qkv,
                                         int B, int T, int C, int NH) {
    int total = B * NH * T;
    int grid = (total + ATT_BLOCK - 1) / ATT_BLOCK;
    attention_forward_kernel<<<grid, ATT_BLOCK>>>(d_atty, d_att, d_qkv,
                                                  B, T, C, NH);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void attention_backward_device(float *d_dqkv, const float *d_datty,
                                          const float *d_qkv, const float *d_att,
                                          int B, int T, int C, int NH) {
    int total = B * NH * T;
    int grid = (total + ATT_BLOCK - 1) / ATT_BLOCK;
    attention_backward_kernel<<<grid, ATT_BLOCK>>>(d_dqkv, d_datty, d_qkv, d_att,
                                                   B, T, C, NH);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */
/*                       host wrappers (oracle pattern)                    */
/* ---------------------------------------------------------------------- */

/*
 * attention_forward_cuda — host wrapper. qkv is the only input; atty and att
 * are both outputs (att is the softmax-weight cache the backward needs), so
 * both are copied back.
 */
extern "C" void attention_forward_cuda(float *atty, float *att,
                                       const float *qkv,
                                       int B, int T, int C, int NH) {
    size_t bytes_qkv  = (size_t)B * T * (3 * C) * sizeof(float);
    size_t bytes_atty = (size_t)B * T * C * sizeof(float);
    size_t bytes_att  = (size_t)B * NH * T * T * sizeof(float);

    float *d_qkv = NULL, *d_atty = NULL, *d_att = NULL;
    CUDA_CHECK(cudaMalloc(&d_qkv,  bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_atty, bytes_atty));
    CUDA_CHECK(cudaMalloc(&d_att,  bytes_att));

    CUDA_CHECK(cudaMemcpy(d_qkv, qkv, bytes_qkv, cudaMemcpyHostToDevice));

    attention_forward_device(d_atty, d_att, d_qkv, B, T, C, NH);

    CUDA_CHECK(cudaMemcpy(atty, d_atty, bytes_atty, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(att,  d_att,  bytes_att,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_atty));
    CUDA_CHECK(cudaFree(d_att));
}

/*
 * attention_backward_cuda — host wrapper. d_qkv accumulates with += so it is
 * copied up first; d_atty, qkv, att are read-only caches.
 */
extern "C" void attention_backward_cuda(float *d_qkv, const float *d_atty,
                                        const float *qkv, const float *att,
                                        int B, int T, int C, int NH) {
    size_t bytes_qkv  = (size_t)B * T * (3 * C) * sizeof(float);
    size_t bytes_atty = (size_t)B * T * C * sizeof(float);
    size_t bytes_att  = (size_t)B * NH * T * T * sizeof(float);

    float *dd_qkv = NULL, *d_datty = NULL, *d_qkv_in = NULL, *d_att_in = NULL;
    CUDA_CHECK(cudaMalloc(&dd_qkv,   bytes_qkv));   /* d_qkv grad (device)  */
    CUDA_CHECK(cudaMalloc(&d_datty,  bytes_atty));  /* upstream d_atty      */
    CUDA_CHECK(cudaMalloc(&d_qkv_in, bytes_qkv));   /* forward qkv cache    */
    CUDA_CHECK(cudaMalloc(&d_att_in, bytes_att));   /* forward att cache    */

    CUDA_CHECK(cudaMemcpy(dd_qkv,   d_qkv,  bytes_qkv,  cudaMemcpyHostToDevice)); /* += base */
    CUDA_CHECK(cudaMemcpy(d_datty,  d_atty, bytes_atty, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qkv_in, qkv,    bytes_qkv,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_att_in, att,    bytes_att,  cudaMemcpyHostToDevice));

    attention_backward_device(dd_qkv, d_datty, d_qkv_in, d_att_in, B, T, C, NH);

    CUDA_CHECK(cudaMemcpy(d_qkv, dd_qkv, bytes_qkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd_qkv));
    CUDA_CHECK(cudaFree(d_datty));
    CUDA_CHECK(cudaFree(d_qkv_in));
    CUDA_CHECK(cudaFree(d_att_in));
}
