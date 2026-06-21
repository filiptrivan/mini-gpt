/*
 * gpt_cuda.cu — the full GPT forward and backward pass on the GPU.
 *
 * This is the payoff of Task 10. Read model/gpt.c first: this file is a
 * step-for-step mirror of its gpt_forward / gpt_backward, with every CPU op
 * (embed, layernorm, matmul, attention, gelu, residual, softmax,
 * cross-entropy) replaced by the matching <op>_<dir>_device launcher from
 * cuda_layers.cuh. The ORDER of operations, the per-layer slicing, and the
 * += accumulation into the residual stream are all identical — so if gpt.c is
 * correct (it is, proven by the Task 6/7 numerical-gradient tests) and each
 * kernel matches its CPU oracle (proven by the Task 9/10 parity tests), this
 * composition is correct too.
 *
 * THE ONE NEW IDEA — tensors stay RESIDENT on the device.
 *   Task 9's _cuda wrappers each did a full cudaMalloc / copy-up / launch /
 *   copy-down / free round trip. That is fine for testing one kernel in
 *   isolation but catastrophic for a real forward pass: ~30 kernels would
 *   bounce every intermediate tensor across the PCIe bus twice. Here we instead
 *   upload the inputs ONCE into big device blocks (mirroring the model's single
 *   contiguous params/acts blocks), run all the kernels back-to-back on those
 *   device buffers, and copy results back ONCE at the end. CUDA serializes
 *   kernel launches on the default stream, so each kernel finishes before the
 *   next starts — no explicit synchronization is needed between them; the final
 *   cudaMemcpy (device->host) blocks until everything is done.
 *
 * Memory model: we allocate the device blocks per call and free them at the
 * end. A production trainer would keep them alive across steps; here per-call
 * alloc keeps the API trivial (just pass the model) and mirrors how the CPU
 * side already re-derives everything from the model each step. The per-call
 * alloc is cheap relative to the kernels for any non-trivial batch.
 *
 * DEVICE SUB-POINTER TRICK (the DP/DA/DG/DGA macros below):
 *   The model already computed, on the host, the offset of every weight and
 *   activation tensor inside its contiguous block (model->w.field points into
 *   model->params, model->a.field into model->acts, etc.). The device blocks
 *   use the IDENTICAL layout, so the device pointer for a field is simply the
 *   device block base plus that same offset: e.g. for a weight,
 *   d_params + (model->w.field - model->params). The macros wrap exactly that,
 *   so the body below reads almost verbatim like gpt.c.
 */

#include "cuda/cuda_layers.cuh"
#include "cuda/gpt_cuda.cuh"
#include "model/gpt.h"

#include <assert.h>

/* Device sub-pointer for a tensor field, derived from its host offset inside
 * the corresponding contiguous block. Each macro is only used where its base
 * device pointer (d_params / d_acts / d_grads / d_grads_acts) is in scope.
 *   DP  — a weight       (d_params      + offset of model->w.field)
 *   DA  — an activation  (d_acts        + offset of model->a.field)
 *   DG  — a weight grad  (d_grads       + offset of model->g.field)
 *   DGA — an act grad    (d_grads_acts  + offset of model->da.field) */
#define DP(field)  (d_params     + (size_t)(model->w.field  - model->params))
#define DA(field)  (d_acts       + (size_t)(model->a.field  - model->acts))
#define DG(field)  (d_grads      + (size_t)(model->g.field  - model->grads))
#define DGA(field) (d_grads_acts + (size_t)(model->da.field - model->grads_acts))

/* ====================================================================== */
/*                            forward pass                                */
/* ====================================================================== */

extern "C" void gpt_forward_cuda(GPTModel *model, const int *tokens,
                                 const int *targets, int B, int T) {
    assert(model != NULL && tokens != NULL);
    assert(B > 0 && T > 0 && T <= model->config.max_seq_len);

    /* size the host activation blocks + set model->a offsets (no GPU work) */
    gpt_ensure_acts(model, B, T);

    GPTConfig *cfg = &model->config;
    int C  = cfg->n_embd;
    int V  = cfg->vocab_size;
    int L  = cfg->n_layer;
    int FF = cfg->ff_dim;
    int NH = cfg->n_head;
    int BT = B * T;
    size_t BTC  = (size_t)BT * C;
    size_t BT3C = (size_t)BT * (3 * C);
    size_t BTFF = (size_t)BT * FF;
    size_t attN = (size_t)B * NH * T * T;
    size_t num_params = (size_t)model->num_params;
    size_t num_acts   = model->num_acts;

    /* ---- device blocks: upload params + tokens once, leave acts on-device -- */
    float *d_params = NULL, *d_acts = NULL;
    int   *d_tokens = NULL, *d_targets = NULL;
    CUDA_CHECK(cudaMalloc(&d_params, num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acts,   num_acts   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tokens, (size_t)BT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_params, model->params, num_params * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens, (size_t)BT * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* token + position embeddings -> initial residual stream */
    embed_forward_device(DA(encoded), d_tokens, DP(wte), DP(wpe), B, T, C);

    /* `resid` points at the residual stream entering the current layer */
    float *resid = DA(encoded);

    for (int l = 0; l < L; l++) {
        /* per-layer activation slices */
        float *ln1     = DA(ln1)       + (size_t)l * BTC;
        float *qkv     = DA(qkv)       + (size_t)l * BT3C;
        float *att     = DA(att)       + (size_t)l * attN;
        float *atty    = DA(atty)      + (size_t)l * BTC;
        float *attproj = DA(attproj)   + (size_t)l * BTC;
        float *resid2  = DA(residual2) + (size_t)l * BTC;
        float *ln2     = DA(ln2)       + (size_t)l * BTC;
        float *fch     = DA(fch)       + (size_t)l * BTFF;
        float *fchg    = DA(fch_gelu)  + (size_t)l * BTFF;
        float *fcproj  = DA(fcproj)    + (size_t)l * BTC;
        float *resid3  = DA(residual3) + (size_t)l * BTC;

        /* per-layer weight slices */
        float *ln1_g     = DP(ln1_g)       + (size_t)l * C;
        float *ln1_b     = DP(ln1_b)       + (size_t)l * C;
        float *qkv_w     = DP(qkv_w)       + (size_t)l * C * (3 * C);
        float *attproj_w = DP(attn_proj_w) + (size_t)l * C * C;
        float *ln2_g     = DP(ln2_g)       + (size_t)l * C;
        float *ln2_b     = DP(ln2_b)       + (size_t)l * C;
        float *fc_w      = DP(fc_w)        + (size_t)l * C * FF;
        float *fc_proj_w = DP(fc_proj_w)   + (size_t)l * FF * C;

        /* attention sub-block: ln -> qkv -> attention -> proj -> residual */
        layernorm_forward_device(ln1, resid, ln1_g, ln1_b, BT, C);
        matmul_forward_device(qkv, ln1, qkv_w, BT, C, 3 * C);
        attention_forward_device(atty, att, qkv, B, T, C, NH);
        matmul_forward_device(attproj, atty, attproj_w, BT, C, C);
        residual_forward_device(resid2, resid, attproj, (int)BTC);

        /* feed-forward sub-block: ln -> up -> gelu -> down -> residual */
        layernorm_forward_device(ln2, resid2, ln2_g, ln2_b, BT, C);
        matmul_forward_device(fch, ln2, fc_w, BT, C, FF);
        gelu_forward_device(fchg, fch, (int)BTFF);
        matmul_forward_device(fcproj, fchg, fc_proj_w, BT, FF, C);
        residual_forward_device(resid3, resid2, fcproj, (int)BTC);

        resid = resid3;  /* feed this layer's output into the next */
    }

    /* final layernorm -> logits -> softmax probabilities */
    layernorm_forward_device(DA(lnf), resid, DP(lnf_g), DP(lnf_b), BT, C);
    matmul_forward_device(DA(logits), DA(lnf), DP(lm_head_w), BT, C, V);
    softmax_forward_device(DA(probs), DA(logits), BT, V);

    /* per-position cross-entropy loss (if targets given) */
    if (targets != NULL) {
        CUDA_CHECK(cudaMalloc(&d_targets, (size_t)BT * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)BT * sizeof(int),
                              cudaMemcpyHostToDevice));
        cross_entropy_forward_device(DA(losses), DA(probs), d_targets, BT, V);
    }

    /* copy ALL activations back so model->a.* is valid for inspection and for
     * a subsequent gpt_backward_cuda (which re-uploads them). This is the only
     * device->host traffic of the whole forward — the "copy down once" half of
     * the resident strategy. */
    CUDA_CHECK(cudaMemcpy(model->acts, d_acts, num_acts * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* mean cross-entropy over all positions (same reduction the CPU does) */
    if (targets != NULL) {
        float loss = 0.0f;
        for (int pos = 0; pos < BT; pos++) loss += model->a.losses[pos];
        model->loss = loss / (float)BT;
    } else {
        model->loss = 0.0f;
    }

    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_acts));
    CUDA_CHECK(cudaFree(d_tokens));
    if (d_targets != NULL) CUDA_CHECK(cudaFree(d_targets));
}

/* ====================================================================== */
/*                            backward pass                               */
/* ====================================================================== */

extern "C" void gpt_backward_cuda(GPTModel *model, const int *tokens,
                                  const int *targets, int B, int T) {
    assert(model != NULL && tokens != NULL && targets != NULL);
    /* a matching forward must have run and sized the activations */
    assert(model->acts != NULL && model->B == B && model->T == T);

    GPTConfig *cfg = &model->config;
    int C  = cfg->n_embd;
    int V  = cfg->vocab_size;
    int L  = cfg->n_layer;
    int FF = cfg->ff_dim;
    int NH = cfg->n_head;
    int BT = B * T;
    size_t BTC  = (size_t)BT * C;
    size_t BT3C = (size_t)BT * (3 * C);
    size_t BTFF = (size_t)BT * FF;
    size_t attN = (size_t)B * NH * T * T;
    size_t num_params = (size_t)model->num_params;
    size_t num_acts   = model->num_acts;

    /* ---- device blocks ---------------------------------------------------
     * params + the cached forward activations are uploaded (read-only inputs);
     * the two gradient blocks are allocated and ZEROED on the device (so the
     * kernels' += accumulation starts clean — this replaces gpt_zero_grad and
     * the grads_acts memset that gpt_backward does on the host). */
    float *d_params = NULL, *d_acts = NULL, *d_grads = NULL, *d_grads_acts = NULL;
    int   *d_tokens = NULL, *d_targets = NULL;
    CUDA_CHECK(cudaMalloc(&d_params,      num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acts,        num_acts   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grads,       num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grads_acts,  num_acts   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tokens,  (size_t)BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)BT * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_params, model->params, num_params * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acts, model->acts, num_acts * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grads,      0, num_params * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grads_acts, 0, num_acts   * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens, (size_t)BT * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)BT * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* --- output: fused softmax + cross-entropy gradient into d_logits ---- */
    cross_entropy_backward_device(DGA(logits), DA(probs), d_targets, BT, V);

    /* --- output projection + final layernorm --------------------------- */
    matmul_backward_device(DGA(lnf), DG(lm_head_w), DGA(logits),
                           DA(lnf), DP(lm_head_w), BT, C, V);

    float *resid_last   = DA(residual3)  + (size_t)(L - 1) * BTC;
    float *d_resid_last = DGA(residual3) + (size_t)(L - 1) * BTC;
    layernorm_backward_device(d_resid_last, DG(lnf_g), DG(lnf_b), DGA(lnf),
                              resid_last, DP(lnf_g), BT, C);

    /* --- transformer layers, in reverse -------------------------------- */
    for (int l = L - 1; l >= 0; l--) {
        /* forward activation slices */
        float *ln1    = DA(ln1)      + (size_t)l * BTC;
        float *qkv    = DA(qkv)      + (size_t)l * BT3C;
        float *att    = DA(att)      + (size_t)l * attN;
        float *atty   = DA(atty)     + (size_t)l * BTC;
        float *resid2 = DA(residual2)+ (size_t)l * BTC;
        float *ln2    = DA(ln2)      + (size_t)l * BTC;
        float *fch    = DA(fch)      + (size_t)l * BTFF;
        float *fchg   = DA(fch_gelu) + (size_t)l * BTFF;

        /* gradient activation slices */
        float *d_ln1     = DGA(ln1)       + (size_t)l * BTC;
        float *d_qkv     = DGA(qkv)       + (size_t)l * BT3C;
        float *d_atty    = DGA(atty)      + (size_t)l * BTC;
        float *d_attproj = DGA(attproj)   + (size_t)l * BTC;
        float *d_resid2  = DGA(residual2) + (size_t)l * BTC;
        float *d_ln2     = DGA(ln2)       + (size_t)l * BTC;
        float *d_fch     = DGA(fch)       + (size_t)l * BTFF;
        float *d_fchg    = DGA(fch_gelu)  + (size_t)l * BTFF;
        float *d_fcproj  = DGA(fcproj)    + (size_t)l * BTC;
        float *d_resid3  = DGA(residual3) + (size_t)l * BTC;  /* grad in */

        /* residual stream entering this layer (and its gradient) */
        float *resid_in   = (l == 0) ? DA(encoded)
                                     : DA(residual3)  + (size_t)(l - 1) * BTC;
        float *d_resid_in = (l == 0) ? DGA(encoded)
                                     : DGA(residual3) + (size_t)(l - 1) * BTC;

        /* weight slices (params + their grads) */
        float *qkv_w       = DP(qkv_w)       + (size_t)l * C * (3 * C);
        float *attproj_w   = DP(attn_proj_w) + (size_t)l * C * C;
        float *fc_w        = DP(fc_w)        + (size_t)l * C * FF;
        float *fc_proj_w   = DP(fc_proj_w)   + (size_t)l * FF * C;
        float *ln1_g       = DP(ln1_g)       + (size_t)l * C;
        float *ln2_g       = DP(ln2_g)       + (size_t)l * C;
        float *d_qkv_w     = DG(qkv_w)       + (size_t)l * C * (3 * C);
        float *d_attproj_w = DG(attn_proj_w) + (size_t)l * C * C;
        float *d_fc_w      = DG(fc_w)        + (size_t)l * C * FF;
        float *d_fc_proj_w = DG(fc_proj_w)   + (size_t)l * FF * C;
        float *d_ln1_g     = DG(ln1_g)       + (size_t)l * C;
        float *d_ln1_b     = DG(ln1_b)       + (size_t)l * C;
        float *d_ln2_g     = DG(ln2_g)       + (size_t)l * C;
        float *d_ln2_b     = DG(ln2_b)       + (size_t)l * C;

        /* feed-forward sub-block, reversed */
        residual_backward_device(d_resid2, d_fcproj, d_resid3, (int)BTC);
        matmul_backward_device(d_fchg, d_fc_proj_w, d_fcproj, fchg, fc_proj_w,
                               BT, FF, C);
        gelu_backward_device(d_fch, d_fchg, fch, (int)BTFF);
        matmul_backward_device(d_ln2, d_fc_w, d_fch, ln2, fc_w, BT, C, FF);
        layernorm_backward_device(d_resid2, d_ln2_g, d_ln2_b, d_ln2, resid2,
                                  ln2_g, BT, C);

        /* attention sub-block, reversed */
        residual_backward_device(d_resid_in, d_attproj, d_resid2, (int)BTC);
        matmul_backward_device(d_atty, d_attproj_w, d_attproj, atty, attproj_w,
                               BT, C, C);
        attention_backward_device(d_qkv, d_atty, qkv, att, B, T, C, NH);
        matmul_backward_device(d_ln1, d_qkv_w, d_qkv, ln1, qkv_w, BT, C, 3 * C);
        layernorm_backward_device(d_resid_in, d_ln1_g, d_ln1_b, d_ln1, resid_in,
                                  ln1_g, BT, C);
    }

    /* --- embeddings: scatter the initial-residual gradient into the tables - */
    embed_backward_device(DG(wte), DG(wpe), DGA(encoded), d_tokens, B, T, C);

    /* copy the parameter gradients back — the only device->host traffic here */
    CUDA_CHECK(cudaMemcpy(model->grads, d_grads, num_params * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_acts));
    CUDA_CHECK(cudaFree(d_grads));
    CUDA_CHECK(cudaFree(d_grads_acts));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_targets));
}
