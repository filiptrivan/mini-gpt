/*
 * gpt.c — the full GPT model, assembled from the layer ops in layers.c.
 *
 * Read gpt.h first: it has the big-picture diagram of the forward pass and
 * explains the single-contiguous-block memory model. This file is the
 * implementation of that diagram plus its mirror image (the backward pass).
 *
 * Two pieces of math are NOT in layers.c and live here as static helpers,
 * because they are specific to a GPT (they're compositions of the basic
 * ops, not basic ops themselves):
 *
 *   - self-attention (attention_forward / attention_backward): the step
 *     where each token mixes in information from earlier tokens.
 *   - the fused softmax + cross-entropy loss (done inline in gpt_forward /
 *     gpt_backward): turning logits into a single loss number, and its
 *     gradient.
 *
 * Everything else (layernorm, matmul, gelu, residual, embedding) is called
 * straight from layers.c.
 */

#include "model/gpt.h"
#include "model/layers.h"

#include <assert.h>
#include <math.h>    /* expf, logf, sqrtf, cosf, INFINITY */
#include <stdlib.h>  /* malloc, free */
#include <string.h>  /* memset */

/* Standard deviation for random weight initialization. 0.02 is the GPT-2
 * default: small enough that stacked layers don't blow up, large enough to
 * break symmetry so different neurons learn different things. */
static const float WEIGHT_INIT_STD = 0.02f;

/* 2*pi, used by the Box-Muller transform in randn(). Written out in full so
 * we don't depend on M_PI (which is not in the C standard, only POSIX). */
static const float TWO_PI = 6.28318530717958648f;

/* ====================================================================== */
/*                       random initialization                            */
/* ====================================================================== */

/*
 * xorshift32 — a tiny, fast pseudo-random number generator.
 *
 * It scrambles a 32-bit state with three shift/xor steps and returns it.
 * Not cryptographically strong, but perfectly fine for initializing weights,
 * and — crucially — fully deterministic: the same starting state always
 * yields the same stream, so a test seeded the same way gets the same model
 * every time. The state is passed by pointer so each call advances it.
 *
 * Note the state must never be 0 (xorshift is stuck at 0 forever); gpt_init
 * guarantees a nonzero seed before calling this.
 */
static unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/*
 * randn — one sample from a standard normal distribution (mean 0, variance 1)
 * via the Box-Muller transform.
 *
 * Box-Muller turns two uniform-random numbers u1, u2 in (0,1] into a normally
 * distributed number with the formula sqrt(-2 ln u1) * cos(2*pi*u2). We only
 * need one of the two values it can produce, so we discard the sin() half.
 *
 * The `+ 1.0` / divide-by-2^32 maps the raw 32-bit integer into (0, 1]; we
 * keep u1 strictly above 0 so logf(u1) never hits log(0) = -infinity.
 */
static float randn(unsigned int *state) {
    float u1 = ((float)xorshift32(state) + 1.0f) / 4294967296.0f; /* (0, 1] */
    float u2 = ((float)xorshift32(state))         / 4294967296.0f; /* [0, 1) */
    return sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
}

/* ====================================================================== */
/*                    parameter / activation layout                       */
/* ====================================================================== */

/*
 * param_layout — the SINGLE source of truth for how weights are packed into
 * the contiguous block, walked in one fixed order.
 *
 * Called two ways:
 *   - param_layout(cfg, NULL, NULL)        -> just returns the total float
 *                                             count (used by gpt_num_params).
 *   - param_layout(cfg, base, &views)      -> also points every field of
 *                                             `views` at its slice of `base`.
 *
 * Keeping size-counting and pointer-assignment in ONE function means the two
 * can never drift apart (a classic bug: allocate N floats but lay out N+k
 * pointers). The PLACE(field, count) macro advances a running offset and, if
 * we were given a destination, records the pointer.
 */
static size_t param_layout(const GPTConfig *c, float *base,
                           ParameterTensors *t) {
    size_t off = 0;
    size_t C    = (size_t)c->n_embd;
    size_t V    = (size_t)c->vocab_size;
    size_t L    = (size_t)c->n_layer;
    size_t FF   = (size_t)c->ff_dim;
    size_t maxT = (size_t)c->max_seq_len;

    /* do-while(0): the standard C idiom that lets a multi-statement macro be
     * used like a single statement (with a trailing semicolon) anywhere. */
    #define PLACE(field, count)                              \
        do {                                                 \
            if (t) t->field = base + off;                    \
            off += (size_t)(count);                          \
        } while (0)

    PLACE(wte,         V * C);
    PLACE(wpe,         maxT * C);
    PLACE(ln1_g,       L * C);
    PLACE(ln1_b,       L * C);
    PLACE(qkv_w,       L * C * (3 * C));
    PLACE(attn_proj_w, L * C * C);
    PLACE(ln2_g,       L * C);
    PLACE(ln2_b,       L * C);
    PLACE(fc_w,        L * C * FF);
    PLACE(fc_proj_w,   L * FF * C);
    PLACE(lnf_g,       C);
    PLACE(lnf_b,       C);
    PLACE(lm_head_w,   C * V);

    #undef PLACE
    return off;
}

/*
 * act_layout — same single-source-of-truth trick for the activation block.
 *
 * Sizes depend on the batch B and sequence length T of the current forward
 * pass, so this is recomputed whenever those change. Returns the total float
 * count; if `base` is non-NULL it also points every field of `t` at its slice.
 */
static size_t act_layout(const GPTConfig *c, int B, int T,
                         float *base, ActivationTensors *t) {
    size_t off = 0;
    size_t C  = (size_t)c->n_embd;
    size_t V  = (size_t)c->vocab_size;
    size_t L  = (size_t)c->n_layer;
    size_t FF = (size_t)c->ff_dim;
    size_t NH = (size_t)c->n_head;
    size_t BT = (size_t)B * (size_t)T;

    #define PLACE(field, count)                              \
        do {                                                 \
            if (t) t->field = base + off;                    \
            off += (size_t)(count);                          \
        } while (0)

    PLACE(encoded,   BT * C);
    PLACE(ln1,       L * BT * C);
    PLACE(qkv,       L * BT * (3 * C));
    PLACE(att,       L * (size_t)B * NH * (size_t)T * (size_t)T);
    PLACE(atty,      L * BT * C);
    PLACE(attproj,   L * BT * C);
    PLACE(residual2, L * BT * C);
    PLACE(ln2,       L * BT * C);
    PLACE(fch,       L * BT * FF);
    PLACE(fch_gelu,  L * BT * FF);
    PLACE(fcproj,    L * BT * C);
    PLACE(residual3, L * BT * C);
    PLACE(lnf,       BT * C);
    PLACE(logits,    BT * V);
    PLACE(probs,     BT * V);
    PLACE(losses,    BT);

    #undef PLACE
    return off;
}

/*
 * ensure_acts — make sure the activation blocks exist and fit (B, T).
 *
 * The first forward pass allocates them; later passes reuse the same buffers
 * unless B or T changed (e.g. a shorter final batch), in which case we free
 * and re-allocate. This lazy approach keeps gpt_init independent of batch
 * size and avoids re-allocating every single step.
 */
static void ensure_acts(GPTModel *model, int B, int T) {
    if (model->acts != NULL && model->B == B && model->T == T) {
        return;  /* already the right size */
    }
    free(model->acts);
    free(model->grads_acts);

    size_t n = act_layout(&model->config, B, T, NULL, NULL);
    model->num_acts   = n;
    model->acts       = malloc(n * sizeof(float));
    model->grads_acts = malloc(n * sizeof(float));
    assert(model->acts != NULL && model->grads_acts != NULL);

    act_layout(&model->config, B, T, model->acts,       &model->a);
    act_layout(&model->config, B, T, model->grads_acts, &model->da);
    model->B = B;
    model->T = T;
}

/*
 * gpt_ensure_acts — public thin wrapper over the static ensure_acts above.
 *
 * gpt_forward already calls ensure_acts internally; this just gives external
 * code (specifically the CUDA wiring in src/cuda/gpt_cuda.cu) a way to trigger
 * the same lazy allocation so it can size its device buffers and read the host
 * activation offsets, without having to run a CPU forward pass first. See the
 * doc comment in gpt.h for the full contract.
 */
void gpt_ensure_acts(GPTModel *model, int B, int T) {
    assert(model != NULL);
    assert(B > 0 && T > 0 && T <= model->config.max_seq_len);
    ensure_acts(model, B, T);
}

/* ====================================================================== */
/*                        public: init / free                             */
/* ====================================================================== */

int gpt_num_params(const GPTConfig *config) {
    assert(config != NULL);
    assert(config->max_seq_len > 0 && config->vocab_size > 0);
    assert(config->n_embd > 0 && config->n_head > 0);
    assert(config->n_layer > 0 && config->ff_dim > 0);
    assert(config->n_embd % config->n_head == 0);  /* heads must tile n_embd */
    return (int)param_layout(config, NULL, NULL);
}

void gpt_init(GPTModel *model, GPTConfig config, unsigned int seed) {
    assert(model != NULL);

    model->config     = config;
    model->num_params = gpt_num_params(&config);  /* also validates config */

    size_t n = (size_t)model->num_params;
    model->params = malloc(n * sizeof(float));
    model->grads  = malloc(n * sizeof(float));
    assert(model->params != NULL && model->grads != NULL);

    /* point the tensor views into the freshly-allocated blocks */
    param_layout(&config, model->params, &model->w);
    param_layout(&config, model->grads,  &model->g);

    /* random weights ~ N(0, 0.02^2) everywhere... */
    unsigned int rng = seed ? seed : 1u;  /* xorshift needs a nonzero seed */
    for (size_t i = 0; i < n; i++) {
        model->params[i] = WEIGHT_INIT_STD * randn(&rng);
    }

    /* ...except layer-norm scale/shift, which start at the identity transform
     * (gamma = 1, beta = 0) so the very first forward pass leaves normalized
     * activations untouched — the standard, stable initialization. */
    int C = config.n_embd, L = config.n_layer;
    for (int i = 0; i < L * C; i++) { model->w.ln1_g[i] = 1.0f; model->w.ln1_b[i] = 0.0f; }
    for (int i = 0; i < L * C; i++) { model->w.ln2_g[i] = 1.0f; model->w.ln2_b[i] = 0.0f; }
    for (int i = 0; i < C;     i++) { model->w.lnf_g[i] = 1.0f; model->w.lnf_b[i] = 0.0f; }

    /* gradients start at zero (backward accumulates with +=) */
    memset(model->grads, 0, n * sizeof(float));

    /* activations are allocated lazily on the first forward pass */
    model->B = 0;
    model->T = 0;
    model->num_acts   = 0;
    model->acts       = NULL;
    model->grads_acts = NULL;
    model->loss       = 0.0f;
}

void gpt_free(GPTModel *model) {
    assert(model != NULL);
    /* free(NULL) is a no-op, so this is safe even if forward never ran */
    free(model->params);
    free(model->grads);
    free(model->acts);
    free(model->grads_acts);
    model->params     = NULL;
    model->grads      = NULL;
    model->acts       = NULL;
    model->grads_acts = NULL;
    model->num_params = 0;
    model->num_acts   = 0;
    model->B = 0;
    model->T = 0;
}

/* ====================================================================== */
/*                    self-attention (forward + backward)                 */
/* ====================================================================== */

/*
 * attention_forward — causal multi-head self-attention.
 *
 * Input is `qkv`, shape (B, T, 3C): for each token we have its Query, Key and
 * Value vectors concatenated (C each). We split the C channels into `n_head`
 * heads of width head_dim = C / n_head and run attention independently per
 * head, then write each head's output into its slice of `atty` (B, T, C).
 *
 * For one head and one query position t:
 *   1. score[t2] = (Q_t . K_t2) / sqrt(head_dim)   for every key position t2.
 *      The 1/sqrt(head_dim) keeps the dot products from growing with width
 *      (so softmax doesn't saturate). We only consider t2 <= t — the "causal"
 *      mask — so a token can attend to itself and earlier tokens but never
 *      peek at the future (that would be cheating at next-token prediction).
 *   2. att[t2] = softmax(score) over the allowed t2 (numerically stabilized
 *      by subtracting the row max before exp, same trick as softmax_forward).
 *   3. out_t = sum_t2 att[t2] * V_t2 — a weighted average of the Values.
 *
 * We cache the softmax weights `att` (shape B, NH, T, T) because the backward
 * pass needs them. Masked-out entries (t2 > t) are written as 0 so the cache
 * is clean.
 */
void attention_forward(float *atty, float *att, const float *qkv,
                       int B, int T, int C, int NH) {
    int head_dim = C / NH;
    float scale = 1.0f / sqrtf((float)head_dim);
    int C3 = 3 * C;  /* row stride of qkv: Q (C) + K (C) + V (C) */

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                /* Q for this (b, t, head): offset 0*C picks the Q block,
                 * h*head_dim picks this head's slice inside it. */
                const float *q = qkv + (b * T + t) * C3 + 0 * C + h * head_dim;
                /* this query's row of attention weights, length T */
                float *att_row = att + ((b * NH + h) * T + t) * T;

                /* 1) raw scores against allowed keys, tracking the max */
                float maxval = -INFINITY;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *k = qkv + (b * T + t2) * C3 + 1 * C + h * head_dim;
                    float dot = 0.0f;
                    for (int i = 0; i < head_dim; i++) dot += q[i] * k[i];
                    dot *= scale;
                    att_row[t2] = dot;
                    if (dot > maxval) maxval = dot;
                }

                /* 2) softmax over the allowed keys */
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float e = expf(att_row[t2] - maxval);
                    att_row[t2] = e;
                    sum += e;
                }
                float inv = 1.0f / sum;
                for (int t2 = 0; t2 <= t; t2++) att_row[t2] *= inv;
                /* zero the masked (future) positions so the cache is clean */
                for (int t2 = t + 1; t2 < T; t2++) att_row[t2] = 0.0f;

                /* 3) output = weighted sum of Values */
                float *out = atty + (b * T + t) * C + h * head_dim;
                for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = qkv + (b * T + t2) * C3 + 2 * C + h * head_dim;
                    float a_ = att_row[t2];
                    for (int i = 0; i < head_dim; i++) out[i] += a_ * v[i];
                }
            }
        }
    }
}

/*
 * attention_backward — gradient of attention w.r.t. its qkv input.
 *
 * Given d_atty (gradient flowing in from above) plus the cached forward
 * `qkv` and softmax weights `att`, we accumulate into d_qkv (the gradient of
 * Q, K and V, packed the same way as qkv). Accumulates with += (d_qkv must be
 * pre-zeroed by the caller — gpt_backward zeros the whole activation-grad
 * block at the start).
 *
 * We undo the three forward steps in reverse:
 *   3') out_t = sum att[t2] * V_t2   =>   d_V_t2 += att[t2] * d_out_t
 *                                          d_att[t2] = d_out_t . V_t2
 *   2') att = softmax(score)         =>   the standard softmax Jacobian:
 *          d_score[t2] = att[t2] * (d_att[t2] - sum_j att[j] * d_att[j])
 *       (same factored O(T) form as softmax_backward in layers.c)
 *   1') score[t2] = scale * (Q_t . K_t2)
 *                                    =>   d_Q_t  += scale * d_score[t2] * K_t2
 *                                          d_K_t2 += scale * d_score[t2] * Q_t
 *
 * `dscratch` is a small per-head-row workspace for d_att (length T); we
 * allocate one buffer up front and reuse it across all rows.
 */
void attention_backward(float *d_qkv, const float *d_atty,
                        const float *qkv, const float *att,
                        int B, int T, int C, int NH) {
    int head_dim = C / NH;
    float scale = 1.0f / sqrtf((float)head_dim);
    int C3 = 3 * C;

    /* workspace for one query row's d_att values (length T). One malloc,
     * reused for every (b, h, t) — avoids T*NH*B tiny allocations. */
    float *dscratch = malloc((size_t)T * sizeof(float));
    assert(dscratch != NULL);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float *d_out = d_atty + (b * T + t) * C + h * head_dim;
                const float *att_row = att + ((b * NH + h) * T + t) * T;

                /* 3') d_V and d_att from the weighted-sum output */
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = qkv   + (b * T + t2) * C3 + 2 * C + h * head_dim;
                    float       *dv = d_qkv + (b * T + t2) * C3 + 2 * C + h * head_dim;
                    float datt = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        datt    += d_out[i] * v[i];          /* d_att[t2]   */
                        dv[i]   += att_row[t2] * d_out[i];   /* d_V_t2 += .. */
                    }
                    dscratch[t2] = datt;
                }

                /* 2') softmax backward: subtract the weighted mean */
                float dsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) dsum += att_row[t2] * dscratch[t2];

                /* 1') scores back into d_Q and d_K */
                const float *q = qkv   + (b * T + t) * C3 + 0 * C + h * head_dim;
                float       *dq = d_qkv + (b * T + t) * C3 + 0 * C + h * head_dim;
                for (int t2 = 0; t2 <= t; t2++) {
                    float dscore = att_row[t2] * (dscratch[t2] - dsum) * scale;
                    const float *k = qkv   + (b * T + t2) * C3 + 1 * C + h * head_dim;
                    float       *dk = d_qkv + (b * T + t2) * C3 + 1 * C + h * head_dim;
                    for (int i = 0; i < head_dim; i++) {
                        dq[i] += dscore * k[i];
                        dk[i] += dscore * q[i];
                    }
                }
            }
        }
    }
    free(dscratch);
}

/* ====================================================================== */
/*                            forward pass                                */
/* ====================================================================== */

void gpt_forward(GPTModel *model, const int *tokens, const int *targets,
                 int B, int T) {
    assert(model != NULL && tokens != NULL);
    assert(B > 0 && T > 0 && T <= model->config.max_seq_len);

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

    ensure_acts(model, B, T);
    ParameterTensors *w = &model->w;
    ActivationTensors *a = &model->a;

    /* token + position embeddings -> the initial residual stream */
    embed_forward(a->encoded, tokens, w->wte, w->wpe, B, T, C);

    /* `resid` always points at the residual stream entering the current
     * layer: encoded for layer 0, then the previous layer's residual3. */
    float *resid = a->encoded;

    for (int l = 0; l < L; l++) {
        /* per-layer activation slices */
        float *ln1     = a->ln1       + (size_t)l * BTC;
        float *qkv     = a->qkv       + (size_t)l * BT3C;
        float *att     = a->att       + (size_t)l * attN;
        float *atty    = a->atty      + (size_t)l * BTC;
        float *attproj = a->attproj   + (size_t)l * BTC;
        float *resid2  = a->residual2 + (size_t)l * BTC;
        float *ln2     = a->ln2       + (size_t)l * BTC;
        float *fch     = a->fch       + (size_t)l * BTFF;
        float *fchg    = a->fch_gelu  + (size_t)l * BTFF;
        float *fcproj  = a->fcproj    + (size_t)l * BTC;
        float *resid3  = a->residual3 + (size_t)l * BTC;

        /* per-layer weight slices */
        float *ln1_g     = w->ln1_g       + (size_t)l * C;
        float *ln1_b     = w->ln1_b       + (size_t)l * C;
        float *qkv_w     = w->qkv_w       + (size_t)l * C * (3 * C);
        float *attproj_w = w->attn_proj_w + (size_t)l * C * C;
        float *ln2_g     = w->ln2_g       + (size_t)l * C;
        float *ln2_b     = w->ln2_b       + (size_t)l * C;
        float *fc_w      = w->fc_w        + (size_t)l * C * FF;
        float *fc_proj_w = w->fc_proj_w   + (size_t)l * FF * C;

        /* attention sub-block: ln -> qkv -> attention -> proj -> residual */
        layernorm_forward(ln1, resid, ln1_g, ln1_b, BT, C);
        matmul_forward(qkv, ln1, qkv_w, BT, C, 3 * C);
        attention_forward(atty, att, qkv, B, T, C, NH);
        matmul_forward(attproj, atty, attproj_w, BT, C, C);
        residual_forward(resid2, resid, attproj, (int)BTC);

        /* feed-forward sub-block: ln -> up -> gelu -> down -> residual */
        layernorm_forward(ln2, resid2, ln2_g, ln2_b, BT, C);
        matmul_forward(fch, ln2, fc_w, BT, C, FF);
        gelu_forward(fchg, fch, (int)BTFF);
        matmul_forward(fcproj, fchg, fc_proj_w, BT, FF, C);
        residual_forward(resid3, resid2, fcproj, (int)BTC);

        resid = resid3;  /* feed this layer's output into the next layer */
    }

    /* final layernorm, then project to per-vocab logits, then softmax */
    layernorm_forward(a->lnf, resid, w->lnf_g, w->lnf_b, BT, C);
    matmul_forward(a->logits, a->lnf, w->lm_head_w, BT, C, V);
    softmax_forward(a->probs, a->logits, BT, V);

    /* cross-entropy loss against the targets (if provided) */
    if (targets != NULL) {
        float loss = 0.0f;
        for (int pos = 0; pos < BT; pos++) {
            /* probability the model assigned to the correct next token */
            float p = a->probs[(size_t)pos * V + targets[pos]];
            float li = -logf(p);   /* cross-entropy = -log(prob of truth) */
            a->losses[pos] = li;
            loss += li;
        }
        model->loss = loss / (float)BT;  /* mean over all positions */
    } else {
        model->loss = 0.0f;
    }
}

/* ====================================================================== */
/*                            backward pass                               */
/* ====================================================================== */

void gpt_zero_grad(GPTModel *model) {
    assert(model != NULL && model->grads != NULL);
    memset(model->grads, 0, (size_t)model->num_params * sizeof(float));
}

void gpt_backward(GPTModel *model, const int *tokens, const int *targets,
                  int B, int T) {
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

    ParameterTensors *w = &model->w;
    ParameterTensors *g = &model->g;
    ActivationTensors *a  = &model->a;
    ActivationTensors *da = &model->da;

    /* the activation gradients are pure scratch — zero them so the += in
     * every backward op starts clean (some, like the residual stream, are
     * written from two paths and rely on starting at zero). */
    memset(model->grads_acts, 0, model->num_acts * sizeof(float));

    /* --- output: fused softmax + cross-entropy gradient ---------------- */
    /* The classic result: d_logits = (probs - onehot(target)) / N. This one
     * line is the combined derivative of softmax followed by -log of the
     * true class; deriving them separately would give the same thing. */
    for (int pos = 0; pos < BT; pos++) {
        for (int j = 0; j < V; j++) {
            float indicator = (j == targets[pos]) ? 1.0f : 0.0f;
            da->logits[(size_t)pos * V + j] =
                (a->probs[(size_t)pos * V + j] - indicator) / (float)BT;
        }
    }

    /* --- output projection + final layernorm --------------------------- */
    matmul_backward(da->lnf, g->lm_head_w, da->logits, a->lnf, w->lm_head_w,
                    BT, C, V);

    /* the residual stream entering the final layernorm is the last layer's
     * residual3 (L >= 1 guaranteed by config validation) */
    float *resid_last   = a->residual3  + (size_t)(L - 1) * BTC;
    float *d_resid_last = da->residual3 + (size_t)(L - 1) * BTC;
    layernorm_backward(d_resid_last, g->lnf_g, g->lnf_b, da->lnf,
                       resid_last, w->lnf_g, BT, C);

    /* --- transformer layers, in reverse -------------------------------- */
    for (int l = L - 1; l >= 0; l--) {
        /* forward activation slices for this layer */
        float *ln1     = a->ln1       + (size_t)l * BTC;
        float *qkv     = a->qkv       + (size_t)l * BT3C;
        float *att     = a->att       + (size_t)l * attN;
        float *atty    = a->atty      + (size_t)l * BTC;
        float *resid2  = a->residual2 + (size_t)l * BTC;
        float *ln2     = a->ln2       + (size_t)l * BTC;
        float *fch     = a->fch       + (size_t)l * BTFF;
        float *fchg    = a->fch_gelu  + (size_t)l * BTFF;

        /* gradient activation slices for this layer */
        float *d_ln1     = da->ln1       + (size_t)l * BTC;
        float *d_qkv     = da->qkv       + (size_t)l * BT3C;
        float *d_atty    = da->atty      + (size_t)l * BTC;
        float *d_attproj = da->attproj   + (size_t)l * BTC;
        float *d_resid2  = da->residual2 + (size_t)l * BTC;
        float *d_ln2     = da->ln2       + (size_t)l * BTC;
        float *d_fch     = da->fch       + (size_t)l * BTFF;
        float *d_fchg    = da->fch_gelu  + (size_t)l * BTFF;
        float *d_fcproj  = da->fcproj    + (size_t)l * BTC;
        float *d_resid3  = da->residual3 + (size_t)l * BTC;  /* grad in */

        /* residual stream entering this layer (and its gradient): encoded
         * for layer 0, otherwise the previous layer's residual3 */
        float *resid_in   = (l == 0) ? a->encoded  : a->residual3  + (size_t)(l - 1) * BTC;
        float *d_resid_in = (l == 0) ? da->encoded : da->residual3 + (size_t)(l - 1) * BTC;

        /* weight slices (params + their grads) */
        float *qkv_w       = w->qkv_w       + (size_t)l * C * (3 * C);
        float *attproj_w   = w->attn_proj_w + (size_t)l * C * C;
        float *fc_w        = w->fc_w        + (size_t)l * C * FF;
        float *fc_proj_w   = w->fc_proj_w   + (size_t)l * FF * C;
        float *ln1_g       = w->ln1_g       + (size_t)l * C;
        float *ln2_g       = w->ln2_g       + (size_t)l * C;
        float *d_qkv_w     = g->qkv_w       + (size_t)l * C * (3 * C);
        float *d_attproj_w = g->attn_proj_w + (size_t)l * C * C;
        float *d_fc_w      = g->fc_w        + (size_t)l * C * FF;
        float *d_fc_proj_w = g->fc_proj_w   + (size_t)l * FF * C;
        float *d_ln1_g     = g->ln1_g       + (size_t)l * C;
        float *d_ln1_b     = g->ln1_b       + (size_t)l * C;
        float *d_ln2_g     = g->ln2_g       + (size_t)l * C;
        float *d_ln2_b     = g->ln2_b       + (size_t)l * C;

        /* feed-forward sub-block, reversed.
         * forward was: resid3 = resid2 + fcproj, fcproj = gelu(fch) @ fc_proj_w,
         *              fch = ln2 @ fc_w, ln2 = layernorm(resid2). */
        residual_backward(d_resid2, d_fcproj, d_resid3, (int)BTC);
        matmul_backward(d_fchg, d_fc_proj_w, d_fcproj, fchg, fc_proj_w, BT, FF, C);
        gelu_backward(d_fch, d_fchg, fch, (int)BTFF);
        matmul_backward(d_ln2, d_fc_w, d_fch, ln2, fc_w, BT, C, FF);
        /* layernorm adds the "through" path into d_resid2 (which already has
         * the skip path from residual_backward above) — hence the += in
         * layernorm_backward. */
        layernorm_backward(d_resid2, d_ln2_g, d_ln2_b, d_ln2, resid2, ln2_g, BT, C);

        /* attention sub-block, reversed.
         * forward was: resid2 = resid_in + attproj, attproj = atty @ attproj_w,
         *              atty = attention(qkv), qkv = ln1 @ qkv_w,
         *              ln1 = layernorm(resid_in). */
        residual_backward(d_resid_in, d_attproj, d_resid2, (int)BTC);
        matmul_backward(d_atty, d_attproj_w, d_attproj, atty, attproj_w, BT, C, C);
        attention_backward(d_qkv, d_atty, qkv, att, B, T, C, NH);
        matmul_backward(d_ln1, d_qkv_w, d_qkv, ln1, qkv_w, BT, C, 3 * C);
        /* second contribution into d_resid_in (the skip path was added above) */
        layernorm_backward(d_resid_in, d_ln1_g, d_ln1_b, d_ln1, resid_in, ln1_g, BT, C);
    }

    /* --- embeddings ---------------------------------------------------- */
    /* da->encoded is now the full gradient of the initial residual stream;
     * scatter it back into the token and position embedding tables. */
    embed_backward(g->wte, g->wpe, da->encoded, tokens, B, T, C);
}
