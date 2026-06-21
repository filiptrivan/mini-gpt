#ifndef GPT_H
#define GPT_H

/*
 * gpt.h — the full GPT model: wiring the individual layer ops from
 * src/model/layers.c into one transformer that can do a forward pass
 * (text -> loss) and a backward pass (loss -> gradients).
 *
 * Include guard above (#ifndef / #define / #endif): if two .c files both
 * include this header, the guard makes the second include a no-op so the
 * structs below are not defined twice (which would be a compile error).
 *
 *
 * BIG PICTURE — what a GPT forward pass does, top to bottom:
 *
 *   tokens (integers)                          shape (B, T)
 *      |  embed: look up a vector per token, add a vector per position
 *      v
 *   x = residual stream                        shape (B, T, C)
 *      |
 *      |  repeat this block n_layer times:
 *      |    a    = layernorm(x)                  "normalize"
 *      |    attn = self_attention(a)             "let tokens look at earlier tokens"
 *      |    x    = x + attn_proj(attn)           "residual add (skip connection)"
 *      |    b    = layernorm(x)
 *      |    ffn  = matmul -> gelu -> matmul (b)  "think per-token"
 *      |    x    = x + ffn
 *      v
 *   x = final residual stream                  shape (B, T, C)
 *      |  layernorm(x)
 *      |  matmul by the output table           -> one score per vocab word
 *      v
 *   logits                                     shape (B, T, V)
 *      |  softmax -> probabilities, then cross-entropy against the targets
 *      v
 *   loss (a single number)
 *
 * The backward pass walks this exact diagram in REVERSE, using the
 * `_backward` functions in layers.c to turn "how wrong was the loss" into
 * "how should each weight change." The result lands in `grads`.
 *
 *
 * TWO DESIGN CHOICES baked into this model (decided for simplicity while
 * learning — both are valid, real GPT variants):
 *
 *   1. SEPARATE output table. The table that turns the final vector back
 *      into per-word scores (`lm_head_w`) is its own set of weights, NOT
 *      the token-embedding table reused. Easier to reason about: every
 *      number has exactly one source of gradient.
 *
 *   2. BIAS-FREE linear layers. Every matmul is just `out = x @ W`, with
 *      no `+ bias` afterward. Fewer weights, fewer gradient paths to get
 *      right. (Layer-norm still has its learned gamma/beta scale+shift —
 *      those are not "biases" in this sense and we keep them.)
 *
 *
 * MEMORY MODEL — the single-block trick:
 *
 *   All trainable weights live in ONE contiguous malloc'd array, `params`.
 *   A second array, `grads`, has the exact same size and layout and holds
 *   each weight's gradient. The `ParameterTensors` struct is just a bundle
 *   of pointers INTO those blocks, one per weight matrix, so the math code
 *   can say `model->w.qkv_w` instead of doing offset arithmetic everywhere.
 *
 *   Why one block instead of 13 separate mallocs? Because later tasks become
 *   one-liners: the AdamW optimizer (Task 8) updates every weight with a
 *   single loop over `params`/`grads`; the MPI gradient sync (Task 11)
 *   averages everyone's gradients with a single AllReduce over `grads`.
 *   This mirrors Karpathy's llm.c, the reference C implementation.
 *
 *   The activations (the intermediate tensors the forward pass produces)
 *   use the same trick: one `acts` block, one `grads_acts` block, bundled
 *   by `ActivationTensors`. These depend on the batch size B and sequence
 *   length T, so they are allocated lazily on the first forward pass.
 */

#include <stddef.h>  /* size_t */

/* These are C functions. The extern "C" guard stops a C++ compiler — e.g.
 * nvcc compiling the .cu CUDA wiring in src/cuda/gpt_cuda.cu, or the CUDA
 * test that uses attention_forward as an oracle — from name-mangling them, so
 * the symbols link against the C-built `gpt` library. It is invisible to plain
 * C and mirrors the guard already used in layers.h and cuda_layers.cuh. Having
 * the header self-describe its linkage means C++/CUDA callers just #include it
 * normally instead of wrapping the include in extern "C" at each call site. */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * GPTConfig — the handful of numbers that define the model's shape.
 *
 * Everything else (parameter count, buffer sizes) is DERIVED from these.
 * Tests use tiny values (e.g. n_embd=8, n_layer=1) so a failure is small
 * enough to read; the real seminar model uses the values in PLAN.md.
 *
 *   max_seq_len : the longest sequence the position table can handle (T_max).
 *                 A forward pass may use any T <= max_seq_len.
 *   vocab_size  : number of distinct tokens (V). Output has one score per token.
 *   n_embd      : width of every token's vector (C). The "residual stream" width.
 *   n_head      : number of attention heads (NH). MUST divide n_embd evenly,
 *                 because each head works on a C/n_head-wide slice.
 *   n_layer     : how many transformer blocks are stacked (L).
 *   ff_dim      : width of the feed-forward hidden layer. Conventionally 4*n_embd.
 */
typedef struct {
    int max_seq_len;
    int vocab_size;
    int n_embd;
    int n_head;
    int n_layer;
    int ff_dim;
} GPTConfig;

/*
 * NUM_PARAM_TENSORS — how many distinct weight tensors the model has.
 * Used to size the internal array of per-tensor sizes when we lay out the
 * single `params` block. If you add or remove a weight below, update this.
 */
#define NUM_PARAM_TENSORS 13

/*
 * ParameterTensors — a bundle of pointers, one per weight tensor, each
 * pointing at the right offset inside the big `params` (or `grads`) block.
 *
 * Shapes are written as (rows, cols) of the row-major matrix each pointer
 * addresses. "(L, ...)" means the tensor is repeated once per layer and the
 * layers are laid out back-to-back, so layer l's data starts at offset
 * l * (size of one layer's slab).
 *
 *   wte         (V, C)        token embedding table: row = a token's vector
 *   wpe         (max_seq_len, C)  position embedding table: row = a position's vector
 *   ln1_g/ln1_b (L, C)        layer-norm #1 scale (gamma) and shift (beta), per layer
 *   qkv_w       (L, C, 3C)    projects each token vector into Query+Key+Value (3C wide)
 *   attn_proj_w (L, C, C)     mixes the attention output back to width C
 *   ln2_g/ln2_b (L, C)        layer-norm #2 scale and shift, per layer
 *   fc_w        (L, C, FF)    feed-forward "up" projection (C -> ff_dim)
 *   fc_proj_w   (L, FF, C)    feed-forward "down" projection (ff_dim -> C)
 *   lnf_g/lnf_b (C)           final layer-norm scale and shift (one, not per layer)
 *   lm_head_w   (C, V)        output table: final vector -> one score per vocab word
 */
typedef struct {
    float *wte;
    float *wpe;
    float *ln1_g;
    float *ln1_b;
    float *qkv_w;
    float *attn_proj_w;
    float *ln2_g;
    float *ln2_b;
    float *fc_w;
    float *fc_proj_w;
    float *lnf_g;
    float *lnf_b;
    float *lm_head_w;
} ParameterTensors;

/*
 * NUM_ACT_TENSORS — number of distinct activation tensors cached per forward.
 * We cache (almost) every intermediate result because the backward pass
 * needs them: a gradient through a matmul needs the matmul's inputs, a
 * gradient through gelu needs gelu's input, and so on (see the per-function
 * notes in layers.h).
 */
#define NUM_ACT_TENSORS 16

/*
 * ActivationTensors — pointers into the `acts` (or `grads_acts`) block, one
 * per intermediate tensor of the forward pass. All sizes scale with the
 * current batch B and sequence length T, so this block is allocated lazily
 * on the first forward (and re-allocated if B or T changes).
 *
 *   encoded    (B,T,C)        token + position embeddings: the initial residual stream
 *   ln1        (L,B,T,C)      output of layer-norm #1, per layer
 *   qkv        (L,B,T,3C)     Query/Key/Value for every token, per layer
 *   att        (L,B,NH,T,T)   attention weights (after softmax) per head, per layer
 *   atty       (L,B,T,C)      attention output (weighted sum of Values), pre-projection
 *   attproj    (L,B,T,C)      attention output after the attn_proj_w matmul
 *   residual2  (L,B,T,C)      residual stream after adding the attention block
 *   ln2        (L,B,T,C)      output of layer-norm #2, per layer
 *   fch        (L,B,T,FF)     feed-forward hidden, BEFORE gelu
 *   fch_gelu   (L,B,T,FF)     feed-forward hidden, AFTER gelu
 *   fcproj     (L,B,T,C)      feed-forward output after the down projection
 *   residual3  (L,B,T,C)      residual stream after adding the FFN block (= next layer's input)
 *   lnf        (B,T,C)        output of the final layer-norm
 *   logits     (B,T,V)        raw per-word scores (before softmax)
 *   probs      (B,T,V)        softmax(logits): a probability per word
 *   losses     (B,T)          per-position cross-entropy (averaged into model->loss)
 */
typedef struct {
    float *encoded;
    float *ln1;
    float *qkv;
    float *att;
    float *atty;
    float *attproj;
    float *residual2;
    float *ln2;
    float *fch;
    float *fch_gelu;
    float *fcproj;
    float *residual3;
    float *lnf;
    float *logits;
    float *probs;
    float *losses;
} ActivationTensors;

/*
 * GPTModel — everything needed to run and train one model.
 *
 *   config      : the shape (see GPTConfig).
 *   num_params  : total number of floats in `params` (== in `grads`).
 *   params      : the single contiguous weight block (owned; freed by gpt_free).
 *   grads       : same size/layout as params; each weight's gradient (owned).
 *   w           : pointers into `params`, one per weight tensor.
 *   g           : pointers into `grads`,  one per weight tensor (same layout).
 *
 *   B, T        : batch size and sequence length the activation blocks are
 *                 currently sized for. 0 before the first forward pass.
 *   num_acts    : total floats in `acts` (== in `grads_acts`).
 *   acts        : contiguous activation block for the last forward (owned).
 *   grads_acts  : gradients of the activations, same layout (owned).
 *   a           : pointers into `acts`.
 *   da          : pointers into `grads_acts`.
 *
 *   loss        : mean cross-entropy from the last forward pass that was
 *                 given targets. Undefined if the last forward had targets==NULL.
 */
typedef struct {
    GPTConfig config;

    int num_params;
    float *params;
    float *grads;
    ParameterTensors w;
    ParameterTensors g;

    int B;
    int T;
    size_t num_acts;
    float *acts;
    float *grads_acts;
    ActivationTensors a;
    ActivationTensors da;

    float loss;
} GPTModel;

/*
 * gpt_num_params — compute how many floats the weight block needs for a
 * given config, WITHOUT allocating anything.
 *
 * Pure function of the config (adds up the size of every tensor in
 * ParameterTensors). Exposed so tests can check the allocation against an
 * independently-derived number, and so callers can budget memory up front.
 *
 * Returns: the total float count (always > 0 for a valid config).
 *
 * Preconditions (asserted): all config dimensions > 0, and
 * n_embd is divisible by n_head.
 */
int gpt_num_params(const GPTConfig *config);

/*
 * gpt_init — allocate a model, point all the tensor views at the right
 * offsets, fill the weights with small random values, and zero the grads.
 *
 *   model  : caller-owned struct to fill in (pass &model). Its current
 *            contents are overwritten; do not pass an already-initialized
 *            model without gpt_free first, or you leak its buffers.
 *   config : the model shape (copied into model->config by value).
 *   seed   : random seed for weight initialization. Same seed + same config
 *            => byte-identical weights, which makes tests reproducible.
 *
 * Weight init: each weight is drawn from a Gaussian with mean 0 and a small
 * standard deviation (0.02, the GPT-2 default). Small random weights break
 * symmetry (so different neurons learn different things) without making the
 * initial activations explode.
 *
 * Does NOT allocate the activation blocks — those depend on B and T and are
 * created on the first gpt_forward call.
 *
 * Ownership: the model now owns `params` and `grads`. Release with gpt_free.
 *
 * Preconditions (asserted): model non-NULL, config valid (see gpt_num_params).
 */
void gpt_init(GPTModel *model, GPTConfig config, unsigned int seed);

/*
 * gpt_free — release every heap buffer the model owns and zero its pointers.
 *
 * Safe to call on a model that never ran a forward pass (the activation
 * blocks are NULL then, and free(NULL) is a documented no-op in C). After
 * this returns, the struct is back to an empty shell; do not use its
 * pointers.
 *
 *   model : the model to tear down (non-NULL).
 */
void gpt_free(GPTModel *model);

/*
 * gpt_forward — run the model on a batch and (optionally) compute the loss.
 *
 *   model   : an initialized model (from gpt_init).
 *   tokens  : input token ids, shape (B, T), row-major. Every id MUST be in
 *             [0, vocab_size) — the embedding lookup trusts this (same
 *             boundary contract as embed_forward in layers.h).
 *   targets : the "correct next token" for each position, shape (B, T), or
 *             NULL. If non-NULL, the cross-entropy loss is computed and
 *             stored in model->loss (and per-position in a.losses). If NULL,
 *             only logits/probs are produced (used later for generation).
 *   B, T    : batch size and sequence length. T MUST be <= max_seq_len.
 *
 * On the first call (or whenever B or T changes from the previous call) the
 * activation blocks are (re)allocated to fit. The intermediate tensors are
 * cached in model->a for the backward pass, so a gpt_backward must be
 * preceded by the matching gpt_forward with the SAME tokens/targets/B/T.
 *
 * Preconditions (asserted): model non-NULL and initialized, tokens non-NULL,
 * B > 0, 0 < T <= max_seq_len.
 */
void gpt_forward(GPTModel *model, const int *tokens, const int *targets,
                 int B, int T);

/*
 * gpt_zero_grad — set every parameter gradient back to zero.
 *
 * Call this once at the start of a training step, BEFORE gpt_backward. The
 * backward functions ACCUMULATE into the gradient buffers (the `+=`
 * convention documented in layers.h), so without zeroing first you would
 * add this step's gradients on top of the previous step's leftovers.
 *
 * Only touches the parameter gradients (`grads`). The activation gradients
 * (`grads_acts`) are pure scratch and are zeroed automatically at the start
 * of every gpt_backward.
 *
 *   model : an initialized model (non-NULL).
 */
void gpt_zero_grad(GPTModel *model);

/*
 * gpt_backward — compute the gradient of the loss w.r.t. every weight.
 *
 * Walks the forward diagram in reverse, calling the `_backward` op for each
 * step, and accumulates the result into model->grads (via the `+=`
 * convention). Must be called AFTER a gpt_forward with the same arguments
 * (it reads the activations that forward cached) and AFTER gpt_zero_grad
 * (so the accumulation starts from zero).
 *
 *   model   : the model whose last forward we are differentiating.
 *   tokens  : the SAME token ids passed to the matching forward.
 *   targets : the SAME targets passed to the matching forward (must be
 *             non-NULL here — there is no gradient without a loss).
 *   B, T    : the SAME batch/sequence dimensions as the matching forward.
 *
 * Preconditions (asserted): a matching forward has run (acts allocated for
 * this B,T), targets non-NULL, dimensions match the cached forward.
 */
void gpt_backward(GPTModel *model, const int *tokens, const int *targets,
                  int B, int T);

/*
 * gpt_ensure_acts — (re)allocate the activation blocks for a given (B, T) and
 * point model->a / model->da at their slices, WITHOUT running a forward pass.
 *
 * gpt_forward calls this internally on its first run; it is also exposed here
 * so the CUDA wiring (src/cuda/gpt_cuda.cu) can size its device buffers and
 * derive device sub-pointers from the host offsets (model->a.field -
 * model->acts) before it has run anything on the GPU. After this returns,
 * model->acts / model->grads_acts are allocated, model->num_acts is set, and
 * model->B / model->T record the current shape.
 *
 *   model : an initialized model (from gpt_init).
 *   B, T  : batch size and sequence length. T MUST be <= max_seq_len.
 *
 * Idempotent: calling it again with the same (B, T) is a no-op (the buffers
 * are reused); a different (B, T) frees and re-allocates.
 */
void gpt_ensure_acts(GPTModel *model, int B, int T);

/*
 * attention_forward — causal multi-head self-attention (forward).
 *
 * Exposed (rather than left static in gpt.c) so the CUDA parity test can use
 * this proven CPU implementation as the oracle for attention_forward_cuda —
 * the same "CPU is the oracle" pattern the layer ops use. The full math is
 * documented at the definition in gpt.c.
 *
 *   atty : (B, T, C)      attention output (weighted sum of Values), per head
 *   att  : (B, NH, T, T)  softmax attention weights (cached for backward;
 *                         masked future positions written as 0)
 *   qkv  : (B, T, 3C)     concatenated Query, Key, Value for every token
 *   B,T,C,NH             batch, sequence length, channels, number of heads
 *                        (C must be divisible by NH; head_dim = C / NH)
 *
 * atty and att are caller-allocated outputs; qkv is read-only. No aliasing.
 */
void attention_forward(float *atty, float *att, const float *qkv,
                       int B, int T, int C, int NH);

/*
 * attention_backward — gradient of causal multi-head self-attention.
 *
 * Exposed for the same CUDA-parity-test reason as attention_forward. Full
 * derivation lives at the definition in gpt.c. Accumulates into d_qkv with +=
 * (caller must pre-zero it), matching the project-wide backward convention.
 *
 *   d_qkv  : (B, T, 3C)   destination gradient of Q/K/V, accumulated with +=
 *   d_atty : (B, T, C)    downstream gradient flowing into the attention output
 *   qkv    : (B, T, 3C)   cached forward Q/K/V
 *   att    : (B, NH, T, T) cached forward softmax weights
 *   B,T,C,NH             same dimensions as the forward call
 */
void attention_backward(float *d_qkv, const float *d_atty,
                        const float *qkv, const float *att,
                        int B, int T, int C, int NH);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* GPT_H */
