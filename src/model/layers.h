#ifndef LAYERS_H
#define LAYERS_H

/* These are C functions. The extern "C" guard stops a C++ compiler — e.g.
 * nvcc compiling the .cu CUDA tests/kernels — from name-mangling them, so the
 * symbols link against the C-built `layers` library. It is invisible to plain
 * C, and mirrors the guard already used in src/cuda/cuda_layers.cuh. Having
 * the header self-describe its linkage means C++/CUDA callers just #include it
 * normally, instead of wrapping the include in extern "C" at each call site. */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Neural network layer operations — CPU forward and backward passes.
 *
 * Every function here takes raw float arrays (no fancy tensor objects) and
 * writes into a caller-owned `out` buffer. The caller is always responsible
 * for allocating both inputs and outputs. We never malloc inside these
 * functions; that keeps the math code small and predictable, and lets the
 * training loop reuse buffers across steps to avoid heap churn.
 *
 * Naming convention: `<op>_forward` for the forward pass, `<op>_backward`
 * for the backward (gradient) pass.
 *
 * Backward-pass convention — IMPORTANT, this affects every caller:
 *   All `_backward` functions ACCUMULATE into their gradient outputs with
 *   `+=`, they do NOT overwrite. The caller MUST zero the gradient buffers
 *   exactly once at the start of a training step. The reason: in a real
 *   transformer the same tensor is reused on multiple paths (residual
 *   connections fan a single gradient into two inputs; the same parameter
 *   matrix is touched by every batch element; one embedding row receives
 *   gradients from every position where that token appears). With `+=`
 *   semantics the caller writes the same one-liner for every layer; with
 *   `=` semantics we would need a wrapping `+=` at every call site, which
 *   is error-prone. This matches the convention used in Karpathy's llm.c
 *   and every well-written hand-rolled C reference implementation.
 *
 *   Tests must zero each gradient buffer before calling backward, OR
 *   pre-fill it with a known value if testing the accumulation property.
 */

/*
 * residual_forward — element-wise addition: out[i] = a[i] + b[i].
 *
 * In transformers, this implements the "residual connection" (a.k.a. skip
 * connection): after a sublayer transforms its input, we add the original
 * input back. Mathematically: y = x + Sublayer(x). This lets gradients flow
 * directly through the addition during training, which is the trick that
 * makes deep networks trainable at all.
 *
 * Parameters:
 *   out — destination buffer of length N (caller-allocated, will be overwritten)
 *   a   — first input array of length N
 *   b   — second input array of length N
 *   N   — number of elements in each array
 *
 * Buffers may NOT overlap unless out == a or out == b exactly (in-place add
 * along one input is safe; partial overlap is undefined behavior). The
 * `const` on a and b promises this function will not modify them — a useful
 * contract both for the compiler (enables optimizations) and for the reader.
 *
 * Preconditions (enforced by assert() in debug builds, undefined behavior
 * otherwise — callers must guarantee these):
 *   - out, a, b are non-NULL
 *   - N > 0
 */
void residual_forward(float *out, const float *a, const float *b, int N);

/*
 * gelu_forward — element-wise GELU activation: out[i] = gelu(in[i]).
 *
 * GELU = Gaussian Error Linear Unit. It's a smooth, non-linear function
 * applied between layers so the network can model non-linear patterns
 * (without any non-linearity, stacked layers collapse to one matrix mult).
 *
 * We use the tanh approximation that GPT-2 / GPT-3 use:
 *   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
 *
 * The "true" GELU uses the error function (erf) but is more expensive to
 * compute; the tanh form is the de-facto standard in the GPT family and
 * what every reference implementation (Karpathy's nanoGPT, OpenAI's GPT-2
 * release) uses, so we match it byte-for-byte to make later comparisons easy.
 *
 * Parameters:
 *   out — destination buffer of length N (caller-allocated)
 *   in  — input array of length N
 *   N   — element count
 *
 * In-place is safe: passing the same pointer for out and in works because
 * each output element depends only on the same-index input element.
 *
 * Preconditions:
 *   - out, in are non-NULL
 *   - N > 0
 */
void gelu_forward(float *out, const float *in, int N);

/*
 * matmul_forward — standard matrix multiply: out = a @ b.
 *
 * Shapes (all matrices stored row-major as flat 1D arrays):
 *   a   : M × K   →  a[i*K + k] is row i, column k
 *   b   : K × N   →  b[k*N + j] is row k, column j
 *   out : M × N   →  out[i*N + j] is row i, column j
 *
 * The "inner" dimension K is the one that gets summed over and must match
 * between a's columns and b's rows. The "outer" dimensions M and N survive
 * into the output shape.
 *
 * Element formula:
 *   out[i][j] = sum over k in [0, K) of a[i][k] * b[k][j]
 *
 * Parameters:
 *   out — destination buffer of length M*N (caller-allocated, overwritten)
 *   a   — first matrix,  M*K floats
 *   b   — second matrix, K*N floats
 *   M, K, N — see shape table above
 *
 * out must NOT alias a or b — the inner loop reads from a and b multiple
 * times per output cell, so writing into them mid-computation corrupts
 * later reads. Use separate buffers.
 *
 * This is the simple O(M*N*K) triple-loop implementation. In Task 9 we'll
 * write a tiled CUDA version that's much faster on GPUs; the math here is
 * the reference output that the CUDA kernel will be compared against.
 *
 * Preconditions:
 *   - out, a, b are non-NULL
 *   - M > 0, K > 0, N > 0
 */
void matmul_forward(float *out, const float *a, const float *b, int M, int K, int N);

/*
 * softmax_forward — row-wise softmax.
 *
 * Treats `in` as N independent rows of length V, applies the softmax
 * function to each row, and writes the result into `out` (same shape).
 *
 *   softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
 *
 * Properties of each output row:
 *   - every element is in (0, 1]
 *   - the row sums to exactly 1 (a probability distribution)
 *   - the position of the largest input is also the position of the largest output
 *
 * Numerical stability: we subtract max(row) from every element before
 * exponentiating. The math is identical (a constant cancels in numerator
 * and denominator) but avoids `expf` overflow on large inputs — without
 * this, training would silently produce NaN losses. This is the standard
 * trick used by every deep learning framework.
 *
 * Layout (row-major, like matmul):
 *   in  : N rows × V cols  →  in[n*V + v]
 *   out : N rows × V cols  →  out[n*V + v]
 *
 * out and in must NOT alias. The implementation reads from in during the
 * same pass that writes to out, so overlap would corrupt later reads.
 *
 * Preconditions:
 *   - out, in are non-NULL
 *   - N > 0
 *   - V > 0  (we read in_row[0] before the loop to seed max; an empty row
 *     would access out-of-bounds memory)
 */
void softmax_forward(float *out, const float *in, int N, int V);

/*
 * layernorm_forward — row-wise layer normalization with learned scale/shift.
 *
 * For each of the N rows independently:
 *   1. Compute row mean μ and row variance σ².
 *   2. Normalize: x_hat[c] = (in[c] - μ) / sqrt(σ² + ε).
 *      After this step the row has mean ≈ 0 and variance ≈ 1.
 *   3. Affine: out[c] = gamma[c] * x_hat[c] + beta[c].
 *      gamma and beta are LEARNED during training, so the network can
 *      undo or amplify the normalization if that helps it minimize loss.
 *
 * Shapes (row-major):
 *   in    : N rows × C cols  →  in[n*C + c]
 *   out   : N rows × C cols  →  out[n*C + c]
 *   gamma : C floats         →  shared across all N rows (one scale per feature)
 *   beta  : C floats         →  shared across all N rows (one shift per feature)
 *
 * Epsilon (ε) is hardcoded to 1e-5, matching PyTorch's nn.LayerNorm default.
 * It prevents division by zero when a row happens to be constant
 * (variance = 0). Without it, training would produce NaN whenever any
 * row collapses to a single value, even momentarily.
 *
 * out must NOT alias in.
 *
 * Preconditions:
 *   - out, in, gamma, beta are non-NULL
 *   - N > 0
 *   - C > 0  (the mean and variance computations divide by C; C == 0 would
 *     produce NaN/Inf and silently poison the rest of the forward pass)
 */
void layernorm_forward(float *out, const float *in,
                       const float *gamma, const float *beta,
                       int N, int C);

/*
 * embed_forward — token embedding + position embedding lookup and add.
 *
 * For every (b, t) position in a (B × T) batch of token-id sequences,
 * looks up the token's row in the token-embedding table, looks up the
 * position's row in the position-embedding table, and writes their sum
 * into the output.
 *
 *   out[b, t, c] = wte[ tokens[b, t], c ]  +  wpe[ t, c ]
 *
 * Shapes (all row-major, output and tables are float; tokens is int):
 *   tokens : B × T            integer token IDs in [0, vocab_size)
 *   wte    : vocab_size × C   token embedding table (caller owns vocab_size)
 *   wpe    : max_seq_len × C  position embedding table (caller owns max_seq_len)
 *   out    : B × T × C        output embeddings
 *
 * We don't pass vocab_size or max_seq_len because this function never
 * iterates over either dimension — it only indexes into them by token ID
 * (read from `tokens`) and by position (the loop variable `t`). It's the
 * caller's responsibility to ensure every value in `tokens` is < vocab_size
 * and that T ≤ max_seq_len. Out-of-bounds IDs would read random memory.
 *
 * This is the FIRST operation in GPT's forward pass: it's what turns the
 * integer tokens produced by the BPE tokenizer into the float vectors
 * that everything downstream (layernorm, attention, matmul, ...) consumes.
 *
 * Preconditions:
 *   - out, tokens, wte, wpe are non-NULL
 *   - B > 0, T > 0, C > 0
 *   - EVERY token id in `tokens` must satisfy 0 <= id < vocab_size.
 *     We cannot assert this here because vocab_size is not passed in
 *     (the function never iterates it — it only indexes into wte by id),
 *     but an out-of-range id reads random memory at `wte + id*C` with no
 *     warning, then poisons every downstream layer with garbage floats.
 *     Validate token ids at the data-loader / tokenizer boundary.
 *   - T must satisfy T <= max_seq_len (the row count of wpe), for the
 *     same reason: we index wpe by t without bounds-checking.
 */
void embed_forward(float *out, const int *tokens,
                   const float *wte, const float *wpe,
                   int B, int T, int C);

/* ====================================================================== */
/*                          BACKWARD PASS                                 */
/* ====================================================================== */

/*
 * residual_backward — gradient of out = a + b w.r.t. its two inputs.
 *
 * The derivative of an element-wise sum is trivial: ∂out[i]/∂a[i] = 1
 * and ∂out[i]/∂b[i] = 1, with zero off-diagonals. By the chain rule,
 *   dL/da[i] = dL/dout[i] * 1 = dL/dout[i]
 *   dL/db[i] = dL/dout[i] * 1 = dL/dout[i]
 *
 * So a residual_backward just copies d_out into both d_a and d_b (with
 * the `+=` accumulation described in the file-level note above).
 *
 * This is the math fact that makes residual connections so important
 * during training: gradients flow through them unchanged. Without the
 * skip path, gradient magnitudes shrink (or explode) layer by layer
 * and deep networks become untrainable.
 *
 * Parameters:
 *   d_a   — destination buffer, length N. Will receive d_out added in.
 *   d_b   — destination buffer, length N. Will receive d_out added in.
 *   d_out — gradient flowing in from downstream, length N. Not modified.
 *   N     — number of elements.
 *
 * Aliasing: d_a, d_b, d_out must NOT overlap each other. Each one is a
 * separate gradient buffer in the caller; the math assumes they're
 * independent. (Specifically, d_a == d_b would double-count d_out, and
 * d_out == d_a would read its own freshly-written values mid-loop.)
 *
 * Preconditions:
 *   - d_a, d_b, d_out are non-NULL
 *   - N > 0
 *   - d_a and d_b have been zeroed (or pre-set) by the caller, since
 *     we accumulate with += per the file-level convention.
 */
void residual_backward(float *d_a, float *d_b, const float *d_out, int N);

/*
 * gelu_backward — element-wise derivative of the tanh-approx GELU.
 *
 * Let s = sqrt(2/π) * (x + 0.044715 * x^3). Then
 *   gelu(x) = 0.5 * x * (1 + tanh(s))
 *
 * Differentiating w.r.t. x (product rule + chain rule on tanh):
 *   gelu'(x) = 0.5 * (1 + tanh(s))
 *            + 0.5 * x * (1 - tanh(s)^2) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
 *
 * The two pieces come from differentiating the two factors of
 * (0.5 * x) * (1 + tanh(s)): the first piece is the product rule's
 * "leave the second factor alone, derive the first"; the second piece is
 * "leave the first alone, derive the second" — which means chain-ruling
 * through tanh (giving sech^2 = 1 - tanh^2) and then through the inner
 * polynomial s(x).
 *
 * By the chain rule from downstream loss L:
 *   dL/dx[i] = dL/dout[i] * gelu'(in[i])
 *
 * Note we need the ORIGINAL `in` (not `out`) here — the derivative is a
 * function of x, not gelu(x). The caller must keep the forward input
 * around until backward runs. That's why every forward pass in a
 * training loop caches its inputs.
 *
 * Parameters:
 *   d_in  — destination buffer, length N. Accumulated with +=.
 *   d_out — gradient from downstream, length N. Not modified.
 *   in    — original forward input, length N. Not modified.
 *   N     — element count.
 *
 * In-place is NOT safe here: this op reads in[i] to compute the local
 * derivative but writes into d_in[i]. They are separate buffers in any
 * sane caller, but if you somehow aliased d_in == in you'd corrupt the
 * upstream forward cache. d_in == d_out is also unsafe — the `+=`
 * conflates the accumulated gradient with the downstream gradient
 * (each iteration both reads d_out[i] and increments d_in[i] at the
 * same address, so the read picks up earlier writes instead of the
 * caller's downstream value).
 *
 * Preconditions:
 *   - d_in, d_out, in are non-NULL
 *   - N > 0
 *   - d_in has been zeroed (or pre-set) by the caller.
 */
void gelu_backward(float *d_in, const float *d_out, const float *in, int N);

/*
 * matmul_backward — gradient of out = a @ b w.r.t. a and b.
 *
 * Forward shapes (row-major flat arrays):
 *   a   : M × K       b   : K × N       out : M × N
 *
 * Differentiating out[i,j] = sum_k a[i,k] * b[k,j] gives:
 *   d_a[i,k] = sum_j  d_out[i,j] * b[k,j]      ← d_out @ b^T  (shape M × K)
 *   d_b[k,j] = sum_i  a[i,k]    * d_out[i,j]   ← a^T @ d_out  (shape K × N)
 *
 * Mnemonic: in the forward, the dimension K is "summed over" (it's the
 * shared dimension). In each backward, a DIFFERENT dimension becomes
 * the shared one — j for d_a, i for d_b. The two halves are independent;
 * neither needs the other's output to compute its own.
 *
 * Why we need a and b separately: each input grad needs the OTHER input
 * (the partial derivative of a product wrt one factor is the other
 * factor). So forward inputs must be cached for backward — same general
 * pattern as gelu_backward needing the forward `in`.
 *
 * Parameters:
 *   d_a   — destination buffer of length M*K. Accumulated with +=.
 *   d_b   — destination buffer of length K*N. Accumulated with +=.
 *   d_out — downstream gradient of length M*N. Not modified.
 *   a     — original forward `a`, length M*K. Not modified.
 *   b     — original forward `b`, length K*N. Not modified.
 *   M, K, N — same dimensions as the forward call.
 *
 * Aliasing: each of {d_a, d_b, d_out, a, b} must be a distinct buffer.
 *
 * Preconditions:
 *   - d_a, d_b, d_out, a, b are non-NULL
 *   - M > 0, K > 0, N > 0
 *   - d_a and d_b have been zeroed (or pre-set) by the caller.
 */
void matmul_backward(float *d_a, float *d_b,
                     const float *d_out, const float *a, const float *b,
                     int M, int K, int N);

/*
 * softmax_backward — Jacobian-vector product through row-wise softmax.
 *
 * Forward shapes (row-major):
 *   in, out : N rows × V cols
 *
 * For one row, let y = softmax(x). Then
 *   ∂y[j]/∂x[i] = y[i] * (δ_ij - y[j])
 *
 * Chain rule:
 *   d_in[i] = sum_j d_out[j] * ∂y[j]/∂x[i]
 *           = sum_j d_out[j] * y[i] * (δ_ij - y[j])
 *           = y[i] * d_out[i]  -  y[i] * sum_j d_out[j] * y[j]
 *           = y[i] * ( d_out[i] - sum_j d_out[j] * y[j] )
 *
 * The factored form on the last line is what we implement: O(V) per row
 * vs. the naive double-sum which would be O(V^2). The trick that makes
 * this efficient is that the sum_j term doesn't depend on i, so we
 * compute it once per row and reuse it across all V outputs.
 *
 * Note: we take `out` (the forward output y), NOT the forward input x.
 * Once you have y, the input is no longer needed for the gradient.
 * Caching y is cheaper than recomputing the softmax from x.
 *
 * Parameters:
 *   d_in  — destination, N*V floats. Accumulated with +=.
 *   d_out — downstream gradient, N*V floats. Not modified.
 *   out   — cached forward output (y), N*V floats. Not modified.
 *   N, V  — same as the forward call.
 *
 * Aliasing: d_in must not overlap d_out or out (we read both during the
 * write to d_in). d_out and out are read-only so they may safely share
 * if the caller wanted — but they won't in practice.
 *
 * Preconditions:
 *   - d_in, d_out, out are non-NULL
 *   - N > 0, V > 0
 *   - d_in has been zeroed (or pre-set) by the caller.
 */
void softmax_backward(float *d_in, const float *d_out, const float *out,
                      int N, int V);

/*
 * layernorm_backward — gradient through row-wise layer normalization,
 * including the learned affine parameters gamma and beta.
 *
 * Forward (per row, dropping the row index for brevity):
 *   μ     = mean(in)
 *   σ²    = var(in)
 *   rstd  = 1 / sqrt(σ² + ε)
 *   x_hat = (in - μ) * rstd
 *   out   = gamma * x_hat + beta
 *
 * Per-row gradients (full derivation: see Kevin Clark's "LayerNorm
 * backward" derivation or the comments in Karpathy's llm.c; this is the
 * same formula every framework uses):
 *   d_beta[c]  += d_out[c]
 *   d_gamma[c] += d_out[c] * x_hat[c]
 *   d_x_hat[c]  = d_out[c] * gamma[c]
 *
 *   Then for d_in we need to back-propagate through the (in - μ) * rstd
 *   step. Both μ and rstd depend on every element of `in`, so d_in[c]
 *   actually mixes contributions from every other column in the row.
 *   The closed form is:
 *
 *   d_in[c] += rstd * ( d_x_hat[c]
 *                       - (1/C) * sum_j d_x_hat[j]
 *                       - x_hat[c] * (1/C) * sum_j (d_x_hat[j] * x_hat[j]) )
 *
 *   Intuitively: the first term is the "direct" gradient; the second
 *   subtracts off the gradient's row-mean (because shifting in by a
 *   constant doesn't change out — μ shifts with it); the third subtracts
 *   the projection of d_x_hat onto x_hat (because scaling in doesn't
 *   change out either — rstd rescales it back).
 *
 * d_beta and d_gamma have shape C (one entry per feature, summed over
 * all N rows). d_in has shape N × C, just like in.
 *
 * Forward caches needed: we take `in` and recompute μ and rstd inside
 * the backward. This is O(C) extra work per row vs. caching them, but
 * keeps the layers.h surface area small. If profiling later shows
 * layernorm is hot, we'll add a layernorm_forward_train variant that
 * also returns mean and rstd buffers.
 *
 * Parameters:
 *   d_in    — destination N*C floats, accumulated with +=.
 *   d_gamma — destination C floats,  accumulated with +=.
 *   d_beta  — destination C floats,  accumulated with +=.
 *   d_out   — downstream gradient, N*C floats. Not modified.
 *   in      — original forward input, N*C floats. Not modified.
 *   gamma   — forward gamma, C floats. Not modified.
 *   N, C    — same as the forward call.
 *
 * Preconditions:
 *   - all six pointers are non-NULL
 *   - N > 0, C > 0
 *   - d_in, d_gamma, d_beta have been zeroed (or pre-set) by the caller.
 */
void layernorm_backward(float *d_in, float *d_gamma, float *d_beta,
                        const float *d_out, const float *in,
                        const float *gamma,
                        int N, int C);

/*
 * embed_backward — scatter d_out into the embedding tables.
 *
 * Forward:
 *   out[b, t, c] = wte[ tokens[b, t], c ] + wpe[ t, c ]
 *
 * The token IDs are integer indices, not floats, so there is NO gradient
 * w.r.t. `tokens` — you can't take a derivative of "which row did I pick."
 * What we DO compute are gradients into the embedding TABLES, which are
 * the trainable parameters:
 *
 *   d_wte[ tokens[b,t], c ] += d_out[b, t, c]   (for every b, t, c)
 *   d_wpe[ t,           c ] += d_out[b, t, c]   (for every b, t, c)
 *
 * The `+=` matters here more than anywhere else, because:
 *   - The SAME token id can appear at many (b, t) positions in a batch.
 *     Every appearance contributes to the same row of d_wte. Without +=
 *     you would overwrite earlier contributions and silently lose
 *     gradient signal — a classic, hard-to-spot bug.
 *   - For d_wpe, every batch element contributes to the SAME row t,
 *     so each d_wpe[t] row accumulates B contributions per call (one
 *     per batch element at position t). Trivially correct with B=1,
 *     but with B>1 the contributions MUST be summed, not overwritten.
 *
 * Forward caches needed: we take `tokens` (to know which rows of wte
 * to write to) and the dimensions. We do NOT need `wte`, `wpe`, or the
 * forward `out` — the gradient is purely a scatter.
 *
 * Parameters:
 *   d_wte  — destination of shape vocab_size × C, accumulated with +=.
 *            Caller MUST know vocab_size and allocate accordingly; we
 *            never iterate it explicitly here (we only index by token id,
 *            same as embed_forward), so the function signature does not
 *            include it. Same validity contract as embed_forward: every
 *            token id must satisfy 0 <= id < vocab_size, or we write
 *            garbage to wherever (d_wte + id*C) lands in memory.
 *   d_wpe  — destination of shape max_seq_len × C, accumulated with +=.
 *            Likewise, T must satisfy T <= max_seq_len.
 *   d_out  — downstream gradient of shape B*T*C. Not modified.
 *   tokens — original integer token IDs of shape B*T. Not modified.
 *   B, T, C — same as the forward call.
 *
 * Preconditions:
 *   - d_wte, d_wpe, d_out, tokens are non-NULL
 *   - B > 0, T > 0, C > 0
 *   - d_wte and d_wpe have been zeroed (or pre-set) by the caller.
 *   - Token IDs are in range and T <= max_seq_len (same boundary
 *     contract as embed_forward — enforced by the data pipeline, not
 *     by an assert here, because vocab_size and max_seq_len are not
 *     parameters of this function).
 */
void embed_backward(float *d_wte, float *d_wpe,
                    const float *d_out, const int *tokens,
                    int B, int T, int C);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* LAYERS_H */
