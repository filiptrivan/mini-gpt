#ifndef LAYERS_H
#define LAYERS_H

/*
 * Neural network layer operations — CPU forward pass.
 *
 * Every function here takes raw float arrays (no fancy tensor objects) and
 * writes into a caller-owned `out` buffer. The caller is always responsible
 * for allocating both inputs and outputs. We never malloc inside these
 * functions; that keeps the math code small and predictable, and lets the
 * training loop reuse buffers across steps to avoid heap churn.
 *
 * Naming convention: `<op>_forward` for the forward pass.
 * Backward passes (`<op>_backward`) come in Task 6.
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

#endif /* LAYERS_H */
