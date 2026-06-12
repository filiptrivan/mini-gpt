/*
 * CPU implementations of neural network layer operations.
 *
 * Functions in this file are declared in layers.h and implement the math
 * for one transformer training/inference step. Each is intentionally small
 * and contains no I/O, no logging, and no allocation — just float math on
 * caller-provided buffers, so the same logic is easy to mirror in CUDA later.
 */

#include <assert.h> /* assert() — compiles to a runtime check in debug builds, a no-op under -DNDEBUG */
#include <math.h>   /* tanhf, expf, sqrtf — single-precision math from the C standard library */
#include <stddef.h> /* NULL — defined here in C99 (also dragged in by stdio/stdlib, but we don't include those) */

#include "layers.h"

/*
 * Why assert() everywhere instead of returning error codes:
 *   - These functions sit on the hot path of every forward pass; a branch
 *     per call would pile up.
 *   - The preconditions (non-NULL pointers, positive dims) are programmer
 *     contracts, not user input — they should never be violated by correct
 *     code. assert() catches bugs during development and disappears in
 *     -DNDEBUG release builds, which is exactly the trade we want.
 *   - This matches the style the professor requested in PR #3 review:
 *     "clearly check or document that pointers aren't NULL and dimensions
 *     are greater than zero." The header docstrings carry the documentation
 *     side; the asserts here carry the runtime check.
 */

/*
 * Shared numerical constants.
 *
 * These live at file scope so each forward/backward pair reads from a single
 * source. Previously each function declared its own copy with a "keep in
 * sync" comment — exactly the smell that justifies hoisting them: if a
 * future change tweaks one copy (e.g., switching GELU to the exact erf form)
 * the partner would silently drift, and the numerical gradient check would
 * still pass because it always runs forward and backward together against
 * each other rather than against an external reference.
 *
 *   GELU_SQRT_2_OVER_PI, GELU_COEF — constants in the tanh-approximation
 *     of GELU used by gelu_forward and gelu_backward.
 *   LAYERNORM_EPS                 — small constant added inside the
 *     sqrt() in layernorm to prevent divide-by-zero on constant rows.
 *     Matches PyTorch's nn.LayerNorm default of 1e-5.
 */
static const float GELU_SQRT_2_OVER_PI = 0.7978845608028654f; /* = sqrt(2/π) */
static const float GELU_COEF           = 0.044715f;           /* magic constant from the GPT-2 paper */
static const float LAYERNORM_EPS       = 1e-5f;

/*
 * residual_forward — element-wise add: out[i] = a[i] + b[i]. See layers.h.
 *
 * The whole function is one for-loop because there is no inter-element
 * dependency: out[3] doesn't need out[2] first. That property is also why
 * this op is trivially parallelizable — in Task 10 the CUDA version will
 * assign one GPU thread per i and run all N additions simultaneously.
 */
void residual_forward(float *out, const float *a, const float *b, int N) {
    assert(out != NULL);
    assert(a   != NULL);
    assert(b   != NULL);
    assert(N    > 0);

    for (int i = 0; i < N; i++) {
        out[i] = a[i] + b[i];
    }
}

/*
 * gelu_forward — GPT-2 tanh-approximation of GELU. See layers.h.
 *
 * Formula: gelu(x) = 0.5 * x * (1 + tanh( SQRT_2_OVER_PI * (x + 0.044715 * x^3) ))
 *
 * Constants are pulled out of the loop as `static const` so the compiler
 * stores them once instead of materializing them every iteration. The `f`
 * suffix (e.g. 0.5f) marks them as `float` rather than `double` — without
 * the suffix the compiler would promote `x` to double, do the math in
 * double precision, then narrow back to float, which is both slower and
 * gives bit-different results from reference implementations.
 *
 * tanhf (not tanh) is the float-precision version; matching precision end
 * to end keeps every kernel comparable to the eventual CUDA version, which
 * also uses single precision.
 */
void gelu_forward(float *out, const float *in, int N) {
    assert(out != NULL);
    assert(in  != NULL);
    assert(N    > 0);

    /* Constants live at file scope so gelu_backward reads from the same
     * declaration — see the "Shared numerical constants" block near the
     * top of this file. */
    for (int i = 0; i < N; i++) {
        float x     = in[i];
        float x_cubed = x * x * x;
        float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/*
 * matmul_forward — out = a @ b, triple-nested loop. See layers.h.
 *
 * Total work is M*N*K multiply-adds either way you slice it, but the
 * order of the three loops matters a lot for speed because of CPU caches.
 *
 * Textbook order (i, j, k): "for each output cell, compute its dot
 * product." Intuitive, but the innermost line reads b[k*N + j] — and as
 * k increments, that address jumps forward by N floats (a whole row of
 * b). For any N bigger than a cache line (~16 floats), every step is a
 * cache miss.
 *
 * Order used here (i, k, j): walk the inner loop along contiguous memory
 * in b and out. For each (i, k), broadcast the scalar a[i,k] across the
 * entire k-th row of b, accumulating into out_row. Same math (the sum
 * over k is just split into K += operations instead of one) but reads
 * are sequential, the prefetcher is happy, and the compiler can
 * auto-vectorize the inner j-loop. Roughly 2-5x faster at our model
 * sizes (e.g. M=64, K=128, N=512 → ~4M MACs per matmul).
 *
 * Because we accumulate with += across the k-loop, the output row must
 * be zeroed first. The textbook version got away with a single `sum`
 * register per cell; we pay one extra pass over out_row for the zero
 * init, recouped many times over by the cache-friendly inner loop.
 *
 * The CUDA version in Task 9 will use yet another structure (tiled
 * blocks loaded into shared memory) — the principle is the same:
 * arrange memory access so the hardware can stream data efficiently.
 */
void matmul_forward(float *out, const float *a, const float *b, int M, int K, int N) {
    assert(out != NULL);
    assert(a   != NULL);
    assert(b   != NULL);
    assert(M    > 0);
    assert(K    > 0);
    assert(N    > 0);

    for (int i = 0; i < M; i++) {
        float       *out_row = out + i * N;   /* output row i */
        const float *a_row   = a   + i * K;   /* input row i of a — its k-th element is the scalar we broadcast below */

        /* Zero out_row so the += accumulation in the k-loop starts from 0. */
        for (int j = 0; j < N; j++) {
            out_row[j] = 0.0f;
        }

        /* For each k, scale b's k-th row by the scalar a_row[k] and add
         * it into out_row. After K iterations, out_row[j] holds the full
         * dot product sum_k a[i,k] * b[k,j] for every j — same result
         * as the textbook order, computed in a different sweep pattern. */
        for (int k = 0; k < K; k++) {
            float a_ik = a_row[k];               /* scalar, reused across the j-loop */
            const float *b_row = b + k * N;      /* contiguous row of b */
            for (int j = 0; j < N; j++) {
                out_row[j] += a_ik * b_row[j];
            }
        }
    }
}

/*
 * softmax_forward — numerically stable row-wise softmax. See layers.h.
 *
 * Three passes over each row:
 *   Pass 1: find max(row). Used as a shift in pass 2 to keep exp() in range.
 *   Pass 2: compute exp(in[v] - max) for each element, and accumulate the
 *           sum into `denom` as we go (saving a separate loop).
 *   Pass 3: divide each output by `denom` to normalize to a probability
 *           distribution that sums to 1.
 *
 * Why three passes and not one: we need max(row) before any exp() can be
 * computed (pass 1 finishes first), and we need the full sum before any
 * division (pass 2 finishes before pass 3). The data dependencies prevent
 * a single-pass formulation. The cost is O(V) per row either way; the
 * three-pass structure just makes the dependencies explicit.
 *
 * `expf` (not `exp`) keeps everything in float precision — same reason as
 * tanhf in gelu_forward: matches single-precision results on GPU later.
 *
 * Row-pointer trick: instead of writing `in[n*V + v]` four times per
 * iteration, we compute `in + n*V` once per row and treat that as a
 * fresh 1D array `in_row[0..V-1]`. Clearer to read and slightly faster
 * (one multiply hoisted out of the inner loops).
 */
void softmax_forward(float *out, const float *in, int N, int V) {
    assert(out != NULL);
    assert(in  != NULL);
    assert(N    > 0);
    assert(V    > 0);  /* V == 0 would crash on the `in_row[0]` seed read below */

    for (int n = 0; n < N; n++) {
        const float *in_row = in  + n * V;   /* start of row n in `in`  */
        float       *out_row = out + n * V;  /* start of row n in `out` */

        /* Pass 1: find max(row). Start with the first element so any
         * row of any sign works (initializing to 0 would be wrong for
         * rows where every element is negative). */
        float max_val = in_row[0];
        for (int v = 1; v < V; v++) {
            if (in_row[v] > max_val) {
                max_val = in_row[v];
            }
        }

        /* Pass 2: shifted exp and running sum. After subtracting max_val,
         * the largest argument is 0, so the largest exponential is 1 —
         * no overflow possible. The denominator is at least 1, so the
         * division in pass 3 is always safe. */
        float denom = 0.0f;
        for (int v = 0; v < V; v++) {
            float e = expf(in_row[v] - max_val);
            out_row[v] = e;     /* stash for pass 3 — saves recomputing expf */
            denom += e;
        }

        /* Pass 3: normalize. Multiplying by 1/denom once and using that
         * inverse inside the loop is faster than dividing each iteration
         * (division is slower than multiplication on most CPUs). */
        float inv_denom = 1.0f / denom;
        for (int v = 0; v < V; v++) {
            out_row[v] *= inv_denom;
        }
    }
}

/*
 * layernorm_forward — see layers.h.
 *
 * Per row, three passes:
 *   1. Sum elements → divide by C → mean (μ).
 *   2. Sum squared deviations → divide by C → variance (σ²).
 *      Then compute reciprocal std-dev: rstd = 1 / sqrt(σ² + ε).
 *      Storing rstd (not stdev) lets pass 3 multiply instead of divide
 *      per element — same micro-optimization we did in softmax.
 *   3. For each c: x_hat = (in[c] - μ) * rstd, then out[c] = γ[c]*x_hat + β[c].
 *
 * ε = 1e-5 matches PyTorch's nn.LayerNorm default — useful so anyone
 * comparing our numbers to a PyTorch reference sees matching values.
 */
void layernorm_forward(float *out, const float *in,
                       const float *gamma, const float *beta,
                       int N, int C) {
    assert(out   != NULL);
    assert(in    != NULL);
    assert(gamma != NULL);
    assert(beta  != NULL);
    assert(N      > 0);
    assert(C      > 0);  /* mean = sum / C and var = sq_sum / C — C == 0 would NaN-poison the row */

    /* Hoist 1/C outside the n-loop so each row pays one divide instead of
     * two. The backward pass uses exactly this pattern; keeping them
     * symmetric also keeps the float results bit-identical between
     * forward and backward, which is important for the gradient check. */
    float inv_C = 1.0f / (float)C;

    for (int n = 0; n < N; n++) {
        const float *in_row  = in  + n * C;
        float       *out_row = out + n * C;

        /* Pass 1: mean of the row. */
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += in_row[c];
        }
        float mean = sum * inv_C;

        /* Pass 2: variance of the row, then reciprocal std-dev.
         * We use the "two-pass" formula (recompute deviations from the
         * mean we just found) rather than the one-pass "sum of squares
         * minus square of sum" formula, because the latter loses
         * precision when the values are large relative to their spread. */
        float sq_sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = in_row[c] - mean;
            sq_sum += d * d;
        }
        float var  = sq_sum * inv_C;
        float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);   /* sqrtf = single-precision sqrt */

        /* Pass 3: normalize, then apply learned scale and shift.
         * γ and β are per-feature (length C), shared across all N rows. */
        for (int c = 0; c < C; c++) {
            float x_hat = (in_row[c] - mean) * rstd;
            out_row[c]  = gamma[c] * x_hat + beta[c];
        }
    }
}

/*
 * embed_forward — see layers.h.
 *
 * Triple loop: batch × time × feature. For every (b, t):
 *   1. Read the token ID at tokens[b*T + t] (an int, not a float).
 *   2. Locate the matching row of wte (the token's embedding vector).
 *   3. Locate row t of wpe (the position embedding for this time step).
 *   4. Add the two vectors element-wise and write into out.
 *
 * Pointer-arithmetic trick: instead of writing `wte[token_id*C + c]` and
 * `wpe[t*C + c]` inside the innermost loop, we compute the row start
 * pointers ONCE per (b,t) and treat them as plain C-length arrays.
 * Same readability + speed reason as in softmax/layernorm.
 *
 * No allocation, no branches inside the inner loop — this op is so
 * cheap that the only thing limiting it is memory bandwidth.
 */
void embed_forward(float *out, const int *tokens,
                   const float *wte, const float *wpe,
                   int B, int T, int C) {
    assert(out    != NULL);
    assert(tokens != NULL);
    assert(wte    != NULL);
    assert(wpe    != NULL);
    assert(B       > 0);
    assert(T       > 0);
    assert(C       > 0);
    /* NOTE: we cannot assert(tokens[i] < vocab_size) here — vocab_size is
     * not a parameter of this function (see layers.h). An out-of-range
     * token id silently reads garbage from wherever `wte + id*C` lands in
     * memory. The data loader and tokenizer must guarantee valid ids. */

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            /* Step 1: which token sits at this batch/position? */
            int token_id = tokens[b * T + t];

            /* Step 2 & 3: locate the two source rows.
             * wte_row points at wte[token_id, 0]; wpe_row at wpe[t, 0]. */
            const float *wte_row = wte + token_id * C;
            const float *wpe_row = wpe + t        * C;

            /* Output row: out[b, t, 0..C-1].
             * Flat index follows the same row-major rule as everywhere
             * else: walk b batches of T*C floats, then t blocks of C. */
            float *out_row = out + (b * T + t) * C;

            /* Step 4: add the two embedding vectors element-wise. */
            for (int c = 0; c < C; c++) {
                out_row[c] = wte_row[c] + wpe_row[c];
            }
        }
    }
}

/* ====================================================================== */
/*                          BACKWARD PASS                                 */
/* ====================================================================== */

/* Every backward ACCUMULATES with `+=` into its destination gradient
 * buffers; the caller is responsible for zeroing them once per training
 * step. Rationale and full contract live in the file-level note at the
 * top of layers.h. */

/*
 * residual_backward — see layers.h.
 *
 * ∂(a+b)/∂a = ∂(a+b)/∂b = 1, so the upstream gradient flows into both
 * inputs unchanged. One pass over the array, two `+=` per i. This is
 * literally the cheapest operation in the entire transformer's
 * backward pass.
 */
void residual_backward(float *d_a, float *d_b, const float *d_out, int N) {
    assert(d_a   != NULL);
    assert(d_b   != NULL);
    assert(d_out != NULL);
    assert(N      > 0);

    for (int i = 0; i < N; i++) {
        d_a[i] += d_out[i];
        d_b[i] += d_out[i];
    }
}

/*
 * gelu_backward — see layers.h.
 *
 * Differentiating  gelu(x) = 0.5 * x * (1 + tanh(s(x)))  where
 * s(x) = SQRT_2_OVER_PI * (x + GELU_COEF * x^3):
 *
 *   s'(x)    = SQRT_2_OVER_PI * (1 + 3 * GELU_COEF * x^2)
 *   tanh'(s) = 1 - tanh^2(s)          (the sech^2 identity)
 *
 * Product rule on 0.5 * x * (1 + tanh(s)):
 *   gelu'(x) = 0.5 * (1 + tanh(s))                                  (1)
 *            + 0.5 * x * (1 - tanh^2(s)) * s'(x)                    (2)
 *
 * Then by the chain rule from downstream loss L:
 *   d_in[i] += d_out[i] * gelu'(in[i])
 *
 * We cache tanh(s) in a local so we don't compute it twice per element.
 * Same constants and the same float-precision (`tanhf` not `tanh`,
 * `0.5f` not `0.5`) as the forward — anything else gives bit-different
 * results from the forward, which would tank the gradcheck.
 */
void gelu_backward(float *d_in, const float *d_out, const float *in, int N) {
    assert(d_in  != NULL);
    assert(d_out != NULL);
    assert(in    != NULL);
    assert(N      > 0);

    /* Constants are declared at file scope (see "Shared numerical
     * constants" near the top of this file) so this function and
     * gelu_forward read from the same source. */
    for (int i = 0; i < N; i++) {
        float x        = in[i];
        float x_cubed  = x * x * x;
        float s        = GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
        float tanh_s   = tanhf(s);
        float sech2_s  = 1.0f - tanh_s * tanh_s;  /* 1 - tanh^2 = sech^2 */
        float s_prime  = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x * x);

        float dgelu_dx = 0.5f * (1.0f + tanh_s)
                       + 0.5f * x * sech2_s * s_prime;

        d_in[i] += d_out[i] * dgelu_dx;
    }
}

/*
 * matmul_backward — see layers.h.
 *
 * The two halves correspond to the two factors of out = a @ b:
 *
 *   d_a[i,k] += sum_j d_out[i,j] * b[k,j]      (shape M × K)
 *   d_b[k,j] += sum_i a[i,k]    * d_out[i,j]   (shape K × N)
 *
 * They are computed in two independent passes here. Could be fused into
 * a single triple loop, but keeping them separate makes each one easy
 * to verify against the formula and to mirror in CUDA later (each will
 * become its own kernel).
 *
 * Pass 1 — d_a, loop order (i, k, j):
 *   For each output cell d_a[i,k], the inner j-loop is a dot product
 *   between row i of d_out (length N, contiguous) and row k of b (length
 *   N, contiguous). Two contiguous streams + one scalar accumulator =
 *   maximally cache-friendly. Same memory pattern as the dot product in
 *   matmul_forward's textbook order, just walked over different rows.
 *
 * Pass 2 — d_b, loop order (i, k, j):
 *   This is identical to matmul_forward's inner pattern: the j-loop
 *   reads d_out[i, 0..N-1] and writes d_b[k, 0..N-1], both contiguous,
 *   with a constant a[i,k] scalar. Same trick the forward uses to win
 *   ~2-5x over the textbook order; reused here for the same reason.
 *
 * Note: pass 2 writes to d_b with +=. That's important when this
 * function is called multiple times accumulating into the same d_b
 * (the file-level convention), but it ALSO matters within this one
 * call: every (i, j, k) contributes to the same d_b[k, j] slot, so we
 * must accumulate across i in pass 2 (and across j in pass 1 via the
 * local `sum`).
 */
void matmul_backward(float *d_a, float *d_b,
                     const float *d_out, const float *a, const float *b,
                     int M, int K, int N) {
    assert(d_a   != NULL);
    assert(d_b   != NULL);
    assert(d_out != NULL);
    assert(a     != NULL);
    assert(b     != NULL);
    assert(M      > 0);
    assert(K      > 0);
    assert(N      > 0);

    /* Pass 1: d_a[i,k] += sum_j d_out[i,j] * b[k,j].
     * Two contiguous row reads per (i, k) cell, one accumulator. */
    for (int i = 0; i < M; i++) {
        const float *d_out_row = d_out + i * N;       /* d_out[i, 0..N-1] */
        float       *d_a_row   = d_a   + i * K;       /* d_a[i, 0..K-1]   */
        for (int k = 0; k < K; k++) {
            const float *b_row = b + k * N;           /* b[k, 0..N-1]     */
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += d_out_row[j] * b_row[j];
            }
            d_a_row[k] += sum;
        }
    }

    /* Pass 2: d_b[k,j] += sum_i a[i,k] * d_out[i,j].
     * Same (i, k, j) memory pattern as the matmul_forward inner loop —
     * a_ik is a broadcast scalar, d_out_row and d_b_row are walked
     * contiguously side by side. */
    for (int i = 0; i < M; i++) {
        const float *a_row     = a     + i * K;       /* a[i, 0..K-1]     */
        const float *d_out_row = d_out + i * N;       /* d_out[i, 0..N-1] */
        for (int k = 0; k < K; k++) {
            float a_ik = a_row[k];
            float *d_b_row = d_b + k * N;             /* d_b[k, 0..N-1]   */
            for (int j = 0; j < N; j++) {
                d_b_row[j] += a_ik * d_out_row[j];
            }
        }
    }
}

/*
 * softmax_backward — see layers.h.
 *
 * Per row, the trick is the factored formula
 *   d_in[v] += y[v] * ( d_out[v] - s ),   where s = sum_j d_out[j] * y[j].
 *
 * Two passes per row:
 *   Pass 1: compute the scalar s = dot(d_out_row, y_row). O(V).
 *   Pass 2: write d_in[v] += y[v] * (d_out[v] - s). O(V).
 *
 * Total O(V) per row, vs. O(V^2) for the naive double-sum
 * d_in[v] = sum_j d_out[j] * y[v] * (delta_vj - y[j]). The trick that
 * makes it linear is recognizing that y[v] factors out of the j-sum.
 * This is exactly the formulation every framework (PyTorch, TF, JAX,
 * Karpathy's llm.c) uses.
 *
 * `out` is the FORWARD output y, cached by the caller. We don't need
 * the forward input here — y carries all the information about the
 * Jacobian we need. Caching y is one float-per-element extra memory
 * but saves us recomputing exp+sum during backward.
 */
void softmax_backward(float *d_in, const float *d_out, const float *out,
                      int N, int V) {
    assert(d_in  != NULL);
    assert(d_out != NULL);
    assert(out   != NULL);
    assert(N      > 0);
    assert(V      > 0);

    for (int n = 0; n < N; n++) {
        const float *out_row   = out   + n * V;
        const float *d_out_row = d_out + n * V;
        float       *d_in_row  = d_in  + n * V;

        /* Pass 1: row-scalar s = sum_j d_out[j] * y[j]. */
        float s = 0.0f;
        for (int v = 0; v < V; v++) {
            s += d_out_row[v] * out_row[v];
        }

        /* Pass 2: scatter d_in. */
        for (int v = 0; v < V; v++) {
            d_in_row[v] += out_row[v] * (d_out_row[v] - s);
        }
    }
}

/*
 * layernorm_backward — see layers.h.
 *
 * Per row:
 *   1. Recompute mean μ and rstd r = 1/sqrt(var+ε) from `in` — same
 *      math as the forward, just inline. We don't cache mean/rstd
 *      from forward (yet); recomputing is O(C) per row, dwarfed by
 *      the rest of the backward.
 *   2. Compute x_hat[c] = (in[c] - μ) * r,  d_x_hat[c] = d_out[c] * γ[c].
 *      These are the per-element ingredients we need.
 *   3. Accumulate d_γ[c] += d_out[c] * x_hat[c] and d_β[c] += d_out[c].
 *      γ and β are shared across all N rows, so these are sums across n.
 *   4. Two row-level scalars:
 *        sum1 = (1/C) * sum_j d_x_hat[j]
 *        sum2 = (1/C) * sum_j d_x_hat[j] * x_hat[j]
 *   5. d_in[c] += r * ( d_x_hat[c] - sum1 - x_hat[c] * sum2 )
 *
 * The closed form in step 5 comes from grinding through chain rule on
 * the full normalization. Intuition: any uniform shift of `in` doesn't
 * change `out` (μ shifts with it) — that's the sum1 subtraction.
 * Any uniform scaling doesn't change `out` either (rstd rescales it
 * back) — that's the sum2 projection along x_hat.
 *
 * Both sums are over the row only — gamma and beta gradients accumulate
 * across rows because γ, β are shared, but the within-row reductions
 * stay row-local because μ, rstd are per-row.
 */
void layernorm_backward(float *d_in, float *d_gamma, float *d_beta,
                        const float *d_out, const float *in,
                        const float *gamma,
                        int N, int C) {
    assert(d_in    != NULL);
    assert(d_gamma != NULL);
    assert(d_beta  != NULL);
    assert(d_out   != NULL);
    assert(in      != NULL);
    assert(gamma   != NULL);
    assert(N        > 0);
    assert(C        > 0);

    /* Epsilon lives at file scope (see "Shared numerical constants" near
     * the top of this file) so this function and layernorm_forward read
     * from the same declaration — drift between forward and backward
     * would silently invalidate the gradient check. */
    float inv_C = 1.0f / (float)C;

    for (int n = 0; n < N; n++) {
        const float *in_row    = in    + n * C;
        const float *d_out_row = d_out + n * C;
        float       *d_in_row  = d_in  + n * C;

        /* Step 1: recompute mean and rstd for this row. Identical to
         * the forward's two-pass formula. */
        float sum = 0.0f;
        for (int c = 0; c < C; c++) sum += in_row[c];
        float mean = sum * inv_C;

        float sq_sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = in_row[c] - mean;
            sq_sum += d * d;
        }
        float var  = sq_sum * inv_C;
        float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);

        /* Step 2-4 fused: walk the row once to compute x_hat, d_x_hat,
         * the two row-scalars sum1 and sum2, and the accumulations into
         * d_gamma / d_beta. We need sum1 and sum2 before we can write
         * d_in, so step 5 takes a second pass over the row.
         *
         * NOTE: x_hat and d_x_hat are stored per-element via local
         * computation — we recompute them in step 5 rather than
         * allocating C-element scratch buffers. Cheap (a multiply +
         * subtract per element) and keeps this function allocation-free,
         * matching the style of every other op in this file. */
        float sum1 = 0.0f;  /* sum_j d_x_hat[j]            */
        float sum2 = 0.0f;  /* sum_j d_x_hat[j] * x_hat[j] */
        for (int c = 0; c < C; c++) {
            float x_hat   = (in_row[c] - mean) * rstd;
            float d_x_hat = d_out_row[c] * gamma[c];

            sum1 += d_x_hat;
            sum2 += d_x_hat * x_hat;

            /* gamma and beta gradients are shared across rows, so they
             * accumulate across n via the outer loop's += semantics. */
            d_gamma[c] += d_out_row[c] * x_hat;
            d_beta[c]  += d_out_row[c];
        }
        sum1 *= inv_C;
        sum2 *= inv_C;

        /* Step 5: write d_in. Second pass over the row; recomputes
         * x_hat and d_x_hat (cheap) instead of allocating scratch. */
        for (int c = 0; c < C; c++) {
            float x_hat   = (in_row[c] - mean) * rstd;
            float d_x_hat = d_out_row[c] * gamma[c];
            d_in_row[c] += rstd * (d_x_hat - sum1 - x_hat * sum2);
        }
    }
}

/*
 * embed_backward — see layers.h.
 *
 * The forward op is a gather (read embedding rows indexed by token id);
 * its backward is the dual operation, a SCATTER: each d_out[b, t, c]
 * gets added to two destination rows, d_wte[tokens[b,t], c] and
 * d_wpe[t, c].
 *
 * The `+=` accumulation matters here more than in any other layer:
 *
 *   - The same token id can appear at many (b, t) positions. Every
 *     appearance writes into the SAME row of d_wte. Without +=, all
 *     but the last write would be silently lost — a real-world bug
 *     that's hard to catch by eyeballing the numbers because the
 *     resulting gradient is still "shaped right," just smaller.
 *
 *   - For d_wpe, every batch element contributes to the same row t,
 *     so each d_wpe[t] row accumulates B contributions per call (one
 *     per batch element at that position). With B>1 the contributions
 *     MUST be summed, not overwritten.
 *
 * Both writes are inside the innermost loop, so the per-element cost
 * is two stores plus one load (d_out_row[c]). No reductions, no
 * cross-element dependencies — trivially parallelizable on GPU (the
 * CUDA version will use atomicAdd for d_wte; d_wpe doesn't need it
 * because each thread writes to a distinct row).
 */
void embed_backward(float *d_wte, float *d_wpe,
                    const float *d_out, const int *tokens,
                    int B, int T, int C) {
    assert(d_wte  != NULL);
    assert(d_wpe  != NULL);
    assert(d_out  != NULL);
    assert(tokens != NULL);
    assert(B       > 0);
    assert(T       > 0);
    assert(C       > 0);
    /* Same caveat as embed_forward: out-of-range token ids would write
     * into garbage memory at d_wte + id*C. The data loader guarantees
     * the contract; we cannot check it here without taking vocab_size. */

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = tokens[b * T + t];

            const float *d_out_row = d_out + (b * T + t) * C;
            float       *d_wte_row = d_wte + token_id    * C;
            float       *d_wpe_row = d_wpe + t           * C;

            for (int c = 0; c < C; c++) {
                d_wte_row[c] += d_out_row[c];
                d_wpe_row[c] += d_out_row[c];
            }
        }
    }
}
