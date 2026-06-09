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

    static const float SQRT_2_OVER_PI = 0.7978845608028654f; /* = sqrt(2/π) */
    static const float GELU_COEF      = 0.044715f;           /* magic constant from the GPT-2 paper */

    for (int i = 0; i < N; i++) {
        float x     = in[i];
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
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

    static const float EPS = 1e-5f;

    for (int n = 0; n < N; n++) {
        const float *in_row  = in  + n * C;
        float       *out_row = out + n * C;

        /* Pass 1: mean of the row. */
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += in_row[c];
        }
        float mean = sum / (float)C;

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
        float var  = sq_sum / (float)C;
        float rstd = 1.0f / sqrtf(var + EPS);   /* sqrtf = single-precision sqrt */

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
