/*
 * Implementation of shared test helpers — see test_utils.h for the API
 * contract and rationale.
 */

#include <assert.h> /* assert() — same use as in src/model/layers.c */
#include <stddef.h> /* NULL */

#include "test_utils.h"

/*
 * numerical_gradient — central-difference loop over each dimension.
 *
 * The sweep is:
 *   for each i:
 *     save x[i]
 *     x[i] = saved + h           → evaluate L_plus  = loss_fn(x, ctx)
 *     x[i] = saved - h           → evaluate L_minus = loss_fn(x, ctx)
 *     x[i] = saved               → restore (exact bit-for-bit)
 *     grad_out[i] = (L_plus - L_minus) / (2 h)
 *
 * Restoring x[i] = saved (rather than x[i] -= h, then x[i] -= h, etc.)
 * matters: repeated additions/subtractions of h in float32 accumulate
 * rounding error and would leave x slightly off its original value.
 * Saving the original and writing it back guarantees byte-identical
 * restoration, so the caller can pass the same x to multiple gradient
 * checks without contamination.
 *
 * h is taken from the caller because the right value depends on the
 * scale of x and the smoothness of the loss; tests for layernorm may
 * want a slightly larger h than tests for residual, etc. Hardcoding
 * one value here would force every test to live with a compromise.
 */
void numerical_gradient(float *grad_out,
                        test_loss_fn_t loss_fn,
                        float *x, int n,
                        void *ctx, float h) {
    assert(grad_out != NULL);
    assert(loss_fn  != NULL);
    assert(x        != NULL);
    assert(n         > 0);
    assert(h         > 0.0f);

    for (int i = 0; i < n; i++) {
        /* Snapshot x[i] so we can restore it exactly after the two
         * perturbed evaluations. Bit-for-bit restoration matters when
         * the same x vector is reused across multiple grad checks. */
        float saved = x[i];

        x[i] = saved + h;
        float L_plus = loss_fn(x, ctx);

        x[i] = saved - h;
        float L_minus = loss_fn(x, ctx);

        x[i] = saved;  /* restore — must use saved, not -=h, to avoid drift */

        /* Central-difference quotient. Multiplying by the inverse once
         * is the same micro-optimization we used in softmax/layernorm —
         * one float divide replaced by one multiply. Order-of-magnitude
         * the same here since n is tiny, but kept for consistency. */
        grad_out[i] = (L_plus - L_minus) * (0.5f / h);
    }
}

/*
 * fill_random_ids — see test_utils.h for the contract and the why.
 *
 * The two magic constants (1664525, 1013904223) are the standard Numerical
 * Recipes LCG multiplier/increment; multiplying and adding in 32-bit unsigned
 * arithmetic wraps around (well-defined for unsigned in C) and walks through a
 * long, repeatable cycle of states. We take the HIGH bits (`*s >> 16`) because
 * the low bits of an LCG are notoriously non-random, then fold into [0, vocab)
 * with the modulo.
 */
void fill_random_ids(int *out, int n, int vocab, unsigned int *s) {
    assert(out   != NULL);
    assert(s     != NULL);
    assert(n     >= 0);
    assert(vocab  > 0);

    for (int i = 0; i < n; i++) {
        *s = (*s) * 1664525u + 1013904223u;            /* advance the LCG state          */
        out[i] = (int)((*s >> 16) % (unsigned)vocab);  /* high bits, then fold into range */
    }
}
