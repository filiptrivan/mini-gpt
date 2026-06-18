/*
 * adamw.c — the AdamW optimizer.
 *
 * Implements the update rule documented in adamw.h: per-parameter momentum (m)
 * and per-parameter gradient-magnitude tracking (v), combined with bias
 * correction and decoupled weight decay. The whole model is updated by a single
 * loop over its flat params/grads arrays — see adamw.h for the big picture and
 * the intuition behind each term.
 */

#include "optimizer/adamw.h"

#include <assert.h>
#include <stdlib.h>  /* calloc, free */
#include <math.h>    /* sqrtf, powf */

/*
 * adamw_init — store the hyperparameters and allocate ZEROED moment buffers.
 *
 * calloc (not malloc) is deliberate: it returns memory already set to all-zero
 * bytes, which for IEEE-754 floats is exactly 0.0f. Adam's bias correction in
 * adamw_step assumes m and v start at zero, so this zeroing is a correctness
 * requirement, not just tidiness.
 *
 * See adamw.h for parameter docs and the ownership contract.
 */
void adamw_init(AdamW *opt, int num_params,
                float lr, float beta1, float beta2,
                float eps, float weight_decay) {
    assert(opt != NULL);
    assert(num_params > 0);
    /* beta1/beta2 must be in [0,1): the bias-correction denominator is
     * (1 - beta^t), which would be <= 0 (division by zero or a sign flip) if a
     * beta were >= 1, and the running average makes no sense for beta < 0. */
    assert(beta1 >= 0.0f && beta1 < 1.0f);
    assert(beta2 >= 0.0f && beta2 < 1.0f);
    assert(eps > 0.0f);  /* eps guards the divide; it must be strictly positive */

    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->t = 0;

    /* calloc(count, size) = malloc(count*size) but zero-initialized. */
    opt->m = calloc((size_t)num_params, sizeof(float));
    opt->v = calloc((size_t)num_params, sizeof(float));
    assert(opt->m != NULL && opt->v != NULL);
}

/*
 * adamw_step — one AdamW update over every parameter.
 *
 * The recurrence (per parameter i, with g = grads[i]) is spelled out in
 * adamw.h. A few implementation notes:
 *
 *   - t is incremented ONCE here, before the loop, because the bias-correction
 *     factors (1 - beta^t) are the same for every parameter on this step.
 *
 *   - The bias-correction denominators are computed once with powf and reused,
 *     rather than recomputed per parameter — same value, much cheaper.
 *
 *   - The hyperparameters and the m/v base pointers are copied into locals
 *     before the loop. They don't change during a step, so this keeps the hot
 *     inner loop (it runs over every parameter, ~half a million of them) from
 *     re-reading them through `opt` and recomputing `1 - beta` each iteration.
 *
 *   - The two terms in the final update are intentionally separate:
 *         m_hat / (sqrt(v_hat) + eps)   <- the Adam step (gradient-driven)
 *         weight_decay * params[i]      <- decoupled decay (param-driven)
 *     Adding the decay to the PARAMETER (not to the gradient before it enters
 *     m/v) is precisely what makes this AdamW rather than Adam-with-L2: the
 *     decay never pollutes the moment estimates.
 */
void adamw_step(AdamW *opt, float *params, const float *grads) {
    assert(opt != NULL && params != NULL && grads != NULL);

    opt->t += 1;

    /* Loop-invariants pulled into locals once (see the block comment). The
     * `1 - beta` mixing coefficients are named here so the inner loop just
     * multiplies, and `m`/`v` alias opt->m/opt->v so we don't deref `opt`
     * every iteration. */
    const float b1 = opt->beta1, b2 = opt->beta2;
    const float one_minus_b1 = 1.0f - b1, one_minus_b2 = 1.0f - b2;
    const float lr = opt->lr, eps = opt->eps, wd = opt->weight_decay;
    float *m = opt->m, *v = opt->v;
    const int n = opt->num_params;

    /* Bias-correction denominators for this step. As t grows, beta^t -> 0 so
     * these approach 1 and the correction fades out (see adamw.h). powf is
     * float-precision pow; t is cast to float for the exponent. */
    const float bc1 = 1.0f - powf(b1, (float)opt->t);
    const float bc2 = 1.0f - powf(b2, (float)opt->t);

    for (int i = 0; i < n; i++) {
        float g = grads[i];

        /* update the running averages (in place, carried to the next step) */
        m[i] = b1 * m[i] + one_minus_b1 * g;
        v[i] = b2 * v[i] + one_minus_b2 * g * g;

        /* bias-corrected estimates */
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;

        /* the Adam step plus the decoupled weight-decay pull */
        float update = m_hat / (sqrtf(v_hat) + eps) + wd * params[i];
        params[i] -= lr * update;
    }
}

/*
 * adamw_free — release the moment buffers and null the pointers.
 *
 * free(NULL) is a no-op, so this is safe on a zero-initialized or already-freed
 * optimizer. Does not touch params/grads — those belong to the model.
 */
void adamw_free(AdamW *opt) {
    assert(opt != NULL);
    free(opt->m);
    free(opt->v);
    opt->m = NULL;
    opt->v = NULL;
}
