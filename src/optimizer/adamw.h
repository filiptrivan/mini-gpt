#ifndef ADAMW_H
#define ADAMW_H

/*
 * adamw.h — the AdamW optimizer: the rule that turns gradients into weight
 * updates, i.e. the part of training that actually makes the model better.
 *
 * Include guard above (#ifndef / #define / #endif): if two .c files both
 * include this header, the guard makes the second include a no-op so the
 * struct below is not defined twice (which would be a compile error).
 *
 *
 * WHERE THIS FITS — one training step, end to end:
 *
 *   gpt_zero_grad(model)                 reset last step's gradients to 0
 *   gpt_forward (model, x, y, B, T)      tokens -> loss   (how wrong are we?)
 *   gpt_backward(model, x, y, B, T)      loss  -> grads   (which way is "less wrong"?)
 *   adamw_step  (opt, model->params,     grads -> weights (take a step that way)
 *                     model->grads)
 *
 * The optimizer is deliberately ignorant of what a "GPT" is. It only sees two
 * flat float arrays of equal length: `params` (the numbers to improve) and
 * `grads` (the slope of the loss w.r.t. each number). Because gpt.c stores ALL
 * weights in one contiguous block (see gpt.h's "single-block trick"), the whole
 * model is optimized by a single call over those two arrays.
 *
 *
 * WHAT "AdamW" MEANS — the intuition behind the math in adamw_step:
 *
 *   Plain gradient descent does `p -= lr * g`: step downhill, step size fixed.
 *   That is fragile — too big a step for a steep direction overshoots, too small
 *   a step for a shallow one crawls. Adam fixes this by tracking, per parameter,
 *   two running averages:
 *
 *     m ("first moment")  — a smoothed average of the gradient itself.
 *                           This is MOMENTUM: it keeps moving in a consistent
 *                           direction and cancels out zig-zag noise.
 *     v ("second moment") — a smoothed average of the gradient SQUARED.
 *                           This estimates how big this parameter's gradients
 *                           tend to be, so we can divide by it and give every
 *                           parameter its own effective step size.
 *
 *   The update is roughly `p -= lr * m / sqrt(v)`: move in the smoothed
 *   direction, but scaled so that consistently-large-gradient parameters and
 *   consistently-tiny-gradient parameters both move a sensible amount.
 *
 *   The "W" is decoupled WEIGHT DECAY: separately from the gradient, nudge every
 *   weight a little toward zero each step (`p -= lr * weight_decay * p`). This is
 *   a mild regularizer that discourages any single weight from growing huge.
 *   "Decoupled" = applied straight to the weight, NOT folded into the gradient
 *   (that distinction is exactly what separates AdamW from plain "Adam + L2").
 *
 *
 * MEMORY MODEL:
 *
 *   The optimizer owns two heap buffers, `m` and `v`, each the same length as
 *   the parameter array. They persist across steps (that is the whole point —
 *   they accumulate history) and are released by adamw_free. The params/grads
 *   arrays themselves are owned by the model, NOT by the optimizer; adamw_step
 *   reads grads and writes params but frees neither.
 */

/*
 * AdamW — optimizer state: the fixed hyperparameters plus the per-parameter
 * running averages that have to survive from one step to the next.
 *
 *   lr           : learning rate — the base step size. The single most
 *                  important knob. Typical: 1e-3 to 3e-4.
 *   beta1        : decay rate for the first moment m (momentum). Closer to 1 =
 *                  smoother/slower-reacting. Typical: 0.9.
 *   beta2        : decay rate for the second moment v. Typical: 0.999.
 *   eps          : tiny constant added before dividing by sqrt(v), so a
 *                  parameter whose v is ~0 doesn't blow up the update. ~1e-8.
 *   weight_decay : decoupled pull toward zero each step. 0 disables it.
 *
 *   num_params   : length of m, v, and the params/grads arrays this optimizer
 *                  is meant to update. Set once at init.
 *   t            : how many steps have been taken. 0 before the first step,
 *                  incremented at the start of each adamw_step. Used for "bias
 *                  correction" (see adamw_step) — early steps would otherwise
 *                  be biased toward zero because m and v start at zero.
 *
 *   m            : first-moment estimate, one float per parameter (owned).
 *   v            : second-moment estimate, one float per parameter (owned).
 */
typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    int   num_params;
    int   t;

    float *m;
    float *v;
} AdamW;

/*
 * adamw_init — set the hyperparameters and allocate the moment buffers.
 *
 *   opt          : caller-owned struct to fill in (pass &opt). Its current
 *                  contents are overwritten; do not pass an already-initialized
 *                  optimizer without adamw_free first, or you leak its buffers.
 *   num_params   : number of parameters this optimizer will update. Must match
 *                  the length of the params/grads arrays passed to adamw_step
 *                  (in practice, model->num_params).
 *   lr, beta1, beta2, eps, weight_decay : the hyperparameters above, copied in.
 *
 * The moment buffers m and v are allocated here and ZEROED — Adam's running
 * averages must start from zero (and the bias correction in adamw_step assumes
 * exactly that). The step counter t is reset to 0.
 *
 * Ownership: the optimizer now owns `m` and `v`. Release with adamw_free.
 *
 * Preconditions (asserted): opt non-NULL, num_params > 0, beta1 and beta2 in
 * [0, 1), eps > 0. (lr and weight_decay are not constrained — 0 is meaningful
 * for both, and a test may use an unusual lr on purpose.)
 */
void adamw_init(AdamW *opt, int num_params,
                float lr, float beta1, float beta2,
                float eps, float weight_decay);

/*
 * adamw_step — apply one AdamW update to every parameter.
 *
 *   opt    : an initialized optimizer (from adamw_init).
 *   params : the parameter array to UPDATE in place, length opt->num_params.
 *            Owned by the caller (the model); we only read and overwrite its
 *            values, never free or reallocate it.
 *   grads  : the gradient of the loss w.r.t. each parameter, same length.
 *            Read only — produced by the backward pass before this call.
 *
 * For each parameter i, using g = grads[i], this computes:
 *
 *     t      += 1                               (done once, not per element)
 *     m[i]    = beta1*m[i] + (1-beta1)*g        update first moment
 *     v[i]    = beta2*v[i] + (1-beta2)*g*g      update second moment
 *     m_hat   = m[i] / (1 - beta1^t)            bias-correct (compensate for
 *     v_hat   = v[i] / (1 - beta2^t)            m,v starting at zero)
 *     params[i] -= lr * ( m_hat/(sqrt(v_hat)+eps) + weight_decay*params[i] )
 *
 * The bias correction (dividing by 1 - beta^t) matters most on the first few
 * steps: m and v start at 0, so without it the early updates would be far too
 * small. As t grows, beta^t -> 0 and the correction fades to 1.
 *
 * Preconditions (asserted): opt, params, grads all non-NULL.
 */
void adamw_step(AdamW *opt, float *params, const float *grads);

/*
 * adamw_free — release the moment buffers and zero the optimizer's pointers.
 *
 * Safe to call on a zero-initialized or already-freed optimizer (free(NULL) is
 * a documented no-op in C). After this returns the struct is an empty shell;
 * do not call adamw_step on it without re-initializing. Does NOT touch the
 * model's params/grads — those are not ours to free.
 *
 *   opt : the optimizer to tear down (non-NULL).
 */
void adamw_free(AdamW *opt);

#endif /* ADAMW_H */
