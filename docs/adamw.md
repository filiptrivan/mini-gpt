# The AdamW Optimizer

How `src/optimizer/adamw.c` turns gradients into weight updates — the part
of training that actually makes the model better.

The forward pass (`docs/forward-pass.md`) measures *how wrong* the model is.
The backward pass (`docs/backward-pass.md`) computes the gradient — *which
way* each weight should move to be less wrong. The optimizer is the third
step: it decides *how big a step* to take, and applies it.

## First, in plain words

### The problem

Training is just repeating this: nudge every weight a little to make the
loss smaller. The gradient points downhill, so the obvious rule is

```
weight = weight - learning_rate * gradient
```

That's **plain gradient descent**, and it's clumsy: it uses the *same* fixed
step size for every weight. Too big for a steep direction (you overshoot the
bottom), too small for a flat one (you crawl). We'd like each weight to get
its own sensible step size — automatically.

### Adam's fix: two memories per weight

Adam keeps two running averages **for every single weight**, updated each
step:

- **`m` — momentum.** A smoothed average of the gradient itself. It cancels
  out the step-to-step zig-zag noise and keeps you rolling in the direction
  that's been consistently downhill — like a ball gaining speed on a slope.

- **`v`.** A smoothed average of the gradient *squared*. This estimates "how
  big are this weight's gradients, typically?"

The update is, in essence:

```
weight = weight - learning_rate * m / sqrt(v)
```

Read it as: **step in the smoothed direction `m`, but divide by how big the
gradients usually are (`sqrt(v)`).** A weight with consistently huge
gradients takes small, careful steps; a weight with tiny gradients gets
boosted. Everyone moves a sensible amount — that's the whole idea.

```
plain descent:   every weight takes the same-size step  →  clumsy
Adam:            every weight takes its own right-size step
                 (smoothed by m, scaled by 1/sqrt(v))
```

### One wrinkle: bias correction

`m` and `v` both **start at zero**, so for the first few steps they're
artificially small (they haven't "filled up" yet). Adam compensates by
dividing them by `(1 - beta^t)`, where `t` is the step number. Early on this
boosts the tiny estimates back to full size; as training continues `beta^t`
shrinks to zero and the correction quietly fades to 1 (no effect).

A neat consequence: on the **very first step**, the correction makes the
update magnitude come out almost exactly equal to the learning rate, no
matter how big the gradient is. (`test_adamw_first_step_is_lr_sized` checks
precisely this.)

### The "W": weight decay

On top of the gradient step, AdamW also gently pulls every weight a little
toward zero each step:

```
weight = weight - learning_rate * weight_decay * weight
```

This is a mild "don't let any weight grow huge" pressure (a regularizer that
helps avoid overfitting). The **W** stands for the fact that this shrink is
applied **straight to the weight**, kept *separate* from the gradient — it
never gets mixed into `m` and `v`. That separation is the only thing that
distinguishes **AdamW** from plain "Adam with L2 regularization," and it's
the version modern GPTs use.

## Putting it together

The full per-weight update each step (this is exactly the loop in
`adamw_step`):

```
t      = t + 1                              # step counter, for bias correction

m      = beta1*m + (1 - beta1)*g            # update momentum        (g = gradient)
v      = beta2*v + (1 - beta2)*g*g          # update squared-grad average

m_hat  = m / (1 - beta1^t)                  # bias-correct both
v_hat  = v / (1 - beta2^t)

weight = weight - lr * ( m_hat / (sqrt(v_hat) + eps)   # the Adam step
                       + weight_decay * weight )        # the decoupled decay
```

The `eps` (a tiny number like `1e-8`) just guards the division so a weight
whose `v` is near zero doesn't blow up.

### Typical hyperparameters

| Knob           | Typical value | What it does                                    |
|----------------|---------------|-------------------------------------------------|
| `lr`           | 1e-3 … 3e-4   | base step size — the most important knob        |
| `beta1`        | 0.9           | momentum smoothing (higher = smoother/slower)   |
| `beta2`        | 0.999         | squared-grad smoothing                          |
| `eps`          | 1e-8          | divide-by-zero guard                            |
| `weight_decay` | 0.0 … 0.1     | pull-toward-zero strength (0 disables it)       |

### Where it fits in one training step

```
gpt_zero_grad(model)                 reset last step's gradients to 0
gpt_forward (model, x, y, B, T)      tokens -> loss   (how wrong are we?)
gpt_backward(model, x, y, B, T)      loss  -> grads   (which way is downhill?)
adamw_step  (opt, params, grads)     grads -> weights (take a right-sized step)
```

The optimizer is deliberately ignorant of what a "GPT" is — it only sees two
flat float arrays of equal length: `params` (the numbers to improve) and
`grads` (their gradients). Because `gpt.c` keeps **all** weights in one
contiguous block (see `docs/gpt-model.md`, the "single-block trick"), the
entire model is optimized by a single loop over those two arrays.

## Concept checks

1. **Why divide by `sqrt(v)` instead of just using `m`?** Because `v` tells
   you how big this weight's gradients tend to be. Dividing by it gives every
   weight its own effective step size, so steep and flat directions both move
   a sensible amount. Without it you're back to one fixed step size for
   everything.

2. **Why do `m` and `v` need bias correction?** They start at zero, so the
   first few steps would be far too small. Dividing by `(1 - beta^t)` cancels
   that startup bias, and the correction fades to nothing as `t` grows.

3. **What makes AdamW different from Adam?** The weight decay is applied
   directly to the weight (`weight -= lr*wd*weight`), *separately* from the
   gradient — it never enters `m` or `v`. In plain Adam, an L2 penalty would
   be folded into the gradient and get distorted by the `1/sqrt(v)` scaling.
   `test_adamw_weight_decay_shrinks` isolates this: with a zero gradient, a
   weight shrinks by exactly the factor `(1 - lr*weight_decay)`.
