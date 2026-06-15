# The GPT Model, Assembled

How `src/model/gpt.c` wires the small layer functions from
`src/model/layers.c` into one complete transformer that can run a forward
pass (tokens → loss) and a backward pass (loss → gradients).

This is the step where everything built in Tasks 5 and 6 comes together.
`docs/forward-pass.md` walks a single FFN block by hand; this doc zooms out
to the whole model.

## The shape of the model

```
tokens (B, T)  ── integers, one per position
      │  embed_forward: token vector + position vector
      ▼
   x  (B, T, C)  ── the "residual stream"
      │
      │   ┌─────────────────  repeat n_layer times  ─────────────────┐
      │   │  a    = layernorm(x)                                      │
      │   │  attn = self_attention(a)     ← tokens look at the past   │
      │   │  x    = x + attn_proj(attn)   ← residual (skip) add       │
      │   │  b    = layernorm(x)                                      │
      │   │  ffn  = matmul → gelu → matmul (b)  ← think per-token     │
      │   │  x    = x + ffn               ← residual add              │
      │   └───────────────────────────────────────────────────────────┘
      ▼
   x  (B, T, C)  ── final residual stream
      │  layernorm, then matmul by the output table
      ▼
 logits (B, T, V)  ── one score per vocabulary word, per position
      │  softmax → probabilities, then cross-entropy against the targets
      ▼
  loss  ── a single number: "how surprised was the model by the real next token?"
```

`B` = batch size, `T` = sequence length, `C` = embedding width,
`V` = vocab size. The seminar model uses `C=128, V=512, n_layer=2,
n_head=4` (~534K parameters at our settings).

## Two design choices (kept simple on purpose)

1. **Separate output table.** The matrix that turns the final vector back
   into per-word scores (`lm_head_w`) is its own set of weights, *not* the
   token-embedding table reused. Every weight then has exactly one source of
   gradient — easier to reason about.

2. **Bias-free linear layers.** Every matmul is just `out = x @ W`, with no
   `+ bias`. Fewer weights, fewer gradient paths. (Layer-norm keeps its
   learned `gamma`/`beta` scale-and-shift — those aren't "biases" in this
   sense.)

Both are valid real-GPT variants; we picked the variants that are simplest
to read and debug while learning.

## The single-block memory trick

All trainable weights live in **one** contiguous array, `params`. A second
array, `grads`, has the same size and layout and holds each weight's
gradient. `ParameterTensors` is just a bundle of pointers *into* those blocks
(`model->w.qkv_w`, etc.) so the math code never does raw offset arithmetic.

Why bother? Because the next tasks become one-liners:

- **AdamW (Task 8)** updates every weight with a single loop over
  `params`/`grads`.
- **MPI sync (Task 11)** averages everyone's gradients with a single
  AllReduce over `grads`.

The intermediate tensors of the forward pass (`acts`) and their gradients
(`grads_acts`) use the same one-block trick. They depend on `B` and `T`, so
they're allocated lazily on the first `gpt_forward` and reused after that.

## Self-attention (the one new piece)

Attention is the only math not already in `layers.c`, because it's a
*composition* of the basic ops, specific to a GPT. For each head and each
query position `t`:

1. `score[t2] = (Q_t · K_t2) / sqrt(head_dim)` for every earlier position
   `t2 ≤ t`. The `1/sqrt(head_dim)` keeps the dot products from growing with
   width. The `t2 ≤ t` restriction is the **causal mask**: a token may look
   at itself and the past, never the future (that would be cheating at
   next-token prediction).
2. `att = softmax(score)` over those allowed positions.
3. `out_t = Σ att[t2] · V_t2` — a weighted average of the Value vectors.

The `C` channels are split into `n_head` independent heads of width
`C / n_head`; each head runs the three steps above on its own slice.

## Why the backward pass uses `+=` everywhere

The residual stream `x` is read **twice** every layer: once through the
layer-norm ("through" path) and once through the skip connection ("skip"
path, the `x + ...`). So its gradient must be the **sum** of what comes back
along both paths. That's exactly why every `_backward` in `layers.c`
accumulates with `+=` instead of overwriting — `gpt_backward` zeroes the
gradient buffers once, then lets contributions pile up correctly.

`gpt_backward` simply walks the forward diagram in reverse, calling the
matching `_backward` op for each arrow.

## How we know it's correct

The headline test is `tests/test_gpt.c::test_gpt_numerical_gradient_*`. It
checks the **entire** backward pass at once by comparing every analytical
gradient against a finite-difference estimate:

```
numerical gradient ≈ ( loss(w + h) − loss(w − h) ) / 2h
```

If any single layer's backward is wrong — a missing term, a sign flip, a
forgotten `1/sqrt(head_dim)` — the analytical and numerical gradient vectors
diverge and the test fails. A correct implementation agrees to a relative
error around `1e-3`.

One subtlety worth knowing: the test first **amplifies** the random weights
(`amplify_weights`) before checking. At the realistic training scale
(std 0.02) the attention scores are nearly zero, so softmax is almost
uniform and the gradient check literally can't feel an attention bug.
Pushing the weights to std ~0.5 makes attention non-uniform, so every term
actually influences the loss — that's the regime where the check has teeth.

## Concept checks

1. **Why a residual stream instead of just stacking layers?** Because
   `x = x + block(x)` means each block computes a small *correction*, and
   gradients flow straight back through the `+`. Without it, gradients shrink
   or explode layer by layer and deep networks won't train.

2. **Why layer-norm *before* each block (not after)?** Pre-norm keeps the
   residual stream itself un-normalized (so the skip path is a clean
   identity), while still feeding each block well-scaled inputs. It's the
   more stable arrangement and what modern GPTs use.

3. **Why is the loss always positive?** Cross-entropy is `-log(probability)`,
   and a probability is at most 1, so its negative log is at least 0 — and
   strictly positive unless the model is perfectly certain, which a freshly
   initialized one never is.
