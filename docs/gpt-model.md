# The GPT Model, Assembled

How `src/model/gpt.c` wires the small layer functions from
`src/model/layers.c` into one complete transformer that can run a forward
pass (tokens → loss) and a backward pass (loss → gradients).

This is the step where everything built in Tasks 5 and 6 comes together.
`docs/forward-pass.md` walks a single FFN block by hand; this doc zooms out
to the whole model.

## First, in plain words

Three ideas you need before the diagram makes sense: what `x` is, the
"residual stream," and how attention differs from feed-forward.

### What is `x`? Just a table of numbers.

A token is a word (or word-piece) written as an ID number — like `"je"` →
token `42`. But a single number like `42` is useless to the model, so
**every token is represented as a list of numbers** instead. In the real
model that list is `C = 128` numbers long; here we use 4 to keep it readable:

```
"je"  →  [ 0.9,  0.1, -0.2,  0.6 ]
```

Take a 3-word sentence. Each word becomes its own list of numbers; stack them
and you get a **table** — and that whole table *is* `x`:

```
                    n1     n2     n3     n4
  "elektronsko"  [  0.2,  -0.5,   1.1,   0.0 ]
  "poslovanje"   [ -0.3,   0.8,   0.4,  -0.1 ]
  "je"           [  0.9,   0.1,  -0.2,   0.6 ]
```

- Each **row** = one token's "meaning so far," written as numbers.
- The number of **columns** is `C` (the embedding width).
- The number of **rows** is `T` (how many tokens are in the sequence).
- (The model usually runs several sentences at once — a *batch* of `B` of
  them — so `x` is really a stack of `B` such tables. Ignore that at first.)

So whenever you see `x` (called `resid` in the code), picture **that table**.

### The residual stream

Picture a **highway** running straight through the model, from the
embeddings to the very end. The table `x` is what drives down it. The key
rule: **each layer doesn't replace `x` — it adds a small correction to it.**

```
x  ──►(+)──────►(+)──────►(+)──►  ... ──►  output
       ▲          ▲          ▲
       │          │          │
   attention   feed-fwd   attention   ← each block computes a "correction"
```

In code that is literally:

```
x = x + attention(x)      ← add what attention figured out
x = x + feedforward(x)    ← add what the feed-forward figured out
```

That `x = x + something` is a **residual connection** (or *skip connection*).
Think of editing an essay: you keep the draft (`x`) and add small edits each
pass, instead of rewriting from scratch. It matters for two reasons:

1. **It trains.** During the backward pass, gradients flow straight back
   through the `+` untouched. Without this path they shrink or explode across
   many layers and deep networks won't train. This is *the* trick that makes
   deep transformers work.
2. **Each layer has an easy job** — compute a small adjustment, not rebuild
   everything.

### Attention vs. feed-forward: two different jobs

Every layer has two blocks, and they do opposite kinds of work on the table:

- **Attention mixes ACROSS rows (tokens look at each other).** A token reads
  the earlier tokens and pulls in relevant context. This is the only place
  information moves *between* positions.

  ```
  [ elektronsko ] ─┐
  [ poslovanje  ] ─┼─►  "je" gathers context from the words before it
  [ je          ] ◄┘
  ```

- **Feed-forward works on EACH row on its own (no looking at neighbors).**
  Every token's vector is processed independently — the same little
  transformation applied row by row, to "think harder" about each token.

  ```
  [ elektronsko ] ─► think ─► [ elektronsko' ]
  [ poslovanje  ] ─► think ─► [ poslovanje'  ]   (rows never see each other)
  [ je          ] ─► think ─► [ je'          ]
  ```

One-line memory hook: **attention = mixing between tokens (horizontal);
feed-forward = thinking per token (vertical).** A transformer layer alternates
the two: gather context, then chew on it.

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

Q, K and V are each a `T × C` table (one vector per token), produced by a
single matmul `qkv = layernorm(x) @ qkv_w` where `qkv_w` is `C × 3C` — the
output's `3C` columns are just Q, K, V glued side by side. The weighted
average in step 3 comes out `T × C` again, gets one more "mixing" matmul
(the output projection `attn_proj_w`), and is then **added back into the
residual stream at the same token's row** (`x = x + attn_proj(out)`).

### Heads: several attentions at once

A *head* is one attention computation. "Multi-head" means we split the `C`
columns into `n_head` equal groups of width `head_dim = C / n_head` and run
attention independently inside each group, then glue the results back
together:

```
  one token's Q vector (C=4, n_head=2):  [  a   b  │  c   d  ]
                                          └ head 0 ┘└ head 1 ┘

  head 0 attends using only columns a,b   →  [ p  q ]  ┐
  head 1 attends using only columns c,d   →  [ r  s ]  ┘
                                   glue →  [ p  q │ r  s ]   (back to C=4)
```

Why split instead of one big attention? So different heads can specialize.
Analogy: reading a sentence with several highlighters at once — one tracks
subject↔verb, another tracks adjective↔noun, another tracks nearby words.
Each head learns to hunt for its own kind of relationship; gluing them
together captures richer structure than a single pass could.

This is why the config requires `n_embd` divisible by `n_head` (the columns
must split evenly — see the assert in `gpt_num_params`). The seminar model
uses `C=128, n_head=4`, so 4 heads of width 32. In `attention_forward` the
`for (h = 0; h < NH; h++)` loop and the `+ h * head_dim` pointer offset are
what pick out each head's slice of columns.

### Why divide the scores by √head_dim

A score is a dot product of two `head_dim`-long vectors, so it's a sum of
`head_dim` products. The more terms you add, the bigger that sum tends to
get — scores grow with the head width. If the scores get large, softmax
turns into a near-hard "pick the single biggest" and its gradient nearly
vanishes (a flat region), so the model can barely learn. Dividing every
score by `√head_dim` cancels that growth and keeps the scores in a sane
range where softmax stays soft and trainable. (It was a missing `1/√head_dim`
that the Task 7 gradient check caught once the weights were amplified.)

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
