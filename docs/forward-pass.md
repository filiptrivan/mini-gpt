# Forward Pass, Walked by Hand

A worked example of how the six layer functions in `src/model/layers.c` turn
an input token ID into a probability distribution over the next token.
Useful as a reference for seeing how the small math pieces fit together
into a single forward step.

## What we're showing

GPT's forward pass alternates two kinds of sub-block: **attention** (mixes
information between tokens) and **FFN** (transforms each token
independently). Attention is Task 7 — not yet built — so this walkthrough
focuses on a single **FFN block** with the embedding step before it and
the output projection after it.

The pipeline:

```
token_id ──► embed ──► layernorm ──► matmul ──► gelu ──► matmul ──► residual ──► matmul ──► softmax ──► probs
                              \__________________ FFN block __________________/    └─── output head ───┘
```

## Setup: a toy 4-token model

The smallest model that still exercises every layer:

| Parameter      | Value | Meaning |
|----------------|-------|---------|
| `vocab_size`   | 4 | only 4 possible tokens (IDs 0..3) |
| `C` (embed dim)| 2 | each token is a 2-d vector |
| `ff_dim`       | 4 | FFN's hidden width (typically 4×C) |
| `B`, `T`       | 1, 1 | single token, single batch (focus on the math) |

**Learned weights** (these would normally come from training; here we
pick them by hand):

```
wte (4 × 2)        wpe (1 × 2)
[ -1.0  -1.0 ]     [ 0.5  0.5 ]   ← position 0
[  0.0   0.0 ]
[  0.5   1.5 ]   ← what we'll look up
[  2.0  -1.0 ]

γ_pre, β_pre   = [1, 1], [0, 0]    (identity affine for layernorm)

W_up (2 × 4)             W_down (4 × 2)        W_out (2 × 4)
[ 1  0 -1  2 ]           [ 1  0 ]              [ 1  0  0.5 -1 ]
[ 0  1  1  1 ]           [ 0  1 ]              [ 0  1  0.5  1 ]
                         [ 1  1 ]
                         [ 0  0 ]
```

**Input:** `tokens = [2]` (a single token with ID 2, at position 0).

## Step 1: embed — token ID becomes a vector

`embed_forward` looks up two rows and adds them:

- `wte[2]` = `[0.5, 1.5]` (this is what "token 2 means" to the model)
- `wpe[0]` = `[0.5, 0.5]` (this is "I'm at position 0")
- `x0 = wte[2] + wpe[0] = [1.0, 2.0]`

This is the first time integer input meets float math. From now on,
everything is floats.

## Step 2: layernorm — center and rescale

`layernorm_forward(x0, γ, β)` runs three passes:

- Pass 1: mean = `(1.0 + 2.0) / 2 = 1.5`
- Pass 2: variance = `((1.0 − 1.5)² + (2.0 − 1.5)²) / 2 = 0.25`, then
  `rstd = 1 / √(0.25 + 1e-5) ≈ 2.0`
- Pass 3: x_hat = `[(1.0 − 1.5)·2.0, (2.0 − 1.5)·2.0] = [−1.0, 1.0]`,
  then `out = γ·x_hat + β = [−1.0, 1.0]`

So `x1 = [−1.0, 1.0]`. Mean dropped to 0, variance went up to 1 —
exactly the layernorm guarantee.

## Step 3: matmul up — project into a wider hidden space

`matmul_forward(x1, W_up)` with shapes `(1 × 2) @ (2 × 4) = (1 × 4)`:

```
x1 = [-1,  1]

         [ 1  0 -1  2 ]
W_up  =  [ 0  1  1  1 ]

x2[j] = -1·W_up[0,j] + 1·W_up[1,j]
x2[0] = -1·1 + 1·0 = -1
x2[1] = -1·0 + 1·1 =  1
x2[2] = -1·(-1) + 1·1 =  2
x2[3] = -1·2 + 1·1 = -1
```

`x2 = [−1, 1, 2, −1]` — now 4-dimensional. The FFN's job is to do its
real "thinking" in this wider space.

## Step 4: gelu — apply non-linearity

`gelu_forward(x2)` applies the GPT-2 tanh approximation element-by-element:

| input | output (≈) |
|-------|-------|
| −1 | −0.159 |
|  1 |  0.841 |
|  2 |  1.955 |
| −1 | −0.159 |

`x3 ≈ [−0.159, 0.841, 1.955, −0.159]`. Without this step the whole FFN
would just be `x @ W_up @ W_down` — one matrix multiply in disguise.
GELU is what makes the network actually non-linear.

## Step 5: matmul down — project back to the original width

`matmul_forward(x3, W_down)` with shapes `(1 × 4) @ (4 × 2) = (1 × 2)`:

```
                  [ 1  0 ]
                  [ 0  1 ]
W_down (4×2)  =   [ 1  1 ]
                  [ 0  0 ]

x4[0] = -0.159·1 + 0.841·0 + 1.955·1 + (-0.159)·0 =  1.796
x4[1] = -0.159·0 + 0.841·1 + 1.955·1 + (-0.159)·0 =  2.796
```

`x4 ≈ [1.796, 2.796]`. We're back to 2-d — same shape as `x0`, which is
the requirement for the next step.

## Step 6: residual — add the original input back

`residual_forward(x0, x4)`:

```
x0 = [1.0,   2.0  ]
x4 = [1.796, 2.796]
x5 = [2.796, 4.796]
```

This is the **skip connection** — the FFN computed a "correction" (`x4`)
to be added to the original (`x0`), rather than replacing it. This is
the trick that lets gradients flow cleanly through dozens of layers
during training.

## Step 7: output projection — back to vocabulary scores

In a real GPT we'd loop through more transformer blocks before this.
Here we go straight to the output head: `matmul_forward(x5, W_out)`
with shapes `(1 × 2) @ (2 × 4) = (1 × 4)`:

```
x5 = [2.796, 4.796]

                  [ 1  0  0.5 -1 ]
W_out (2×4)  =    [ 0  1  0.5  1 ]

logits[0] = 2.796·1 + 4.796·0     =  2.796
logits[1] = 2.796·0 + 4.796·1     =  4.796
logits[2] = 2.796·0.5 + 4.796·0.5 =  3.796
logits[3] = 2.796·(-1) + 4.796·1  =  2.000
```

These are **logits** — one score per vocabulary token. They're not
probabilities yet (some are above 1, the sum is 13+).

## Step 8: softmax — logits become probabilities

`softmax_forward(logits, N=1, V=4)`:

- max = 4.796
- shifted = `[−2.0, 0, −1.0, −2.796]`
- exp of shifted ≈ `[0.135, 1.000, 0.368, 0.061]`, sum ≈ 1.564
- divide by sum ≈ `[0.087, 0.639, 0.235, 0.039]`

The probability distribution over what the next token should be:

| Next token ID | Probability |
|---|---|
| 0 |  8.7% |
| 1 | **63.9%** |
| 2 | 23.5% |
| 3 |  3.9% |

Given that we input token 2 at position 0, the model thinks the next
token is most likely token 1. (With untrained random weights it's not
meaningful — but the *mechanism* is exactly what a trained model uses.)

## The whole picture

```
  token_id = 2
        |
        v
  +-----------+   wte[2] = [0.5, 1.5]
  |   embed   |   wpe[0] = [0.5, 0.5]
  +-----+-----+
        |  x0 = [1.0, 2.0]
        +--------------------------+   (saved for residual)
        |                          |
        v                          |
  +-----------+                    |
  | layernorm |  → [-1.0, 1.0]     |
  +-----+-----+                    |
        |                          |
        v                          |
  +-----------+                    |
  | matmul ↑  |  → [-1, 1, 2, -1]  |   (project to 4-d)
  +-----+-----+                    |
        |                          |
        v                          |
  +-----------+                    |
  |   gelu    |  → [-0.16, 0.84,   |
  +-----+-----+      1.95, -0.16]  |
        |                          |
        v                          |
  +-----------+                    |
  | matmul ↓  |  → [1.80, 2.80]    |   (project back to 2-d)
  +-----+-----+                    |
        |                          |
        v                          |
  +-----------+                    |
  |  residual | <------------------+
  +-----+-----+
        |  x5 = [2.80, 4.80]
        v
  +-----------+
  | matmul →V |  → [2.80, 4.80, 3.80, 2.00]   (logits over vocab)
  +-----+-----+
        |
        v
  +-----------+
  |  softmax  |  → [0.09, 0.64, 0.24, 0.04]   (probabilities)
  +-----+-----+
        |
        v
   next-token
   probabilities
```

## What changes in a real GPT

For our seminar model (C=128, num_layers=2, vocab=512) the same diagram
applies, but:

- The FFN sub-block repeats inside each transformer layer, **after** an
  attention sub-block.
- Each transformer layer is
  `layernorm → attention → residual → layernorm → FFN → residual`.
- Multiple layers stack on top of each other.
- B and T are real batches and sequences, not 1×1 — but every operation
  we wrote handles them by being row-wise. Stacking more rows = same
  code path.

The total parameter count grows fast, but the *operations* don't change
at all — just bigger weight matrices and longer loops. Everything built
in Task 5 is also what runs inside GPT-3 at scale.

## Concept checks

Three questions to sanity-check understanding:

1. **Why `layernorm → matmul → gelu → matmul → residual` in that order?**
   Layernorm puts every input on a stable scale so the matmuls don't see
   wildly different magnitudes between examples. GELU sits *between* the
   two matmuls because two back-to-back linear ops are mathematically
   equivalent to one — the non-linearity in the middle is what gives the
   FFN expressive power. The residual at the end means the block
   computes a *correction*, not a replacement.

2. **Why does the FFN project up to a wider dim and then back down?**
   The intermediate hidden space (here 4-d, normally 4×C) is where the
   actual computation happens. Going up gives the layer "room to think"
   — more channels for richer combinations of the input. Going back down
   compresses the result to the same shape as the input so the residual
   add is shape-compatible.

3. **Why apply softmax only at the very end, never inside the network?**
   Softmax forces outputs into probabilities (sum to 1). That's useful
   when you want to *interpret* the output, but it loses a lot of
   information (relative magnitudes get squished). Inside the network
   we want to preserve full numerical detail; softmax appears only when
   we need a probability — inside attention (for weighting tokens) and
   at the output (for sampling the next token).
