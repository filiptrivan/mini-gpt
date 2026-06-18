# Backward Pass, Walked by Hand

A worked explanation of how `src/model/layers.c`'s six `_backward`
functions compute gradients — the numbers that tell the optimizer how
to nudge each weight to make the loss go down. This is the companion
document to `forward-pass.md`; read that one first if you haven't, and
keep its toy example open in another window because we'll trace
gradients back through it here.

## What "backward" means and why we need it

A neural network learns by **adjusting its weights to reduce a loss**.
For each weight `w`, the optimizer needs one number:

> `dL/dw` — how much the loss `L` changes if I nudge `w` up by a tiny
> amount.

If `dL/dw = +5`, nudging `w` up makes `L` go up (bad), so the optimizer
nudges `w` *down*. If `dL/dw = -2`, nudging `w` up makes `L` go down
(good), so the optimizer nudges `w` *up*. The simplest update rule —
**gradient descent** — is one line:

```
w_new = w_old  −  learning_rate × dL/dw
```

That minus sign is the entire mechanism. Run it for thousands of steps
and the loss drifts downward.

The hard part: our model has ~534,000 weights, all stacked through many
layers. We can't compute `dL/dw` for each by hand. The **backward pass**
is the algorithm that computes all of them at once, layer by layer,
walking from the output back to the input — hence "back-propagation."

## A 60-second refresher on derivatives

A **derivative** answers one question:

> "If I change `x` by a tiny amount, how much does `f(x)` change?"

Take `f(x) = x²` at `x = 3`. Then `f(3) = 9`. If we bump `x` to `3.001`:

```
f(3.001)  =  (3.001)²  ≈  9.006
```

A change of `0.001` in `x` caused a change of `0.006` in `f`. The ratio
is `0.006 / 0.001 = 6` — and that ratio (in the limit, as the bump
goes to zero) is the **derivative**:

```
df/dx = 2x       at x = 3,  df/dx = 6 ✓
```

Visualised, the derivative is the **slope** of the function at that
point. Steep slope → small change in `x` causes a big change in `f`.
Flat slope → `f` barely moves.

For our network, the "function" is the loss `L`, and "x" is each
weight `w`. The derivative `dL/dw` tells us "if I bump this weight a
tiny bit, will the loss go up (bad) or down (good), and by how much?"
That single number per weight is everything the optimizer needs.

## The only math trick: the chain rule

The chain rule answers a follow-up question: **what if the bump has
to travel through two functions before it reaches the output?**

Let:

```
y  =  2·x       (y is double x)
z  =  y²        (z is y squared)
```

Sit at `x = 3`, so `y = 6` and `z = 36`. Now bump `x` by `0.001`:

| variable | value before | value after | change | ratio to x's change |
|----------|--------------|-------------|--------|---------------------|
| x        | 3            | 3.001       | 0.001  | 1×                  |
| y        | 6            | 6.002       | 0.002  | **2×**              |
| z        | 36           | 36.024      | 0.024  | **24×**             |

Two things to notice:

- `y` changed twice as much as `x`. That's `dy/dx = 2`.
- `z` changed 12 times as much as `y`. That's `dz/dy = 2y = 12`.
- `z` changed 24 times as much as `x`. And `24 = 12 × 2`.

**The sensitivities multiplied.** That's the chain rule:

```
dz/dx  =  (dz/dy)  ×  (dy/dx)
  24   =    12     ×     2
```

In words: if doubling `x` doubles `y`, and doubling `y` triples `z`,
then doubling `x` triples `z`. Effects chain by multiplication.

You can verify by expanding directly: `z = (2x)² = 4x²`, so
`dz/dx = 8x`, which at `x = 3` is `24`. Same answer. The chain-rule
version is the one that scales — because nobody ever needs to know `z`
as a single formula in `x`. Each link in the chain only needs to know
**its own local derivative**.

## One layer, in plain English: what `d_out` and `d_in` actually are

Before drawing the multi-layer picture, do it for ONE layer with real
numbers — that's where the confusion usually clears up. Pick `gelu`.
Forget the rest of the network for a second.

The names first, in their full English meaning (the abbreviations are
short for these phrases — that's all):

- `d_out` means: **"if I bump the layer's OUTPUT up by a tiny amount,
  how much does the loss change?"**
- `d_in` means: **"if I bump the layer's INPUT up by a tiny amount,
  how much does the loss change?"**

Both are just numbers. Both answer "how sensitive is the loss to a
nudge at this spot?" — they just nudge at different spots.

### Real numbers

Set `x = 1`. Then `gelu(1) ≈ 0.84`.

Suppose someone has already figured out (from whatever lives downstream
of gelu — more layers, then the loss):

> "If you bump gelu's output (the 0.84) up by `0.001`, the loss goes up
> by `0.002`."

So the sensitivity AT THE OUTPUT is `0.002 / 0.001 = 2`. **That number,
`2`, is `d_out`.** That's literally all it is.

Now gelu's job is to compute the sensitivity at its INPUT — `d_in`. In
other words: if we bump `x` (the input) by `0.001`, what does the loss
do?

Two things happen when we bump `x`:

1. The output `gelu(x)` changes. By how much? By `0.001 × gelu's slope
   at x=1`. The slope `gelu'(1) ≈ 1.08`. So the output changes by
   `0.001 × 1.08 = 0.00108`.
2. That output change then ripples to the loss. We already know:
   output bumps of `0.001` cause loss bumps of `0.002` (sensitivity 2).
   So an output bump of `0.00108` causes a loss bump of
   `0.00108 × 2 = 0.00216`.

Combine: a `0.001` bump in `x` causes a `0.00216` bump in `L`.
Sensitivity at the input is `0.00216 / 0.001 = 2.16`. **That `2.16` is
`d_in`.**

And look at the relationship:

```
d_in   =   d_out   ×   gelu'(x)
2.16   =     2     ×     1.08
```

That's the chain rule, applied to a single layer. **That single line is
literally the entire body of `gelu_backward`.**

### Plain-English recap

- Someone tells me "the loss is `d_out`-sensitive to my output."
- I figure out "then the loss is `d_in`-sensitive to my input."
- The way I figure it out: multiply `d_out` by my own slope.

That's it. That's all any backward function in `layers.c` does. The
formulas get fancier when the input/output are vectors instead of single
numbers, but the structure is always this same "multiply by my own
local slope."

## Applied to a multi-layer network

Now stretch the single-layer picture out. Our forward pass is a chain:

```
x  →  layer1  →  y  →  layer2  →  z  →  loss  →  L
```

We want `dL/dx`. By the chain rule, applied layer-by-layer just like
the gelu example above:

```
dL/dx  =  (dL/dz) × (dz/dy) × (dy/dx)
            ↑          ↑          ↑
       gradient    layer2's    layer1's
       from loss   own job     own job
```

Each layer's backward function does **the same one thing** the gelu
example did: take the gradient at MY output (`d_out`), multiply by my
own local slope, write the gradient at MY input (`d_in`). The forward
pass walks left-to-right; the backward pass walks right-to-left,
accumulating the chain-rule product one layer at a time.

A piece of vocabulary to nail down: the gradient sitting at the
boundary between two layers has **two names** — one from each layer's
point of view. If `y` is the boundary between layer1 and layer2, then
the same number is:

- `d_out` from layer1's point of view (it sits on layer1's output side)
- `d_in` from layer2's point of view (it sits on layer2's input side)

That's why backward functions can be chained: each layer's `d_in`
becomes the next layer's `d_out` for free, no translation needed.

This is why every backward function in `layers.c` has the same shape:

```c
op_backward(d_in, ..., d_out, ...);
//          ↑              ↑
//   what we write    what we receive
```

`d_out` is the gradient from downstream (the rest of the network on
the loss's side). `d_in` is the gradient we hand upstream (the rest
of the network on the input's side). Whatever's between is the local
derivative of THIS layer.

## The "contraction with d_out" trick

Most of our layers output a vector (or a 2-D tensor), not a single
number. So strictly speaking the "derivative" of `out` with respect to
`in` is a whole **Jacobian matrix** — one partial derivative per
(output element, input element) pair. For a softmax over 512 vocab
tokens, that Jacobian has 512² ≈ 260,000 entries per row. We never
want to materialize that.

Trick: we don't need the full Jacobian. We only need its **product**
with `d_out`:

```
d_in[i]  =  sum_j  d_out[j] × ∂out[j]/∂in[i]
```

For most layers, that sum collapses to a simple closed-form that costs
O(N), not O(N²). The whole backward pass is built on knowing the
closed form for each layer. We never compute Jacobians, only their
contractions with `d_out` — which is exactly what the chain rule
needs anyway.

## Backward for each layer

Each subsection is one of our six layers. Forward formula on top,
derivative on the bottom, intuition in plain English. Numerical
walked-through example for the simple ones; just the formula for the
hairier ones.

### residual_backward

**Forward**: `out = a + b` (element-wise).

**Backward**: gradient flows unchanged into both inputs.

```
d_a[i] += d_out[i]
d_b[i] += d_out[i]
```

**Why**: differentiating `a + b` with respect to `a` (or `b`) gives 1.
The chain rule multiplies `d_out` by 1 — so nothing changes; we just
copy `d_out` into both `d_a` and `d_b`. This is the math fact that
makes skip connections so important during training: gradients flow
through them undiluted, layer after layer.

### gelu_backward

**Forward**: `out = 0.5 · x · (1 + tanh(s))` where
`s = √(2/π) · (x + 0.044715·x³)`.

**Backward**: element-wise multiplication by the derivative of GELU.

```
gelu'(x) =  0.5 · (1 + tanh(s))
         +  0.5 · x · (1 − tanh²(s)) · s'(x)

s'(x)    =  √(2/π) · (1 + 3·0.044715·x²)

d_in[i] += d_out[i] × gelu'(in[i])
```

**Why**: two pieces because of the product rule on `0.5·x · (1+tanh(s))`.
The first piece is "leave the `tanh` alone, differentiate the `x`";
the second is "leave the `x` alone, differentiate the `tanh`," and that
inner differentiation chains through `tanh` (giving `1−tanh²`, the
sech² identity) and through the polynomial `s(x)`.

**Forward cache needed**: the original input `in`. The derivative
depends on `x`, not on `gelu(x)`. Every layer that needs its forward
input cached has the same reason: the local derivative is a function
of that input.

### matmul_backward

**Forward**: `out = a @ b`, shapes `(M×K) @ (K×N) = (M×N)`.

**Backward**: two matmuls, one for each input.

```
d_a += d_out  @  bᵀ      (shape M×K)
d_b += aᵀ     @  d_out   (shape K×N)
```

**Why**: differentiate `out[i,j] = sum_k a[i,k] · b[k,j]` with respect
to `a[i,k]` — the only term that survives is `b[k,j]`. Sum that across
all output positions weighted by `d_out`, and you get the formula
above. The symmetry for `d_b` is the same argument with `a` and `b`
swapped.

**Mnemonic**: forward has a "shared" dimension `K`. Each backward has
a *different* shared dimension (`j` for `d_a`, `i` for `d_b`). You
always sum over the dim that disappears between input and output.

**Forward cache needed**: both `a` and `b`. The gradient w.r.t. one
factor uses the OTHER factor — same idea as the derivative of a
product `f·g`: `f'·g + f·g'`.

### softmax_backward

**Forward**: `y = softmax(x)`, row-wise.

**Backward** (per row): the Jacobian of softmax is dense (every output
depends on every input), but its product with `d_out` collapses to a
clean closed form.

```
s = sum_j  d_out[j] · y[j]      (a single scalar, per row)
d_in[v] += y[v] · (d_out[v] − s)
```

**Why**: the softmax Jacobian is
`∂y[j]/∂x[i] = y[i] · (δ_ij − y[j])`. Plug into the chain rule, factor
`y[i]` out of the inner sum, and the `s` term is what's left. The
collapse turns O(V²) work per row into O(V) — the trick that makes
softmax backward practical.

**Forward cache needed**: the forward OUTPUT `y` (not the input `x`).
Once you have `y`, the input is no longer needed. Caching one vector
instead of two saves memory and matches what every framework does.

### layernorm_backward

**Forward** (per row): center to mean 0, scale to variance 1, then
apply learned affine.

```
μ      = mean(in)
r      = 1 / √(var(in) + ε)
x_hat  = (in − μ) · r
out    = γ · x_hat + β
```

**Backward** (per row): the closed-form gradient through this whole
recipe.

```
d_γ[c]  += d_out[c] · x_hat[c]                    (sum across rows)
d_β[c]  += d_out[c]                                (sum across rows)
d_x_hat = d_out · γ                                (per-element)

sum1    = (1/C) · sum_j d_x_hat[j]                 (scalar, per row)
sum2    = (1/C) · sum_j d_x_hat[j] · x_hat[j]      (scalar, per row)

d_in[c] += r · ( d_x_hat[c] − sum1 − x_hat[c]·sum2 )
```

**Why** the last line is the way it is, intuitively:
- The `d_x_hat[c]` term is the "direct" gradient — what you'd write if
  μ and r didn't depend on `in`.
- The `sum1` subtraction handles the fact that shifting `in` by any
  constant doesn't change `out` (because μ shifts with it). The
  gradient in the "uniform shift" direction must be zero, which is
  what subtracting the row-mean of `d_x_hat` achieves.
- The `sum2` projection handles the fact that scaling `in` uniformly
  doesn't change `out` either (because `r` rescales it back). The
  gradient in the "uniform scale" direction must also be zero.

**Forward cache needed**: the input `in` (so we can recompute μ and
`r`). We chose to recompute rather than cache for simplicity — O(C)
extra work per row, negligible.

### embed_backward

**Forward**: gather. `out[b, t, c] = wte[tokens[b,t], c] + wpe[t, c]`.

**Backward**: scatter. The dual of gather.

```
d_wte[ tokens[b,t], c ] += d_out[b, t, c]
d_wpe[ t,           c ] += d_out[b, t, c]
```

**Why**: there is no gradient with respect to `tokens` (they're
integer indices — you can't differentiate "which row did I pick").
The gradient flows entirely into the embedding TABLES, which are the
trainable parameters.

**The accumulation matters here more than anywhere else**: if the same
token id appears at multiple `(b, t)` positions in a batch (very
common in real text), each appearance contributes to the same row of
`d_wte`. With `=` (overwrite) only the last contribution survives —
a silent bug that's hard to spot because the resulting gradient is
still "shaped right," just systematically smaller. With `+=`, every
contribution is preserved.

**Forward cache needed**: `tokens` (the indices) and the dimensions.
We don't need the forward `out` or the embedding tables themselves —
the gradient is a pure scatter.

## The `+=` rule (the one extra thing to know)

There's a tiny extra rule in the backward pass. It's simple, but it
matters.

### The setup: one value used in two places

Imagine `x = 3` and we compute `y = x + x`. Then `y = 6`, and obviously
`dy/dx = 2` — bump `x` by 1, `y` bumps by 2 because `x` shows up twice.

But our `residual_forward` treats this as `y = a + b`, where `a = x`
and `b = x` are *separately named inputs* even though they hold the
same value. When the messenger walks backward:

- The first input `a` gets some gradient contribution.
- The second input `b` gets some gradient contribution.
- **Both contributions belong to the same `x`**, because `a` and `b`
  are both `x`.

If we wrote `dx = (contribution from a)`, we'd miss the contribution
from `b`. They have to be **added**:

```
dx  =  (contribution from a)  +  (contribution from b)
```

That's the entire rule.

### How this looks in code

Every `_backward` function in `layers.c` writes its gradients with
`+=` instead of `=`:

```c
d_in[i]  +=  d_out[i] * (this layer's slope);   // ADD, don't overwrite
```

The function doesn't know in advance whether something earlier wrote
into `d_in[i]` from a different path. It just adds its contribution
to whatever's already there. `=` would erase earlier contributions.

### Where else this matters in our network

1. **Residual.** One tensor feeds two paths (skip path + sublayer).
   Both paths produce a gradient that lands on the same input. They
   add.

2. **Shared weights (matmul).** A weight matrix is reused for every
   batch element and every time step. Each use makes its own gradient
   contribution to the same matrix. They all add.

3. **Repeated embeddings.** The same token id (say `42`) often appears
   at many positions in a sentence. Each appearance contributes a
   gradient to the **same row** of the embedding table. They add.

With `+=` baked into every backward function, the caller just calls
them one after another and everything sums correctly.

### What the caller has to do

Because every backward adds, the caller has to **zero the gradient
buffers once at the start of a training step**. Otherwise gradients
from this step pile onto gradients from the previous step — wrong.

Every backward test in `tests/test_layers.c` starts with exactly that:

```c
memset(d_in_ana, 0, sizeof(d_in_ana));   // start clean
backward_function(d_in_ana, ...);        // then accumulate
```

### Summary of the `+=` rule

- One value can be used in multiple places.
- Each use produces its own gradient contribution.
- All contributions to the same value must **add up**.
- That's why backward writes with `+=` instead of `=`.
- The caller zeroes the buffers once at the start.

## How we KNOW the backward is correct: numerical gradient check

A wrong backward is the worst kind of bug because the model still
"trains" — it just trains the wrong direction, very slowly, and you
can't tell from the loss curve. So every backward function has to be
verified mechanically. The technique is the same one every framework
uses.

**The idea**: pick any scalar loss `L(x)`. By definition of the
derivative,

```
dL/dx[i]  ≈  ( L(x + h·e_i)  −  L(x − h·e_i) )  /  (2h)
```

— a "central difference," for small `h` like 10⁻³. This is the
*numerical* gradient. It's slow (one forward pass per parameter per
direction) but it's directly derived from the forward function we
already trust, so it's "ground truth."

If the analytical backward and the numerical gradient agree to a few
thousandths, the backward is correct. If they disagree, the backward
has a bug.

**The L we use in tests**: the "contraction with `d_out`":

```
L(x)  =  sum_k  d_out[k] · forward(x)[k]
```

The chain rule says `dL/dx = (forward's Jacobian)ᵀ @ d_out` — which is
*exactly* what every backward function computes. So checking that
`analytical_backward(d_out) ≈ numerical_grad(L)` is the right test.

The helper that does the central-difference sweep is
`numerical_gradient()` in `tests/helpers/test_utils.c`. Every backward
test in `tests/test_layers.c` follows the same pattern:

1. Build a tiny input.
2. Run analytical backward → analytical gradient.
3. Build the loss `L(x) = sum(d_out · forward(x))`.
4. Run `numerical_gradient(L)` → numerical gradient.
5. Compare element-wise. Tolerance: 1e-3.

A 1e-3 difference is the float32 floor — the central difference has
~1e-4 round-off error baked in. Tighter tolerances would fail on the
math we trust; looser would let real bugs through.

## Concept checks

Three questions to sanity-check understanding:

1. **Why do we walk backward instead of forward to compute gradients?**
   We could compute `dL/dw` for one weight `w` by perturbing `w`,
   re-running the entire forward, and measuring the change. That's
   one forward pass per weight — 534,000 forward passes per training
   step. Walking backward computes all gradients in *one* backward
   pass, by reusing the chain-rule product as we go. That's the whole
   reason "back-propagation" exists.

2. **Why does each backward need its forward inputs (or outputs)?**
   Because the local derivative at a layer depends on what the layer
   saw. `gelu'(x)` depends on `x`. `matmul`'s gradient w.r.t. `a`
   depends on `b`. Without the forward cache, we'd have to re-run
   forward to get them — defeating the whole point of caching them
   during the forward pass we already ran.

3. **Why does the numerical gradient check use a fake `d_out`?**
   Because backward functions compute `(Jacobian)ᵀ @ d_out`, not the
   Jacobian itself. To verify them we need a loss whose gradient is
   exactly that quantity. The contraction `L = sum(d_out · forward(x))`
   has gradient `dL/dx = (Jacobian)ᵀ @ d_out` by the chain rule —
   which matches what backward computes, so the test is a clean
   apples-to-apples comparison.
