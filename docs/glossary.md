# Glossary

A quick reference for every term, variable, and concept used in the
project. Look here when you see an unfamiliar symbol or buzzword and
want a one-line refresher.

Terms are grouped by topic, not alphabet, so related ideas sit next to
each other. Within each table, terms are listed in the order they would
naturally appear in a forward pass of the model.

---

## Tokens and vocabulary

Text gets chopped into small pieces, and each piece is replaced by an
integer. Everything downstream in the model works on those integers.

| Term | What it is | Example |
|------|------------|---------|
| **token** | A small piece of text, decided by the BPE tokenizer | `"hello"`, `" world"`, `"ić"` |
| **token ID** | The integer name of a token. Indexes into `wte`. | `42` |
| **vocab_size** | How many distinct token IDs exist | 4 (toy), **512** (seminar), 50257 (GPT-2) |
| **BPE** | Byte Pair Encoding — the algorithm that learned which substrings deserve their own token | See `docs/bpe-training.md` |
| **encode** | Turn text into a list of token IDs | `"hello"` → `[42, ...]` |
| **decode** | Turn a list of token IDs back into text | `[42, ...]` → `"hello"` |

---

## Shapes and dimensions

Almost every variable in the project has a shape — the list of its
dimensions. These letters appear everywhere.

| Symbol | What it means | Typical value |
|--------|---------------|---------------|
| **B** | batch size — how many independent sequences are processed in parallel | 1–32 |
| **T** | sequence length ("time") — number of tokens per sequence | 1–64 |
| **C** | embed dim — width of one token's vector representation | 2 (toy), **128** (seminar) |
| **V** / **vocab_size** | size of the token vocabulary | **512** (seminar) |
| **ff_dim** | width of the FFN's intermediate "scratch space" (usually 4 × C) | 8 (toy), **512** (seminar) |
| **num_heads** | how many parallel attention "heads" (Task 7) | **4** (seminar) |
| **num_layers** | how many stacked transformer blocks | **2** (seminar) |
| **max_seq_len** | longest sequence the model can handle (sets `wpe`'s row count) | **64** (seminar) |
| **M, K, N** | matmul-specific: `(M × K) @ (K × N) = (M × N)` | varies |
| **N, V** (in softmax) | row count and row width for row-wise softmax | varies |

The seminar model in `PLAN.md`: B varies, T=64, C=128, ff_dim=512, V=512, num_heads=4, num_layers=2 → ~534K parameters (533,760 exactly).

---

## Layout in memory

Arrays are stored flat — these terms describe how multi-dimensional data
fits into a 1-D block of memory.

| Term | What it means |
|------|---------------|
| **row-major** | 2-D matrices are stored row by row in memory. `M[r][c] = flat[r * num_cols + c]` |
| **stride** | How many floats you skip to move one row down. Equal to the row's width. |
| **flat index** | The 1-D position in memory: `row * stride + col`. Used in every layer function. |
| **contiguous** | Memory that is laid out without gaps — what every weight buffer in this project is |
| **batch dimension** | The leading (outer) dimension. Rows are independent; the model treats them in parallel. |

---

## Embeddings

The first thing the model does: turn integer token IDs into float
vectors.

| Term | What it is | Shape |
|------|------------|-------|
| **wte** | **W**ord **T**oken **E**mbedding table — one row per token in the vocabulary | `vocab_size × C` |
| **wpe** | **W**ord **P**osition **E**mbedding table — one row per position in the sequence | `max_seq_len × C` |
| **embedding lookup** | The act of selecting row `wte[token_id]` from the table | implemented in `embed_forward` |
| **embedded vector** | The C-wide float vector that represents one token + its position | `wte[id] + wpe[pos]` |

---

## Layer operations (Task 5 — built)

The 6 forward functions in `src/model/layers.c`.

| Function | What it computes | Shape transform |
|----------|------------------|-----------------|
| **embed_forward** | `out = wte[token] + wpe[pos]` per (b, t) | `(B, T)` → `(B, T, C)` |
| **residual_forward** | `out[i] = a[i] + b[i]` | same |
| **gelu_forward** | smooth non-linearity, GPT-2 tanh approximation | same |
| **matmul_forward** | `out = a @ b`, classic triple loop | `(M, K) @ (K, N) = (M, N)` |
| **softmax_forward** | row → probability distribution (sums to 1) | same |
| **layernorm_forward** | per-row normalize then learned affine | same |

See `docs/forward-pass.md` for a worked example chaining them together.

---

## Weights and parameters

Anything the model **learns** during training (Task 8 onward). All are
just arrays of floats.

| Term | What it is | Shape (for one layer) |
|------|------------|-----------------------|
| **parameter** | A single learnable float | one entry in any weight buffer |
| **weights** | Generic name for the model's learnable arrays | varies |
| **W_up** | FFN matrix that projects from `C` to `ff_dim` | `C × ff_dim` |
| **W_down** | FFN matrix that projects from `ff_dim` back to `C` | `ff_dim × C` |
| **W_out** | Output projection from `C` to `vocab_size` (the "language model head") | `C × vocab_size` |
| **gamma** (γ) | Per-feature scale in layernorm | `C` |
| **beta** (β) | Per-feature shift in layernorm | `C` |
| **bias** | An additive constant per output unit (often skipped in GPT) | `output_dim` |
| **logits** | Raw scores from the final matmul, before softmax | `(B, T, vocab_size)` |

---

## Math concepts

The recurring patterns and tricks.

| Term | What it means |
|------|---------------|
| **affine transformation** | `output = scale * input + shift`. A linear step followed by an additive shift. Layernorm's γ·x_hat + β is one. |
| **linear transformation** | Same as affine but forced to send 0 to 0 (no shift). A plain matmul is linear. |
| **non-linearity** / **activation function** | A non-linear element-wise function (GELU, ReLU, sigmoid). Without one, stacked layers collapse to a single matmul. |
| **dot product** | Multiply two vectors element-wise and sum the results. The building block of matmul. |
| **probability distribution** | A list of non-negative numbers that sum to 1. Softmax produces one per row. |
| **numerical stability** | Designing a formula so floats don't overflow or underflow. The max-subtract in softmax and the ε in layernorm are examples. |
| **epsilon** (ε) | A small positive constant added to avoid dividing by zero. We use `1e-5` in layernorm. |
| **rstd** | "reciprocal standard deviation," i.e. `1 / std`. Multiplying by `rstd` is faster than dividing by `std`. |
| **variance** | Average squared deviation from the mean. Measures "spread" of a row. |
| **two-pass variance** | Compute mean first, then sum squared deviations. Slower than one-pass but numerically stable. Used in `layernorm_forward`. |
| **mean / centering** | Subtracting the mean from every element so the result has mean = 0. |
| **scale / shift** | Multiplying by a constant (scale) and adding a constant (shift). The affine step of layernorm. |
| **residual / skip connection** | `output = input + transform(input)`. Lets gradients flow through deep stacks. |

---

## Training mechanics (Task 6+ — coming)

We have *not* built these yet, but they're referenced in PLAN.md and CLAUDE.md.

| Term | What it will mean |
|------|---------------|
| **forward pass** | Run the model on input, produce predictions. (Built in Task 5.) |
| **backward pass** | Compute gradients: how much each parameter should change to reduce loss. (Task 6.) |
| **gradient** | A float telling each parameter "increase me / decrease me to lower the loss." Same shape as the parameter. |
| **loss** | A single number measuring how wrong the model's predictions are. Lower = better. |
| **cross-entropy loss** | The specific loss function used for next-token prediction. |
| **chain rule** | The calculus rule that lets backward pass run layer by layer. |
| **numerical gradient check** | Finite-difference test that compares analytical gradients to `(f(x+ε) − f(x−ε)) / 2ε`. Gold-standard correctness check. |
| **optimizer** | The algorithm that uses gradients to update parameters. We'll use AdamW (Task 8). |
| **AdamW** | An adaptive optimizer that tracks per-parameter momentum and applies weight decay. |
| **learning rate** | A small scalar (e.g. `1e-3`) controlling how big each parameter update is. |
| **training step** | One iteration: forward → loss → backward → optimizer update. |
| **epoch** | One full pass over the training corpus. |

---

## Attention (Task 7 — coming)

Mentioned in PLAN.md; not yet implemented. Including here so the
terms aren't surprising when they show up.

| Term | What it will mean |
|------|---------------|
| **attention** | The mechanism that lets each token mix information from previous tokens |
| **head** | One attention computation. A multi-head layer runs `num_heads` of them in parallel. |
| **head_dim** | Width of one head's space, equal to `C / num_heads` |
| **query, key, value** (Q, K, V) | The three vectors derived from each token to compute attention scores |
| **attention scores** | Dot products of queries with keys; turned into weights by softmax |
| **causal mask** | Prevents a token from attending to future tokens (so GPT can predict left-to-right) |

---

## Data pipeline

Loading data so the trainer can practice on it.

| Term | What it means |
|------|---------------|
| **corpus** | All the training text. For us: a Serbian e-business textbook (PDF). |
| **text_parser** | Cleans raw PDF text — strips page numbers, captions, bullet glyphs |
| **tokenize** | Turn cleaned text into a stream of token IDs via the trained BPE tokenizer |
| **`.bin` file** | The tokenized corpus on disk — packed `int32_t` token IDs, no header |
| **DataLoader** | Reads the `.bin` into RAM and hands out batches |
| **batch** | One `(inputs, targets)` pair: a `B × T` grid of inputs plus the same grid shifted by one for targets |
| **inputs** | Token IDs the model is allowed to see, shape `(B, T)` |
| **targets** | The correct next-token at each position, shape `(B, T)`. Equals inputs shifted by 1. |
| **cursor** | The DataLoader's current read position in the token buffer |
| **wraparound** | When the cursor hits end-of-file, reset to 0 and keep going |

See `docs/dataloader.md` for the full picture.

---

## C language idioms used in this project

If you trip over a piece of C syntax, this section explains the recurring patterns.

| Idiom | Meaning |
|-------|---------|
| **pointer** | An address of something in memory. `float *p` reads "p is a pointer to a float." |
| **dereference** | `*p` means "the value at the address p." `p[i]` means `*(p + i)`. |
| **`const` parameter** | Promise to the compiler that the function will not modify the data through this pointer |
| **`static` function** | File-local — invisible from other `.c` files. Prevents name clashes. |
| **`(void)x;`** | A no-op statement that uses `x` just enough to silence "unused parameter" warnings |
| **malloc / free** | Manual heap allocation. Every `malloc` needs a matching `free`, otherwise = memory leak. |
| **include guard** | `#ifndef FOO_H / #define FOO_H / ... / #endif` — prevents a header from being included twice |
| **`f` suffix** | `0.5f` is a float; `0.5` is a double. Keep the `f` for single-precision consistency. |
| **`tanhf`, `expf`, `sqrtf`** | Single-precision versions of `tanh`, `exp`, `sqrt`. Match the `float` type, not `double`. |
| **flat-index trick** | `arr[row * NCOLS + col]` — turn a 2-D coordinate into a 1-D index. |
| **`{0}` initializer** | `float buf[4] = {0};` zero-initializes the whole array. |
| **forward declaration** | A function's signature in a `.h` file, without the body. Lets callers compile against it. |

---

## Build and test tools

The infrastructure that runs the code and tests.

| Tool | What it is |
|------|------------|
| **CMake** | The build system. Reads `CMakeLists.txt` and generates platform-specific build files. |
| **`cmake -B build`** | Configure step: produce files in `build/` |
| **`cmake --build build`** | Compile step: actually run the C compiler |
| **CMocka** | The C unit-test framework. Pattern: `assert_*` calls inside `static void test_*(void**)` functions. |
| **ctest** | The CMake test runner. `ctest --test-dir build --output-on-failure` runs everything in `tests/`. |
| **TDD** | Test-Driven Development: write the test first, watch it fail, implement, watch it pass. The project's required workflow. |
| **fixture** | A small test data file (e.g. `tests/helpers/fixtures/sample_text.txt`) |
| **assertion** | A line like `assert_float_equal(actual, expected, tol)` that fails the test if the condition is false |
| **`-Wall -Werror`** | Compiler flags: enable all warnings, treat warnings as errors. Keeps the codebase clean. |
| **`ENABLE_CUDA`, `ENABLE_MPI`** | CMake options. Off on Mac (default), on for Google Colab. |

---

## Hardware and parallelism (Task 9+ — coming)

The "concurrent and distributed" part of the seminar.

| Term | What it will mean |
|------|---------------|
| **CPU** | The main processor. Where everything runs during Mac development. |
| **GPU** | Graphics card with thousands of small cores. Where training actually happens on Colab. |
| **CUDA** | NVIDIA's API for writing GPU code in C-like syntax. Files end in `.cu`. |
| **kernel** | One GPU function, launched in parallel across many threads |
| **thread** | The smallest unit of GPU work. Thousands run concurrently per kernel launch. |
| **block** | A group of threads that share fast on-chip memory. Sized e.g. 256. |
| **grid** | The whole launch — all blocks together. |
| **shared memory** | Fast memory shared within one block. Used in tiled matmul and softmax. |
| **MPI** | Message Passing Interface — lets multiple processes coordinate. The "distributed" part of the project. |
| **rank** | The ID of one MPI process (0, 1, 2, ...) |
| **world_size** | Total number of MPI processes |
| **AllReduce** | MPI operation that sums (or averages) a value across all ranks. Used to combine gradients. |
| **data parallelism** | Every rank has the full model; each processes different data; gradients are summed across ranks. |

---

## Where to look next

- `PLAN.md` — full project roadmap, task by task
- `CLAUDE.md` — project conventions (C style, TDD, test naming)
- `docs/bpe-training.md` — BPE algorithm walked by hand
- `docs/dataloader.md` — data loader walked by hand
- `docs/forward-pass.md` — embed → FFN → output, walked by hand
- `docs/backward-pass.md` — gradients flowing back through each layer
- `docs/gpt-model.md` — wiring the layers into one full transformer
- `docs/adamw.md` — the AdamW optimizer: gradients → weight updates
