# Data Loader, Walked by Hand

A worked example of what the data loader in `src/data/dataloader.c` does and
why GPT training needs it. Useful as a reference for total beginners to both
C and language models.

## What the data loader is for

The GPT model learns by playing one guessing game over and over:

> "I'll show you a few tokens. Guess what the next token is."

Every wrong guess nudges the model's internal numbers a tiny bit so it
guesses better next time. Millions of nudges later, it has learned the
patterns of the training text.

The data loader is the **teacher** that hands the model practice questions
during training. It does three jobs:

1. Read the tokenized training file from disk into memory once.
2. On request, hand back the next "practice question": one batch of inputs
   and the matching targets.
3. When it reaches the end of the file, wrap back to the beginning so
   training can keep cycling through the data.

That is the whole job. No randomness, no shuffling — just a moving window
over a long array of integers.

## Step 0: what is in the `.bin` file

`./tokenize` (from Task 3) wrote the training corpus as a flat array of
`int32_t` token ids — no header, no separators, just numbers packed back to
back. Conceptually:

```
disk file: train.bin

[10, 20, 30, 40, 50, 60, 70, 80, 90, ...thousands more]
```

Each integer is one token (a piece of text that BPE learned). For this
walkthrough we will pretend the file is short and sequential:
`[0, 1, 2, 3, ..., 19]` — 20 tokens total. The numbers are chosen so the
"shifted by one" property is obvious by eye.

## Step 1: at init, load the whole file into RAM

The loader opens the file once, reads every byte into a `malloc`'d buffer,
then closes the file. From that point on the loader works entirely in memory,
never touching disk again.

```
disk (train.bin)                 RAM buffer (loader->tokens)
+------------------------+       +-------------------------------+
| 0 1 2 3 ... 19         | ----> | 0 1 2 3 4 5 6 7 ... 18 19     |
+------------------------+       +-------------------------------+
                                  ^
                                  loader->cursor = 0
```

State held by the loader after init:

| Field          | Value | Meaning |
|----------------|-------|---------|
| `tokens`       | pointer to RAM buffer | the whole file, in memory |
| `num_tokens`   | 20 | how many ints are in the buffer |
| `batch_size`   | 2  | rows per batch (chosen by the trainer) |
| `seq_len`      | 4  | tokens per row |
| `cursor`       | 0  | next read position in `tokens` |

`batch_size` and `seq_len` come from the trainer (Task 8). For this example
we use `batch_size=2`, `seq_len=4` so every batch is a tiny 2x4 grid.

## Step 2: what one batch looks like

The model wants two arrays per question:

- `inputs[batch_size][seq_len]` — tokens it is allowed to see
- `targets[batch_size][seq_len]` — the correct next-token at each position

`targets` is just `inputs` shifted right by one position. To fill both
arrays the loader needs `batch_size * seq_len + 1` consecutive tokens from
the buffer (the `+ 1` is the extra rightmost token used only for targets).

With `B = 2`, `T = 4`, one batch consumes 9 tokens of the stream:

```
cursor=0:
  buffer:  [ 0  1  2  3  4  5  6  7  8 | 9 10 11 12 ... 19 ]
             |---------- B*T -----------|
             |------- inputs window -------|
                |------ targets window ------|

  read positions 0..7 -> fill inputs (row by row)
  read positions 1..8 -> fill targets (row by row)
```

Reshaped into the two 2x4 grids the trainer expects:

```
inputs (B=2, T=4)              targets (B=2, T=4)
+-----------------+            +-----------------+
| 0  1  2  3      |  row 0     | 1  2  3  4      |  row 0
| 4  5  6  7      |  row 1     | 5  6  7  8      |  row 1
+-----------------+            +-----------------+
```

Read column-by-column at row 0:

- Position 0: input `0`, target `1`. "Given token 0, predict 1."
- Position 1: input `1`, target `2`. "Given tokens 0,1, predict 2."
- Position 2: input `2`, target `3`. "Given tokens 0,1,2, predict 3."
- Position 3: input `3`, target `4`. "Given tokens 0,1,2,3, predict 4."

That is GPT in one sentence: **predict the next token from the previous
tokens, at every position, in parallel.** One 2x4 batch is really `B*T = 8`
independent practice questions packed into one tensor for speed.

## Step 3: advance the cursor

After filling the batch, the loader moves the cursor forward by `B*T` (not
by `B*T + 1` — the extra token at the end is only borrowed for the targets;
the next batch's inputs start where this batch's inputs ended):

```
before:  cursor = 0
buffer:  [ 0  1  2  3  4  5  6  7  8 | 9 10 11 12 13 14 15 16 17 18 19 ]
           |--- batch 1 inputs -----|

after:   cursor = 8
buffer:  [ 0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15 16 17 18 19 ]
                                    ^
                                    next read starts here
```

The second batch then produces:

```
inputs (from positions 8..15)  targets (from positions 9..16)
+-----------------+            +------------------+
|  8  9 10 11     |  row 0     |  9 10 11 12      |  row 0
| 12 13 14 15     |  row 1     | 13 14 15 16      |  row 1
+-----------------+            +------------------+
```

Cursor advances to 16. Same `targets[i] == inputs[i] + 1` property holds
because the synthetic file is sequential.

## Step 4: wraparound at end of file

A third batch would need positions 16..24, but the buffer only has 20
tokens. Instead of erroring, the loader **resets the cursor to 0** and reads
from the start again:

```
cursor = 16, would need positions 16..24 (9 tokens), only 20 exist
=> not enough room, wrap: cursor = 0
=> read positions 0..8 again, producing the SAME batch as the very first call
```

This is intentional. Training is supposed to cycle through the data many
times ("epochs"). The loader's job is to make the file feel infinite — when
it ends, just loop. The trainer never has to special-case end-of-file.

The exact wrap condition, written out:

```
if cursor + B*T + 1 > num_tokens:
    cursor = 0
fill inputs from cursor .. cursor + B*T - 1
fill targets from cursor + 1 .. cursor + B*T
cursor += B*T
```

A subtle point: we check **before** the read, not after. That guarantees we
never read past the end of the buffer, which in C would be undefined
behavior (a memory bug, possibly a crash).

## Step 5: free memory

In C there is no garbage collector. Every `malloc` must be matched by a
`free`. The loader allocates one buffer (the `tokens` array) at init, so
`dataloader_free` releases that buffer and the `DataLoader` struct itself.

```
dataloader_init  -> malloc tokens buffer, malloc loader struct
... training ...
dataloader_free  -> free tokens buffer, free loader struct
```

Forgetting `free` causes a "memory leak": the program still works, but it
holds onto memory it no longer needs. Not a crash, but bad hygiene.

## The whole picture

```
  +--------------------------------------------------------+
  |  train.bin (on disk, written by ./tokenize)            |
  |  [10, 20, 30, 40, 50, 60, 70, 80, 90, ...]             |
  +-----------------------+--------------------------------+
                          | fread() once at init
                          v
  +--------------------------------------------------------+
  |  loader->tokens (in RAM)                               |
  |  [10, 20, 30, 40, 50, 60, 70, 80, 90, ...]             |
  |   ^                                                    |
  |   loader->cursor (moves forward each batch, wraps)     |
  +-----------------------+--------------------------------+
                          | every call to dataloader_next_batch()
                          v
  +------------------+   +------------------+
  | inputs  [B][T]   |   | targets [B][T]   |
  | (caller-owned)   |   | (caller-owned)   |
  +--------+---------+   +--------+---------+
           |                      |
           +----- fed into GPT forward + backward (Task 8) -----+
```

## Why this design (not the other choices)

A few alternatives, and why we picked what we did:

- **Whole file in RAM vs streaming with fread.** Our training corpus is a
  textbook, a few MB at most. Loading once is simpler, faster, and easier
  to teach. Streaming only matters for datasets that do not fit in memory,
  which is not us.

- **Sequential cursor vs random offsets.** Picking random start positions
  per batch gives a slightly better training signal in practice, but it
  makes tests non-deterministic (random output) and is harder to reason
  about for a first version. Sequential is good enough and trivially
  testable.

- **Targets shifted by one vs separate next-token files.** The shift
  trick means the input file is the only thing we need on disk. The
  "labels" are the same data, just read with an offset. No preprocessing
  step needed.

## Concept checks

Three questions to sanity-check understanding:

1. **Why does one batch need `B*T + 1` tokens, not `B*T`?** Because targets
   are inputs shifted right by one position, and the rightmost target in
   the bottom row needs a token from one step past the inputs window.

2. **Why move the cursor by `B*T` after a batch, not `B*T + 1`?** The
   extra `+ 1` token at the end is shared: it was the last target of this
   batch, and it should also be the first input of the next batch (the
   next batch will then re-borrow its own `+ 1` neighbor as a target).
   Advancing by `B*T` keeps the stream continuous.

3. **What happens if the file is shorter than `B*T + 1`?** Then no valid
   batch can be made — `init` should reject this case with an error,
   because wrapping does not help (even a wrap-around window would not
   fit). This is a precondition we check up front.
