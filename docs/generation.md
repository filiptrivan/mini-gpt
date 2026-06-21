# Text Generation (Task 13)

How a trained model turns a prompt into text — `tools/generate.c` — and the
checkpoint format that gets the weights from training to inference
(`src/model/gpt_io.c`).

## First, the checkpoint

Training builds good weights in RAM; they vanish when the process exits. So
`train.c` (when given a 4th argument) calls **`gpt_save`** to write them to disk,
and `generate` calls **`gpt_load`** to read them back. This mirrors the
tokenizer's `bpe_save`/`bpe_load`.

The file is deliberately tiny and simple:

```
header : 9 × int32  →  magic, version, the 6 GPTConfig dimensions, num_params
body   : num_params × float32  →  the contiguous parameter block, verbatim
```

Only the **parameters** are saved. Activations and gradients are scratch that the
forward/backward passes rebuild, so there's no reason to store them. On load we
read the config, rebuild the exact model layout from it via `gpt_init` (the
layout is a pure function of the config), and read the floats straight back into
`model->params`. The `num_params` field is a sanity check: if it doesn't match
what the config implies, the file is rejected.

## Autoregressive generation

A GPT models **the probability of the next token given all previous tokens**. It
doesn't generate a whole sentence at once — it generates one token, appends it to
the input, and predicts again. That loop is the entire idea:

```
tokens = encode(prompt)
repeat max_new_tokens times:
    logits = model.forward(last min(len(tokens), ctx) tokens)   # no targets -> no loss
    next   = sample(logits at the final position, temperature)
    append next to tokens
    print decode(next)
```

A few details that matter:

- **The last position is the prediction.** The forward pass produces logits at
  *every* position, but only the final one is the model's guess for what comes
  next. We read `logits[(T-1) * vocab_size ...]`.
- **The context window slides.** The model has positional embeddings for only
  `max_seq_len` (64) positions. Once the generated sequence is longer than that,
  we feed the model the *last* 64 tokens, not the whole history.
- **No loss on this path.** `gpt_forward` is called with `targets == NULL`, so it
  computes logits/probabilities but skips the cross-entropy — there's nothing to
  score against during generation.
- **The tokenizer must match the model.** `generate` refuses a tokenizer whose
  vocab is larger than the model's, because an out-of-range id would index past
  the embedding table. Use the same `.bpe` file the training tokens came from.

## Temperature — the randomness knob

Instead of always taking the single most likely token (which loops and repeats),
we **sample** from the distribution, scaled by a `temperature`:

```
p(token) ∝ exp(logit / temperature)
```

- **`temperature = 0`** → greedy: always the argmax. Deterministic, often repetitive.
- **`temperature < 1`** → sharper: the model's confident choices dominate.
- **`temperature = 1`** → the model's raw distribution.
- **`temperature > 1`** → flatter: rarer tokens get a real chance; more surprising,
  more incoherent.

We compute it stably by subtracting the max logit before `exp` (so it never
overflows — a constant shift cancels in the normalization) and sample by walking
the cumulative distribution until it passes a uniform draw. A seeded
`xorshift64*` RNG makes a given `(prompt, temperature, seed)` reproducible.

## Running it

```bash
# train, saving a checkpoint (4th arg)
./build/src/train data/processed/tokens.bin 500 1e-3 model.ckpt

# generate: <checkpoint> <tokenizer> <prompt> [max_new_tokens] [temperature] [seed]
./build/tools/generate model.ckpt data/processed/tok.bpe "Elektronsko poslovanje" 200 0.8
```

Generated text goes to **stdout** (so you can redirect it); model info and
settings go to **stderr**. The prompt is echoed first, so the output reads as one
continuous passage, and tokens stream out as they're produced.

## What to expect

Recognizably Serbian *characters* and some common fragments and short words —
but not coherent sentences. 534K parameters trained on a few hundred KB is far
too small for fluency, and that was never the point: generation is the proof that
the whole hand-built stack — tokenizer, model, training, checkpointing — composes
into something that runs end to end. Lower the temperature for safer, more
word-like output; raise it for more adventurous (and more garbled) text.
```
