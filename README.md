# Mini GPT — a GPT from scratch in C, CUDA, and MPI

A small GPT-style language model built **from scratch in C** — every layer, the
forward and backward passes, the optimizer, the tokenizer, the data pipeline —
with **CUDA** kernels for GPU training (the *concurrent* part) and **MPI** for
multi-process gradient averaging (the *distributed* part). It trains on Serbian
academic text (an e-business textbook, Latin script).

Seminar project for *Konkurentno i distribuirano programiranje* (Concurrent and
Distributed Programming) at FON Belgrade. MIT licensed.

> **Expectations, up front:** with ~534K parameters on a few hundred KB of text,
> the model will **not** write coherent Serbian. The goal was never output
> quality — it was building the whole system by hand and watching the loss go
> down. It does. That's the win.

## What's in it

- **No frameworks.** No PyTorch, no cuBLAS, no autograd. The matmuls, attention,
  layernorm, GELU, softmax, cross-entropy, and all their gradients are written
  out by hand and checked against finite-difference numerical gradients.
- **Three build modes from one codebase**, selected by compile-time flags:
  CPU (Mac, daily dev), CPU+GPU (CUDA), and distributed (MPI) — composable.
- **Test-driven.** Every component has a test; the CUDA kernels are checked
  against the CPU implementations as oracles, and the MPI layer is checked across
  two ranks.

## Model

Small GPT-2 style transformer:

| Hyperparameter | Value |
|----------------|-------|
| Layers         | 2     |
| Embedding dim  | 128   |
| Attention heads| 4     |
| Feed-forward   | 512   |
| Context length | 64    |
| Vocab size     | 512 (256 byte tokens + 256 BPE merges) |
| **Parameters** | **~534K** |

Bias-free linear layers, a separate (untied) output head, learned positional
embeddings. All weights live in one contiguous block, which is what lets the
optimizer update everything in one loop and MPI average everything in one call.

## Build

Requires CMake ≥ 3.18 and CMocka. CUDA and MPI builds additionally need the
NVIDIA toolkit and an MPI implementation (both present on Google Colab).

```bash
# Mac / CPU — daily development
cmake -B build -DENABLE_CUDA=OFF -DENABLE_MPI=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Google Colab — CUDA + MPI
cmake -B build -DENABLE_CUDA=ON -DENABLE_MPI=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The MPI test runs under `mpirun`:

```bash
mpirun --allow-run-as-root --oversubscribe -np 2 ./build/tests/test_distributed
```

## The full pipeline

```
PDF ──► extract_pdf.py ──► preprocess ──► train_bpe ──► tokenize ──► train ──► generate
        (raw text)         (clean text)   (tokenizer)   (tokens)    (weights)  (text)
```

The cleaned corpus (`data/processed/skripta_clean.txt`) is committed, so you can
start from `train_bpe`. End to end:

```bash
# 1. Tokenizer: train on a sample (BPE training is O(n²)), 256 merges -> vocab 512
head -c 32768 data/processed/skripta_clean.txt > data/processed/sample.txt
./build/tools/train_bpe data/processed/sample.txt data/processed/tok.bpe 256

# 2. Tokenize the FULL corpus into int32 tokens
./build/tools/tokenize data/processed/skripta_clean.txt data/processed/tok.bpe data/processed/tokens.bin

# 3. Train, saving a checkpoint (add -np 2 under mpirun for distributed)
./build/src/train data/processed/tokens.bin 500 1e-3 model.ckpt

# 4. Generate from a prompt
./build/tools/generate model.ckpt data/processed/tok.bpe "Elektronsko poslovanje" 200 0.8
```

`generate` arguments: `<checkpoint> <tokenizer> <prompt> [max_new_tokens]
[temperature] [seed]`. Temperature 0 is greedy; higher is more random.

The easiest way to run the GPU + distributed path is the Colab notebook,
`colab/train.ipynb` (open it from GitHub, *Runtime → T4 GPU*, *Run all*).

## How the "concurrent" and "distributed" parts work

- **Concurrent (CUDA).** Each layer op has a GPU kernel (tiled matmul, per-row
  layernorm/softmax reductions, per-element GELU/residual, causal attention).
  `gpt_forward_cuda`/`gpt_backward_cuda` chain them with tensors kept resident on
  the device. Correctness is proven by comparing every kernel — and the whole
  forward/backward — against the CPU version within tolerance.
- **Distributed (MPI).** Data parallelism: each process trains on a different
  slice of the corpus, then `MPI_Allreduce` averages the gradients so every
  process applies the identical update and the model replicas stay in sync.

## Layout

```
src/
  tokenizer/   BPE train/encode/decode + save/load
  data/        PDF text cleaner + batch DataLoader
  model/       layers.c (all ops, fwd+bwd) · gpt.c (the transformer) · gpt_io.c (checkpoints)
  cuda/        one .cu per op + gpt_cuda.cu (full GPU fwd/bwd)
  optimizer/   AdamW
  distributed/ MPI init / all-reduce / broadcast
  train.c      the training loop (CPU / GPU / MPI)
tools/         extract_pdf.py · preprocess · train_bpe · tokenize · generate
tests/         one test file per component (CMocka)
docs/          a plain-language write-up per component
colab/         setup.sh + train.ipynb (end-to-end on a T4)
```

## Docs

Each component has a beginner-oriented explainer in [`docs/`](docs/):
[forward](docs/forward-pass.md) · [backward](docs/backward-pass.md) ·
[GPT model](docs/gpt-model.md) · [BPE](docs/bpe-training.md) ·
[DataLoader](docs/dataloader.md) · [AdamW](docs/adamw.md) ·
[CUDA kernels](docs/cuda-kernels.md) · [MPI](docs/mpi-distributed.md) ·
[Colab training](docs/colab-training.md) · [generation](docs/generation.md) ·
and a [glossary](docs/glossary.md).

## License

MIT — see [LICENSE](LICENSE).
