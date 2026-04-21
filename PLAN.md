# Mini GPT in C/CUDA/MPI — Implementation Plan

## Context

This is a seminar project for "Konkurentno i distribuirano programiranje" (Concurrent and Distributed Programming) at FON Belgrade. The goal is to build a small GPT-style language model from scratch in C, using CUDA for parallel computation (concurrent) and MPI for multi-process gradient synchronization (distributed). The model trains on Serbian academic text (e-business textbook PDF, Latin script). Development happens on Mac M2 (CPU-only), training on Google Colab (T4 GPU). Everything is beginner-friendly, TDD-driven, MIT licensed, and public on GitHub.

---

## Project Structure

```
KP/
├── CMakeLists.txt                    # Root build: CPU/CUDA/MPI toggles
├── CLAUDE.md                         # Project conventions for Claude Code
├── .gitignore
├── colab/
│   ├── setup.sh                      # Colab dependency installer
│   └── train.ipynb                   # Notebook: build, test, train, generate
├── data/
│   ├── raw/                          # PDF + raw extracted text (gitignored)
│   └── processed/                    # Cleaned text + tokenized binary files
├── src/
│   ├── CMakeLists.txt
│   ├── tokenizer/
│   │   ├── bpe.h / bpe.c            # BPE train/encode/decode
│   │   └── bpe_io.h / bpe_io.c      # Save/load tokenizer
│   ├── data/
│   │   ├── text_parser.h / .c       # Clean extracted PDF text for tokenization
│   │   └── dataloader.h / .c        # Batch loading from binary tokens
│   ├── model/
│   │   ├── layers.h / layers.c      # All CPU layer ops (forward + backward)
│   │   └── gpt.h / gpt.c            # Full GPT model wiring
│   ├── cuda/
│   │   ├── cuda_layers.cuh           # CUDA kernel declarations
│   │   ├── matmul.cu                 # Tiled matrix multiply
│   │   ├── softmax.cu                # Row-wise softmax
│   │   ├── layernorm.cu              # Layer normalization
│   │   ├── attention.cu              # Attention score + aggregate
│   │   ├── elementwise.cu            # GELU, residual add
│   │   ├── embedding.cu              # Embedding lookup
│   │   ├── cross_entropy.cu          # Loss computation
│   │   └── gpt_cuda.cu              # GPU forward/backward wiring
│   ├── optimizer/
│   │   └── adamw.h / adamw.c        # AdamW optimizer
│   ├── distributed/
│   │   └── mpi_utils.h / mpi_utils.c # MPI init, AllReduce, broadcast
│   └── train.c                       # Main training loop
├── tests/
│   ├── CMakeLists.txt
│   ├── helpers/
│   │   ├── test_utils.h / .c        # Float comparison, numerical grad check
│   │   └── fixtures/
│   │       └── sample_text.txt       # PDF text artifacts for parser tests
│   ├── test_text_parser.c
│   ├── test_bpe.c
│   ├── test_dataloader.c
│   ├── test_layers.c
│   ├── test_gpt.c
│   ├── test_adamw.c
│   ├── test_cuda_layers.cu           # CUDA vs CPU comparison tests
│   └── test_distributed.c            # MPI gradient sync tests
└── tools/
    ├── extract_pdf.py                # Python: PDF → raw text extraction
    ├── preprocess.c                  # CLI: raw text → cleaned text
    ├── train_bpe.c                   # CLI: train BPE tokenizer
    ├── tokenize.c                    # CLI: text → binary tokens
    └── generate.c                    # CLI: load model + generate text
```

---

## Model Specs (tiny but real)

| Parameter      | Value          |
|---------------|----------------|
| max_seq_len   | 64             |
| vocab_size    | 512            |
| embed_dim     | 128            |
| num_heads     | 4              |
| num_layers    | 2              |
| ff_dim        | 512 (4×C)      |
| **Total params** | **~270K**   |

---

## Task Breakdown (13 sessions, ~2-3h each)

Each task follows TDD: write test → watch it fail → implement → pass → commit.

### Week 1: Data Pipeline + Foundations

**Task 1: Project Skeleton + Build System (~1.5h)**
- Create directory structure, `.gitignore`, `CLAUDE.md`
- Write `CMakeLists.txt` with `option(ENABLE_CUDA OFF)` and `option(ENABLE_MPI OFF)`
- Install CMocka (`brew install cmocka` on Mac)
- Write a trivial hello-world test to verify CMocka + CMake work
- Verify: `cmake -B build && cmake --build build && ctest --test-dir build`
- Git init + first commit

**Task 2: Text Parser (~2h)**
- Test first: `tests/test_text_parser.c`
  - Load and parse text from fixture file
  - Remove page numbers (standalone digit lines)
  - Remove page markers (`---PAGE N---`)
  - Remove figure captions (`Slika N ...`)
  - Remove table captions (`Tabela N ...`)
  - Strip Word bullet Unicode chars (U+F0B7, U+F02D)
  - Normalize whitespace (no blank lines, no trailing spaces)
  - UTF-8 Serbian characters (č, ž, š, đ, ć) pass through intact
  - Empty input → length=0, no crash
- Implement: `src/data/text_parser.c`
- Python extraction: `tools/extract_pdf.py` (PDF → raw text)
- Implement: `tools/preprocess.c` (CLI wrapper, raw text → clean text)

**Task 3: BPE Tokenizer (~3h)**
- Test first: `tests/test_bpe.c`
  - 256 base byte tokens exist after init
  - Training on known string produces expected merge count
  - `decode(encode(text)) == text` (roundtrip)
  - Serbian multi-byte UTF-8 handled correctly
  - Empty string → len=0
  - Deterministic: same input → same tokens
- Implement: `src/tokenizer/bpe.c`, `src/tokenizer/bpe_io.c`
- Implement: `tools/train_bpe.c`, `tools/tokenize.c`

**Task 4: Data Loader (~1.5h)**
- Test first: `tests/test_dataloader.c`
  - Create small binary token file in test setup
  - Batch has correct shape (batch_size × seq_len)
  - Targets = inputs shifted by 1
  - Wraparound at end of data
- Implement: `src/data/dataloader.c`

### Week 2: Neural Network Layers (CPU)

**Task 5: Layer Functions — Forward Pass (~3h)**
- Test first: `tests/test_layers.c`
  - `matmul_forward`: hand-computed 2×2 case
  - `layernorm_forward`: output has mean≈0, variance≈1
  - `softmax_forward`: sums to 1, all positive, handles large values (numerical stability)
  - `gelu_forward`: known values
  - `residual_forward`: a + b
  - `embed_forward`: correct lookup + position embedding add
- Implement: all forward functions in `src/model/layers.c`

**Task 6: Layer Functions — Backward Pass (~3h)**
- Test: extend `tests/test_layers.c` with numerical gradient checks
  - For each layer: perturb input by ε, recompute forward, compare `(f(x+ε) - f(x-ε)) / 2ε` with analytical gradient
  - This is the gold-standard correctness test for gradients
- Implement: all backward functions in `src/model/layers.c`
- Helper: `numerical_gradient()` function in `tests/helpers/test_utils.c`

**Task 7: GPT Model Assembly (~3h)**
- Test first: `tests/test_gpt.c`
  - Model init: correct param count, no null pointers
  - Forward: produces a loss (not NaN, not Inf)
  - Backward: all gradients non-zero
  - Numerical gradient check on tiny model (C=8, L=1, NH=1, V=32, T=4)
- Implement: `src/model/gpt.c` — wire layers into full forward/backward pass
- Key: single contiguous memory block for all parameters and gradients

### Week 3: Optimizer, CUDA, Training

**Task 8: AdamW + CPU Training Loop (~2h)**
- Test first: `tests/test_adamw.c`
  - Optimizer step reduces loss on trivial problem
  - Weight decay shrinks weights toward zero
  - Moment estimates accumulate correctly
- Implement: `src/optimizer/adamw.c`
- Wire up: `src/train.c` (CPU-only), verify loss decreases on small dataset
- **This is the "acid test"** — if loss decreases, the entire forward/backward/update pipeline works

**Task 9: CUDA Kernels — MatMul + LayerNorm (~3h, on Colab)**
- Test: `tests/test_cuda_layers.cu`
  - Pattern: run CPU version, run CUDA kernel, compare outputs (tolerance ~1e-5)
  - Test `matmul_forward_cuda` vs `matmul_forward`
  - Test `layernorm_forward_cuda` vs `layernorm_forward`
- Implement: `src/cuda/matmul.cu` (tiled, TILE=16), `src/cuda/layernorm.cu`

**Task 10: CUDA Kernels — Remaining (~3h, on Colab)**
- Extend `tests/test_cuda_layers.cu`:
  - softmax, attention, GELU, residual, embedding, cross-entropy
- Implement: remaining `.cu` files
- Wire up: `src/cuda/gpt_cuda.cu` — GPU forward/backward

### Week 4: Distributed + Polish

**Task 11: MPI Data Parallelism (~3h, on Colab)**
- Test first: `tests/test_distributed.c` (run with `mpirun -np 2`)
  - AllReduce on known values: rank 0 has [1,2], rank 1 has [3,4] → both get [2,3]
  - All ranks get identical results
  - After one training step from identical copies, parameters still match across ranks
- Implement: `src/distributed/mpi_utils.c`
- Integrate into `src/train.c`:
  - Each rank gets different batch (offset DataLoader by rank)
  - AllReduce gradients after backward → average → optimizer step

**Task 12: End-to-End Training on Colab (~3h)**
- Create `colab/setup.sh` + `colab/train.ipynb`
- Full pipeline: upload data → preprocess → train BPE → tokenize → train model
- Run single-GPU training, verify loss curve
- Run `mpirun -np 2` training, compare loss curves
- Benchmark: 1 process vs 2 processes speed comparison

**Task 13: Text Generation + README (~2h)**
- Implement: `tools/generate.c`
  - Autoregressive sampling with temperature parameter
  - Load checkpoint + tokenizer, generate from prompt
- Write `README.md`: what it is, how to build, architecture overview
- Clean up, final commit

---

## CUDA Kernels Summary

| Kernel | What it parallelizes | Grid | Block | Shared Mem |
|--------|---------------------|------|-------|------------|
| embedding_fwd | per-element (B×T×C) | (N+255)/256 | 256 | No |
| layernorm_fwd | per-(b,t) position | B×T | min(C,256) | Yes |
| matmul_fwd | tiled output tiles | 2D | (16,16) | Yes |
| attention_scores | per-(b,nh,t) | B×NH×T | min(T,256) | No |
| attention_softmax | per-row | B×NH×T | min(T,256) | Yes |
| attention_aggregate | per-(b,nh,t) | B×NH×T | head_dim | No |
| gelu_fwd | per-element | (N+255)/256 | 256 | No |
| residual_fwd | per-element | (N+255)/256 | 256 | No |
| softmax_fwd | per-row over vocab | B×T | min(V,256) | Yes |
| crossentropy_fwd | per-(b,t) | (N+255)/256 | 256 | No |

Each kernel also has a backward variant (same parallelization strategy). Backward for embedding uses `atomicAdd`.

---

## MPI Distributed Strategy

**Architecture**: Data parallelism. Each rank has full model copy, processes different batches.

```
For each training step:
  1. Each rank: load its own batch (DataLoader offset by rank)
  2. Each rank: forward pass (CPU or CUDA)
  3. Each rank: backward pass → local gradients
  4. ALL ranks: MPI_Allreduce(grads, MPI_SUM) → divide by world_size
  5. Each rank: AdamW step with averaged gradients (identical update)
```

On Colab: `mpirun --allow-run-as-root -np 2 ./build/train ...`
Multiple processes share one GPU — inefficient but valid for demonstrating distributed concepts.

---

## Development Workflow

**On Mac M2 (daily development):**
```bash
cmake -B build -DENABLE_CUDA=OFF -DENABLE_MPI=OFF
cmake --build build && ctest --test-dir build --output-on-failure
```

**On Google Colab (CUDA + MPI testing):**
```bash
git clone <repo> && bash colab/setup.sh
cmake -B build -DENABLE_CUDA=ON -DENABLE_MPI=ON
cmake --build build && ctest --test-dir build --output-on-failure
mpirun --allow-run-as-root -np 2 ./build/tests/test_distributed
```

Source code uses `#ifdef USE_CUDA` / `#ifdef USE_MPI` guards for conditional compilation.

---

## Testing Strategy

- **~60-70 tests** across 8 test files
- **Numerical gradient checking** is the key technique: for every backward pass, compare analytical gradient to finite-difference approximation
- **CUDA tests**: run CPU version as oracle, compare CUDA output within tolerance (1e-5)
- **MPI tests**: run with `mpirun -np 2`, verify all ranks agree after AllReduce
- **Integration test** ("acid test"): train 50 steps on tiny data, assert loss[49] < loss[0]
- **All test dimensions are tiny**: d_model=4-8, seq_len=4, vocab=32 in tests — fast, debuggable

---

## Verification Plan

1. **After each task**: `ctest --test-dir build --output-on-failure` (all tests pass)
2. **After Task 8**: CPU training loop shows loss decreasing over 100 steps
3. **After Task 10**: CUDA forward/backward matches CPU within 1e-5 tolerance
4. **After Task 11**: `mpirun -np 2` training produces same loss as `mpirun -np 1` (with equivalent batch sizes)
5. **After Task 13**: `./generate --prompt "elektronsko" --length 200` produces text that resembles Serbian academic prose

---

## Critical Files (in order of importance)

1. `CMakeLists.txt` — tri-mode build enables the Mac→Colab workflow
2. `src/model/layers.c` — mathematical core, every operation's forward+backward
3. `src/model/gpt.c` — wires layers into a complete transformer
4. `src/tokenizer/bpe.c` — only path from text→tokens→text
5. `src/distributed/mpi_utils.c` — core deliverable for the "distributed" aspect
6. `src/cuda/matmul.cu` — most performance-critical CUDA kernel

---

## What to Expect from the Model

With ~270K parameters on a small Serbian text corpus:
- It will NOT produce coherent Serbian sentences
- It WILL learn character/word-fragment patterns
- Output will "look like" Serbian text at a glance (correct character distributions, common words)
- Clear loss decrease during training = success
- The value is in building the system, not output quality
