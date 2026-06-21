# End-to-End Training on Colab (Task 12)

How the whole project comes together into one training run on a Google Colab T4:
raw text → tokens → a model whose loss actually goes down — on the GPU, and
across two MPI processes. The runnable version is `colab/train.ipynb`; this doc
is the map and the "why."

## The two deliverables

- **`colab/setup.sh`** — installs the Colab dependencies that aren't preinstalled:
  CMocka (unit tests), OpenMPI (the distributed build/tests), and PyPDF2 (only for
  the optional "bring your own PDF" path). `nvcc` already ships with Colab's GPU
  runtime.
- **`colab/train.ipynb`** — opens straight from GitHub, clones the repo, builds
  with CUDA + MPI, runs the full test suite, then walks the pipeline below.

## The pipeline

```
data/processed/skripta_clean.txt   (committed — a fresh clone already has it)
        │  head -c 32768            (BPE trains on a SAMPLE — see below)
        ▼
   sample.txt
        │  train_bpe  <sample.txt> <out.bpe> <num_merges>
        ▼
   tok.bpe  (the trained BPE tokenizer)
        │  tokenize  <skripta_clean.txt> <tok.bpe> <out.bin>   (FULL corpus)
        ▼
   tokens.bin  (flat int32 token ids)
        │  train  <tokens.bin> [steps] [lr]
        ▼
   loss falling  =  success
```

The cleaned Serbian corpus is checked into the repo, so the notebook needs **no
upload** — it trains straight from a clone. (To start from your own PDF instead:
`extract_pdf.py` → `preprocess` produces the same cleaned text; see the optional
cell in the notebook.)

### One contract that must hold: vocab size

`src/train.c` is hard-configured for **`vocab_size = 512`**, and the DataLoader
rejects any token id `>= vocab_size`. BPE produces `256 + num_merges` tokens
(256 base byte tokens plus the merges it learns), so the tokenizer is trained
with **256 merges**:

```
256 byte tokens + 256 merges = 512 = the model's vocab_size
```

Use a different merge count and either the ids overflow the embedding table
(init fails) or you waste vocab rows. 256 is the number that matches.

### Why BPE trains on a subsample

`bpe_train` counts pairs by brute force, making training **O(num_merges ×
length²)** (see the block comment in `src/tokenizer/bpe.c` and
`docs/bpe-training.md`). On the full 778 KB corpus that is ~10¹⁴ operations —
*hours*. So the notebook trains the tokenizer on a small slice (`head -c 32768`,
~16 s) and then tokenizes the **full** corpus with it. The merges learned from a
representative sample generalize, and **encoding** is only `O(num_merges ×
length)` (a second or two on the whole text), so the model still trains on every
token of the full corpus. Replacing the brute-force pair counter with an
incrementally-updated hash map is the documented future improvement.

## How `train.c` picks CPU / GPU / distributed

One source file, three build modes, selected entirely by compile-time guards —
no runtime flags:

| Build | Forward/backward | Gradient sync |
|-------|------------------|---------------|
| `cmake` (default, Mac) | CPU `gpt_forward` / `gpt_backward` | none |
| `-DENABLE_CUDA=ON` | GPU `gpt_forward_cuda` / `gpt_backward_cuda` (Task 10) | none |
| `-DENABLE_MPI=ON`  | (CPU or GPU, per above) | `mpi_allreduce_mean` over `model.grads` |

- **`#ifdef USE_CUDA`** swaps the training loop to the CUDA forward/backward. Those
  functions re-upload `model.params` to the device each call and copy
  `model.loss` / `model.grads` back to the host, so the host-side AdamW step and
  the MPI averaging below work unchanged. (`gpt_backward_cuda` zeroes the device
  gradients itself, so there is no `gpt_zero_grad` on the GPU path.)
- **`#ifdef USE_MPI`** adds the per-rank data offset, the broadcast of rank 0's
  initial weights, and the `mpi_allreduce_mean(model.grads, …)` before the step.

CUDA and MPI compose: under `-DENABLE_CUDA=ON -DENABLE_MPI=ON` each rank runs the
GPU pass, then they average gradients on the host. On Colab the ranks share the
one T4 (valid, just not faster — see below).

## The benchmark caveat

`mpirun -np 2` will usually be **no faster** than one process here, because both
ranks share a single GPU and add MPI communication on top. That is expected and
not a bug. Data parallelism speeds things up when each rank owns its own
accelerator; on one shared T4 there is no extra hardware to spread the work over.
What the two-process run *does* demonstrate — and what the seminar is about — is
the **distributed mechanism**: gradients averaged across ranks every step so the
replicas stay bit-for-bit in sync (proven by `test_distributed`, Task 11).

## Commands (Colab)

```bash
bash colab/setup.sh
cmake -B build -DENABLE_CUDA=ON -DENABLE_MPI=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

# tokenizer trained on a 32 KB sample (BPE training is O(merges*length^2)),
# then the FULL corpus is tokenized (encoding is O(merges*length)). 256 -> vocab 512
head -c 32768 data/processed/skripta_clean.txt > data/processed/sample.txt
./build/tools/train_bpe data/processed/sample.txt        data/processed/tok.bpe    256
./build/tools/tokenize  data/processed/skripta_clean.txt data/processed/tok.bpe    data/processed/tokens.bin

# single-process GPU training
./build/src/train data/processed/tokens.bin 500 1e-3

# distributed (two ranks share the GPU; flags are the Colab-root + single-slot fix)
mpirun --allow-run-as-root --oversubscribe -np 2 ./build/src/train data/processed/tokens.bin 500 1e-3
```

## What to expect

The loss curve slopes down — that is the headline result and means the entire
hand-built stack (BPE, DataLoader, the layer math, the GPU kernels, AdamW, and
MPI) is correct end to end. The model will **not** produce coherent Serbian:
534K parameters on a few hundred KB of text is far too small. Coherence was never
the goal; building the system was. Text generation from a checkpoint is **Task 13**.
