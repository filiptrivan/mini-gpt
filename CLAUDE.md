# Mini GPT — Project Conventions

## What is this?

A small GPT-style transformer implemented from scratch in C with CUDA and MPI.
Seminar project for "Konkurentno i distribuirano programiranje" at FON Belgrade.
Training data: Serbian academic text (PDF textbook). MIT license.

## Build Commands

```bash
# Mac M2 (CPU only) — daily development
cmake -B build -DENABLE_CUDA=OFF -DENABLE_MPI=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Google Colab (CUDA + MPI) — GPU training
cmake -B build -DENABLE_CUDA=ON -DENABLE_MPI=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Run MPI tests (Colab only)
mpirun --allow-run-as-root -np 2 ./build/tests/test_distributed
```

## Development Workflow

1. **TDD**: Always write the test FIRST in `tests/test_*.c`, watch it fail, then implement
2. **Test dimensions**: Use tiny sizes in tests (d_model=4-8, seq_len=4, vocab=32) — fast and debuggable
3. **Numerical gradient check**: For every backward pass, verify analytical gradient matches finite-difference approximation
4. **CPU first**: All logic works on CPU (Mac). CUDA kernels are tested by comparing output to CPU reference
5. **Conditional compilation**: Use `#ifdef USE_CUDA` and `#ifdef USE_MPI` guards

## Code Style

- C99 standard
- Functions: `snake_case` (e.g., `matmul_forward`, `bpe_encode`)
- Types: `PascalCase` (e.g., `GPTConfig`, `BPETokenizer`, `DataLoader`)
- Constants: `UPPER_SNAKE_CASE`
- All memory: explicit `malloc`/`free`, no memory leaks
- Keep functions short and single-purpose
- Comments explain "why", not "what"

## Testing

- Framework: CMocka
- Test naming: `test_<component>_<behavior>` (e.g., `test_matmul_identity_matrix`)
- One test file per component: `tests/test_<component>.c`
- Shared helpers: `tests/helpers/test_utils.h`
- Test fixtures: `tests/helpers/fixtures/`
- CUDA tests: `tests/test_cuda_layers.cu` (Colab only)
- MPI tests: `tests/test_distributed.c` (run with `mpirun -np 2`)

## Project Structure

- `src/` — all source code, organized by component
- `src/model/layers.c` — CPU implementations of all neural network operations
- `src/model/gpt.c` — full GPT model (forward + backward)
- `src/cuda/` — CUDA kernel implementations (one `.cu` per operation)
- `src/distributed/` — MPI utilities
- `tests/` — all tests
- `tools/` — CLI utilities (preprocess, train_bpe, tokenize, generate)
- `colab/` — Google Colab setup and notebook
- `src/data/text_parser.c` — PDF text cleaner (removes page numbers, captions, bullets)
- `data/raw/` — PDF and raw extracted text (gitignored, never commit)
- `data/processed/` — tokenized binary files

## Model Architecture

Small GPT-2 style: 2 layers, 128 embed dim, 4 heads, 512 ff dim, 512 vocab size.
~270K parameters total. Trained on Serbian academic text (Latin script).

## Session Workflow

This project is built task-by-task across separate Claude sessions.
Each session tackles one task from the implementation plan.
Always run `ctest` after changes to verify nothing is broken.

## Important Notes

- Never commit files from `data/raw/` (PDF and raw text)
- The model will NOT produce coherent text — that's expected with 270K params
- Loss decreasing during training = success
- Keep explanations simple — this is a learning project
