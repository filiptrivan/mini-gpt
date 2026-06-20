# CUDA Kernels — MatMul + LayerNorm (Task 9)

How `src/cuda/matmul.cu` and `src/cuda/layernorm.cu` run the same math as the
CPU layers, but on the GPU — and how we prove they're correct.

So far everything has run on the CPU (your Mac). This is the first task where
the **"concurrent"** half of the seminar shows up: we write code that thousands
of GPU threads run at the same time. The catch is that a Mac M2 has no NVIDIA
GPU, so this code only compiles and runs on **Google Colab** (a free Tesla T4).

## First, in plain words

### What a GPU is good at

A CPU has a handful of fast cores. A GPU has thousands of slow ones. It wins
when you have the *same* simple operation to do over a huge pile of data — like
"multiply these two matrices" or "normalize these 1000 rows." You hand each
little piece of the work to its own **thread**, and they all run together.

CUDA organizes threads into a **grid of blocks**, and each **block** holds many
threads. The two things that make GPU code fast (or slow):

- **Threads in the same block can share a tiny, very fast scratchpad** called
  *shared memory*. Reading global GPU memory (DRAM) is slow; shared memory is
  ~100× faster. The whole game is "load once from slow memory into shared, then
  reuse it many times."
- **Threads in a block can synchronize** with `__syncthreads()` — a barrier
  where every thread waits until all of them arrive. You need this whenever one
  thread reads a value another thread just wrote.

### How we know the GPU code is right

We already wrote and *proved* the CPU versions correct in Tasks 5–6 (the
numerical gradient checks). So for the GPU we don't re-derive the math. We just
ask one question:

> Given the same input, does the GPU produce the same answer as the CPU?

That's the entire `tests/test_cuda_layers.cu` strategy: run the CPU function
(the **oracle**), run the CUDA kernel, compare every element. They won't be
*bit*-identical — floating-point addition isn't associative, and the GPU sums
things in a different order — so we allow a small tolerance (`1e-4`).

## MatMul: shared-memory tiling

`out = a @ b`. To compute one output cell `out[i][j]` you need row `i` of `a`
and column `j` of `b`. A naive GPU kernel would re-read those rows/columns from
slow global memory for *every* output cell — the same numbers pulled from DRAM
over and over. That makes the kernel memory-bound and wastes the GPU.

**Tiling** fixes it. We chop the matrices into 16×16 **tiles**. A block of
16×16 = 256 threads:

1. Cooperatively copies one 16×16 tile of `a` and one of `b` into shared memory
   (each thread loads exactly one element of each).
2. `__syncthreads()` so the tiles are fully loaded before anyone reads them.
3. Each thread multiplies its row-slice by its column-slice *out of fast shared
   memory* and adds to a running total.
4. Slides to the next pair of tiles along the shared `K` dimension and repeats.

Each value of `a` and `b` is now read from slow memory **once per tile** instead
of once per output cell. That's the whole speedup.

```
        b (K×N), tiled
        ┌────┬────┬────┐
        │ B0 │ B1 │ .. │     a block walks one tile-row of `a`
        ├────┼────┼────┤     and one tile-column of `b`, summing
a (M×K) │ B0'│    │    │     partial products as it slides along K.
┌────┐  └────┴────┴────┘
│ A0 │ A0'..   the 256 threads of one block fill A0/B0 into shared
├────┤         memory, multiply, then move to A1/B1, A2/B2, ...
│ .. │
└────┘
```

**Why `TILE = 16`:** a block is then 256 threads (good occupancy), and the two
shared tiles cost `16*16*4*2 = 2 KB` — comfortably inside every GPU's per-block
shared-memory budget.

**Boundary handling:** the matrix dimensions don't have to be multiples of 16.
Threads that fall off the edge still take part in the cooperative loads (so
`__syncthreads()` stays uniform — if some threads skipped the barrier the kernel
would hang), but they load `0.0f` and skip the final write. Padding with zeros
is safe because `0 × anything = 0` adds nothing to the real sums. The test
deliberately uses ugly sizes like `33×17×40` to exercise exactly this.

## LayerNorm: one block per row, parallel reduction

LayerNorm normalizes each row independently:

```
mean = average(row)
var  = average((row - mean)²)
out  = gamma * (row - mean) / sqrt(var + 1e-5) + beta
```

Computing a mean and a variance is a **reduction**: combine `C` numbers into
one. We give **one block to each row**, and the threads in that block work
together to reduce the row.

The reduction is a **tree**: with 128 threads holding 128 partial sums, step 1
adds the upper 64 into the lower 64, step 2 adds 32 into 32, then 16, 8, 4, 2,
1 — `log₂(128) = 7` steps instead of 128. After each step a `__syncthreads()`
makes sure the writes landed before the next step reads them.

```
[ p0 p1 p2 p3 p4 p5 p6 p7 ]   8 partial sums in shared memory
   +  +  +  +   ← add upper half into lower half
[ s0 s1 s2 s3 ]               (each = p_i + p_{i+4})
   +  +                       ← again
[ t0 t1 ]
   +                          ← again
[ total ]                     sdata[0] now holds the row's sum
```

Two such reductions run per row: one for the sum (→ mean), one for the sum of
squared deviations (→ variance).

**Two subtleties the code handles:**

- **The tree reduction needs a power-of-two number of threads** (it keeps
  halving). So the wrapper picks the largest power of two ≤ `min(C, 256)`.
  Examples: `C=128 → 128` threads, `C=50 → 32`, `C=6 → 4`.
- **A row can be longer than the block.** If `C = 200` but we launched 128
  threads, each thread first sums a *strided* slice of the row
  (`for (c = tid; c < C; c += nthreads)`) into its private partial, and only
  then do the partials get tree-reduced. The test's `C=50` case exercises this.

We use `1.0f / sqrtf(...)` (not the faster `rsqrtf`) specifically to match the
CPU's formula as closely as possible, so the comparison stays within tolerance.

## The host wrappers

Each kernel has a host wrapper (`matmul_forward_cuda`, `layernorm_forward_cuda`)
with the **same signature as its CPU twin** and taking plain CPU pointers. The
wrapper does the full round trip: `cudaMalloc` device buffers → copy inputs
host→device → launch the kernel → copy the result device→host → free.

This per-call allocate-and-copy is deliberately simple and a bit wasteful — but
it's all we need to *prove the kernels are correct*. In Task 10, when we wire up
the real GPU training loop, tensors will stay resident on the GPU across many
kernels instead of bouncing back and forth every call.

Every CUDA call is wrapped in the `CUDA_CHECK` macro (in `cuda_layers.cuh`),
which aborts with a file/line and a readable message if anything fails — far
better than silently computing garbage.

## Building and running (Colab only)

```bash
cmake -B build -DENABLE_CUDA=ON -DENABLE_MPI=OFF
cmake --build build
ctest --test-dir build --output-on-failure -R test_cuda_layers
```

On your Mac, leave `ENABLE_CUDA=OFF` — the `cuda_layers` library and the
`test_cuda_layers` test are skipped entirely, and the CPU build is unaffected.
The default GPU architecture is `sm_75` (Colab's T4); override with
`-DCMAKE_CUDA_ARCHITECTURES=80` for an A100, etc.
