# MPI Data Parallelism (Task 11)

How `src/distributed/mpi_utils.c` lets the **same** training program run across
several processes at once, each chewing through different data, while they all
keep one shared model perfectly in sync — and how we prove it's correct.

This is the **"distributed"** half of the seminar (the **"concurrent"** half was
the CUDA kernels). Like CUDA, it needs software the Mac doesn't ship, so it only
builds and runs on **Google Colab** (which has Open MPI). Your daily Mac build
leaves `ENABLE_MPI=OFF` and is completely unaffected — every MPI line in the
code is wrapped in `#ifdef USE_MPI`.

## First, in plain words

### What problem are we solving?

Training looks at a *batch* of text, measures how wrong the model is (the loss),
and nudges the weights to be a little less wrong. Bigger batches give a steadier
nudge — but one process can only do so much work per step. **Data parallelism**
is the simplest way to go bigger: run *N* copies of the program, give each copy
a *different* batch, and combine their results so it's as if one giant batch was
processed.

### What is MPI?

MPI (Message Passing Interface) is the standard toolkit for getting separate
*processes* — possibly on different machines — to cooperate. You launch them
together with `mpirun -np N ./program`, which starts **N copies** of the same
program. Each copy is a **rank** (rank 0, rank 1, …, like a process id within
the job). They don't share memory; they coordinate by *sending messages*. The
two messages we need:

- **Broadcast** — one rank's array is copied to everyone (`MPI_Bcast`).
- **All-reduce** — everyone's arrays are combined element-wise (sum, max, …) and
  the *combined result is handed back to every rank* (`MPI_Allreduce`). "All-"
  means everyone gets the answer, not just a leader.

### The whole strategy in one picture

```
                  rank 0                         rank 1
   ┌───────────────────────────┐  ┌───────────────────────────┐
   │ identical weights (bcast) │  │ identical weights (bcast) │   ← start in sync
   ├───────────────────────────┤  ├───────────────────────────┤
   │ batch A → fwd → bwd        │  │ batch B → fwd → bwd        │   ← different data
   │            ↓ grads_A       │  │            ↓ grads_B       │
   └───────────────┬───────────┘  └───────────┬───────────────┘
                   └───────  allreduce-mean  ──┘                   ← average grads
                   ┌───────────────────────────┐
            (grads_A + grads_B) / 2  on BOTH ranks                 ← identical grads
                   └───────────────────────────┘
   ┌───────────────────────────┐  ┌───────────────────────────┐
   │ adamw_step (same update)   │  │ adamw_step (same update)   │   ← identical step
   └───────────────────────────┘  └───────────────────────────┘
            still identical weights → repeat next step            ← stay in sync
```

Because both ranks start identical, receive the *same* averaged gradient, and
run the *same* optimizer, they apply the *same* update — so the model copies
never drift apart. The math is exactly equivalent to training on `batch A`
concatenated with `batch B` in one big step.

## Why averaging gradients is the right move

The loss over a big batch is the **average** of the per-example losses, so its
gradient is the average of the per-example gradients. Splitting the big batch
across ranks and averaging each rank's gradient reconstructs that same overall
gradient. We average (divide by `world_size`) rather than just sum so the
gradient magnitude — and therefore the learning rate you should use — doesn't
change when you add more ranks.

This leans entirely on the model's **single-block trick** (see `gpt.h`): every
weight gradient lives in one contiguous `float` array (`model->grads`), so
averaging the *whole* model is a single call:

```c
mpi_allreduce_mean(model->grads, model->num_params);
```

## The four functions (`src/distributed/mpi_utils.c`)

| Function | Wraps | Job |
|----------|-------|-----|
| `mpi_setup`         | `MPI_Init` + `Comm_rank`/`Comm_size` | start MPI, report this rank's id and the total count |
| `mpi_teardown`      | `MPI_Finalize`     | shut MPI down so `mpirun` exits cleanly |
| `mpi_allreduce_mean`| `MPI_Allreduce(SUM)`+ divide | average a float array across ranks, in place |
| `mpi_broadcast`     | `MPI_Bcast`        | copy one rank's array to everyone |

`mpi_allreduce_mean` uses **`MPI_IN_PLACE`**: instead of a separate send and
receive buffer, it tells MPI "use this one array as both" — every rank
contributes its `data` and gets the summed result back into the same `data`.
Then a single multiply by `1/world_size` turns the sum into the mean.

### The no-MPI stubs

The same file has an `#else` branch (compiled when `USE_MPI` is undefined) with
trivial single-process versions: `setup` reports "rank 0 of 1", `allreduce_mean`
and `broadcast` are no-ops (averaging one rank's data with itself, or copying to
nobody, changes nothing). This means `train.c` can call these functions without
its own `#ifdef`s around every line — though in practice CMake only compiles the
library at all when `ENABLE_MPI` is on.

## How `train.c` uses it

The training loop changes in four small, `#ifdef USE_MPI`-guarded ways:

1. **`mpi_setup`** at the top of `main` (before parsing args — MPI may strip its
   own launcher flags out of `argv`).
2. **`mpi_broadcast(model.params, …)`** right after `gpt_init`, so every rank
   trains from rank 0's exact starting weights.
3. **Per-rank data offset**: `loader->cursor = rank * B * T;` so each rank reads
   a different slice of the corpus (the loader treats the file as a cyclic
   stream, so any starting cursor is safe).
4. **`mpi_allreduce_mean(model.grads, …)`** after `gpt_backward` and before
   `adamw_step`, so all ranks step identically.

Plus: only rank 0 prints, so the log isn't duplicated `N` times, and
`mpi_teardown` runs at the very end.

## How we know it's correct

`tests/test_distributed.c` is launched with `mpirun -np 2` (the only test in the
project that runs under `mpirun`), so the test body executes on every rank and
checks that the ranks **agree**. Four checks:

1. **AllReduce computes the mean.** Hand-picked inputs whose average we know on
   paper: rank 0 holds `[1,2]`, rank 1 holds `[3,4]` → result `[2,3]` on both.
2. **Every rank gets the same result.** Average rank-dependent data, then verify
   the result is identical everywhere by reducing its per-rank `max` and `min`
   and asserting they coincide (if any rank disagreed, `max > min`).
3. **Broadcast shares one rank's data.** Rank 0 fills an array, others fill a
   sentinel; after the broadcast everyone holds rank 0's values.
4. **One step keeps params in sync** (the headline). Start identical models,
   feed each rank a *different* batch (so their local gradients genuinely
   differ — the test asserts this too), average the gradients, take one AdamW
   step, and confirm the parameters are *still* identical across ranks.

A note on cmocka + MPI: a failed assert does a `longjmp` out of the test, so the
test is written so every collective call (`Allreduce`/`Bcast`) happens **before**
any assert that depends on it. Because reductions return identical results on all
ranks, a failing assert fires on *all* ranks together — they never split up and
deadlock on a later collective.

## Building and running (Colab only)

```bash
cmake -B build -DENABLE_MPI=ON          # add -DENABLE_CUDA=ON for the GPU too
cmake --build build

# Run the distributed unit tests directly...
mpirun --allow-run-as-root --oversubscribe -np 2 ./build/tests/test_distributed

# ...or via ctest (registered to launch under mpirun automatically):
ctest --test-dir build --output-on-failure -R test_distributed

# Data-parallel training across 2 processes (binary is at build/src/train):
mpirun --allow-run-as-root --oversubscribe -np 2 ./build/src/train data/processed/tokens.bin 100
```

Two Colab-specific `mpirun` flags:

- `--allow-run-as-root` — Colab runs as root and Open MPI refuses that by default.
- `--oversubscribe` — a Colab VM reports only **one** allocatable "slot", so
  launching 2 ranks otherwise fails with *"not enough slots available."* This
  flag lets the ranks share the available core(s); fine for our tiny job. (The
  `ctest` registration in `tests/CMakeLists.txt` already includes both flags.)

On your Mac, leave `ENABLE_MPI=OFF`: the `mpi_utils` library and the
`test_distributed` test are skipped entirely and the CPU build is unchanged.

## What to expect

Single-process and two-process runs with equivalent total batch sizes should
produce **similar loss curves** — two processes isn't "twice as accurate," it's
"the same training with a bigger effective batch per step." On Colab the two
ranks share one GPU, so it won't be twice as *fast* either; the point is to
demonstrate the distributed mechanism (gradient averaging that keeps replicas in
sync), which is exactly how real multi-GPU training scales out.
```
