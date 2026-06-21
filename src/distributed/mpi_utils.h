#ifndef MPI_UTILS_H
#define MPI_UTILS_H

/*
 * mpi_utils.h — the tiny MPI layer that turns single-process training into
 * DATA-PARALLEL training across several processes ("ranks").
 *
 * Include guard above (#ifndef / #define / #endif): if two .c files both
 * include this header, the guard makes the second include a no-op so the
 * declarations below are not seen twice (which would be a compile error).
 *
 *
 * WHAT "DISTRIBUTED" MEANS HERE — data parallelism in one picture:
 *
 *   We launch the SAME program N times (e.g. `mpirun -np 2 ./train ...`).
 *   Each copy is a "rank" (rank 0, rank 1, ...). Every rank holds a FULL copy
 *   of the model and runs the whole forward/backward pass — but each rank
 *   feeds the model a DIFFERENT slice of the data. After backward, every rank
 *   has its own local gradients (computed from its own batch). One MPI step
 *   then AVERAGES those gradients across all ranks, so every rank applies the
 *   exact same averaged update and the model copies stay byte-for-byte in sync.
 *
 *       rank 0:  batch A  -> forward -> backward -> grads_0  \
 *                                                              >- average -> identical update
 *       rank 1:  batch B  -> forward -> backward -> grads_1  /
 *
 *   Net effect: we processed (A and B) in one step instead of one batch — the
 *   effective batch size scales with the number of ranks. This is exactly how
 *   real multi-GPU training works; here it is the "distributed" deliverable of
 *   the seminar (the "concurrent" half being the CUDA kernels).
 *
 *
 * WHY THE SINGLE-BLOCK TRICK MATTERS (see gpt.h): because every weight lives in
 * one contiguous float array (model->params) and every gradient in another
 * (model->grads), syncing the whole model is ONE call over a flat array — no
 * per-tensor bookkeeping. mpi_allreduce_mean(model->grads, model->num_params)
 * averages the entire model's gradients in a single line.
 *
 *
 * CONDITIONAL COMPILATION: the real bodies (in mpi_utils.c) are guarded by
 * #ifdef USE_MPI. When MPI is disabled (the Mac CPU build), the same functions
 * compile as harmless single-process stubs — rank 0 of 1, with "average across
 * one rank" being a no-op — so callers never need their own #ifdefs around
 * every call. The CMake build only links this library when ENABLE_MPI is set,
 * which also defines USE_MPI.
 */

/*
 * mpi_setup — initialize MPI and report this process's identity.
 *
 * Call ONCE at the very start of main, before any other MPI use. MPI needs to
 * see the program's argc/argv (it may consume its own launcher flags), so we
 * pass their addresses through.
 *
 *   argc           : address of main's argc (pass &argc). MPI may modify it.
 *   argv           : address of main's argv (pass &argv). MPI may modify it.
 *   rank_out       : written with THIS process's rank, an int in
 *                    [0, world_size). Rank 0 is conventionally the "leader"
 *                    (does the printing, owns the canonical initial weights).
 *                    May be NULL if the caller doesn't need it.
 *   world_size_out : written with the total number of ranks launched
 *                    (the N in `-np N`). May be NULL.
 *
 * In the no-MPI build this reports rank 0 of world_size 1 and touches nothing
 * else, so a CPU run behaves exactly like a single distributed rank.
 */
void mpi_setup(int *argc, char ***argv, int *rank_out, int *world_size_out);

/*
 * mpi_teardown — shut MPI down cleanly.
 *
 * Call ONCE at the very end of main, after all MPI calls and ideally after
 * freeing your buffers. Skipping it can leave the launcher (mpirun) hanging or
 * reporting an abnormal exit. No-op in the no-MPI build.
 */
void mpi_teardown(void);

/*
 * mpi_allreduce_mean — average a float array element-wise across ALL ranks,
 *                      in place, leaving every rank with the same result.
 *
 * This is the heart of data-parallel training: call it on the gradient block
 * after backward so every rank steps the optimizer with the SAME averaged
 * gradient and the model copies never drift apart.
 *
 *   data  : the array to average, length `count`. On return, data[i] holds the
 *           mean of every rank's original data[i]. Both input and output —
 *           the old local values are overwritten.
 *   count : number of floats in `data` (e.g. model->num_params).
 *
 * Implementation: an MPI_Allreduce with MPI_SUM (so all ranks receive the
 * element-wise sum), then a divide by world_size to turn the sum into a mean.
 * "All-reduce" = reduce + broadcast in one: the combined result is delivered
 * back to every rank, not just to a root. The result is identical on all ranks.
 *
 * In the no-MPI build this is a no-op (the mean of one rank's data is itself).
 */
void mpi_allreduce_mean(float *data, int count);

/*
 * mpi_broadcast — copy one rank's array to every other rank, in place.
 *
 * Used once right after gpt_init so every rank starts training from the SAME
 * weights: rank `root` initialized them, this overwrites all other ranks'
 * weights with rank root's copy. Without it, each rank would random-init
 * differently (or identically only by luck) and the "stay in sync" guarantee
 * would never hold.
 *
 *   data  : the array to share, length `count`. On rank `root` it is the
 *           source (read, unchanged); on every other rank it is the
 *           destination (overwritten with root's values).
 *   count : number of floats in `data`.
 *   root  : the rank whose copy wins (usually 0).
 *
 * No-op in the no-MPI build (with one rank there is nobody to copy to).
 */
void mpi_broadcast(float *data, int count, int root);

#endif /* MPI_UTILS_H */
