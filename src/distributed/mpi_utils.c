/*
 * mpi_utils.c — implementation of the small MPI data-parallel layer.
 * See distributed/mpi_utils.h for the contract and the big-picture overview,
 * and docs/mpi-distributed.md for a walked-by-hand example.
 *
 * Everything that actually talks to MPI lives behind #ifdef USE_MPI. When MPI
 * is disabled (the Mac CPU build, where this file is not even compiled by the
 * default CMake config) the #else branch provides single-process stubs so the
 * functions still have a valid definition: one rank, nothing to communicate.
 * Keeping both halves here means train.c calls these functions unconditionally
 * and never sprinkles its own #ifdefs around every line.
 */

#include "distributed/mpi_utils.h"

#ifdef USE_MPI

#include <mpi.h>  /* MPI_Init, MPI_Comm_rank/size, MPI_Allreduce, MPI_Bcast, ... */

/*
 * mpi_setup — MPI_Init, then ask MPI who we are and how many of us there are.
 * See header for the parameter contract.
 */
void mpi_setup(int *argc, char ***argv, int *rank_out, int *world_size_out) {
    /* MPI_Init must run before any other MPI call. It wires up the process into
     * the job's communication world and may strip its own launcher arguments
     * out of argv, which is why we hand it the addresses of argc/argv. */
    MPI_Init(argc, argv);

    int rank = 0, world_size = 1;
    /* MPI_COMM_WORLD is the default communicator containing every rank in the
     * job. Comm_rank = this process's id; Comm_size = how many processes. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Report back only what the caller asked for (NULL means "don't care"). */
    if (rank_out)       *rank_out       = rank;
    if (world_size_out) *world_size_out = world_size;
}

/*
 * mpi_teardown — MPI_Finalize. Releases MPI's internal resources and lets
 * mpirun exit cleanly. After this no MPI call is legal.
 */
void mpi_teardown(void) {
    MPI_Finalize();
}

/*
 * mpi_allreduce_mean — sum `data` across all ranks, then divide by the rank
 * count so each element becomes the cross-rank average. See header.
 */
void mpi_allreduce_mean(float *data, int count) {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* MPI_IN_PLACE tells MPI to use `data` as BOTH the send and the receive
     * buffer: every rank contributes its current `data` and receives the
     * element-wise MPI_SUM back into the same array. MPI_Allreduce (vs the
     * plain MPI_Reduce) delivers the combined result to EVERY rank, not just a
     * root — exactly what we need so all ranks step the optimizer identically.
     * MPI guarantees all ranks get a bit-identical result here. */
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    /* Sum -> mean. Dividing by world_size keeps the averaged gradient on the
     * same scale regardless of how many processes ran, so the learning rate
     * doesn't have to change with -np. Multiply by the reciprocal once instead
     * of dividing in the loop (a divide is far costlier than a multiply). */
    const float inv = 1.0f / (float)world_size;
    for (int i = 0; i < count; i++) {
        data[i] *= inv;
    }
}

/*
 * mpi_broadcast — copy rank `root`'s `data` to every other rank. See header.
 */
void mpi_broadcast(float *data, int count, int root) {
    /* MPI_Bcast: on the root rank `data` is the source; on every other rank it
     * is overwritten with the root's bytes. One call synchronizes the array
     * across the whole world. */
    MPI_Bcast(data, count, MPI_FLOAT, root, MPI_COMM_WORLD);
}

#else  /* !USE_MPI — single-process stubs so the API exists without MPI */

#include <stddef.h>  /* NULL */

/* With no MPI there is exactly one process: it is rank 0 of a world of 1. */
void mpi_setup(int *argc, char ***argv, int *rank_out, int *world_size_out) {
    (void)argc;  /* (void)x silences "unused parameter" warnings — there is */
    (void)argv;  /* genuinely nothing to initialize in a single process.    */
    if (rank_out)       *rank_out       = 0;
    if (world_size_out) *world_size_out = 1;
}

void mpi_teardown(void) {
    /* Nothing was initialized, so nothing to finalize. */
}

void mpi_allreduce_mean(float *data, int count) {
    /* The average of a single rank's data with itself is the data unchanged. */
    (void)data;
    (void)count;
}

void mpi_broadcast(float *data, int count, int root) {
    /* With one rank there is no other process to copy to. */
    (void)data;
    (void)count;
    (void)root;
}

#endif /* USE_MPI */
