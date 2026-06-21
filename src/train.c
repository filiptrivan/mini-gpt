/*
 * train.c — the CPU training loop. This is the program that actually LEARNS:
 * it loads tokenized text, builds the model and optimizer, and repeats the
 * four-step training cycle until the loss comes down.
 *
 *   Usage:
 *     train <tokens.bin> [steps] [lr]
 *
 *       tokens.bin : a flat file of int32 token ids (the output of
 *                    tools/tokenize.c), every id in [0, vocab_size).
 *       steps      : how many training steps to run        (default 50)
 *       lr         : AdamW learning rate                   (default 1e-3)
 *
 * One training step is the whole project in miniature:
 *
 *     dataloader_next_batch   get the next chunk of tokens + their next-token labels
 *     gpt_zero_grad           clear last step's gradients (backward ACCUMULATES)
 *     gpt_forward             tokens -> loss      (how wrong are we on this batch?)
 *     gpt_backward            loss  -> gradients  (which way is downhill?)
 *     adamw_step              gradients -> weights (take a right-sized step)
 *
 * If the printed loss trends downward, the entire forward/backward/update
 * pipeline is correct — that is the "acid test" this task exists to pass.
 *
 * DISTRIBUTED (Task 11): when built with -DENABLE_MPI=ON and launched under
 * `mpirun -np N`, this same file runs DATA-PARALLEL across N processes ("ranks").
 * The MPI-specific lines are all guarded by #ifdef USE_MPI, so the Mac CPU build
 * compiles and runs exactly as before. What changes under MPI:
 *
 *   - rank 0 broadcasts its freshly-initialized weights so every rank starts
 *     from the SAME model;
 *   - each rank offsets the DataLoader by its rank, so the ranks read different
 *     slices of the corpus (different batches);
 *   - after backward, all ranks average their gradients (allreduce-mean) before
 *     the optimizer step, so every rank applies the identical update and the
 *     model copies stay in sync;
 *   - only rank 0 prints, so the log isn't duplicated N times.
 */

#include <stdio.h>
#include <stdlib.h>   /* atoi, atof, malloc, free */

#include "model/gpt.h"
#include "optimizer/adamw.h"
#include "data/dataloader.h"

#ifdef USE_MPI
#include "distributed/mpi_utils.h"  /* mpi_setup/teardown, allreduce_mean, broadcast */
#endif

#ifdef USE_CUDA
#include "cuda/gpt_cuda.cuh"  /* gpt_forward_cuda / gpt_backward_cuda (Task 10) */
#endif

/*
 * mpi_teardown_if_enabled — finalize MPI on an MPI build, do nothing otherwise.
 *
 * main() leaves through several early returns (bad args, load failure, out of
 * memory) plus the normal path, and every one must finalize MPI when built with
 * it. Wrapping the #ifdef once here lets each exit call a single plain function
 * instead of repeating the guard, and keeps the CPU build (which doesn't link
 * the MPI library) from referencing the mpi_teardown symbol at all.
 */
static void mpi_teardown_if_enabled(void) {
#ifdef USE_MPI
    mpi_teardown();
#endif
}

int main(int argc, char **argv) {
    /* Process identity. Without MPI there is one process: rank 0 of 1, so all
     * the rank-gated logic below collapses to "just do it" on the CPU build. */
    int rank = 0, world_size = 1;
#ifdef USE_MPI
    /* Must run before any other MPI call; may also strip mpirun's own flags
     * out of argc/argv, so it happens before we parse our arguments. */
    mpi_setup(&argc, &argv, &rank, &world_size);
#endif

    if (argc < 2 || argc > 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <tokens.bin> [steps] [lr]\n", argv[0]);
        }
        mpi_teardown_if_enabled();
        return 1;
    }
    const char *data_path = argv[1];
    int   steps = (argc >= 3) ? atoi(argv[2])         : 50;
    float lr    = (argc >= 4) ? (float)atof(argv[3])  : 1e-3f;
    if (steps <= 0) {
        if (rank == 0) {
            fprintf(stderr, "steps must be positive (got %d)\n", steps);
        }
        mpi_teardown_if_enabled();
        return 1;
    }

    /* Model shape: the seminar's real configuration (see PLAN.md / CLAUDE.md).
     * vocab_size MUST match the BPE tokenizer that produced tokens.bin. */
    GPTConfig cfg = {.max_seq_len = 64, .vocab_size = 512, .n_embd = 128,
                     .n_head = 4, .n_layer = 2, .ff_dim = 512};

    int B = 4;                  /* sequences per batch                       */
    int T = cfg.max_seq_len;    /* tokens per sequence (use the full window) */

    /* ---- Data ---- */
    /* dataloader_init validates that every token id is in [0, vocab_size) and
     * that the file holds at least one full batch (B*T + 1 tokens). */
    DataLoader *loader = dataloader_init(data_path, B, T, cfg.vocab_size);
    if (loader == NULL) {
        if (rank == 0) {
            fprintf(stderr,
                    "Cannot load %s: need >= %d tokens, all ids in [0, %d).\n",
                    data_path, B * T + 1, cfg.vocab_size);
        }
        mpi_teardown_if_enabled();
        return 1;
    }

#ifdef USE_MPI
    /* Per-rank data offset: start each rank's cursor a different B*T-sized
     * window into the corpus so the ranks read DIFFERENT batches (that is what
     * makes this data-parallel rather than N copies of the same work). The
     * loader treats the corpus as a cyclic stream and wraps to 0 when a batch
     * no longer fits, so any starting cursor is safe. cursor is a documented
     * public field of DataLoader (see dataloader.h). */
    loader->cursor = (size_t)rank * (size_t)B * (size_t)T;
#endif

    /* ---- Model ---- */
    GPTModel model;
    gpt_init(&model, cfg, /*seed=*/1337u);

#ifdef USE_MPI
    /* Make every rank start from rank 0's exact weights. With one shared seed
     * the random init already matches, but broadcasting removes any dependence
     * on that coincidence (e.g. if init ever became rank-aware) — this is the
     * canonical "all ranks begin identical" step. */
    mpi_broadcast(model.params, model.num_params, /*root=*/0);
#endif

    if (rank == 0) {
        /* Which compute path this binary was built with — makes the 1-vs-2 and
         * CPU-vs-GPU benchmark logs self-explanatory. */
#ifdef USE_CUDA
        const char *device = "CUDA (GPU)";
#else
        const char *device = "CPU";
#endif
        /* Only the leader prints, so the header appears once, not once per rank.
         * world_size and the effective batch (B*T per rank x ranks) make it
         * clear how much data each step actually consumes under MPI. */
        printf("Model: %d parameters | %s | batch %dx%d x %d rank(s) "
               "(effective %d tok/step) | %d steps | lr %g\n",
               model.num_params, device, B, T, world_size,
               B * T * world_size, steps, lr);
    }

    /* ---- Optimizer ---- */
    /* GPT-2 default betas; a small weight decay to keep weights tame. */
    AdamW opt;
    adamw_init(&opt, model.num_params, lr,
               /*beta1=*/0.9f, /*beta2=*/0.999f,
               /*eps=*/1e-8f, /*weight_decay=*/0.01f);

    /* Reused batch buffers — allocate once, refill every step (the loader
     * writes into these; it never allocates per batch). */
    int *inputs  = malloc((size_t)B * T * sizeof(int));
    int *targets = malloc((size_t)B * T * sizeof(int));
    if (inputs == NULL || targets == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Out of memory allocating batch buffers\n");
        }
        free(inputs); free(targets);
        adamw_free(&opt); gpt_free(&model); dataloader_free(loader);
        mpi_teardown_if_enabled();
        return 1;
    }

    /* ---- Training loop ---- */
    float first_loss = 0.0f;
    for (int step = 0; step < steps; step++) {
        dataloader_next_batch(loader, inputs, targets);

#ifdef USE_CUDA
        /* GPU path: forward + backward run as CUDA kernels with the tensors kept
         * resident on the device between kernels (see src/cuda/gpt_cuda.cu).
         * gpt_backward_cuda zeroes the device gradients itself, so there is NO
         * gpt_zero_grad here. Both calls re-upload model.params and copy
         * model.loss / model.grads back to the host, so the host-side MPI
         * averaging and adamw_step below operate on fresh values unchanged. */
        gpt_forward_cuda(&model, inputs, targets, B, T);
        gpt_backward_cuda(&model, inputs, targets, B, T);
#else
        gpt_zero_grad(&model);
        gpt_forward(&model, inputs, targets, B, T);
        gpt_backward(&model, inputs, targets, B, T);
#endif

#ifdef USE_MPI
        /* The data-parallel core: average every rank's gradients before the
         * step. Because the averaged gradient is identical on all ranks and the
         * weights started identical, adamw_step applies the SAME update on every
         * rank — the model copies never drift apart. Equivalent to training on
         * the concatenation of all ranks' batches at once. */
        mpi_allreduce_mean(model.grads, model.num_params);
#endif
        adamw_step(&opt, model.params, model.grads);

        if (step == 0) first_loss = model.loss;
        /* Only rank 0 prints the per-step loss so logs aren't duplicated. Each
         * rank's loss is for its own batch; rank 0's is a representative sample
         * (the gradients — not the losses — are what get averaged). */
        if (rank == 0) {
            printf("step %4d | loss %.4f\n", step, model.loss);
        }
    }

    /* The headline result: did the loss come down? model.loss still holds the
     * final batch's loss — nothing overwrites it after the last forward. */
    if (rank == 0) {
        printf("\nloss: %.4f -> %.4f  (%s)\n",
               first_loss, model.loss,
               model.loss < first_loss ? "DECREASED" : "did NOT decrease");
    }

    /* ---- Cleanup ---- */
    free(inputs);
    free(targets);
    adamw_free(&opt);
    gpt_free(&model);
    dataloader_free(loader);
    /* Last MPI call: release MPI's resources and let mpirun exit cleanly. */
    mpi_teardown_if_enabled();
    return 0;
}
