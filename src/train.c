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
 * (CUDA and MPI come in later tasks; this build is pure CPU.)
 */

#include <stdio.h>
#include <stdlib.h>   /* atoi, atof, malloc, free */

#include "model/gpt.h"
#include "optimizer/adamw.h"
#include "data/dataloader.h"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s <tokens.bin> [steps] [lr]\n", argv[0]);
        return 1;
    }
    const char *data_path = argv[1];
    int   steps = (argc >= 3) ? atoi(argv[2])         : 50;
    float lr    = (argc >= 4) ? (float)atof(argv[3])  : 1e-3f;
    if (steps <= 0) {
        fprintf(stderr, "steps must be positive (got %d)\n", steps);
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
        fprintf(stderr,
                "Cannot load %s: need >= %d tokens, all ids in [0, %d).\n",
                data_path, B * T + 1, cfg.vocab_size);
        return 1;
    }

    /* ---- Model ---- */
    GPTModel model;
    gpt_init(&model, cfg, /*seed=*/1337u);
    printf("Model: %d parameters | batch %dx%d | %d steps | lr %g\n",
           model.num_params, B, T, steps, lr);

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
        fprintf(stderr, "Out of memory allocating batch buffers\n");
        free(inputs); free(targets);
        adamw_free(&opt); gpt_free(&model); dataloader_free(loader);
        return 1;
    }

    /* ---- Training loop ---- */
    float first_loss = 0.0f;
    for (int step = 0; step < steps; step++) {
        dataloader_next_batch(loader, inputs, targets);

        gpt_zero_grad(&model);
        gpt_forward(&model, inputs, targets, B, T);
        gpt_backward(&model, inputs, targets, B, T);
        adamw_step(&opt, model.params, model.grads);

        if (step == 0) first_loss = model.loss;
        printf("step %4d | loss %.4f\n", step, model.loss);
    }

    /* The headline result: did the loss come down? model.loss still holds the
     * final batch's loss — nothing overwrites it after the last forward. */
    printf("\nloss: %.4f -> %.4f  (%s)\n",
           first_loss, model.loss,
           model.loss < first_loss ? "DECREASED" : "did NOT decrease");

    /* ---- Cleanup ---- */
    free(inputs);
    free(targets);
    adamw_free(&opt);
    gpt_free(&model);
    dataloader_free(loader);
    return 0;
}
