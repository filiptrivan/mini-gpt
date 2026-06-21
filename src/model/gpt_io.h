#ifndef GPT_IO_H
#define GPT_IO_H

/*
 * gpt_io.h — save and load a trained GPT model to/from a binary checkpoint file.
 *
 * Include guard above (#ifndef / #define / #endif): if two .c files both include
 * this header, the guard makes the second include a no-op so the declarations
 * below are not seen twice (which would be a compile error).
 *
 *
 * WHY THIS EXISTS — the bridge from training to generation:
 *
 *   train.c runs the loop that makes the weights good, but those weights live in
 *   RAM and vanish when the process exits. To actually USE the model later
 *   (tools/generate.c), training has to write the weights to disk and generation
 *   has to read them back. That is exactly what gpt_save / gpt_load do — the same
 *   role bpe_save / bpe_load play for the tokenizer (see tokenizer/bpe_io.h),
 *   and they follow the same simple binary-file conventions.
 *
 *
 * FILE FORMAT (all little-endian, the only platforms we target — Mac M2, Linux):
 *
 *   header : 9 × int32
 *       [0] magic        — 0x00545047 ("GPT\0"), identifies the file type
 *       [1] version      — format version (1)
 *       [2] max_seq_len  \
 *       [3] vocab_size    |
 *       [4] n_embd        | the GPTConfig — everything needed to rebuild the
 *       [5] n_head        | model's shape before reading the weights
 *       [6] n_layer       |
 *       [7] ff_dim       /
 *       [8] num_params   — the float count that follows (a redundancy check:
 *                          it must equal the count derived from the config)
 *   body   : num_params × float32 — the contiguous parameter block, in the
 *            exact order param_layout lays it out (gpt.c). Because the layout is
 *            a pure function of the config, the reader reproduces it from the
 *            header and the floats drop straight back into place.
 *
 * The activations and gradients are NOT saved: they are scratch that the forward
 * and backward passes rebuild from the weights, so only `params` is persistent.
 */

#include "model/gpt.h"  /* GPTModel */

/*
 * gpt_save — write a model's weights to a binary checkpoint file.
 *
 *   model    : an initialized model (from gpt_init / gpt_load) to serialize.
 *              Only its config and params block are written; read-only here.
 *   filepath : NUL-terminated destination path (overwritten if it exists).
 *
 * Returns: 0 on success, -1 on any failure (cannot open the file, or a short
 * write). On failure the file may be partially written — callers that care can
 * delete it, but training simply reports the error.
 *
 * Preconditions: model and filepath non-NULL, model initialized.
 */
int gpt_save(const GPTModel *model, const char *filepath);

/*
 * gpt_load — rebuild a model from a checkpoint written by gpt_save.
 *
 *   model    : caller-owned struct to populate (pass &model). On success it is a
 *              fully-initialized GPTModel — identical to one from gpt_init except
 *              the weights are the saved ones instead of random. Release it with
 *              gpt_free, exactly as you would an gpt_init'd model.
 *   filepath : NUL-terminated path to a file produced by gpt_save.
 *
 * Reads the header, validates the magic/version and that the config is sane and
 * self-consistent, reconstructs the model layout via gpt_init (so all the tensor
 * views point at the right offsets), then overwrites the weights with the file's
 * floats. The gradients are left zeroed (generation never uses them).
 *
 * Returns: 0 on success, -1 on any failure (cannot open, bad magic/version,
 * corrupt/invalid config, num_params mismatch, or a short read). On failure the
 * model is NOT left allocated — there is nothing for the caller to free.
 *
 * Preconditions: model and filepath non-NULL.
 */
int gpt_load(GPTModel *model, const char *filepath);

#endif /* GPT_IO_H */
