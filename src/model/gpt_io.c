/*
 * gpt_io.c — model checkpoint save/load. See model/gpt_io.h for the file format
 * and the contract; this mirrors tokenizer/bpe_io.c for the tokenizer.
 *
 * The whole job is small because the model already keeps every weight in one
 * contiguous block (model->params) whose layout is a pure function of the config
 * (param_layout in gpt.c). So a checkpoint is just "the config, then the float
 * block": on load we rebuild the layout from the config via gpt_init and read
 * the floats straight back into place.
 */

#include "model/gpt_io.h"

#include <stdio.h>   /* fopen, fread, fwrite, fclose */

/* Magic number identifying a GPT checkpoint: the four bytes 'G','P','T',0 read
 * as a little-endian int32 = 0x00545047. Stored in the file's native byte order;
 * we only target little-endian platforms (Mac M2, Linux x86_64). */
#define GPT_MAGIC   0x00545047
#define GPT_VERSION 1

/* Number of int32 fields in the header: magic, version, the 6 GPTConfig
 * dimensions, and num_params. Kept as a named constant so the fwrite/fread
 * counts and the index math below can't silently drift apart. */
#define GPT_HEADER_INTS 9

/*
 * gpt_save — see header for the contract and file layout.
 *
 * We track success in `rc` and reach a single fclose at the end on every exit
 * path — the same simple cleanup style as bpe_save (no goto needed).
 */
int gpt_save(const GPTModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (f == NULL) return -1;

    int rc = 0;

    /* Build the header in a local array, then write it in one call. Field order
     * must match the documented format (and gpt_load's reader below). */
    const GPTConfig *c = &model->config;
    int header[GPT_HEADER_INTS] = {
        GPT_MAGIC, GPT_VERSION,
        c->max_seq_len, c->vocab_size, c->n_embd,
        c->n_head, c->n_layer, c->ff_dim,
        model->num_params
    };
    if (fwrite(header, sizeof(int), GPT_HEADER_INTS, f) != GPT_HEADER_INTS) {
        rc = -1;
    }

    /* Write the parameter block: num_params float32 in param_layout order. */
    if (rc == 0) {
        size_t n = (size_t)model->num_params;
        if (fwrite(model->params, sizeof(float), n, f) != n) rc = -1;
    }

    fclose(f);
    return rc;
}

/*
 * gpt_load — see header for the contract.
 *
 * Order matters: we validate the config BEFORE calling gpt_init, because
 * gpt_init asserts on a malformed config (e.g. n_embd not divisible by n_head)
 * and we want a corrupt file to return -1, not abort the program.
 */
int gpt_load(GPTModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (f == NULL) return -1;

    int header[GPT_HEADER_INTS];
    if (fread(header, sizeof(int), GPT_HEADER_INTS, f) != GPT_HEADER_INTS) {
        fclose(f);
        return -1;
    }

    /* Magic + version gate: reject anything that isn't our format. */
    if (header[0] != GPT_MAGIC || header[1] != GPT_VERSION) {
        fclose(f);
        return -1;
    }

    GPTConfig cfg = {
        .max_seq_len = header[2], .vocab_size = header[3], .n_embd = header[4],
        .n_head      = header[5], .n_layer    = header[6], .ff_dim = header[7],
    };
    int saved_num_params = header[8];

    /* Validate the config ourselves so gpt_init's asserts never fire on a
     * corrupt file. These are the same invariants gpt_num_params requires:
     * every dimension positive and n_embd evenly divisible into n_head heads. */
    if (cfg.max_seq_len <= 0 || cfg.vocab_size <= 0 || cfg.n_embd <= 0 ||
        cfg.n_head <= 0 || cfg.n_layer <= 0 || cfg.ff_dim <= 0 ||
        cfg.n_embd % cfg.n_head != 0) {
        fclose(f);
        return -1;
    }

    /* Rebuild the model layout from the config (allocates params/grads, points
     * the tensor views at the right offsets, seeds random weights we are about
     * to overwrite). seed is irrelevant — the file's weights replace them. */
    gpt_init(model, cfg, /*seed=*/1u);

    /* The float count the file claims must match what this config actually
     * needs; if not, the file is corrupt or from a different build. */
    if (model->num_params != saved_num_params) {
        gpt_free(model);
        fclose(f);
        return -1;
    }

    /* Overwrite the random init with the saved weights. */
    size_t n = (size_t)model->num_params;
    if (fread(model->params, sizeof(float), n, f) != n) {
        gpt_free(model);   /* leave nothing allocated on failure */
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}
