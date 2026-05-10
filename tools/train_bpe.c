/* train_bpe — CLI: read a text file, train a BPE tokenizer, save it to disk.
 *
 * Usage:
 *   train_bpe <input.txt> <output.bpe> <num_merges>
 *
 * The input file is read in raw binary mode (no character decoding) so that
 * UTF-8 bytes — including Serbian Latin diacritics — pass through intact.
 * num_merges is how many merge rules to learn; final vocab_size will be
 * 256 + num_merges (or fewer, if the corpus is too small).
 */

#include <stdio.h>
#include <stdlib.h>

#include "tokenizer/bpe.h"
#include "tokenizer/bpe_io.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.txt> <output.bpe> <num_merges>\n",
                argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    int num_merges = atoi(argv[3]);

    if (num_merges <= 0) {
        fprintf(stderr, "num_merges must be a positive integer (got '%s')\n",
                argv[3]);
        return 1;
    }

    /* ---- Read the entire input file into memory ----
     * Strategy: open binary, seek to end to learn the size, allocate, seek
     * back, fread the whole thing. Works for files that comfortably fit in
     * RAM, which is fine for our seminar corpus (a few hundred KB at most). */
    FILE *f = fopen(input_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Cannot open input file: %s\n", input_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (length <= 0) {
        fprintf(stderr, "Input file is empty or unreadable: %s\n", input_path);
        fclose(f);
        return 1;
    }

    unsigned char *text = malloc((size_t)length);
    size_t got = fread(text, 1, (size_t)length, f);
    fclose(f);
    if (got != (size_t)length) {
        fprintf(stderr, "Short read on %s (got %zu of %ld bytes)\n",
                input_path, got, length);
        free(text);
        return 1;
    }

    /* ---- Train ---- */
    BPETokenizer *tok = bpe_create();
    printf("Training BPE on %ld bytes, target %d merges...\n",
           length, num_merges);
    bpe_train(tok, text, (size_t)length, num_merges);
    printf("Trained: vocab_size=%d, num_merges=%d\n",
           tok->vocab_size, tok->num_merges);

    /* ---- Save ---- */
    if (bpe_save(tok, output_path) != 0) {
        fprintf(stderr, "Failed to save tokenizer to %s\n", output_path);
        bpe_free(tok);
        free(text);
        return 1;
    }
    printf("Saved tokenizer to %s\n", output_path);

    bpe_free(tok);
    free(text);
    return 0;
}
