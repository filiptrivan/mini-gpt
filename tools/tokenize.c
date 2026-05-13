/* tokenize — CLI: encode a text file into a binary token file using a
 * previously-trained BPE tokenizer.
 *
 * Usage:
 *   tokenize <input.txt> <tokenizer.bpe> <output.bin>
 *
 * The output file is a flat array of int32 token ids in native endian.
 * Length is derivable from the file size: file_bytes / sizeof(int).
 * The model's DataLoader (Task 4) reads this file directly into memory.
 */

#include <stdio.h>
#include <stdlib.h>

#include "tokenizer/bpe.h"
#include "tokenizer/bpe_io.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.txt> <tokenizer.bpe> <output.bin>\n",
                argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *bpe_path    = argv[2];
    const char *output_path = argv[3];

    /* ---- Load the trained tokenizer ---- */
    BPETokenizer *tok = bpe_load(bpe_path);
    if (tok == NULL) {
        fprintf(stderr, "Cannot load tokenizer from %s\n", bpe_path);
        return 1;
    }

    /* ---- Read input file into memory (same pattern as train_bpe) ---- */
    FILE *f = fopen(input_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Cannot open input file: %s\n", input_path);
        bpe_free(tok);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);

    unsigned char *text = NULL;
    if (length > 0) {
        text = malloc((size_t)length);
        if (fread(text, 1, (size_t)length, f) != (size_t)length) {
            fprintf(stderr, "Short read on %s\n", input_path);
            fclose(f);
            free(text);
            bpe_free(tok);
            return 1;
        }
    }
    fclose(f);

    /* ---- Encode ---- */
    size_t n = 0;
    int *toks = bpe_encode(tok, text, (size_t)length, &n);

    /* Compression ratio: input bytes / output tokens. >1 means BPE found
     * structure; ==1 means it fell back to one token per byte. */
    double ratio = (n > 0) ? (double)length / (double)n : 0.0;
    printf("Encoded %ld bytes -> %zu tokens (%.2fx compression)\n",
           length, n, ratio);

    /* ---- Write tokens to output file ---- */
    FILE *out = fopen(output_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "Cannot open output file: %s\n", output_path);
        free(toks);
        free(text);
        bpe_free(tok);
        return 1;
    }
    if (n > 0) {
        if (fwrite(toks, sizeof(int), n, out) != n) {
            fprintf(stderr, "Short write on %s\n", output_path);
            fclose(out);
            free(toks);
            free(text);
            bpe_free(tok);
            return 1;
        }
    }
    fclose(out);

    printf("Wrote %zu tokens to %s\n", n, output_path);

    free(toks);
    free(text);
    bpe_free(tok);
    return 0;
}
