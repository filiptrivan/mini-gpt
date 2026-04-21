#include <stdio.h>
#include <stdlib.h>
#include "data/text_parser.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <output.txt>\n", argv[0]);
        return 1;
    }

    ParsedText *pt = text_parser_parse_file(argv[1]);
    if (!pt) {
        fprintf(stderr, "Error: could not read '%s'\n", argv[1]);
        return 1;
    }

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "Error: could not open '%s' for writing\n", argv[2]);
        parsed_text_free(pt);
        return 1;
    }

    fwrite(pt->text, 1, pt->length, out);
    fclose(out);

    printf("Input:  %s\n", argv[1]);
    printf("Output: %s (%zu bytes)\n", argv[2], pt->length);

    parsed_text_free(pt);
    return 0;
}
