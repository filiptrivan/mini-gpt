#include <stdio.h>
#include <stdlib.h>
#include "data/text_parser.h"

/*
 * preprocess — small CLI wrapper around the text parser.
 *
 * Usage: preprocess <input.txt> <output.txt>
 *
 * Reads the raw extracted text from <input.txt>, runs the cleaning pipeline,
 * and writes the cleaned text to <output.txt>. Exit code 0 on success, 1 on
 * any error (bad args, unreadable input, unwritable output).
 *
 * `argc` is the argument count and `argv` is the argument vector. By
 * convention argv[0] is the program name and argv[1..] are the user args, so
 * "two real arguments" means argc must equal 3.
 */
int main(int argc, char *argv[]) {
    if (argc != 3) {
        /* fprintf to stderr (not stdout) so usage messages don't pollute the
         * output if someone pipes this program into another tool. */
        fprintf(stderr, "Usage: %s <input.txt> <output.txt>\n", argv[0]);
        return 1;
    }

    /* Hand off all the parsing/cleaning work to the library. We get back
     * either a populated ParsedText* or NULL on failure. */
    ParsedText *pt = text_parser_parse_file(argv[1]);
    if (!pt) {
        fprintf(stderr, "Error: could not read '%s'\n", argv[1]);
        return 1;
    }

    /* "wb" = write, binary. Binary mode keeps '\n' as a single byte and
     * matches the "rb" we used to read, so the cleaned bytes round-trip
     * faithfully. */
    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "Error: could not open '%s' for writing\n", argv[2]);
        /* Important: free the ParsedText on every error path or it leaks. */
        parsed_text_free(pt);
        return 1;
    }

    /* fwrite(src, elem_size, num_elems, file). With elem_size=1 we're writing
     * raw bytes; pt->length is already the exact byte count (no trailing NUL
     * is included, which is what we want for a text file). */
    fwrite(pt->text, 1, pt->length, out);
    fclose(out);

    /* "%zu" is the printf format specifier for `size_t` (added in C99). Using
     * "%d" or "%lu" here would be wrong on platforms where size_t differs
     * from int / unsigned long. */
    printf("Input:  %s\n", argv[1]);
    printf("Output: %s (%zu bytes)\n", argv[2], pt->length);

    parsed_text_free(pt);
    return 0;
}
