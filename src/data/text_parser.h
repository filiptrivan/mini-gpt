#ifndef TEXT_PARSER_H
#define TEXT_PARSER_H

#include <stddef.h>

typedef struct {
    char *text;
    size_t length;
} ParsedText;

/* Parse and clean text from a file. Returns NULL on read failure.
 * Caller must free with parsed_text_free(). */
ParsedText *text_parser_parse_file(const char *filepath);

/* Parse and clean an in-memory string. Makes a copy.
 * Caller must free with parsed_text_free(). */
ParsedText *text_parser_parse_string(const char *raw_text, size_t length);

/* Free a ParsedText. Safe to call with NULL. */
void parsed_text_free(ParsedText *pt);

#endif /* TEXT_PARSER_H */
