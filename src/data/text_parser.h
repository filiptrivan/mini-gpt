#ifndef TEXT_PARSER_H
#define TEXT_PARSER_H

/* Include guard above (#ifndef / #define / #endif) prevents this header from
 * being included twice in the same translation unit. Without it, the typedef
 * below would be redefined and the compiler would error. */

#include <stddef.h>  /* for size_t */

/*
 * ParsedText — a small "owning" struct returned by the parser.
 *
 *   text   : pointer to a heap-allocated, NUL-terminated string with the cleaned
 *            text. The buffer is owned by this struct; do not free `text`
 *            yourself — call parsed_text_free() on the whole struct.
 *   length : number of bytes in `text` NOT counting the trailing '\0'.
 *
 * `typedef struct { ... } Name;` lets us write `ParsedText` instead of
 * `struct ParsedText` everywhere.
 */
typedef struct {
    char *text;
    size_t length;
} ParsedText;

/*
 * text_parser_parse_file — read a UTF-8 text file from disk and clean it.
 *
 *   filepath : NUL-terminated path to the input file (read in binary mode so
 *              UTF-8 bytes pass through untouched).
 *
 * Returns: a freshly malloc'd ParsedText on success, or NULL if the file
 *          cannot be opened or memory allocation fails.
 *
 * Ownership: caller must free the returned pointer with parsed_text_free().
 */
ParsedText *text_parser_parse_file(const char *filepath);

/*
 * text_parser_parse_string — clean an in-memory buffer (no disk I/O).
 *
 *   raw_text : pointer to the input bytes. Does NOT need to be NUL-terminated;
 *              we use `length` to know where it ends.
 *   length   : number of bytes in raw_text.
 *
 * Returns: a freshly malloc'd ParsedText. The output is a copy — the input
 *          buffer is not modified or kept.
 *
 * Ownership: caller must free the returned pointer with parsed_text_free().
 */
ParsedText *text_parser_parse_string(const char *raw_text, size_t length);

/*
 * parsed_text_free — release a ParsedText and its inner `text` buffer.
 *
 * Safe to call with NULL (does nothing), which mirrors how free() itself behaves.
 * Always pair every successful parse_* call with exactly one parsed_text_free().
 */
void parsed_text_free(ParsedText *pt);

#endif /* TEXT_PARSER_H */
