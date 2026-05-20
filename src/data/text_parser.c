#include "text_parser.h"
#include <stdlib.h>  /* malloc, free */
#include <stdio.h>   /* FILE, fopen, fread, fseek, ftell, fclose */
#include <string.h>  /* strncmp, memchr */
#include <ctype.h>   /* isdigit */

/* --- internal helpers --- */

/*
 * is_page_marker_line — does this line look like "---PAGE 31---"?
 */
static int is_page_marker_line(const char *line, size_t len) {
    // len >= 10 is a cheap guard: the shortest valid marker "---PAGE 1---"
    return len >= 10 && strncmp(line, "---PAGE ", 8) == 0;
}

/*
 * is_page_number_line — does this line consist only of digits (and spaces)?
 *
 * Many PDFs have a footer with just the page number, e.g. "  31  ". After
 * extraction those land in the text as standalone numeric lines and we
 * strip them so they don't pollute the training corpus.
 *
 * Returns 1 if the line is non-empty after trimming AND contains only digits.
 */
static int is_page_number_line(const char *line, size_t len) {
    size_t i = 0;
    /* Skip leading spaces. */
    while (i < len && line[i] == ' ') i++;
    /* Line was only spaces — not a page number, just blank. */
    if (i == len) return 0;
    size_t start = i;
    /* isdigit takes an int but expects a value in the range of unsigned char
     * (or EOF). On platforms where `char` is signed, passing it directly with
     * a high-bit byte (e.g. UTF-8) is undefined behavior — the cast to
     * `unsigned char` first is the standard fix. */
    while (i < len && isdigit((unsigned char)line[i])) i++;
    /* Skip trailing spaces. */
    while (i < len && line[i] == ' ') i++;
    /* Match only if we consumed every byte AND we saw at least one digit. */
    return i == len && (i - start) > 0;
}

/*
 * is_caption_line — does this line start with "<prefix><digit>"?
 *
 * Used to drop figure/table captions like "Slika 7: ..." or "Tabela 3: ...",
 * which are noise for a language model. We match on prefix + a digit so that
 * a sentence happening to contain the word "Slika" mid-text isn't dropped.
 *
 *   prefix     : caption keyword including the trailing space, e.g. "Slika ".
 *   prefix_len : strlen(prefix) — passed in so we don't recompute it per call.
 */
static int is_caption_line(const char *line, size_t len,
                           const char *prefix, size_t prefix_len) {
    /* Pointer arithmetic: `line + len` is the address one past the last byte.
     * It's legal to compare to it but NOT to dereference it. */
    const char *p = line;
    const char *line_end = line + len;
    /* Skip leading spaces by advancing the pointer (same as line[i++]
     * but written in pointer style). */
    while (p < line_end && *p == ' ') p++;
    size_t remaining = line_end - p;
    /* Need strictly more than prefix_len bytes so there is at least one byte
     * after the prefix to inspect (the digit). */
    if (remaining > prefix_len &&
        strncmp(p, prefix, prefix_len) == 0 &&
        isdigit((unsigned char)p[prefix_len]))
        return 1;
    return 0;
}

/*
 * is_bullet_byte — is the 3-byte sequence at `p` a known bullet glyph?
 *
 * UTF-8 encodes code points above U+007F as multi-byte sequences. Two private-
 * use code points appear as bullets in our source PDF and need stripping:
 *   U+F0B7  →  0xEF 0x82 0xB7
 *   U+F02D  →  0xEF 0x80 0xAD
 *
 *   p         : pointer into the line, treated as raw bytes.
 *   remaining : bytes left in the line from p onward (so we don't read past it).
 *
 * Returns 1 if the next 3 bytes match either bullet, 0 otherwise.
 */
static int is_bullet_byte(const unsigned char *p, size_t remaining) {
    /* Need at least 3 bytes, and all our targets start with 0xEF —
     * checking that byte first is a fast reject for normal ASCII. */
    if (remaining < 3 || p[0] != 0xEF) return 0;
    if (p[1] == 0x82 && p[2] == 0xB7) return 1;
    if (p[1] == 0x80 && p[2] == 0xAD) return 1;
    return 0;
}

/* --- core cleaning --- */

/*
 * clean_text — the heart of the parser. Walks the raw input line by line,
 * drops noise lines (page markers, page numbers, captions), strips bullet
 * glyphs, trims trailing whitespace, and collapses blank lines.
 *
 *   raw     : pointer to input bytes (not NUL-terminated; we use raw_len).
 *   raw_len : number of bytes in raw.
 *
 * Returns: a fresh ParsedText* (heap-allocated) or NULL on allocation failure.
 *
 * Memory layout: we allocate ONE output buffer up-front sized at raw_len + 1
 * (output is always <= input because we only ever delete bytes), then write
 * to it as we walk. The "+1" leaves room for the final NUL terminator.
 */
static ParsedText *clean_text(const char *raw, size_t raw_len) {
    /* sizeof(ParsedText) gives the struct size; malloc returns void* which
     * implicitly converts to ParsedText* in C (this is one of the few places
     * C and C++ disagree — C++ requires a cast). */
    ParsedText *result = malloc(sizeof(ParsedText));
    if (!result) return NULL;  /* `!result` == `result == NULL` */

    /* Special case empty input: still return a valid (empty) ParsedText so
     * callers don't have to special-case NULL vs "empty result". */
    if (raw_len == 0) {
        result->text = malloc(1);
        result->text[0] = '\0';
        result->length = 0;
        return result;
    }

    /* Output can only be <= input size. */
    char *out = malloc(raw_len + 1);
    /* If this allocation fails, we must free the earlier one to avoid a leak
     * before returning NULL. */
    if (!out) { free(result); return NULL; }

    size_t out_len = 0;          /* how many bytes we've written into `out` */
    const char *p = raw;         /* current read position */
    const char *end = raw + raw_len;

    /* Loop once per line of input. */
    while (p < end) {
        /* Find end of current line. memchr searches for a single byte and
         * returns a pointer to it, or NULL if not found. */
        const char *line_start = p;
        const char *line_end = memchr(p, '\n', end - p);
        if (!line_end) line_end = end;  /* last line may not end in '\n' */
        size_t line_len = line_end - line_start;

        /* Advance past the newline so the next iteration starts on the next
         * line. The ternary guards against stepping past `end` on the last line. */
        p = (line_end < end) ? line_end + 1 : end;

        /* Skip filtered lines entirely. `continue` jumps to the next while iter. */
        if (is_page_marker_line(line_start, line_len)) continue;
        if (is_page_number_line(line_start, line_len)) continue;
        if (is_caption_line(line_start, line_len, "Slika ", 6)) continue;
        if (is_caption_line(line_start, line_len, "Tabela ", 7)) continue;

        /* Copy this line into `out`, stripping bullet bytes as we go.
         * We remember where this line started in the output (line_out_start)
         * so we can later detect "did this line produce any content?" and
         * trim trailing whitespace just from this line. */
        size_t line_out_start = out_len;
        const unsigned char *lp = (const unsigned char *)line_start;
        const unsigned char *lend = (const unsigned char *)(line_start + line_len);

        while (lp < lend) {
            if (is_bullet_byte(lp, lend - lp)) {
                /* Drop the 3 bullet bytes by skipping over them. */
                lp += 3;
                continue;
            }
            /* Copy one byte. The `out_len++` is a post-increment: it uses the
             * current value as the index, then increments. Same as:
             *     out[out_len] = (char)*lp;
             *     out_len++;                                                   */
            out[out_len++] = (char)*lp;
            lp++;
        }

        /* Trim trailing whitespace from this line only (we walk back from the
         * end of what we just wrote, but never past line_out_start). */
        while (out_len > line_out_start &&
               (out[out_len - 1] == ' ' || out[out_len - 1] == '\t')) {
            out_len--;
        }

        /* If after stripping bullets and trimming we wrote no bytes, the line
         * was effectively blank — skip emitting a newline so we don't end up
         * with consecutive blank lines. */
        int is_blank = (out_len == line_out_start);
        if (is_blank) continue;

        /* Terminate this line. */
        out[out_len++] = '\n';
    }

    /* Remove trailing blank lines: while the last byte is '\n' AND either the
     * buffer is too small to have a non-newline before it OR the byte before
     * the last is also '\n', drop the last byte. This collapses any tail of
     * "...\n\n\n" down to a single "\n". */
    while (out_len > 0 && out[out_len - 1] == '\n' &&
           (out_len < 2 || out[out_len - 2] == '\n')) {
        out_len--;
    }

    /* C strings are NUL-terminated. We sized the buffer as raw_len + 1, so
     * writing here is always within bounds. */
    out[out_len] = '\0';
    result->text = out;
    result->length = out_len;
    return result;
}

/* --- public API --- */

/*
 * text_parser_parse_file — see text_parser.h for the contract.
 *
 * Implementation: read the whole file into a heap buffer, hand it to
 * clean_text, then free the raw buffer (clean_text already produced its own
 * output buffer that lives inside the returned ParsedText).
 *
 * KNOWN LIMITATION (alpha version): peak memory is roughly 2 * file_size,
 * because the raw input buffer and the cleaned output buffer both exist at
 * the same time during clean_text. Acceptable for the seminar corpus (a
 * textbook is a few MB; 2x is single-digit MB). A future improvement: since
 * cleaning only ever REMOVES bytes (never adds), it could be done in place
 * with a two-pointer read/write scheme over a single buffer — same pattern
 * the BPE training already uses to compact merged pairs. Deferred.
 */
ParsedText *text_parser_parse_file(const char *filepath) {
    /* "rb" = read, binary. Binary mode matters on Windows so '\n' isn't
     * translated to "\r\n"; on Unix it's the same as "r". We want raw bytes. */
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    /* Standard "size of an open file" trick: seek to end, ask the position,
     * seek back to start. Works for regular files; not reliable for pipes. */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Empty file: return an empty ParsedText rather than NULL, so callers
     * don't have to distinguish "missing file" from "empty file". */
    if (fsize <= 0) {
        fclose(f);
        return text_parser_parse_string("", 0);
    }

    char *buf = malloc(fsize);
    if (!buf) { fclose(f); return NULL; }

    /* fread(dst, elem_size, num_elems, file) returns number of elements read.
     * With elem_size=1 it returns the byte count, which may be less than fsize
     * on partial reads — we use `nread` rather than fsize for safety. */
    size_t nread = fread(buf, 1, fsize, f);
    fclose(f);

    ParsedText *result = clean_text(buf, nread);
    /* clean_text copies the bytes it keeps into its own output buffer, so we
     * can safely free the raw input buffer here. */
    free(buf);
    return result;
}

/*
 * text_parser_parse_string — see text_parser.h. Just a thin wrapper that
 * forwards to clean_text, since there's no file I/O to do.
 */
ParsedText *text_parser_parse_string(const char *raw_text, size_t length) {
    return clean_text(raw_text, length);
}

/*
 * parsed_text_free — release a ParsedText and the buffer it owns.
 *
 * Order matters: free the inner `text` first, then the struct. Freeing the
 * struct first would lose the `text` pointer and leak that buffer.
 *
 * The early-return on NULL mirrors free()'s own behavior, so callers can
 * unconditionally call this without a guard.
 */
void parsed_text_free(ParsedText *pt) {
    if (!pt) return;
    free(pt->text);
    free(pt);
}
