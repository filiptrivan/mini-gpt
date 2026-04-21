#include "text_parser.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* --- internal helpers --- */

static int is_page_marker_line(const char *line, size_t len) {
    /* Matches "---PAGE N---" */
    return len >= 10 && strncmp(line, "---PAGE ", 8) == 0;
}

static int is_page_number_line(const char *line, size_t len) {
    /* Trimmed line is only digits (e.g. "31 ") */
    size_t i = 0;
    while (i < len && line[i] == ' ') i++;
    if (i == len) return 0;
    size_t start = i;
    while (i < len && isdigit((unsigned char)line[i])) i++;
    while (i < len && line[i] == ' ') i++;
    return i == len && (i - start) > 0;
}

static int is_caption_line(const char *line, size_t len,
                           const char *prefix, size_t prefix_len) {
    const char *p = line;
    const char *line_end = line + len;
    while (p < line_end && *p == ' ') p++;
    size_t remaining = line_end - p;
    if (remaining > prefix_len &&
        strncmp(p, prefix, prefix_len) == 0 &&
        isdigit((unsigned char)p[prefix_len]))
        return 1;
    return 0;
}

static int is_bullet_byte(const unsigned char *p, size_t remaining) {
    /* U+F0B7 = EF 82 B7, U+F02D = EF 80 AD */
    if (remaining < 3 || p[0] != 0xEF) return 0;
    if (p[1] == 0x82 && p[2] == 0xB7) return 1;
    if (p[1] == 0x80 && p[2] == 0xAD) return 1;
    return 0;
}

/* --- core cleaning --- */

static ParsedText *clean_text(const char *raw, size_t raw_len) {
    ParsedText *result = malloc(sizeof(ParsedText));
    if (!result) return NULL;

    if (raw_len == 0) {
        result->text = malloc(1);
        result->text[0] = '\0';
        result->length = 0;
        return result;
    }

    /* Output can only be <= input size */
    char *out = malloc(raw_len + 1);
    if (!out) { free(result); return NULL; }

    size_t out_len = 0;
    const char *p = raw;
    const char *end = raw + raw_len;

    while (p < end) {
        /* Find end of current line */
        const char *line_start = p;
        const char *line_end = memchr(p, '\n', end - p);
        if (!line_end) line_end = end;
        size_t line_len = line_end - line_start;

        /* Advance past the newline */
        p = (line_end < end) ? line_end + 1 : end;

        /* Skip filtered lines */
        if (is_page_marker_line(line_start, line_len)) continue;
        if (is_page_number_line(line_start, line_len)) continue;
        if (is_caption_line(line_start, line_len, "Slika ", 6)) continue;
        if (is_caption_line(line_start, line_len, "Tabela ", 7)) continue;

        /* Copy line, stripping bullet chars and trailing whitespace */
        size_t line_out_start = out_len;
        const unsigned char *lp = (const unsigned char *)line_start;
        const unsigned char *lend = (const unsigned char *)(line_start + line_len);

        while (lp < lend) {
            if (is_bullet_byte(lp, lend - lp)) {
                lp += 3;
                continue;
            }
            out[out_len++] = (char)*lp;
            lp++;
        }

        /* Trim trailing whitespace from this line */
        while (out_len > line_out_start &&
               (out[out_len - 1] == ' ' || out[out_len - 1] == '\t')) {
            out_len--;
        }

        /* Skip blank lines entirely */
        int is_blank = (out_len == line_out_start);
        if (is_blank) continue;

        /* Add newline */
        out[out_len++] = '\n';
    }

    /* Remove trailing blank lines */
    while (out_len > 0 && out[out_len - 1] == '\n' &&
           (out_len < 2 || out[out_len - 2] == '\n')) {
        out_len--;
    }

    out[out_len] = '\0';
    result->text = out;
    result->length = out_len;
    return result;
}

/* --- public API --- */

ParsedText *text_parser_parse_file(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize <= 0) {
        fclose(f);
        return text_parser_parse_string("", 0);
    }

    char *buf = malloc(fsize);
    if (!buf) { fclose(f); return NULL; }

    size_t nread = fread(buf, 1, fsize, f);
    fclose(f);

    ParsedText *result = clean_text(buf, nread);
    free(buf);
    return result;
}

ParsedText *text_parser_parse_string(const char *raw_text, size_t length) {
    return clean_text(raw_text, length);
}

void parsed_text_free(ParsedText *pt) {
    if (!pt) return;
    free(pt->text);
    free(pt);
}
