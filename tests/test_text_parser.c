/* Standard CMocka prelude — required headers, in this order, before <cmocka.h>. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>   /* strstr, strlen */
#include <cmocka.h>
#include "data/text_parser.h"

/* --- helpers --- */

/*
 * contains — does `haystack` contain `needle` as a substring?
 *
 * `strstr(h, n)` returns a pointer to the first occurrence of `n` in `h`, or
 * NULL if not found. We just normalize that to a 0/1 int.
 *
 * Note: both arguments must be NUL-terminated C strings. ParsedText.text is
 * always NUL-terminated, so passing pt->text here is safe.
 */
static int contains(const char *haystack, const char *needle) {
    return strstr(haystack, needle) != NULL;
}

/*
 * contains_standalone_line — is `line` present in `text` as a complete line,
 * i.e. surrounded by '\n' (or by start/end of buffer)?
 *
 * We can't just use `contains` for things like "31", because the digit string
 * "31" appears legitimately inside many words/numbers. This walks every
 * occurrence and checks the byte just before and after to confirm both ends
 * are line boundaries.
 */
static int contains_standalone_line(const char *text, const char *line) {
    const char *p = text;
    size_t line_len = strlen(line);
    /* Assignment-in-condition: `p = strstr(...)` both stores the result and
     * yields it. The outer `!= NULL` is the actual loop condition. Common
     * idiom in C, but worth flagging because it's syntactically unusual. */
    while ((p = strstr(p, line)) != NULL) {
        /* `p == text` covers the case where the match is at offset 0 (no
         * preceding byte to look at). Otherwise the byte at p-1 must be '\n'. */
        int at_start = (p == text) || (*(p - 1) == '\n');
        /* `p[line_len]` is the byte just past the match. '\0' covers
         * "match goes to end of string". */
        int at_end = (p[line_len] == '\n') || (p[line_len] == '\0');
        if (at_start && at_end) return 1;
        /* Not a clean line — skip past this match and keep searching. */
        p += line_len;
    }
    return 0;
}

/* --- tests ---
 *
 * Each test follows the same shape:
 *   1. Build/load some input (file fixture or in-memory string).
 *   2. Parse it.
 *   3. Assert something about the cleaned output.
 *   4. Free the ParsedText so we don't leak memory across tests.
 *
 * `(void)state;` silences the unused-parameter warning on the CMocka
 * `void **state` argument that we never use.
 *
 * `FIXTURE_DIR` is a string literal injected by CMake at compile time
 * (`-DFIXTURE_DIR="..."`), so it expands to a real path on disk. */

/*
 * test_text_parser_load_from_file — sanity-check that we can read the
 * fixture file and get back a non-empty result. If this fails, every other
 * test that uses the file is meaningless, so it runs first.
 */
static void test_text_parser_load_from_file(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_non_null(pt->text);
    assert_true(pt->length > 0);
    parsed_text_free(pt);
}

/*
 * test_text_parser_removes_page_numbers — the fixture has "31" and "32" on
 * their own lines (footers from a PDF). After cleaning, those should be gone
 * even though the digits "31"/"32" may still appear inside other text.
 */
static void test_text_parser_removes_page_numbers(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    /* `assert_false` fails the test if the expression is non-zero/true. */
    assert_false(contains_standalone_line(pt->text, "31"));
    assert_false(contains_standalone_line(pt->text, "32"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_removes_page_markers — synthetic "---PAGE N---" lines
 * inserted by our PDF extractor must be stripped completely. Substring check
 * is fine here because "---PAGE" never appears legitimately.
 */
static void test_text_parser_removes_page_markers(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "---PAGE"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_removes_figure_captions — "Slika 7..." lines (Serbian for
 * "Figure 7") should be dropped. We assert the specific caption "Slika 7" is
 * not in the output.
 */
static void test_text_parser_removes_figure_captions(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "Slika 7"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_removes_table_captions — same idea as figures but for
 * "Tabela 3" ("Table 3" in Serbian).
 */
static void test_text_parser_removes_table_captions(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "Tabela 3"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_removes_bullet_chars — the two private-use bullet glyphs
 * (UTF-8 sequences 0xEF 0x82 0xB7 and 0xEF 0x80 0xAD) must be stripped, but
 * the surrounding text is preserved.
 *
 * The "\xEF\x82\xB7" syntax in a C string literal embeds raw bytes by their
 * hex value — useful for non-ASCII content where you want exact bytes.
 */
static void test_text_parser_removes_bullet_chars(void **state) {
    (void)state;
    /* U+F0B7 = 0xEF 0x82 0xB7, U+F02D = 0xEF 0x80 0xAD */
    const char *input = "pre \xEF\x82\xB7 bullet one\n\xEF\x80\xAD bullet two\n";
    ParsedText *pt = text_parser_parse_string(input, strlen(input));
    assert_non_null(pt);
    assert_false(contains(pt->text, "\xEF\x82\xB7"));
    assert_false(contains(pt->text, "\xEF\x80\xAD"));
    /* The text after bullets is preserved */
    assert_true(contains(pt->text, "bullet one"));
    assert_true(contains(pt->text, "bullet two"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_normalizes_whitespace — checks two collapsing rules:
 *   1. Multiple consecutive blank lines collapse to a single newline boundary.
 *   2. Trailing spaces before a newline are trimmed.
 * The actual text content of each line is preserved.
 */
static void test_text_parser_normalizes_whitespace(void **state) {
    (void)state;
    const char *input = "line one  \n\n\n\nline two   \n\nline three\n";
    ParsedText *pt = text_parser_parse_string(input, strlen(input));
    assert_non_null(pt);
    /* No consecutive newlines */
    assert_false(contains(pt->text, "\n\n"));
    /* No trailing spaces before newlines */
    assert_false(contains(pt->text, "  \n"));
    /* Content preserved */
    assert_true(contains(pt->text, "line one"));
    assert_true(contains(pt->text, "line two"));
    assert_true(contains(pt->text, "line three"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_preserves_serbian_chars — Serbian Latin-script letters
 * (č, ž, š, đ, ć) are multi-byte UTF-8 sequences. The parser must NOT mangle
 * them (a naive byte-level filter could easily corrupt these). All five must
 * survive a round-trip through the parser unchanged.
 */
static void test_text_parser_preserves_serbian_chars(void **state) {
    (void)state;
    const char *input = "Karakteristike: č, ž, š, đ, ć.\n";
    ParsedText *pt = text_parser_parse_string(input, strlen(input));
    assert_non_null(pt);
    assert_true(contains(pt->text, "č"));
    assert_true(contains(pt->text, "ž"));
    assert_true(contains(pt->text, "š"));
    assert_true(contains(pt->text, "đ"));
    assert_true(contains(pt->text, "ć"));
    parsed_text_free(pt);
}

/*
 * test_text_parser_empty_input — passing an empty buffer must return a valid
 * (but empty) ParsedText, NOT NULL. This makes the API safer to use because
 * callers don't need a special branch for "empty file".
 */
static void test_text_parser_empty_input(void **state) {
    (void)state;
    ParsedText *pt = text_parser_parse_string("", 0);
    assert_non_null(pt);
    assert_int_equal(pt->length, 0);
    parsed_text_free(pt);
}

/*
 * main — register all tests with CMocka and run them. Returns the number of
 * failed tests, which becomes the program's exit code (0 = all green).
 */
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_text_parser_load_from_file),
        cmocka_unit_test(test_text_parser_removes_page_numbers),
        cmocka_unit_test(test_text_parser_removes_page_markers),
        cmocka_unit_test(test_text_parser_removes_figure_captions),
        cmocka_unit_test(test_text_parser_removes_table_captions),
        cmocka_unit_test(test_text_parser_removes_bullet_chars),
        cmocka_unit_test(test_text_parser_normalizes_whitespace),
        cmocka_unit_test(test_text_parser_preserves_serbian_chars),
        cmocka_unit_test(test_text_parser_empty_input),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
