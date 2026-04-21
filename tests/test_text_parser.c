#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
#include <cmocka.h>
#include "data/text_parser.h"

/* --- helpers --- */

static int contains(const char *haystack, const char *needle) {
    return strstr(haystack, needle) != NULL;
}

static int contains_standalone_line(const char *text, const char *line) {
    /* Check if 'line' appears as a complete line (between newlines or at start/end) */
    const char *p = text;
    size_t line_len = strlen(line);
    while ((p = strstr(p, line)) != NULL) {
        int at_start = (p == text) || (*(p - 1) == '\n');
        int at_end = (p[line_len] == '\n') || (p[line_len] == '\0');
        if (at_start && at_end) return 1;
        p += line_len;
    }
    return 0;
}

/* --- tests --- */

static void test_text_parser_load_from_file(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_non_null(pt->text);
    assert_true(pt->length > 0);
    parsed_text_free(pt);
}

static void test_text_parser_removes_page_numbers(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    /* "31" and "32" as standalone lines should be gone */
    assert_false(contains_standalone_line(pt->text, "31"));
    assert_false(contains_standalone_line(pt->text, "32"));
    parsed_text_free(pt);
}

static void test_text_parser_removes_page_markers(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "---PAGE"));
    parsed_text_free(pt);
}

static void test_text_parser_removes_figure_captions(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "Slika 7"));
    parsed_text_free(pt);
}

static void test_text_parser_removes_table_captions(void **state) {
    (void)state;
    const char *path = FIXTURE_DIR "/sample_text.txt";
    ParsedText *pt = text_parser_parse_file(path);
    assert_non_null(pt);
    assert_false(contains(pt->text, "Tabela 3"));
    parsed_text_free(pt);
}

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

static void test_text_parser_empty_input(void **state) {
    (void)state;
    ParsedText *pt = text_parser_parse_string("", 0);
    assert_non_null(pt);
    assert_int_equal(pt->length, 0);
    parsed_text_free(pt);
}

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
