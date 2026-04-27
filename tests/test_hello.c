/* Standard CMocka prelude: these four headers must be included in this exact
 * order. CMocka's macros use jmp_buf/va_list internally, so <stdarg.h>,
 * <stddef.h>, and <setjmp.h> must come before <cmocka.h>. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

/*
 * test_sanity — smoke test that the build/test pipeline is wired up.
 *
 * CMocka test signature is always `void test_name(void **state)`. The `state`
 * pointer is a per-test scratch slot we don't use here, so we cast to (void)
 * to silence "unused parameter" warnings.
 *
 * `assert_int_equal` is a CMocka macro that fails the test (and prints a
 * helpful message) if the two ints differ. We compare 1 == 1 just to prove
 * the harness can run a passing assertion.
 */
static void test_sanity(void **state) {
    (void)state;
    assert_int_equal(1, 1);
}

/*
 * main — entry point for the test executable.
 *
 * CMocka groups tests into an array of CMUnitTest entries. The
 * `cmocka_unit_test(fn)` macro wraps a test function into one such entry.
 * `cmocka_run_group_tests` runs them all and returns the number of failures
 * (0 = all passed), which becomes the program's exit code — exactly what
 * `ctest` looks at.
 *
 * Why is `cmocka_unit_test` a macro and not a regular function? Two reasons:
 *   1. It uses the preprocessor's `#` stringification operator to capture the
 *      function's NAME as a string literal at compile time (e.g. turning the
 *      token `test_sanity` into "test_sanity" so CMocka can print it in pass/
 *      fail messages). A regular C function can't see the caller's source
 *      text — only the preprocessor can.
 *   2. It expands to a brace-enclosed struct initializer like
 *      `{ "test_sanity", test_sanity, NULL, NULL, NULL }`, which slots
 *      directly into the `CMUnitTest tests[]` array literal below. A function
 *      can only return a value, not initializer syntax, so it couldn't be
 *      used inside an array initializer this way.
 */
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sanity),
    };
    /* The two NULLs are optional group setup/teardown callbacks — we don't
     * need any shared fixture, so we pass NULL for both. */
    return cmocka_run_group_tests(tests, NULL, NULL);
}
