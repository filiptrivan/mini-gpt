/* Test executable for the BPE tokenizer.
 *
 * For now this file contains only a single trivial test that always passes.
 * Its purpose at this stage is to verify that:
 *   - the bpe library builds and links,
 *   - CMake registers the executable as a CTest target,
 *   - CMocka's test runner is wired up correctly.
 *
 * Real BPE tests (init, encode/decode roundtrip, training, Serbian UTF-8,
 * determinism, ...) will be added one-by-one in the TDD step. */

#include <stdarg.h>  /* required by cmocka.h before its own includes */
#include <stddef.h>  /* size_t */
#include <setjmp.h>  /* required by cmocka.h */
#include <cmocka.h>

#include "tokenizer/bpe.h"

/* test_placeholder — proves the test harness runs. We reference the BPE
 * header above to also confirm the include path is set up correctly; the
 * actual assertion is trivially true. */
static void test_placeholder(void **state) {
    (void)state;  /* unused — CMocka passes per-test state we don't need */
    assert_int_equal(1, 1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_placeholder),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
