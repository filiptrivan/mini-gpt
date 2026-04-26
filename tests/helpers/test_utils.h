#ifndef TEST_UTILS_H
#define TEST_UTILS_H

/*
 * Shared test helpers — currently empty, populated from Task 6 onward.
 *
 * Planned contents:
 *   float_eq(a, b, eps)        — compare floats within tolerance, since '==' on
 *                                floats is unreliable due to rounding error.
 *   numerical_gradient(...)    — finite-difference gradient estimator used to
 *                                verify analytical gradients in backward-pass
 *                                tests (see TDD workflow in CLAUDE.md).
 *
 * The include guard (#ifndef / #define / #endif) is in place already so that
 * once we add declarations, including this header from multiple test files
 * won't cause duplicate-symbol errors.
 */

#endif /* TEST_UTILS_H */
