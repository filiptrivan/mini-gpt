#ifndef TEST_UTILS_H
#define TEST_UTILS_H

/*
 * Shared test helpers.
 *
 * Currently provides numerical_gradient(): a finite-difference gradient
 * estimator used to verify that the analytical backward functions in
 * src/model/layers.c match the true derivative of the corresponding
 * forward function. This is the gold-standard correctness test for any
 * hand-written backward pass — Karpathy's nanoGPT, Andrej Karpathy's
 * micrograd, PyTorch's `gradcheck`, every framework does this.
 *
 * The include guard (#ifndef / #define / #endif) prevents duplicate-symbol
 * errors when this header is pulled in from multiple test .c files.
 */

/*
 * test_loss_fn_t — function pointer type for a scalar loss L(x).
 *
 * In C, `typedef float (*name_t)(args)` defines `name_t` as "a type that
 * holds a pointer to a function taking those args and returning float."
 * Once declared, you can pass `name_t fn` around like any other variable
 * and call it with `fn(...)`. We use this so numerical_gradient can be
 * generic — each test supplies its own loss closure (forward op + a fixed
 * upstream gradient).
 *
 * Parameters of the callback:
 *   x   — current input vector (length n is known to the caller; the
 *         loss function captures it via ctx if needed).
 *   ctx — opaque "user data" pointer — the standard C idiom for closing
 *         over extra state (think: lambda capture). The loss function
 *         casts it to whatever struct the caller actually passed in.
 *         numerical_gradient never touches ctx — it just forwards it.
 *
 * Returns:
 *   The scalar loss at this x. Convention used in our tests:
 *     L(x) = sum_k d_out[k] * forward(x)[k]
 *   With that contraction, dL/dx is exactly what the analytical
 *   backward function should compute.
 */
typedef float (*test_loss_fn_t)(const float *x, void *ctx);

/*
 * numerical_gradient — central-difference estimate of dL/dx[i].
 *
 * For each i in [0, n):
 *   grad_out[i] = (L(x + h e_i) - L(x - h e_i)) / (2 h)
 *
 * where e_i is the unit vector in dimension i. The central difference
 * (both sides perturbed, divided by 2h) has O(h^2) error, vs. O(h) for
 * a one-sided difference — important because we work in float32 where
 * the round-off floor is already around h^-1 * machine_epsilon.
 *
 * Practical h: 1e-3 is the sweet spot in single precision. Smaller h
 * (e.g. 1e-6) cancels almost all the signal in (L+ - L-) before the
 * division, leaving only round-off noise; larger h (e.g. 1e-1) makes
 * the linear approximation inaccurate.
 *
 * Parameters:
 *   grad_out — destination buffer, length n (caller-allocated, overwritten)
 *   loss_fn  — scalar loss callback (see test_loss_fn_t above)
 *   x        — input vector, length n. TEMPORARILY mutated during the
 *              sweep (each x[i] gets +h, then -h) but every perturbation
 *              is reverted before moving to the next index, so x is
 *              byte-identical on return.
 *   n        — number of elements (and length of grad_out)
 *   ctx      — opaque user data forwarded to loss_fn unchanged
 *   h        — perturbation step; pass 1e-3 unless you have a specific
 *              reason to use something else
 *
 * Why `x` is non-const: we mutate-then-restore each element in place.
 * Cloning x into a scratch buffer would also work, but in-place is what
 * every reference implementation does (PyTorch's gradcheck, Karpathy's
 * llm.c numerical grad helper) — it avoids allocation in the inner loop
 * and the byte-identical restore makes the side effect invisible.
 */
void numerical_gradient(float *grad_out,
                        test_loss_fn_t loss_fn,
                        float *x, int n,
                        void *ctx, float h);

#endif /* TEST_UTILS_H */
