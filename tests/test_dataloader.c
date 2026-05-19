/* Test executable for the data loader.
 *
 * Strategy: write a tiny synthetic .bin file containing tokens 0,1,2,...,19
 * into a temp path, then exercise the loader against it. Sequential token
 * ids make every assertion trivially checkable by eye (e.g. targets[i]
 * must equal inputs[i] + 1).
 */

#include <stdarg.h>  /* required by cmocka.h before its own includes */
#include <stddef.h>  /* size_t */
#include <setjmp.h>  /* required by cmocka.h */
#include <stdlib.h>  /* free, malloc */
#include <stdio.h>   /* fopen, fwrite, fclose, remove, tmpnam */
#include <string.h>  /* memset */
#include <cmocka.h>

#include "data/dataloader.h"

/* Path used by every test for the synthetic input file. We keep it in /tmp
 * so the tests don't pollute the project directory. Each test creates the
 * file in its setup and removes it at the end. */
#define SYNTHETIC_PATH "/tmp/test_dataloader_synthetic.bin"

/*
 * write_synthetic_bin — create a .bin file containing the integers
 *                       0, 1, 2, ..., n-1 (each as one int).
 *
 *   path : where to write the file.
 *   n    : how many ints to write.
 *
 * Mirrors the format produced by `tools/tokenize.c`: a flat sequence of
 * `int`s, no header. fwrite returns the number of items written, so we
 * assert it equals `n` to catch disk-full or permission errors early.
 */
static void write_synthetic_bin(const char *path, int n) {
    /* "wb" = write, binary. On Unix the 'b' is a no-op, but we keep it
     * explicit because the loader will open the same file with "rb". */
    FILE *f = fopen(path, "wb");
    assert_non_null(f);

    /* Write the ints one loop iteration at a time. We could write them all
     * at once with a single fwrite, but the loop is easier to read and the
     * file is tiny. */
    for (int i = 0; i < n; i++) {
        size_t written = fwrite(&i, sizeof(int), 1, f);
        assert_int_equal(written, 1);
    }
    fclose(f);
}

/*
 * test_init_rejects_missing_file — opening a non-existent file should fail
 * gracefully with NULL, not crash. This guards the "user gave a wrong path"
 * case.
 */
static void test_init_rejects_missing_file(void **state) {
    (void)state;  /* per-test state — unused */

    DataLoader *loader = dataloader_init("/no/such/file.bin",
                                         /*batch_size=*/2,
                                         /*seq_len=*/4);
    assert_null(loader);
}

/*
 * test_init_rejects_too_small_file — a file shorter than B*T+1 tokens can't
 * produce even one batch (wrapping doesn't help: even a full wrap-around
 * window doesn't have enough room). init must refuse with NULL.
 *
 * With batch_size=2, seq_len=4, we need >= 9 tokens. We give it 5.
 */
static void test_init_rejects_too_small_file(void **state) {
    (void)state;

    write_synthetic_bin(SYNTHETIC_PATH, 5);

    DataLoader *loader = dataloader_init(SYNTHETIC_PATH,
                                         /*batch_size=*/2,
                                         /*seq_len=*/4);
    assert_null(loader);

    remove(SYNTHETIC_PATH);
}

/*
 * test_init_loads_file_correctly — after a successful init, the loader's
 * in-memory buffer should hold exactly the bytes we wrote: 20 tokens, with
 * tokens[0]==0 and tokens[19]==19.
 *
 * We poke at the struct fields directly. They're exposed in the header so
 * tests can verify state — production callers won't need to.
 */
static void test_init_loads_file_correctly(void **state) {
    (void)state;

    write_synthetic_bin(SYNTHETIC_PATH, 20);

    DataLoader *loader = dataloader_init(SYNTHETIC_PATH,
                                         /*batch_size=*/2,
                                         /*seq_len=*/4);
    assert_non_null(loader);

    assert_int_equal(loader->num_tokens, 20);
    assert_int_equal(loader->batch_size, 2);
    assert_int_equal(loader->seq_len, 4);
    assert_int_equal(loader->cursor, 0);
    assert_non_null(loader->tokens);
    assert_int_equal(loader->tokens[0], 0);
    assert_int_equal(loader->tokens[19], 19);

    dataloader_free(loader);
    remove(SYNTHETIC_PATH);
}

/*
 * test_batch_shape_and_shift — the heart of the loader's contract.
 *
 * With B=2, T=4 against [0..19]:
 *   inputs  must be [[0,1,2,3], [4,5,6,7]]
 *   targets must be [[1,2,3,4], [5,6,7,8]]
 *
 * In flat row-major layout that's:
 *   inputs  = {0,1,2,3, 4,5,6,7}
 *   targets = {1,2,3,4, 5,6,7,8}
 *
 * The loop also verifies the universal property targets[i] == inputs[i] + 1,
 * which is what makes the dataset "self-labelling" for next-token prediction.
 */
static void test_batch_shape_and_shift(void **state) {
    (void)state;
    const int B = 2, T = 4;

    write_synthetic_bin(SYNTHETIC_PATH, 20);
    DataLoader *loader = dataloader_init(SYNTHETIC_PATH, B, T);
    assert_non_null(loader);

    /* Caller-owned buffers — the loader fills them, doesn't allocate them.
     * Static arrays here because B*T is a compile-time constant. */
    int inputs[2 * 4];
    int targets[2 * 4];
    dataloader_next_batch(loader, inputs, targets);

    /* Spot-check every element row-by-row, then re-verify with the shift
     * property as a second redundant check (different way to be wrong). */
    int expected_inputs[]  = {0, 1, 2, 3, 4, 5, 6, 7};
    int expected_targets[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < B * T; i++) {
        assert_int_equal(inputs[i],  expected_inputs[i]);
        assert_int_equal(targets[i], expected_targets[i]);
        assert_int_equal(targets[i], inputs[i] + 1);
    }

    dataloader_free(loader);
    remove(SYNTHETIC_PATH);
}

/*
 * test_cursor_advances — after a batch the cursor moves forward by B*T so
 * the next batch picks up where this one ended.
 *
 * Batch 1 consumes positions 0..7 (and borrows 8 for targets).
 * Batch 2 should produce inputs from positions 8..15.
 */
static void test_cursor_advances(void **state) {
    (void)state;
    const int B = 2, T = 4;

    write_synthetic_bin(SYNTHETIC_PATH, 20);
    DataLoader *loader = dataloader_init(SYNTHETIC_PATH, B, T);

    int inputs[2 * 4];
    int targets[2 * 4];

    /* First call: just to advance the cursor; we already tested its
     * contents in the previous test. */
    dataloader_next_batch(loader, inputs, targets);
    assert_int_equal(loader->cursor, 8);

    /* Second call: inputs should now be 8..15, targets 9..16. */
    dataloader_next_batch(loader, inputs, targets);
    int expected_inputs[]  = { 8,  9, 10, 11, 12, 13, 14, 15};
    int expected_targets[] = { 9, 10, 11, 12, 13, 14, 15, 16};
    for (int i = 0; i < B * T; i++) {
        assert_int_equal(inputs[i],  expected_inputs[i]);
        assert_int_equal(targets[i], expected_targets[i]);
    }
    assert_int_equal(loader->cursor, 16);

    dataloader_free(loader);
    remove(SYNTHETIC_PATH);
}

/*
 * test_wraparound_at_eof — once the cursor can't fit a full batch in what's
 * left of the file, it must wrap to 0 and start over. With a 20-token file
 * and B*T=8 per batch + 1 borrowed for targets:
 *
 *   batch 1: cursor 0 -> 8     (reads 0..8)
 *   batch 2: cursor 8 -> 16    (reads 8..16)
 *   batch 3: cursor 16, would need 16..24 but only 20 tokens exist
 *            -> wrap cursor to 0, read 0..8 again, cursor ends at 8
 *
 * So the third batch should produce the same inputs as the very first batch.
 */
static void test_wraparound_at_eof(void **state) {
    (void)state;
    const int B = 2, T = 4;

    write_synthetic_bin(SYNTHETIC_PATH, 20);
    DataLoader *loader = dataloader_init(SYNTHETIC_PATH, B, T);

    int inputs[2 * 4];
    int targets[2 * 4];

    dataloader_next_batch(loader, inputs, targets);  /* batch 1 */
    dataloader_next_batch(loader, inputs, targets);  /* batch 2 */

    /* Batch 3: this is the wrap. */
    dataloader_next_batch(loader, inputs, targets);
    int expected_inputs[]  = {0, 1, 2, 3, 4, 5, 6, 7};
    int expected_targets[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < B * T; i++) {
        assert_int_equal(inputs[i],  expected_inputs[i]);
        assert_int_equal(targets[i], expected_targets[i]);
    }
    assert_int_equal(loader->cursor, 8);  /* cursor was reset to 0, then advanced */

    dataloader_free(loader);
    remove(SYNTHETIC_PATH);
}

/*
 * test_free_handles_null — calling dataloader_free(NULL) must be a no-op,
 * matching how libc's free() handles NULL. This makes cleanup code in
 * callers simpler (no need to check before freeing).
 */
static void test_free_handles_null(void **state) {
    (void)state;
    dataloader_free(NULL);  /* must not crash */
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_init_rejects_missing_file),
        cmocka_unit_test(test_init_rejects_too_small_file),
        cmocka_unit_test(test_init_loads_file_correctly),
        cmocka_unit_test(test_batch_shape_and_shift),
        cmocka_unit_test(test_cursor_advances),
        cmocka_unit_test(test_wraparound_at_eof),
        cmocka_unit_test(test_free_handles_null),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
