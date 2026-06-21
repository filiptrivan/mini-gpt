/*
 * Tests for src/distributed/mpi_utils.c — the MPI data-parallel layer.
 *
 * HOW TO RUN (Colab only — needs MPI):
 *
 *     cmake -B build -DENABLE_MPI=ON
 *     cmake --build build
 *     mpirun --allow-run-as-root -np 2 ./build/tests/test_distributed
 *
 * Unlike every other test in this project, this one is launched with `mpirun`:
 * MPI starts N copies of the program (here -np 2 => two "ranks"), and the test
 * body runs in every copy at once. The asserts therefore run on each rank, and
 * the whole point is to check that the ranks AGREE after our collective calls.
 *
 * The four checks (the three from PLAN.md plus a broadcast smoke test):
 *
 *   1. ALLREDUCE COMPUTES THE MEAN (test_allreduce_mean_known_values)
 *      Hand-picked per-rank inputs whose average we know on paper. For -np 2,
 *      rank 0 holds [1,2], rank 1 holds [3,4]; the averaged result is [2,3].
 *
 *   2. ALL RANKS GET THE SAME RESULT (test_allreduce_all_ranks_identical)
 *      Average rank-dependent data, then prove every rank ended up with a
 *      byte-identical result by reducing the per-rank max and min and checking
 *      they coincide. (A reduce that returned different answers per rank would
 *      make max != min.)
 *
 *   3. BROADCAST SHARES ONE RANK'S DATA (test_broadcast_shares_root_values)
 *      Rank 0 fills an array, the others fill garbage; after broadcast from
 *      rank 0 everyone holds rank 0's values. This is how train.c makes every
 *      rank start from identical weights.
 *
 *   4. ONE STEP KEEPS PARAMS IN SYNC (test_one_step_keeps_params_in_sync)
 *      The headline guarantee of data parallelism. Start identical models,
 *      feed each rank a DIFFERENT batch (=> different local gradients), average
 *      the gradients, take one AdamW step — and the parameters must STILL be
 *      identical across ranks afterward.
 *
 * A note on cmocka + MPI: a failing assert does a longjmp out of the test, so
 * we are careful to put every collective (Allreduce/Bcast) BEFORE the asserts.
 * If an assert could fire between two collectives, a failing rank would skip
 * the second collective while a passing rank waited on it forever (deadlock).
 * On success all ranks march through the collectives in lockstep, then assert.
 */

/* CMocka's required four includes, in this exact order (see test_layers.c for
 * the full explanation). */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <mpi.h>      /* MPI_Comm_rank/size, MPI_Allreduce, MPI_MAX/MIN (verify) */
#include <stdlib.h>   /* malloc, free */
#include <math.h>     /* fabsf */

#include "distributed/mpi_utils.h"
#include "model/gpt.h"
#include "optimizer/adamw.h"
#include "test_utils.h"  /* fill_random_ids */

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

/*
 * max_cross_rank_spread — the largest disagreement, over all elements, between
 * the ranks' copies of `local`.
 *
 * The trick: reduce the element-wise MAX and the element-wise MIN of `local`
 * across all ranks. For element i, (max - min) is exactly how far apart the
 * ranks are on that element: 0 means every rank agrees, anything bigger means
 * they differ. We return the worst (largest) such gap. Both "are the ranks in
 * sync?" and "did the ranks genuinely differ?" reduce to a threshold on this
 * one number, so both callers below share this single reduction instead of each
 * open-coding the malloc + two MPI_Allreduce + scan. We use separate send/recv
 * buffers (local -> gmax/gmin) so `local` itself is untouched.
 *
 *   local : this rank's array, length n (read-only).
 *   n     : number of elements.
 * Returns: max over i of |max_rank(local[i]) - min_rank(local[i])|.
 */
static float max_cross_rank_spread(const float *local, int n) {
    float *gmax = (float *)malloc((size_t)n * sizeof(float));
    float *gmin = (float *)malloc((size_t)n * sizeof(float));
    assert_non_null(gmax);
    assert_non_null(gmin);

    MPI_Allreduce(local, gmax, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local, gmin, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        float gap = fabsf(gmax[i] - gmin[i]);
        if (gap > worst) worst = gap;
    }
    free(gmax);
    free(gmin);
    return worst;
}

/*
 * assert_synced_across_ranks — fail unless `local[i]` is the same value on
 * every rank, for all i. The defining property we check after a broadcast or an
 * all-reduce: if the worst cross-rank gap is within tol, the ranks agree.
 *
 *   tol : allowed gap (use a tiny epsilon; floating-point reductions are
 *         bit-identical across ranks for the same op, so 0 would usually work,
 *         but a hair of slack keeps the intent clear).
 */
static void assert_synced_across_ranks(const float *local, int n, float tol) {
    assert_true(max_cross_rank_spread(local, n) <= tol);
}

/* ------------------------------------------------------------------ */
/* tests                                                              */
/* ------------------------------------------------------------------ */

/*
 * AllReduce must produce the element-wise MEAN, not the sum. Each rank r starts
 * with [1+2r, 2+2r], so across P ranks element 0 averages {1,3,5,...} and
 * element 1 averages {2,4,6,...}. Those means work out to exactly P and P+1
 * (e.g. P=2: element 0 = (1+3)/2 = 2, element 1 = (2+4)/2 = 3 -> [2,3]). If the
 * code summed without dividing, or divided by the wrong count, these would miss.
 */
static void test_allreduce_mean_known_values(void **state) {
    (void)state;

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    float data[2] = {1.0f + 2.0f * (float)rank, 2.0f + 2.0f * (float)rank};
    mpi_allreduce_mean(data, 2);

    /* Mean of {1,3,5,...,1+2(P-1)} = P; mean of {2,4,...,2+2(P-1)} = P+1. */
    assert_float_equal(data[0], (float)world,        1e-5f);
    assert_float_equal(data[1], (float)world + 1.0f, 1e-5f);
}

/*
 * The "reduce" in AllReduce must hand the SAME answer to every rank (that is
 * what distinguishes it from a plain reduce-to-root). We feed rank-dependent
 * data, average it, and then verify two things: the result matches the
 * hand-computed mean, AND it is identical on all ranks (via max == min).
 *
 * Rank r contributes value[i] = (r+1)*(i+1), so element i averages to
 * (i+1) * mean(1..P) = (i+1) * (P+1)/2.
 */
static void test_allreduce_all_ranks_identical(void **state) {
    (void)state;

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    enum { N = 5 };
    float data[N];
    for (int i = 0; i < N; i++) {
        data[i] = (float)(rank + 1) * (float)(i + 1);
    }

    mpi_allreduce_mean(data, N);

    /* Identical on every rank — the defining property of all-reduce. */
    assert_synced_across_ranks(data, N, 1e-5f);

    /* ...and equal to the value we computed by hand. */
    const float mean_of_ranks = (float)(world + 1) / 2.0f;  /* mean(1..P) */
    for (int i = 0; i < N; i++) {
        assert_float_equal(data[i], (float)(i + 1) * mean_of_ranks, 1e-5f);
    }
}

/*
 * Broadcast: rank 0 owns the canonical values; everyone else starts with a
 * sentinel and must be overwritten. This is exactly the move train.c uses after
 * gpt_init so all ranks begin from rank 0's weights.
 */
static void test_broadcast_shares_root_values(void **state) {
    (void)state;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    enum { N = 4 };
    const float expected[N] = {10.0f, 20.0f, 30.0f, 40.0f};

    float data[N];
    for (int i = 0; i < N; i++) {
        /* Only rank 0 holds the real values; the others hold garbage that the
         * broadcast must replace. */
        data[i] = (rank == 0) ? expected[i] : -1.0f;
    }

    mpi_broadcast(data, N, /*root=*/0);

    for (int i = 0; i < N; i++) {
        assert_float_equal(data[i], expected[i], 0.0f);
    }
    /* And of course everyone now agrees. */
    assert_synced_across_ranks(data, N, 0.0f);
}

/*
 * The integration check that justifies this whole task. Every rank:
 *   - builds the SAME model (same seed) and broadcasts to be certain,
 *   - runs a forward+backward on its OWN distinct batch (=> distinct grads),
 *   - averages the gradients across ranks,
 *   - takes one AdamW step.
 * Because the averaged gradient is identical on every rank and the starting
 * weights were identical, the post-step weights MUST be identical too. That is
 * the invariant that lets N processes train one shared model.
 */
static void test_one_step_keeps_params_in_sync(void **state) {
    (void)state;

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    /* Tiny model so the test is fast and the param block is small. */
    GPTConfig cfg = {.max_seq_len = 8, .vocab_size = 32, .n_embd = 8,
                     .n_head = 2, .n_layer = 1, .ff_dim = 16};
    const int B = 2, T = 4;

    GPTModel model;
    gpt_init(&model, cfg, /*seed=*/1337u);  /* same seed => identical weights */

    /* Belt and suspenders: force every rank to rank 0's exact weights, the
     * same call train.c makes. (With one shared seed they already match; this
     * proves the broadcast path works and removes any doubt.) */
    mpi_broadcast(model.params, model.num_params, /*root=*/0);
    assert_synced_across_ranks(model.params, model.num_params, 0.0f);

    AdamW opt;
    adamw_init(&opt, model.num_params, /*lr=*/1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* Each rank gets a DIFFERENT batch: seed the id generator by rank so the
     * tokens (and therefore the local gradients) genuinely differ per rank. */
    int *inputs  = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    assert_non_null(inputs);
    assert_non_null(targets);
    unsigned int seed = 100u + (unsigned int)rank;
    fill_random_ids(inputs,  B * T, cfg.vocab_size, &seed);
    fill_random_ids(targets, B * T, cfg.vocab_size, &seed);

    gpt_zero_grad(&model);
    gpt_forward(&model, inputs, targets, B, T);
    gpt_backward(&model, inputs, targets, B, T);

    /* With more than one rank the LOCAL gradients should differ — otherwise the
     * test would pass trivially without ever exercising the averaging. Confirm
     * at least one gradient element disagrees across ranks before we average:
     * a nonzero cross-rank spread means the per-rank batches really differed. */
    if (world > 1) {
        assert_true(max_cross_rank_spread(model.grads, model.num_params) > 1e-6f);
    }

    /* The data-parallel core: average gradients, then everyone steps the same. */
    mpi_allreduce_mean(model.grads, model.num_params);
    assert_synced_across_ranks(model.grads, model.num_params, 1e-5f);

    adamw_step(&opt, model.params, model.grads);

    /* The headline assertion: identical models in, identical models out. */
    assert_synced_across_ranks(model.params, model.num_params, 1e-5f);

    free(inputs);
    free(targets);
    adamw_free(&opt);
    gpt_free(&model);
}

int main(int argc, char **argv) {
    /* MPI must be initialized before any collective call and finalized after
     * the last one — so we wrap the cmocka run, not each test. mpi_setup also
     * gives us our rank/size, though the tests query MPI directly themselves. */
    int rank = 0, world = 1;
    mpi_setup(&argc, &argv, &rank, &world);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_allreduce_mean_known_values),
        cmocka_unit_test(test_allreduce_all_ranks_identical),
        cmocka_unit_test(test_broadcast_shares_root_values),
        cmocka_unit_test(test_one_step_keeps_params_in_sync),
    };

    int failures = cmocka_run_group_tests(tests, NULL, NULL);

    mpi_teardown();
    return failures;
}
