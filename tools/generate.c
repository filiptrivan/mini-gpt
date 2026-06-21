/*
 * generate — load a trained checkpoint + tokenizer and autoregressively
 * generate text from a prompt.
 *
 * Usage:
 *   generate <checkpoint.ckpt> <tokenizer.bpe> <prompt> [max_new_tokens] [temperature] [seed]
 *
 *     checkpoint.ckpt : a model saved by train.c / gpt_save (see model/gpt_io.h)
 *     tokenizer.bpe   : the BPE tokenizer used to make the training tokens
 *                       (its vocab MUST match the model's vocab_size)
 *     prompt          : the seed text to continue (quote it if it has spaces)
 *     max_new_tokens  : how many tokens to generate         (default 200)
 *     temperature     : sampling randomness                 (default 0.8)
 *                         0      => greedy (always the most likely token)
 *                         <1     => sharper, more confident/repetitive
 *                         1      => the model's raw distribution
 *                         >1     => flatter, more random
 *     seed            : RNG seed for reproducible sampling   (default 1234)
 *
 * HOW AUTOREGRESSIVE GENERATION WORKS:
 *   A GPT predicts a probability distribution for the NEXT token given the
 *   tokens so far. To generate, we run the model on the prompt, sample one token
 *   from the distribution at the last position, append it, and repeat — each new
 *   token becomes part of the input for the next step. The model only sees up to
 *   max_seq_len tokens of context, so once the sequence grows past that we feed
 *   it a sliding window of the most recent tokens.
 *
 * Generated text goes to STDOUT (so it can be redirected/piped); all status and
 * settings go to STDERR. Generation runs on the CPU — it's lightweight inference,
 * no GPU needed.
 */

#include <stdio.h>
#include <stdlib.h>   /* malloc, free, atoi, atof */
#include <string.h>   /* strlen, memcpy */
#include <stdint.h>   /* uint64_t */
#include <math.h>     /* expf */

#include "model/gpt.h"
#include "model/gpt_io.h"
#include "tokenizer/bpe.h"
#include "tokenizer/bpe_io.h"

/*
 * xorshift64* — a tiny, fast pseudo-random generator. We only need cheap,
 * reproducible randomness for sampling, not cryptographic quality. The state is
 * advanced in place; it must be seeded nonzero (the caller guarantees this).
 */
static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* A uniform double in [0, 1). Take the top 53 bits (the mantissa width of a
 * double) and scale, the standard way to turn a 64-bit integer into a uniform
 * float without bias in the low bits. */
static double rng_uniform(uint64_t *state) {
    return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * sample_token — pick the next token id from a row of `V` logits.
 *
 *   logits      : unnormalized scores for each token (length V), from the model.
 *   V           : vocab size.
 *   temperature : see the usage block. <= 0 means greedy (argmax).
 *   rng         : RNG state for stochastic sampling.
 *
 * For temperature sampling we form probabilities ∝ exp(logit / temperature) and
 * draw from that categorical distribution. We subtract the max logit first for
 * numerical stability (so exp() never overflows — it shifts every probability by
 * the same constant factor, which cancels in the normalization). We do it in two
 * passes — first the normalizer (sum), then a walk of the cumulative
 * distribution until it passes a uniform draw scaled by that sum — but cache
 * each token's exp() in `weights` so the second pass reuses it instead of
 * recomputing (exp is by far the costliest operation here).
 *
 *   weights : caller-provided scratch of length >= V, reused across calls so we
 *             don't malloc per generated token. Filled and consumed internally.
 */
static int sample_token(const float *logits, int V, float temperature,
                        uint64_t *rng, double *weights) {
    /* Greedy: the single most likely token. Deterministic, ignores the RNG. */
    if (temperature <= 0.0f) {
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < V; i++) {
            if (logits[i] > best_logit) { best_logit = logits[i]; best = i; }
        }
        return best;
    }

    /* Max logit for the stability shift. */
    float maxv = logits[0];
    for (int i = 1; i < V; i++) {
        if (logits[i] > maxv) maxv = logits[i];
    }

    /* Pass 1: compute each token's weight exp((logit-max)/temp) ONCE, caching it
     * in `weights`, while summing to get the normalizer. */
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        double w = exp((double)(logits[i] - maxv) / (double)temperature);
        weights[i] = w;
        sum += w;
    }

    /* Draw a target point in [0, sum) and walk the cached weights until the
     * cumulative mass crosses it — that's the sampled token. */
    double target = rng_uniform(rng) * sum;
    double acc = 0.0;
    for (int i = 0; i < V; i++) {
        acc += weights[i];
        if (acc >= target) return i;
    }
    return V - 1;  /* floating-point fallback: target == sum exactly */
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) {
        fprintf(stderr,
                "Usage: %s <checkpoint.ckpt> <tokenizer.bpe> <prompt> "
                "[max_new_tokens] [temperature] [seed]\n", argv[0]);
        return 1;
    }
    const char *ckpt_path   = argv[1];
    const char *bpe_path    = argv[2];
    const char *prompt      = argv[3];
    int   max_new_tokens    = (argc >= 5) ? atoi(argv[4])        : 200;
    float temperature       = (argc >= 6) ? (float)atof(argv[5]) : 0.8f;
    uint64_t seed           = (argc >= 7) ? (uint64_t)strtoull(argv[6], NULL, 10)
                                          : 1234ull;
    if (seed == 0) seed = 1ull;          /* xorshift needs a nonzero seed */
    if (max_new_tokens <= 0) {
        fprintf(stderr, "max_new_tokens must be positive (got %d)\n",
                max_new_tokens);
        return 1;
    }

    /* ---- Load the model checkpoint ---- */
    GPTModel model;
    if (gpt_load(&model, ckpt_path) != 0) {
        fprintf(stderr, "Cannot load checkpoint: %s\n", ckpt_path);
        return 1;
    }
    int V = model.config.vocab_size;
    int max_seq_len = model.config.max_seq_len;

    /* ---- Load the tokenizer ---- */
    BPETokenizer *tok = bpe_load(bpe_path);
    if (tok == NULL) {
        fprintf(stderr, "Cannot load tokenizer: %s\n", bpe_path);
        gpt_free(&model);
        return 1;
    }
    /* The tokenizer must not emit ids the model can't embed. Equal vocab is the
     * normal case (both 512); a larger tokenizer vocab would index past the
     * embedding table, so we refuse it rather than read out of bounds. */
    if (tok->vocab_size > V) {
        fprintf(stderr,
                "Tokenizer vocab (%d) exceeds model vocab (%d) — mismatched files\n",
                tok->vocab_size, V);
        bpe_free(tok);
        gpt_free(&model);
        return 1;
    }

    /* ---- Encode the prompt into starting tokens ---- */
    size_t prompt_count = 0;
    int *prompt_tokens = bpe_encode(tok, (const unsigned char *)prompt,
                                    strlen(prompt), &prompt_count);
    if (prompt_count == 0) {
        fprintf(stderr, "Prompt is empty — give some seed text to continue\n");
        free(prompt_tokens);
        bpe_free(tok);
        gpt_free(&model);
        return 1;
    }

    /* One buffer holds the whole sequence (prompt + everything we generate), so
     * we never reallocate mid-loop. */
    size_t capacity = prompt_count + (size_t)max_new_tokens;
    int *tokens = malloc(capacity * sizeof(int));
    /* Reused scratch the sampler fills with per-token weights — allocated once
     * here rather than once per generated token. */
    double *weights = malloc((size_t)V * sizeof(double));
    if (tokens == NULL || weights == NULL) {
        fprintf(stderr, "Out of memory\n");
        free(tokens);
        free(weights);
        free(prompt_tokens);
        bpe_free(tok);
        gpt_free(&model);
        return 1;
    }
    memcpy(tokens, prompt_tokens, prompt_count * sizeof(int));
    free(prompt_tokens);
    size_t count = prompt_count;

    fprintf(stderr,
            "Model: %d params | vocab %d | ctx %d | prompt %zu tokens | "
            "generating %d | temp %.2f | seed %llu\n",
            model.num_params, V, max_seq_len, prompt_count, max_new_tokens,
            (double)temperature, (unsigned long long)seed);

    /* Echo the prompt to stdout so the output reads as one continuous passage
     * (prompt followed by the model's continuation). */
    fputs(prompt, stdout);
    fflush(stdout);

    /* ---- Autoregressive generation loop ---- */
    for (int step = 0; step < max_new_tokens; step++) {
        /* Feed the last min(count, max_seq_len) tokens — the model's context
         * window. Once the sequence is longer than the window we slide it. */
        int T = (count < (size_t)max_seq_len) ? (int)count : max_seq_len;
        const int *ctx = tokens + (count - (size_t)T);

        /* Forward with targets == NULL: no loss, just logits/probs. B = 1. */
        gpt_forward(&model, ctx, NULL, /*B=*/1, T);

        /* Logits for the LAST position are the prediction for the next token.
         * a.logits has shape (B, T, V) with B = 1, so row T-1 starts at (T-1)*V. */
        const float *logits_last = model.a.logits + (size_t)(T - 1) * V;
        int next = sample_token(logits_last, V, temperature, &seed, weights);

        tokens[count++] = next;

        /* Stream the new token's bytes immediately so generation is visible as
         * it happens. Decoding one id gives that token's byte string. */
        size_t blen = 0;
        unsigned char *bytes = bpe_decode(tok, &next, 1, &blen);
        if (bytes != NULL) {
            fwrite(bytes, 1, blen, stdout);
            free(bytes);
        }
        fflush(stdout);
    }
    fputc('\n', stdout);

    /* ---- Cleanup ---- */
    free(tokens);
    free(weights);
    bpe_free(tok);
    gpt_free(&model);
    return 0;
}
