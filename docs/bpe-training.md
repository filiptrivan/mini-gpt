# BPE Training, Walked by Hand

A worked example of how the BPE training algorithm in `src/tokenizer/bpe.c`
turns a raw byte string into a learned vocabulary of merged tokens. Useful as
a reference when reading the implementation.

## What "training" actually does

In one sentence: it looks at the input data, finds the most frequent adjacent
pair of tokens, and creates a new token that represents that pair. Then
repeats. Each repetition shrinks the input sequence by replacing pairs with
their merged id.

That is the whole algorithm. The "intelligence" is that by always picking the
most frequent pair, BPE naturally discovers common substrings in the data —
first single bigrams (`aa`), then trigrams (`aaa`), then whole common words.

## Worked example: train on `aaabdaaabac` with 3 merges

### Setup: convert input bytes to a working sequence

Input string `aaabdaaabac` is 11 bytes. Each byte's value:

- `'a'` = 97
- `'b'` = 98
- `'c'` = 99
- `'d'` = 100

Initial working sequence (a copy that gets mutated during training):

```
[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
 a   a   a   b   d   a   a   a   b   a   c
```

### Iteration 1: find the most frequent pair

Walk through the sequence and tally every adjacent pair:

| Pair | Count |
|------|-------|
| (97, 97) = `aa` | **4** |
| (97, 98) = `ab` | 2 |
| (98, 100) = `bd` | 1 |
| (100, 97) = `da` | 1 |
| (98, 97) = `ba` | 1 |
| (97, 99) = `ac` | 1 |

Winner: **`(97, 97)`** with count 4.

Update tokenizer state:

- `merges[0] = {97, 97}` (the rule we just learned).
- New token id: `256` (because `merges[0]` always produces id `256 + 0`).
- `vocab[256]` is set to the byte sequence `"aa"` (2 bytes), formed by
  concatenating `vocab[97].bytes` + `vocab[97].bytes`.

Replace every `(97, 97)` in the working sequence with `256`:

```
Before: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
After:  [256, 97, 98, 100, 256, 97, 98, 97, 99]
```

The replacement is greedy left-to-right: the first `(97, 97)` at positions
0–1 becomes `256`, then we skip past it. The third `97` at the original
position 2 is left alone — no second `97` next to it after the merge consumed
positions 0–1. The next match is at original positions 5–6.

### Iteration 2: find the most frequent pair in the new sequence

Working sequence: `[256, 97, 98, 100, 256, 97, 98, 97, 99]`

| Pair | Count |
|------|-------|
| (256, 97) | **2** |
| (97, 98)  | **2** |
| (98, 100) | 1 |
| (100, 256) | 1 |
| (98, 97) | 1 |
| (97, 99) | 1 |

Two pairs tied at 2. Tie-break rule: **first one encountered when scanning
left-to-right.** `(256, 97)` appears at position 0; `(97, 98)` first appears
at position 1. Winner: **`(256, 97)`**.

Update:

- `merges[1] = {256, 97}`.
- New token id: `257`.
- `vocab[257]` = `vocab[256].bytes` + `vocab[97].bytes` = `"aa"` + `"a"` = `"aaa"` (3 bytes).

Replace:

```
Before: [256, 97, 98, 100, 256, 97, 98, 97, 99]
After:  [257, 98, 100, 257, 98, 97, 99]
```

### Iteration 3: final round

Working sequence: `[257, 98, 100, 257, 98, 97, 99]`

| Pair | Count |
|------|-------|
| (257, 98) | **2** |
| (98, 100) | 1 |
| (100, 257) | 1 |
| (98, 97) | 1 |
| (97, 99) | 1 |

Winner: **`(257, 98)`**.

Update:

- `merges[2] = {257, 98}`.
- `vocab[258]` = `"aaa"` + `"b"` = `"aaab"` (4 bytes).

Replace:

```
Before: [257, 98, 100, 257, 98, 97, 99]
After:  [258, 100, 258, 97, 99]
```

## Final state after 3 merges

- `tok->num_merges == 3`
- `tok->vocab_size == 259` (256 base + 3 merged)

| Merge index | Rule | Resulting token id | Vocab entry bytes | Length |
|-------------|------|--------------------|-------------------|--------|
| 0 | (97, 97)  | 256 | `aa`   | 2 |
| 1 | (256, 97) | 257 | `aaa`  | 3 |
| 2 | (257, 98) | 258 | `aaab` | 4 |

Notice the elegance: each merge can use *previously merged* tokens. Merge 1
uses token 256 (created by merge 0). Merge 2 uses token 257 (created by
merge 1). This is how BPE builds up to long substrings even though each step
only merges a pair.

## How this gets used at encode time

Given the trained tokenizer above, encoding `"aaabd"` would:

1. Start as bytes: `[97, 97, 97, 98, 100]`.
2. Apply `merges[0] = (97, 97) → 256`: `[256, 97, 98, 100]`.
3. Apply `merges[1] = (256, 97) → 257`: `[257, 98, 100]`.
4. Apply `merges[2] = (257, 98) → 258`: `[258, 100]`.
5. Done. Result: `[258, 100]` — just 2 tokens for 5 bytes.

That is *compression*: the model now sees 2 token slots instead of 5. The
substring `aaab` got "learned" as a single concept (token 258).

## Concept checks

Three questions to sanity-check understanding:

1. **Why is the merge index also the result token id?** Because the first
   merge produces the first new token (id 256), the second produces the next
   one (id 257), and so on. Index `i` always produces id `256 + i` — no
   separate field needed on the `Merge` struct.

2. **Why is BPE training "greedy"?** Each step picks the locally best merge
   (most frequent pair) without looking ahead. Greedy is not guaranteed
   optimal in general, but for BPE it is simple, fast, deterministic, and
   works well enough in practice. The same algorithm powers GPT-2's
   tokenizer.

3. **What does the `merges` array represent at encode time?** An ordered
   list of "find this pair, replace with this new id" rules, applied in
   priority order. The index in the array IS the priority.
