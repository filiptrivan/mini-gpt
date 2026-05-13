#ifndef BPE_IO_H
#define BPE_IO_H

/* Persistence layer for BPETokenizer: write a trained tokenizer to a
 * binary file and read it back. The file format only stores the merge
 * rules — the vocab is fully reconstructible by replaying the merges
 * on top of the 256 base byte tokens, which keeps the file small.
 *
 * File layout (all integers are 32-bit native endian):
 *
 *   Offset  Bytes   Field
 *   ------  -----   ---------------------------------------------------
 *      0      4    magic = 'B','P','E','\0' = 0x00455042 little-endian
 *      4      4    version (currently 1)
 *      8      4    num_merges (N)
 *     12     8*N   merges (N pairs of int32 a,b)
 *
 * Endianness: native. The project targets Mac M2 + Linux x86_64 (Colab),
 * both little-endian, so we don't byte-swap. A file written on one
 * little-endian machine reads correctly on another. */

#include "tokenizer/bpe.h"

/*
 * bpe_save — serialize a (possibly untrained) tokenizer to a file.
 *
 *   tok      : tokenizer to save. May be untrained (num_merges == 0);
 *              the file will then contain only the header.
 *   filepath : NUL-terminated path to write. File is overwritten if it
 *              already exists.
 *
 * Returns: 0 on success, -1 on any error (cannot open file, short write).
 *
 * Does not take ownership of `tok` — the caller still owns it.
 */
int bpe_save(const BPETokenizer *tok, const char *filepath);

/*
 * bpe_load — deserialize a tokenizer from a file produced by bpe_save.
 *
 *   filepath : NUL-terminated path to read.
 *
 * On success: returns a freshly malloc'd BPETokenizer with vocab and
 * merges fully reconstructed. Caller must call bpe_free() exactly once.
 *
 * Returns NULL on any error: file missing, bad magic, version mismatch,
 * truncated/short read, or allocation failure.
 */
BPETokenizer *bpe_load(const char *filepath);

#endif /* BPE_IO_H */
