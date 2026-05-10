add_library(text_parser data/text_parser.c)
target_include_directories(text_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# BPE tokenizer — converts raw bytes <-> token id sequences. Trained from a
# Serbian text corpus. bpe_io adds save/load to a binary file format so we
# don't have to retrain every run.
add_library(bpe tokenizer/bpe.c tokenizer/bpe_io.c)
target_include_directories(bpe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
