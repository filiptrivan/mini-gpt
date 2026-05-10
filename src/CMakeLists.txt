add_library(text_parser data/text_parser.c)
target_include_directories(text_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# BPE tokenizer — converts raw bytes <-> token id sequences. Trained from a
# Serbian text corpus. Currently a stub; real implementations land in Task 3.
add_library(bpe tokenizer/bpe.c)
target_include_directories(bpe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
