add_library(text_parser data/text_parser.c)
target_include_directories(text_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# BPE tokenizer — converts raw bytes <-> token id sequences. Trained from a
# Serbian text corpus. bpe_io adds save/load to a binary file format so we
# don't have to retrain every run.
add_library(bpe tokenizer/bpe.c tokenizer/bpe_io.c)
target_include_directories(bpe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Data loader — loads a tokenized .bin file into RAM and hands out training
# batches with shifted targets for next-token prediction.
add_library(dataloader data/dataloader.c)
target_include_directories(dataloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Neural network layer ops (CPU) — matmul, layernorm, softmax, gelu,
# residual, embedding. Same buffer-in / buffer-out style as the rest of
# the project so the same logic mirrors easily to CUDA later.
add_library(layers model/layers.c)
target_include_directories(layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Full GPT model — wires the layer ops above into one transformer with a
# complete forward pass (tokens -> loss) and backward pass (loss -> grads).
# Links `layers` (the math) and `m` (libm: expf/tanhf/sqrtf/logf used in
# attention, gelu, layernorm and cross-entropy).
add_library(gpt model/gpt.c)
target_include_directories(gpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpt PUBLIC layers m)
