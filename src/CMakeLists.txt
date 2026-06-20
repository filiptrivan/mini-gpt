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

# AdamW optimizer — turns gradients into weight updates (the part of training
# that improves the model). Operates on the model's flat params/grads blocks,
# so it has no dependency on the GPT model itself. Links `m` for sqrtf/powf.
add_library(adamw optimizer/adamw.c)
target_include_directories(adamw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adamw PUBLIC m)

# CUDA layer kernels (Task 9) — GPU forward passes for matmul and layernorm.
# Built only when ENABLE_CUDA is set (Colab); the Mac CPU build skips this
# entirely so day-to-day development never needs a CUDA toolchain. The Task 9
# tests link this alongside the CPU `layers` lib and compare their outputs.
if(ENABLE_CUDA)
    add_library(cuda_layers cuda/matmul.cu cuda/layernorm.cu)
    # PUBLIC export of src/ so includes like "cuda/cuda_layers.cuh" and
    # "model/layers.h" resolve for anything that links this target.
    target_include_directories(cuda_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# train — the CPU training program (the "acid test"): load tokens, then loop
# dataloader -> forward -> backward -> adamw_step and watch the loss fall.
# Pure CPU for now; CUDA/MPI wiring arrives in later tasks.
add_executable(train train.c)
target_link_libraries(train gpt adamw dataloader)
