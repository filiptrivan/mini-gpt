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
# layers.c calls tanhf/expf/sqrtf from the math library (libm). On macOS libm
# is part of the default C runtime so linking "just works"; on Linux (e.g.
# Colab) it must be linked explicitly or the link fails with "undefined
# reference to tanhf/expf/sqrtf". PUBLIC so every target that links `layers`
# (test_layers, gpt, test_cuda_layers, ...) inherits -lm transitively.
target_link_libraries(layers PUBLIC m)

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

# CUDA layer kernels (Task 9 + Task 10) — GPU forward AND backward passes for
# every layer op. Built only when ENABLE_CUDA is set (Colab); the Mac CPU build
# skips this entirely so day-to-day development never needs a CUDA toolchain.
# The parity tests link this alongside the CPU `layers`/`gpt` libs and compare
# their outputs kernel by kernel.
if(ENABLE_CUDA)
    add_library(cuda_layers
        cuda/matmul.cu          # tiled matmul (fwd) + the two gradient matmuls (bwd)
        cuda/layernorm.cu       # row-wise layernorm (fwd/bwd, atomicAdd on gamma/beta)
        cuda/elementwise.cu     # gelu + residual (fwd/bwd)
        cuda/softmax.cu         # row-wise softmax (fwd/bwd)
        cuda/embedding.cu       # token+pos embedding (fwd, bwd scatter via atomicAdd)
        cuda/cross_entropy.cu   # per-position loss (fwd) + fused softmax+CE grad (bwd)
        cuda/attention.cu)      # causal multi-head self-attention (fwd/bwd)
    # PUBLIC export of src/ so includes like "cuda/cuda_layers.cuh" and
    # "model/layers.h" resolve for anything that links this target.
    target_include_directories(cuda_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # Full-model GPU forward/backward (Task 10): wires the _device launchers in
    # cuda_layers into one resident pass. Reads the model's layout (GPTModel,
    # gpt_ensure_acts) from the `gpt` library, so it links both.
    add_library(cuda_gpt cuda/gpt_cuda.cu)
    target_include_directories(cuda_gpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cuda_gpt PUBLIC cuda_layers gpt)
endif()

# MPI data-parallel layer (Task 11) — MPI_Init/Finalize plus the two collective
# operations training needs: average gradients across ranks (allreduce-mean) and
# share rank 0's initial weights (broadcast). Built only when ENABLE_MPI is set
# (Colab); the Mac CPU build skips it, and train.c's MPI calls are #ifdef'd out.
# find_package(MPI) in the root CMakeLists already located the headers/libs; the
# MPI::MPI_C imported target carries the right include path and link flags.
if(ENABLE_MPI)
    add_library(mpi_utils distributed/mpi_utils.c)
    target_include_directories(mpi_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(mpi_utils PUBLIC MPI::MPI_C)
endif()

# train — the training program (the "acid test"): load tokens, then loop
# dataloader -> forward -> backward -> adamw_step and watch the loss fall. When
# ENABLE_MPI is on it also links the MPI layer and runs data-parallel under
# mpirun (broadcast weights, average gradients each step); otherwise it is the
# plain single-process CPU loop.
add_executable(train train.c)
target_link_libraries(train gpt adamw dataloader)
if(ENABLE_MPI)
    target_link_libraries(train mpi_utils MPI::MPI_C)
endif()
