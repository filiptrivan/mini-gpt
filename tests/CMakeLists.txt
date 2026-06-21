include_directories(${CMOCKA_INCLUDE_DIRS})
link_directories(${CMOCKA_LIBRARY_DIRS})

# Shared test helpers (numerical_gradient, etc). Tests that need these
# link against test_utils and #include "test_utils.h" — the helpers/
# directory is exported PUBLIC on this target so the prefix is implicit.
add_library(test_utils helpers/test_utils.c)
target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/helpers)

add_executable(test_hello test_hello.c)
target_link_libraries(test_hello ${CMOCKA_LIBRARIES})
add_test(NAME test_hello COMMAND test_hello)

add_executable(test_text_parser test_text_parser.c)
target_link_libraries(test_text_parser text_parser ${CMOCKA_LIBRARIES})
target_compile_definitions(test_text_parser PRIVATE
    FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/helpers/fixtures")
add_test(NAME test_text_parser COMMAND test_text_parser)

# BPE tokenizer tests.
add_executable(test_bpe test_bpe.c)
target_link_libraries(test_bpe bpe ${CMOCKA_LIBRARIES})
add_test(NAME test_bpe COMMAND test_bpe)

# Data loader tests — exercise init / next_batch / free against a synthetic
# .bin file written to /tmp during each test.
add_executable(test_dataloader test_dataloader.c)
target_link_libraries(test_dataloader dataloader ${CMOCKA_LIBRARIES})
add_test(NAME test_dataloader COMMAND test_dataloader)

# Layer op tests — one test per forward function (residual, gelu, matmul,
# softmax, layernorm, embed).
add_executable(test_layers test_layers.c)
target_link_libraries(test_layers layers test_utils ${CMOCKA_LIBRARIES})
add_test(NAME test_layers COMMAND test_layers)

# Full-model tests — param count, finite loss, nonzero grads, and the
# whole-model numerical gradient check (analytical vs finite-difference).
add_executable(test_gpt test_gpt.c)
target_link_libraries(test_gpt gpt test_utils ${CMOCKA_LIBRARIES})
add_test(NAME test_gpt COMMAND test_gpt)

# AdamW optimizer tests — moment accumulation against hand-computed values,
# loss reduction on a quadratic bowl, and decoupled weight decay.
add_executable(test_adamw test_adamw.c)
target_link_libraries(test_adamw adamw ${CMOCKA_LIBRARIES})
add_test(NAME test_adamw COMMAND test_adamw)

# Training "acid test" — compose the full pipeline (forward -> backward ->
# adamw_step) on a tiny model and assert it overfits one batch (loss falls).
add_executable(test_train test_train.c)
target_link_libraries(test_train gpt adamw test_utils ${CMOCKA_LIBRARIES})
add_test(NAME test_train COMMAND test_train)

# CUDA kernel tests (Task 9 + Task 10) — compiled only on a machine with a CUDA
# toolchain (Colab). Each test runs the CPU function as the oracle and the
# matching CUDA kernel under test, then asserts they agree within tolerance.
if(ENABLE_CUDA)
    # Per-kernel parity. Links `gpt` too because attention_forward/backward (the
    # oracles for the attention kernels) live in gpt.c, not layers.c, and
    # `test_utils` for fill_random_ids (the shared reproducible id generator).
    add_executable(test_cuda_layers test_cuda_layers.cu)
    target_link_libraries(test_cuda_layers layers gpt cuda_layers test_utils ${CMOCKA_LIBRARIES})
    add_test(NAME test_cuda_layers COMMAND test_cuda_layers)

    # Full-model parity: gpt_forward/gpt_backward (CPU) vs gpt_forward_cuda/
    # gpt_backward_cuda, on two identically-seeded models. `test_utils` supplies
    # fill_random_ids for the token batches.
    add_executable(test_gpt_cuda test_gpt_cuda.cu)
    target_link_libraries(test_gpt_cuda gpt cuda_gpt cuda_layers test_utils ${CMOCKA_LIBRARIES})
    add_test(NAME test_gpt_cuda COMMAND test_gpt_cuda)
endif()

# MPI distributed tests (Task 11) — built only when ENABLE_MPI is set (Colab).
# This test is special: it MUST be launched with `mpirun -np 2` so MPI starts
# two ranks that then check they agree after allreduce/broadcast. We register it
# with `add_test` so `ctest` runs it under mpirun automatically; MPIEXEC_*
# variables come from find_package(MPI) and resolve to the local mpirun. Links
# `mpi_utils` (the layer under test), plus `gpt`/`adamw` (the in-sync step test)
# and `test_utils` (fill_random_ids for the per-rank batches).
if(ENABLE_MPI)
    add_executable(test_distributed test_distributed.c)
    target_link_libraries(test_distributed
        mpi_utils gpt adamw test_utils MPI::MPI_C ${CMOCKA_LIBRARIES})
    # ctest invocation: mpirun --allow-run-as-root -np 2 ./test_distributed
    # MPIEXEC_EXECUTABLE / _NUMPROC_FLAG / _PREFLAGS come from FindMPI. Colab
    # runs as root and OpenMPI refuses that without --allow-run-as-root, so we
    # add it here to match the project's documented manual command (CLAUDE.md);
    # OpenMPI accepts the flag whether or not you are actually root.
    add_test(NAME test_distributed
             COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root ${MPIEXEC_PREFLAGS}
                     ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_distributed>)
endif()
