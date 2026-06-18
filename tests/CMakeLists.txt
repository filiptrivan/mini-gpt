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
