include_directories(${CMOCKA_INCLUDE_DIRS})
link_directories(${CMOCKA_LIBRARY_DIRS})

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
