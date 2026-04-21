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
