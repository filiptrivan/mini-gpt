add_library(text_parser data/text_parser.c)
target_include_directories(text_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
