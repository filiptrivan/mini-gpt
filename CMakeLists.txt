cmake_minimum_required(VERSION 3.18)
project(mini_gpt C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Build CUDA kernels" OFF)
option(ENABLE_MPI  "Enable MPI data parallelism" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(USE_MPI)
endif()

find_package(PkgConfig REQUIRED)
# Ask the system tool `pkg-config` where the CMocka test framework is installed
# (Homebrew on Mac, apt on Linux/Colab — different paths on each machine).
# This auto-creates variables prefixed with CMOCKA_ that we use in tests/CMakeLists.txt:
#   CMOCKA_INCLUDE_DIRS  → folder containing cmocka.h        (passed via include_directories)
#   CMOCKA_LIBRARY_DIRS  → folder containing libcmocka       (passed via link_directories)
#   CMOCKA_LIBRARIES     → library name to link against      (passed via target_link_libraries)
# REQUIRED makes configure fail immediately with a clear error if CMocka isn't installed.
pkg_check_modules(CMOCKA REQUIRED cmocka)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)
