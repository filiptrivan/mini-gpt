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
pkg_check_modules(CMOCKA REQUIRED cmocka)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
