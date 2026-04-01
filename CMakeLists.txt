cmake_minimum_required(VERSION 3.10)

# FindCUDA （ ）
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# project() CUDA ， CMake CUDA ID
# cmath
set(CMAKE_CUDA_FLAGS_INIT "-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -std=c++17 -Xcompiler=-std=c++17")

project(cuCompressor-all)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)


# CUDA
# CUDA C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# FindCUDA （ ）
find_package(CUDA REQUIRED)

# CUDA C++ （ cmath ）
# C++
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/x86_64-linux-gnu/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -Xcompiler=-std=c++17")
set(CMAKE_CUDA_ARCHITECTURES 70) # 89 86 90
# translated comment
include_directories(include)
include_directories(src/utils)

set(SRC_GPU
    
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconCompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconDecompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_pipeline.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_pipeline.cu
    src/utils/output_bit_stream.cc
    src/utils/input_bit_stream.cc
)

# translated comment
add_subdirectory(test)


# GTest Google Benchmark
find_package(GTest REQUIRED)
# find_package(benchmark REQUIRED)


# translated comment
if (NOT GTest_FOUND)
    message(STATUS "GTest not found. Downloading and building...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0  # translated comment
    )
    FetchContent_MakeAvailable(googletest)
endif()


# if (NOT benchmark_FOUND)
#     message(STATUS "Google Benchmark library not found. Downloading and building it locally...")
    
# translated comment
#     include(FetchContent)
#     FetchContent_Declare(
#         benchmark
#         GIT_REPOSITORY https://github.com/google/benchmark.git
# GIT_TAG v1.6.1 #
#     )
#     FetchContent_MakeAvailable(benchmark)
# endif()







