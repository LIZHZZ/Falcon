cmake_minimum_required(VERSION 3.10)

# 设置策略以允许使用旧的FindCUDA模块（如果需要）
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# 在project()之前设置CUDA标志，确保CMake确定CUDA编译器ID时能使用
# 这修复了cmath找不到的问题
set(CMAKE_CUDA_FLAGS_INIT "-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -std=c++17 -Xcompiler=-std=c++17")

project(cuCompressor-all)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)


# CUDA 设置
# 设置 CUDA 编译器使用 C++17 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 使用旧的FindCUDA模块（兼容性）
find_package(CUDA REQUIRED)

# 确保CUDA编译器能找到C++标准库（修复cmath找不到的问题）
# 添加C++标准库包含路径
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/x86_64-linux-gnu/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -Xcompiler=-std=c++17")
set(CMAKE_CUDA_ARCHITECTURES 70) # 89 86 90
# 包含目录
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

# 添加测试模块
add_subdirectory(test)


# 查找 GTest 和 Google Benchmark
find_package(GTest REQUIRED)
# find_package(benchmark REQUIRED)


# 添加库不存在的下载方法
if (NOT GTest_FOUND)
    message(STATUS "GTest not found. Downloading and building...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0  # 目前我的版本
    )
    FetchContent_MakeAvailable(googletest)
endif()


# if (NOT benchmark_FOUND)
#     message(STATUS "Google Benchmark library not found. Downloading and building it locally...")
    
#     # 设置下载目录
#     include(FetchContent)
#     FetchContent_Declare(
#         benchmark
#         GIT_REPOSITORY https://github.com/google/benchmark.git
#         GIT_TAG v1.6.1 # 目前我的版本
#     )
#     FetchContent_MakeAvailable(benchmark)
# endif()







