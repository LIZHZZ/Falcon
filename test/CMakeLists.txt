cmake_minimum_required(VERSION 3.11)

# 设置策略以允许使用旧的FindCUDA模块
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# 在project()之前设置CUDA标志，确保CMake确定CUDA编译器ID时能使用
# 这修复了cmath找不到的问题
set(CMAKE_CUDA_FLAGS_INIT "-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -std=c++17 -Xcompiler=-std=c++17")

project(CompressionTests CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 标志
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler=-std=c++17")

# 确保CUDA编译器能找到C++标准库（修复cmath找不到的问题）
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/x86_64-linux-gnu/c++/11")

# 查找 nvcomp
find_package(nvcomp CONFIG PATHS 
    /usr/lib/x86_64-linux-gnu/nvcomp/11/cmake/nvcomp
    /usr/lib/x86_64-linux-gnu/cmake/nvcomp
    NO_DEFAULT_PATH
)

if(nvcomp_FOUND)
    message(STATUS "Found nvcomp using CMake config")
    set(NVCOMP_LIBRARIES nvcomp::nvcomp)
    set(NVCOMP_INCLUDE_DIRS "")  # 由目标自己处理
    # get_target_property(NVCOMP_INCLUDE_DIRS nvcomp::nvcomp INTERFACE_INCLUDE_DIRECTORIES)
else()
    set(NVCOMP_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include)
endif()

# 添加子项目
add_subdirectory(baseline/ALP)
add_subdirectory(baseline/ndzip)
add_subdirectory(baseline/elf)
add_subdirectory(baseline/elf_star)
add_subdirectory(baseline/elf_star_g)
add_subdirectory(baseline/ALP-G)
add_subdirectory(baseline/gorilla)
add_subdirectory(baseline/chimp128)

add_subdirectory(baseline/patas)

message(STATUS "✓ ALP-G library added: alpg_lib target is available")

# 查找依赖包
find_package(GTest REQUIRED)
# find_package(benchmark REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDA REQUIRED)

message(STATUS "CUDA Include Directories: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# 设置 CUDA 架构
# set(CMAKE_CUDA_ARCHITECTURES 70)

# CPU源文件库
add_library(Falcon_cpu_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/cpu/Falcon_basic_compressor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/Falcon_basic_decompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/output_bit_stream.cc
    ${CMAKE_SOURCE_DIR}/src/utils/input_bit_stream.cc
)

target_include_directories(Falcon_cpu_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

# ==================== 重要修复：创建共享的CUDA对象库 ====================
# 这避免了多个目标重复编译相同的CUDA文件


# GPU源文件编译为对象库（关键修复）
cuda_add_library(Falcon_gpu_lib STATIC
    # ${CMAKE_SOURCE_DIR}/src/gpu/FalconCompressor_1024.cu
    # ${CMAKE_SOURCE_DIR}/src/gpu/FalconDecompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_pipeline.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_pipeline.cu
)

target_include_directories(Falcon_gpu_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

# 为 Falcon_gpu_lib 设置编译选项以降低内存使用
target_compile_options(Falcon_gpu_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-O2
        -lineinfo
        --dopt=on
    >
)

# 数据工具库
add_library(dataset_utils_lib STATIC
    data/dataset_utils.cpp
)

target_include_directories(dataset_utils_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/test/data
)

# ==================== 全局包含目录 ====================
include_directories(
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include
    ${CMAKE_SOURCE_DIR}/test/data
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/elf
    ${CMAKE_SOURCE_DIR}/test/baseline/elf_star
    ${CMAKE_SOURCE_DIR}/test/baseline/elf_star_g
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
)

# ==================== CPU测试目标 ====================
add_executable(test_Falcon_cpu 
    Falcon_basic_test.cpp
)
target_link_libraries(test_Falcon_cpu PRIVATE 
    Falcon_cpu_lib
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
)

# ==================== GPU测试目标（使用共享库） ====================

# test_gpu_nopack
cuda_add_executable(test_gpu_nopack 
    Falcon_test_nopack.cu
)
target_link_libraries(test_gpu_nopack 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu_spare
cuda_add_executable(test_gpu_spare
    Falcon_test_spare.cu
)
target_link_libraries(test_gpu_spare
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu_br
cuda_add_executable(test_gpu_br 
    Falcon_test_br.cu
)
target_link_libraries(test_gpu_br 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu
cuda_add_executable(test_gpu 
    Falcon_test.cu
)
target_link_libraries(test_gpu 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)


# test_muti_3step_block
cuda_add_executable(test_muti_3step_block 
    muti_type_3step_block.cu
)
target_link_libraries(test_muti_3step_block 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# test_muti_3step_noblock
cuda_add_executable(test_muti_3step_noblock 
    muti_type_3step_noblock.cu
)
target_link_libraries(test_muti_3step_noblock 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# ==================== 其他测试目标 ====================

# ALP测试
add_executable(test_ALP 
    test_ALP.cpp
)
target_link_libraries(test_ALP PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

cuda_add_executable(test_ALP_GPU 
    test_ALP_GPU.cu
)
target_link_libraries(test_ALP_GPU 
    alpg_lib
    alpg_headers
    dataset_utils_lib
    ${CUDA_LIBRARIES}
    GTest::GTest 
    GTest::Main 
    pthread
    # ALP
)

# 添加 ALP-G 包含目录到 test_ALP_GPU
target_include_directories(test_ALP_GPU PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# 设置CUDA编译属性
set_target_properties(test_ALP_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 新版本：使用ALP-G GPU接口的测试
cuda_add_executable(test_ALP_GPU_v2 
    test_ALP_GPU_v2.cu
)
target_link_libraries(test_ALP_GPU_v2 
    alpg_lib
    alpg_headers
    dataset_utils_lib
    ${CUDA_LIBRARIES}
    GTest::GTest 
    GTest::Main 
    pthread
)

# 添加 ALP-G 包含目录到 test_ALP_GPU_v2
target_include_directories(test_ALP_GPU_v2 PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# 设置CUDA编译属性
set_target_properties(test_ALP_GPU_v2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)



# ndzip测试
cuda_add_executable(test_ndzip 
    test_ndzip.cpp
)
target_link_libraries(test_ndzip 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    ndzip-cuda 
    io 
    Boost::program_options 
    GTest::Main 
    ndzip 
    pthread
)
target_include_directories(test_ndzip PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io
)
target_compile_options(test_ndzip PRIVATE ${NDZIP_CUDA_FLAGS})

# ELF测试
add_executable(test_elf 
    test_elf.cpp
)
target_link_libraries(test_elf PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf
)

# ELF Star测试
add_executable(test_elf_star 
    test_elf_star.cpp
)
target_link_libraries(test_elf_star PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star
)


# ELF Star GPU测试
cuda_add_executable(test_elf_star_g 
    test_elf_star_g.cu
)
target_link_libraries(test_elf_star_g 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star_g
)

# Gorilla 测试
add_executable(test_gorilla
    test_gorilla.cpp
)
target_link_libraries(test_gorilla PRIVATE
    dataset_utils_lib
    pthread
    gorilla
)
target_include_directories(test_gorilla PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/gorilla
)

# Chimp128 测试
add_executable(test_chimp
    test_chimp.cpp
)
target_link_libraries(test_chimp PRIVATE
    dataset_utils_lib
    pthread
    chimp
)
target_include_directories(test_chimp PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/chimp128
)

    # Patas 测试
    add_executable(test_patas
        test_patas.cpp
    )
    target_link_libraries(test_patas PRIVATE
        dataset_utils_lib
        pthread
    )
    target_include_directories(test_patas PRIVATE
        ${CMAKE_SOURCE_DIR}/test/baseline/patas
    )

#=====================float=============================
# test_float_muti_stream
cuda_add_executable(test_muti_stream
    Falcon_test_muti_steam_f64.cu
)
target_link_libraries(test_muti_stream
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)


# test_float_muti_stream
cuda_add_executable(test_float_muti_stream 
    test_float_falcon.cu
)
target_link_libraries(test_float_muti_stream 
    Falcon_gpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# ALP测试
add_executable(test_float_ALP 
    test_float_ALP.cpp
)
target_link_libraries(test_float_ALP PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# cuda_add_executable(test_float_ALP_GPU_v2 
#     test_float_ALP_GPU_v2.cu
# )
# target_link_libraries(test_float_ALP_GPU_v2 
#     alpg_lib
#     alpg_headers
#     dataset_utils_lib
#     ${CUDA_LIBRARIES}
#     GTest::GTest 
#     GTest::Main 
#     pthread
# )

cuda_add_executable(test_float_ALP_GPU 
    test_float_ALP_GPU.cu
)
target_link_libraries(test_float_ALP_GPU 
    alpg_lib
    alpg_headers
    dataset_utils_lib
    ${CUDA_LIBRARIES}
    GTest::GTest 
    GTest::Main 
    pthread
    # ALP
)

# 添加 ALP-G 包含目录到 test_float_ALP_GPU
target_include_directories(test_float_ALP_GPU PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# 设置CUDA编译属性
set_target_properties(test_float_ALP_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# test_float_muti_stream
cuda_add_executable(test_float_Lz4 
    test_float_Lz4.cu
)
target_link_libraries(test_float_Lz4
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
if(NVCOMP_INCLUDE_DIRS)
    target_include_directories(test_float_Lz4 BEFORE PRIVATE 
        ${NVCOMP_INCLUDE_DIRS}
    )
endif()

add_executable(test_float_elf
test_float_elf.cpp
)
target_link_libraries(test_float_elf PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf
)

# ELF Star测试
add_executable(test_float_elf_star 
    test_float_elf_star.cpp
)
target_link_libraries(test_float_elf_star PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star
)


# ELF Star GPU测试
cuda_add_executable(test_float_elf_star_g 
    test_float_elf_star_g.cu
)
target_link_libraries(test_float_elf_star_g 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star_g
)

# gdeflate测试
    # 暂时禁用(API兼容性问题)
    cuda_add_executable(test_float_gdeflate 
        test_float_gdeflate.cu
    )
    target_link_libraries(test_float_gdeflate
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_float_gdeflate BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

    

        # Snappy测试
    add_executable(test_float_Snappy 
        test_float_Snappy.cpp
    )
    target_link_libraries(test_float_Snappy
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_float_Snappy BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

    cuda_add_executable(test_float_ndzip 
    test_float_ndzip.cpp
)
target_link_libraries(test_float_ndzip 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    ndzip-cuda 
    io 
    Boost::program_options 
    GTest::Main 
    ndzip 
    pthread
)
target_include_directories(test_float_ndzip PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io
)

# bitcomp测试
    cuda_add_executable(test_float_bitcomp 
        test_float_bitcomp.cu
    )
    target_link_libraries(test_float_bitcomp
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_float_bitcomp BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

# ==================== nvcomp相关测试（如果找到） ====================
if(nvcomp_FOUND)
    # LZ4测试
    cuda_add_executable(test_LZ4 
        test_LZ4.cu
    )
    target_link_libraries(test_LZ4
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_LZ4 BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

    # gdeflate测试 - 暂时禁用(API兼容性问题)
    cuda_add_executable(test_gdeflate 
        test_gdeflate.cu
    )
    target_link_libraries(test_gdeflate
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_gdeflate BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

    # Snappy测试
    add_executable(test_Snappy 
        test_Snappy.cpp
    )
    target_link_libraries(test_Snappy
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_Snappy BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()

    # bitcomp测试
    cuda_add_executable(test_bitcomp 
        test_bitcomp.cu
    )
    target_link_libraries(test_bitcomp
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    if(NVCOMP_INCLUDE_DIRS)
        target_include_directories(test_bitcomp BEFORE PRIVATE 
            ${NVCOMP_INCLUDE_DIRS}
        )
    endif()
endif()

# ==================== 设置输出目录（避免冲突） ====================
set_target_properties(
    test_gpu test_gpu_nopack test_gpu_spare test_gpu_br 
    # test_gpu_string
    test_muti_stream test_muti_3step_block test_muti_3step_noblock
    test_ALP test_ndzip test_elf test_elf_star test_elf_star_g
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# ==================== 添加自定义命令用于清理 ====================
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/test
    COMMENT "Cleaning all build files..."
)

# ==================== 打印配置信息 ====================
message(STATUS "=================================")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  nvcomp Found: ${nvcomp_FOUND}")
message(STATUS "=================================")