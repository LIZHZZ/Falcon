cmake_minimum_required(VERSION 3.11)

# Set policy to allow using the legacy FindCUDA module
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# Set CUDA flags before project() so CMake can determine the CUDA compiler ID
# This fixes the issue of not finding <cmath>
set(CMAKE_CUDA_FLAGS_INIT "-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -std=c++17 -Xcompiler=-std=c++17")

project(CompressionTests CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler=-std=c++17")

# Ensure the CUDA compiler can find the C++ standard library (fixes <cmath> not found)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/c++/11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I/usr/include/x86_64-linux-gnu/c++/11")

# Find nvcomp
find_package(nvcomp CONFIG PATHS 
    /usr/lib/x86_64-linux-gnu/nvcomp/11/cmake/nvcomp
    /usr/lib/x86_64-linux-gnu/cmake/nvcomp
    NO_DEFAULT_PATH
)

if(nvcomp_FOUND)
    message(STATUS "Found nvcomp using CMake config")
    set(NVCOMP_LIBRARIES nvcomp::nvcomp)
    set(NVCOMP_INCLUDE_DIRS "")  # Handled by the target itself
    # get_target_property(NVCOMP_INCLUDE_DIRS nvcomp::nvcomp INTERFACE_INCLUDE_DIRECTORIES)
else()
    set(NVCOMP_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include)
endif()

# Add subprojects
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

# Find dependencies
find_package(GTest REQUIRED)
# find_package(benchmark REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDA REQUIRED)

message(STATUS "CUDA Include Directories: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures
# set(CMAKE_CUDA_ARCHITECTURES 70)



# ==================== Important fix: create a shared CUDA object library ====================
# This avoids recompiling the same CUDA files for multiple targets


# Compile GPU source files into an object library (key fix)
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

# Set compile options for Falcon_gpu_lib to reduce memory usage
target_compile_options(Falcon_gpu_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-O2
        -lineinfo
        --dopt=on
    >
)

# Data utility library
add_library(dataset_utils_lib STATIC
    data/dataset_utils.cpp
)

target_include_directories(dataset_utils_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/test/data
)

# ==================== Global include directories ====================
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


# ==================== GPU test targets (using shared library) ====================

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

# ==================== Other test targets ====================

# ALP tests
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

# Add ALP-G include directories to test_ALP_GPU
target_include_directories(test_ALP_GPU PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# Set CUDA compilation properties
set_target_properties(test_ALP_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# New version: tests using ALP-G GPU interface
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

# Add ALP-G include directories to test_ALP_GPU_v2
target_include_directories(test_ALP_GPU_v2 PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# Set CUDA compilation properties
set_target_properties(test_ALP_GPU_v2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)



# ndzip tests
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

# ELF tests
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

# ELF Star tests
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


# ELF Star GPU tests
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

# Gorilla tests
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

# Chimp128 tests
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

    # Patas tests
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

# ALP tests
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

# Add ALP-G include directories to test_float_ALP_GPU
target_include_directories(test_float_ALP_GPU PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/alp
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/engine
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/flsgpu
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/fls
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/generated-bindings
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP-G/src/nvcomp
)

# Set CUDA compilation properties
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

# ELF Star tests
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


# ELF Star GPU tests
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

# gdeflate tests
    # Temporarily disabled (API compatibility issue)
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

    

        # Snappy tests
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

# bitcomp tests
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

# ==================== nvcomp-related tests (if found) ====================
if(nvcomp_FOUND)
    # LZ4 tests
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

    # gdeflate tests - temporarily disabled (API compatibility issue)
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

    # Snappy tests
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

    # bitcomp tests
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

# ==================== Set output directory (avoid conflicts) ====================
set_target_properties(
    test_gpu test_gpu_nopack test_gpu_spare test_gpu_br 
    # test_gpu_string
    test_muti_stream test_muti_3step_block test_muti_3step_noblock
    test_ALP test_ndzip test_elf test_elf_star test_elf_star_g
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# ==================== Add custom command for cleanup ====================
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/test
    COMMENT "Cleaning all build files..."
)

# ==================== Print configuration summary ====================
message(STATUS "=================================")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  nvcomp Found: ${nvcomp_FOUND}")
message(STATUS "=================================")