# ALP-G (GPU-accelerated ALP) CMakeLists
# CUDA , ALP GPU

cmake_minimum_required(VERSION 3.18)
project(ALP_G CUDA CXX)

# C++ CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

# CUDA
find_package(CUDA REQUIRED)

# ALP-G
set(ALPG_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flsgpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fls
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvcomp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/generated-bindings
)

set(ALPG_SOURCE_FILES
    src/engine/enums.cu
    src/alp/falp.cpp
    src/alp/alp-bindings.cu
    src/fls/ffor.cpp
    src/fls/pack.cpp
    src/fls/unffor.cpp
    src/fls/unpack.cpp
)

# generated-bindings .cu
file(GLOB ALPG_GENERATED_BINDINGS_FILES
    # "src/generated-bindings/*-bindings.cu"
    "src/generated-bindings/*-decompress_column-bindings.cu"
)
list(APPEND ALPG_SOURCE_FILES ${ALPG_GENERATED_BINDINGS_FILES})

# interface (header-only)
add_library(alpg_headers INTERFACE)

target_include_directories(alpg_headers INTERFACE
    ${ALPG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_compile_options(alpg_headers INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
    >
)

# translated comment
cuda_add_library(alpg_lib STATIC
    ${ALPG_SOURCE_FILES}
)

target_include_directories(alpg_lib PUBLIC
    ${ALPG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_compile_options(alpg_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        -Xcompiler=-O2
        --dopt=on
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -fPIC
        -O2
    >
)

# CUDA
set_target_properties(alpg_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# translated comment
set(ALPG_INCLUDE_DIRS ${ALPG_INCLUDE_DIRS} PARENT_SCOPE)
set(alpg_lib alpg_lib PARENT_SCOPE)

message(STATUS "ALP-G Library Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Include Directories: ${ALPG_INCLUDE_DIRS}")
message(STATUS "  Core Source Files: ${ALPG_SOURCE_FILES}")
message(STATUS "  Generated Bindings Files: ${ALPG_GENERATED_BINDINGS_FILES}")
message(STATUS "  Separable Compilation: ON")
message(STATUS "  Resolve Device Symbols: ON")