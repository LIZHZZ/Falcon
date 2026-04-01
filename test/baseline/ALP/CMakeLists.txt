cmake_minimum_required(VERSION 3.10)

project(ALP)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# translated comment
include_directories(${PROJECT_SOURCE_DIR}/include)

# translated comment
file(GLOB_RECURSE ALP_SOURCES "src/*.cpp")

# ALP
add_library(ALP STATIC ${ALP_SOURCES})
