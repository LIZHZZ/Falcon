cmake_minimum_required(VERSION 3.11)

# Patas header-only ，
# translated comment

add_library(patas INTERFACE)

target_include_directories(patas INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "Patas baseline library configured (header-only)")
