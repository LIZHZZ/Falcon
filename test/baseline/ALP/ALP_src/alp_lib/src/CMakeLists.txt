# ALP
add_library(ALP SHARED
    falp.cpp
    fastlanes_ffor.cpp
    fastlanes_generated_ffor.cpp
    fastlanes_generated_unffor.cpp
    fastlanes_unffor.cpp
)

# translated comment
target_include_directories(ALP PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# translated comment
set_target_properties(ALP PROPERTIES CXX_STANDARD 17)

# Clang-Tidy （ ）
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
    set_target_properties(ALP PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()
