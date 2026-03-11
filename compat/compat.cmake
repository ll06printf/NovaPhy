# The module detects the environment and applying some patch.

# TODO

# enable ccache is available

find_program(CCACHE ccache)
if (CCACHE)
    message(STATUS "Using ccache: ${CCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
else()
    message(STATUS "ccache not found, not using it.")
endif()