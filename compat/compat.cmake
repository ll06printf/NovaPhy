# The module detects the environment and applying some tools or 
# patches to improve the build performance and stability.

# enable compiler cache is available

find_program(SCCACHE sccache)
find_program(CCACHE ccache)
if (SCCACHE)
    message(STATUS "Using sccache: ${SCCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${SCCACHE})

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86")
    endif()
    if (NOT DEFINED UIPC_CUDA_ARCHITECTURES)
        set(UIPC_CUDA_ARCHITECTURES "86")
    endif()
elseif (CCACHE)
    message(STATUS "Using ccache: ${CCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
else()
    message(STATUS "compiler cache not found, not using it.")
endif()