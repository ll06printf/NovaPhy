# NovaPhy IPC adapter library (C++20, links libuipc)

add_library(novaphy_ipc STATIC
    ipc_world.cpp
    shape_converter.cpp
)

# This target must compile as C++20 to match libuipc
set_target_properties(novaphy_ipc PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(novaphy_ipc PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(novaphy_ipc 
    PUBLIC
        novaphy_core
    PRIVATE
        $<BUILD_INTERFACE:uipc::uipc>
        $<INSTALL_INTERFACE:uipc::uipc>
)

# Ensure libuipc backend DLLs are built (they are MODULE libraries
# not linked into uipc::uipc, so EXCLUDE_FROM_ALL would skip them)
if(TARGET uipc::backends)
    add_dependencies(novaphy_ipc uipc::backends)
endif()

if (NOVAPHY_ENABLE_INSTALL)

    # TODO Disable CMake config export since libuipc doesn't export targets, 
    # and we don't want to force all users to use find_package(uipc) themselves. 
    # Instead, we can provide a custom Finduipc.cmake that sets the necessary 
    # variables for users to link against libuipc. This is a temporary 
    # workaround until libuipc provides proper CMake targets for its components.
    novaphy_header_set(novaphy_ipc
        BASE_DIR ${PROJECT_SOURCE_DIR}/include
        PATTERNS
            ${PROJECT_SOURCE_DIR}/include/novaphy/ipc/*.h
    )
    novaphy_export_library(novaphy_ipc COMPONENT ipc XNAME ipc WITHOUT_CMAKE_CONFIG)

endif()
