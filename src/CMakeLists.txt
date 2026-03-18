# NovaPhy core static library

configure_file(
    ${PROJECT_SOURCE_DIR}/include/novaphy/novaphy_config.h.in
    ${PROJECT_BINARY_DIR}/include/novaphy/novaphy_config.h
)

add_library(novaphy_core STATIC
    math/spatial.cpp
    math/math_utils.cpp
    core/body.cpp
    core/shape.cpp
    core/model_builder.cpp
    core/joint.cpp
    collision/broadphase.cpp
    collision/narrowphase.cpp
    dynamics/integrator.cpp
    dynamics/free_body_solver.cpp
    dynamics/featherstone.cpp
    dynamics/articulated_solver.cpp
    sim/state.cpp
    sim/world.cpp
    fluid/particle_state.cpp
    fluid/neighbor_search.cpp
    fluid/pbf_solver.cpp
    fluid/fluid_world.cpp
    fluid/boundary.cpp
)

target_include_directories(novaphy_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

target_link_libraries(novaphy_core PUBLIC
    Eigen3::Eigen
)

target_compile_definitions(novaphy_core PUBLIC
    EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
)

if (NOVAPHY_ENABLE_INSTALL)
    novaphy_header_set(novaphy_core
        BASE_DIR ${PROJECT_SOURCE_DIR}/include
        PATTERNS
            ${PROJECT_SOURCE_DIR}/include/novaphy/collision/*.h
            ${PROJECT_SOURCE_DIR}/include/novaphy/core/*.h
            ${PROJECT_SOURCE_DIR}/include/novaphy/dynamics/*.h
            ${PROJECT_SOURCE_DIR}/include/novaphy/fluid/*.h
            ${PROJECT_SOURCE_DIR}/include/novaphy/math/*.h
            ${PROJECT_SOURCE_DIR}/include/novaphy/sim/*.h
        FILES 
            ${PROJECT_SOURCE_DIR}/include/novaphy/novaphy.h
    )
    novaphy_header_set(novaphy_core
        BASE_DIR ${PROJECT_BINARY_DIR}/include
        FILES ${PROJECT_BINARY_DIR}/include/novaphy/novaphy_config.h
    )
    novaphy_export_library(novaphy_core COMPONENT core XNAME core)

    if (SKBUILD AND NOVAPHY_BUNDLE_DEPENDENCIES)
        novaphy_bundle_dependencies(novaphy_core)
    endif()
endif()