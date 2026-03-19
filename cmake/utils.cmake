include_guard()
include(${PROJECT_SOURCE_DIR}/cmake/find_target_dlls.cmake)

# NovaPhy Buildsystem utilities

# CMake package layout
# - lib
#   - cmake/novaphy
#       - <CMake configuration files>
#   - novaphy-bundled-libs
#       - <novaphy bundled libs>
#   - <novaphy library>
# - include/novaphy
#   - <public headers>

# Python Wheel package layout
# - novaphy
#   - <_core python extension>
#   - <python modules>
#   - lib
#       - <novaphy library>
#       - novaphy-bundled-libs "just for Linux"
#           - bundled libs
#   - <bundled libs> "just for Windows"

# On linux, the layout of Wheel package is similar to CMake 
# package, except the python extension library. But on Windows, 
# the runtime files of the novaphy dynamic library will be placed
# in the same directory as the python extension library.

# Enable detailed packaging logs for troubleshooting install layout issues.
set(NOVAPHY_PACKAGE_LOG ON CACHE BOOL
    "Enable verbose logs for NovaPhy packaging utilities"
)

# Internal logger used by this file only.
# Usage: _novaphy_log("message")
# Messages are emitted only when NOVAPHY_PACKAGE_LOG is ON.
function(_novaphy_log message_text)
    if (NOVAPHY_PACKAGE_LOG)
        message(STATUS "[novaphy-packager] ${message_text}")
    endif()
endfunction()

############################################################
# Helper functions
############################################################

# Configure install destinations for a classic CMake package layout.
#
# Output (cache variables):
# - NOVAPHY_INCLUDE_DST: root for public headers
# - NOVAPHY_LIB_DST: root for shared/static/import libraries
# - NOVAPHY_BUNDLED_DST: root for bundled third-party runtime libraries
# - NOVAPHY_CMAKE_CONFIG_DST: root for generated CMake package config files
function(setup_cmake_package_layout)
    set(NOVAPHY_INCLUDE_DST include CACHE PATH
        "The install path of public headers"
    )
    set(NOVAPHY_LIB_DST lib CACHE PATH
        "The install path of libraries"
    )
    set(NOVAPHY_BUNDLED_DST ${NOVAPHY_LIB_DST}/novaphy-bundled-libs CACHE PATH 
        "The install path of bundled libraries"
    )
    set(NOVAPHY_CMAKE_CONFIG_DST ${NOVAPHY_LIB_DST}/cmake/novaphy CACHE PATH 
        "The install path of CMake config files"
    )
    _novaphy_log("CMake layout: include='${NOVAPHY_INCLUDE_DST}', lib='${NOVAPHY_LIB_DST}', bundled='${NOVAPHY_BUNDLED_DST}', config='${NOVAPHY_CMAKE_CONFIG_DST}'")
endfunction()

# Configure install destinations for Python wheel packaging.
#
# Linux wheel layout keeps runtime libraries under novaphy/lib.
# Windows wheel layout places runtime artifacts directly in the top-level
# package folder to simplify DLL loading for the extension module.
function(setup_wheel_layout)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(NOVAPHY_WHL_DST_PREFIX novaphy CACHE PATH
            "The install path prefix of wheel package for scikit-build"
        )
        set(NOVAPHY_WHL_LIB_DST ${NOVAPHY_WHL_DST_PREFIX}/lib CACHE PATH
            "The install path of library for scikit-build"
        )
        set(NOVAPHY_WHL_BUNEDLED_DST ${NOVAPHY_WHL_LIB_DST}/novaphy-bundled-libs CACHE PATH
            "The install path of bundled third-party libraries for wheel scikit-build"
        )
        _novaphy_log("Wheel layout (Linux): prefix='${NOVAPHY_WHL_DST_PREFIX}', lib='${NOVAPHY_WHL_LIB_DST}', bundled='${NOVAPHY_WHL_BUNEDLED_DST}'")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(NOVAPHY_WHL_DST_PREFIX novaphy CACHE PATH
            "The install path prefix of wheel package for scikit-build"
        )
        set(NOVAPHY_WHL_LIB_DST ${NOVAPHY_WHL_DST_PREFIX} CACHE PATH
            "The install path of library for scikit-build"
        )
        set(NOVAPHY_WHL_BUNEDLED_DST ${NOVAPHY_WHL_DST_PREFIX} CACHE PATH
            "The install path of bundled third-party libraries for wheel scikit-build"
        )
        _novaphy_log("Wheel layout (Windows): prefix='${NOVAPHY_WHL_DST_PREFIX}', lib='${NOVAPHY_WHL_LIB_DST}', bundled='${NOVAPHY_WHL_BUNEDLED_DST}'")
    else()
        message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
    endif()
endfunction()

########################################################
# Package and install configuration
########################################################

# Main entry point for package layout setup.
# Chooses layout based on the active build system.
# Initialize package type, package metadata properties, and helper properties
# used by installation/export functions in this file.
#
# Behavior:
# - If SKBUILD is enabled, selects Wheel layout.
# - Otherwise selects CMake package layout.
# - Defines global/target properties used for component export and headers.
function(setup_novaphy_packager)

    # Package type selection

    if (SKBUILD)
        set(NOVAPHY_PACKAGE_TYPE Wheel CACHE STRING
            "The type of package to build, either CMake or Wheel"
        )
        _novaphy_log("Package type selected: Wheel (SKBUILD enabled)")
        setup_wheel_layout()
    else()
        set(NOVAPHY_PACKAGE_TYPE CMake CACHE STRING
            "The type of package to build, either CMake or Wheel"
        )
        _novaphy_log("Package type selected: CMake")
        setup_cmake_package_layout()
    endif()

    # Other variables

    set(NOVAPHY_CONFIG_TYPE "$<CONFIG>" PARENT_SCOPE)
    if(NOT WIN32)
        if("${CMAKE_BUILD_TYPE}" STREQUAL "")
            set(NOVAPHY_CONFIG_TYPE "Release" PARENT_SCOPE)
        else()
            set(NOVAPHY_CONFIG_TYPE "${CMAKE_BUILD_TYPE}" PARENT_SCOPE)
        endif()
    endif()

    # Properties

    define_property(GLOBAL PROPERTY NOVAPHY_PACKAGE_COMPONENTS
        BRIEF_DOCS "The list of components of novaphy package"
        FULL_DOCS "The list of components of novaphy package, used for CMake export and Wheel package."
    )
    define_property(GLOBAL PROPERTY NOVAPHY_PACKAGE_DEPENDENCIES
        BRIEF_DOCS "The list of dependencies of novaphy package"
        FULL_DOCS "The list of dependencies of novaphy package, used for CMake export and Wheel package."
    )

    # The following properties simulate FILE_SET HEADERS from CMake 3.23,
    # which is not supported by the minimum required CMake version of this project (3.18). 
    # Public headers are needed for both CMake and Wheel packages:
    # CMake export and wheel include-file generation.
    define_property(TARGET PROPERTY NOVAPHY_PUBLIC_HEADERS
        BRIEF_DOCS "The list of public headers of the target"
        FULL_DOCS "The list of public headers of the target, used for CMake export and Wheel package."
    )
    define_property(TARGET PROPERTY NOVAPHY_HEADERS_PREFIX
        BRIEF_DOCS "The base directory of public headers of the target"
        FULL_DOCS "The base directory of public headers of the target, used for CMake export and Wheel package."
    )
endfunction()

# Register public header files for a target.
#
# Arguments:
#   novaphy_header_set(<target>
#       [BASE_DIR <dir>]
#       [FILES <file1> <file2> ...]
#       [PATTERNS <glob1> <glob2> ...])
#
# Notes:
# - At least one of FILES or PATTERNS must be provided.
# - BASE_DIR is used to compute relative install subdirectories.
# - This function only registers metadata; installation happens elsewhere.
function(novaphy_header_set target)
    cmake_parse_arguments(hset 
        "" 
        "BASE_DIR" 
        "FILES;PATTERNS" 
        ${ARGN}
    )
    if (NOT hset_FILES AND NOT hset_PATTERNS)
        message(FATAL_ERROR "Either FILES or PATTERNS must be specified for novaphy_header_set")
    endif()

    set(headers)
    if (hset_FILES)
        set(headers ${hset_FILES})
    endif()
    if (hset_PATTERNS)
        foreach(pattern IN LISTS hset_PATTERNS)
            file(GLOB_RECURSE pattern_headers CONFIGURE_DEPENDS ${pattern})
            list(APPEND headers ${pattern_headers})
        endforeach()
    endif()

    list(REMOVE_DUPLICATES headers)
    set(include_prefix "")
    foreach (header IN LISTS headers)
        set(header_prefix ".")
        if (hset_BASE_DIR)
            file(RELATIVE_PATH header_prefix ${hset_BASE_DIR} ${header})
            if (header_prefix MATCHES "^[.][.]")
                message(FATAL_ERROR "Header ${header} is not under the base directory ${hset_BASE_DIR}")
            endif()
            get_filename_component(header_prefix ${header_prefix} DIRECTORY)
        endif()
        list(APPEND include_prefix ${header_prefix})
    endforeach()
    set_property(TARGET ${target} APPEND PROPERTY
        NOVAPHY_PUBLIC_HEADERS "${headers}"
    )
    set_property(TARGET ${target} APPEND PROPERTY
        NOVAPHY_HEADERS_PREFIX "${include_prefix}"
    )
    list(LENGTH headers header_count)
    _novaphy_log("Registered ${header_count} public header(s) for target '${target}'")
endfunction()

# Append one package component into the global component registry.
# Duplicates are removed to keep exports deterministic.
function(_novaphy_add_component component)
    get_property(components GLOBAL PROPERTY NOVAPHY_PACKAGE_COMPONENTS)
    list(APPEND components ${component})
    list(REMOVE_DUPLICATES components)
    set_property(GLOBAL PROPERTY NOVAPHY_PACKAGE_COMPONENTS ${components})
    _novaphy_log("Added package component '${component}'")
endfunction()

# Append one dependency name into the global dependency registry.
# Duplicates are removed before persisting back to the global property.
function(_novaphy_add_dependency dependency)
    get_property(dependencies GLOBAL PROPERTY NOVAPHY_PACKAGE_DEPENDENCIES)
    list(APPEND dependencies ${dependency})
    list(REMOVE_DUPLICATES dependencies)
    set_property(GLOBAL PROPERTY NOVAPHY_PACKAGE_DEPENDENCIES ${dependencies})
    _novaphy_log("Recorded package dependency '${dependency}'")
endfunction()

# Install the python extension module, enabled if the target 
# is built as a python extension module. 
# Install a pybind extension target for wheel packages.
#
# This function is intentionally a no-op for non-wheel packaging.
# On Linux, RPATH is configured so the extension can resolve bundled
# runtime libraries from relative wheel locations.
function(novaphy_export_pybind target)
    if(NOVAPHY_PACKAGE_TYPE STREQUAL "Wheel")
        _novaphy_add_component(pybind)
        _novaphy_log("Installing python extension target '${target}' to '${NOVAPHY_WHL_DST_PREFIX}'")
        install(TARGETS ${target}
            RUNTIME DESTINATION ${NOVAPHY_WHL_DST_PREFIX}
            LIBRARY DESTINATION ${NOVAPHY_WHL_DST_PREFIX}
            ARCHIVE DESTINATION ${NOVAPHY_WHL_DST_PREFIX}
            COMPONENT pybind
        )

        # DLL Search
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set_target_properties(${target} PROPERTIES
                # TODO The Build RPATH is handcoded to the output directory of libuipc
                BUILD_RPATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin"
                INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/lib/novaphy-bundled-libs"
            )
            _novaphy_log("Configured Linux RPATH for python extension target '${target}'")
        endif()
    endif()

endfunction()

# Install the library target; implementation differs for
# different for CMake and Wheel package.
# Export and install a NovaPhy library target.
#
# Arguments:
#   novaphy_export_library(<target>
#       [WITHOUT_CMAKE_CONFIG]
#       [XNAME <export-name>]
#       [COMPONENT <component-name>]
#       [DEPENDS <dep1> <dep2> ...])
#
# Behavior summary:
# - CMake package: installs binaries/headers and optionally CMake exports.
# - Wheel package: installs binaries only (headers/config are not exported).
function(novaphy_export_library target)
    cmake_parse_arguments(exlab
        "WITHOUT_CMAKE_CONFIG"
        "XNAME;COMPONENT"
        "DEPENDS"
        ${ARGN}
    )

    # Installation config
    if (NOVAPHY_PACKAGE_TYPE STREQUAL "CMake")
        if (exlab_XNAME) 
            set_target_properties(${target} PROPERTIES
                EXPORT_NAME ${exlab_XNAME}
            )
            add_library(novaphy::${exlab_XNAME} ALIAS ${target})
        endif()

        _novaphy_log("Exporting target '${target}' to CMake package, component='${exlab_COMPONENT}'")

        # Install binary
        install(TARGETS ${target}
            RUNTIME DESTINATION ${NOVAPHY_LIB_DST}
            LIBRARY DESTINATION ${NOVAPHY_LIB_DST}
            ARCHIVE DESTINATION ${NOVAPHY_LIB_DST}
            COMPONENT ${exlab_COMPONENT}
        )
        _novaphy_log("Installed CMake library target '${target}' to '${NOVAPHY_LIB_DST}'")

        # Install headers
        get_target_property(headers ${target} NOVAPHY_PUBLIC_HEADERS)
        get_target_property(headers_prefix ${target} NOVAPHY_HEADERS_PREFIX)
        list(LENGTH headers num_headers)
        list(LENGTH headers_prefix num_headers_prefix)
        if (NOT num_headers EQUAL num_headers_prefix)
            message(FATAL_ERROR "The number of headers and headers prefixes must be the same for target ${target}")
        endif()
        foreach(header header_prefix IN ZIP_LISTS headers headers_prefix)
            install(FILES ${header}
                DESTINATION ${NOVAPHY_INCLUDE_DST}/${header_prefix}
                COMPONENT ${exlab_COMPONENT}
            )
        endforeach()
        _novaphy_log("Installed public headers for target '${target}' to '${NOVAPHY_INCLUDE_DST}'")

        # Install configure
        if (NOT exlab_WITHOUT_CMAKE_CONFIG)
            _novaphy_add_component(${exlab_COMPONENT})
            install(TARGETS ${target}
                EXPORT novaphy-${exlab_COMPONENT}-targets
                COMPONENT ${exlab_COMPONENT}
            )
            _novaphy_log("Enabled CMake export for target '${target}', component='${exlab_COMPONENT}'")
        endif()

        if (exlab_DEPENDENCIES)
            foreach(dependency IN LISTS exlab_DEPENDENCIES)
                _novaphy_add_dependency(${dependency})
            endforeach()
        endif()
    elseif(NOVAPHY_PACKAGE_TYPE STREQUAL "Wheel")
        install(TARGETS ${target}
            RUNTIME DESTINATION ${NOVAPHY_WHL_LIB_DST}
            LIBRARY DESTINATION ${NOVAPHY_WHL_LIB_DST}
            ARCHIVE DESTINATION ${NOVAPHY_WHL_LIB_DST}
            COMPONENT ${exlab_COMPONENT}
        )
        _novaphy_log("Installed Wheel library target '${target}' to '${NOVAPHY_WHL_LIB_DST}'")
    else()
        message(FATAL_ERROR "Unexpected package type: ${NOVAPHY_PACKAGE_TYPE}")
    endif()

endfunction()

# Bundled third-party library; implementation differs by package type.
# be different for CMake and Wheel package.
# Install a third-party library target into the bundled runtime location.
# Destination differs between CMake package and Wheel package layouts.
function(novaphy_bundle_library target)
    if (NOVAPHY_PACKAGE_TYPE STREQUAL "CMake")
        install(TARGETS ${target}
            RUNTIME DESTINATION ${NOVAPHY_BUNDLED_DST}
            LIBRARY DESTINATION ${NOVAPHY_BUNDLED_DST}
            ARCHIVE DESTINATION ${NOVAPHY_BUNDLED_DST}
            COMPONENT bundled
        )
        _novaphy_log("Installed bundled target '${target}' to '${NOVAPHY_BUNDLED_DST}' (CMake package)")
    elseif(NOVAPHY_PACKAGE_TYPE STREQUAL "Wheel")
        install(TARGETS ${target}
            RUNTIME DESTINATION ${NOVAPHY_WHL_BUNEDLED_DST}
            LIBRARY DESTINATION ${NOVAPHY_WHL_BUNEDLED_DST}
            ARCHIVE DESTINATION ${NOVAPHY_WHL_BUNEDLED_DST}
            COMPONENT bundled
        )
        _novaphy_log("Installed bundled target '${target}' to '${NOVAPHY_WHL_BUNEDLED_DST}' (Wheel package)")
    else()
        message(FATAL_ERROR "Unexpected package type: ${NOVAPHY_PACKAGE_TYPE}")
    endif()
endfunction()

# Bundle arbitrary files into the package-specific bundled directory.
#
# Arguments:
#   novaphy_bundle_files(
#       [FILES <file1> <file2> ...]
#       [GLOB <glob-expression>])
#
# FILES are installed at configure/generate time through install(FILES).
# GLOB is resolved at install time through install(CODE), which is useful
# for cases where files are generated after configuration.
function(novaphy_bundle_files)

    cmake_parse_arguments(bfiles 
        ""
        ""
        "GLOB;FILES"
        ${ARGN}
    )

    set(dst "")
    if (NOVAPHY_PACKAGE_TYPE STREQUAL "CMake")
        set(dst ${NOVAPHY_BUNDLED_DST})
    elseif(NOVAPHY_PACKAGE_TYPE STREQUAL "Wheel")
        set(dst ${NOVAPHY_WHL_BUNEDLED_DST})
    else()
        message(FATAL_ERROR "Unexpected package type: ${NOVAPHY_PACKAGE_TYPE}")
    endif()

    list(REMOVE_DUPLICATES bfiles_FILES)
    if (bfiles_FILES)
        install(FILES ${bfiles_FILES}
            DESTINATION ${dst}
            COMPONENT bundled
        )
    endif()

    if (bfiles_GLOB)
        install(CODE "
            file(GLOB files_to_install ${bfiles_GLOB})
            file(INSTALL \${files_to_install} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${dst}\")
        ")
    endif()
endfunction()

# Discover and bundle runtime DLL/shared-library dependencies of a target.
#
# Supported target kinds: EXECUTABLE, SHARED_LIBRARY, MODULE_LIBRARY.
# Dependencies are resolved via find_target_dlls() and installed into the
# package-specific bundled runtime directory.
function(novaphy_bundle_dependencies target)
    get_target_property(target_type ${target} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE" 
        AND NOT target_type STREQUAL "SHARED_LIBRARY" 
        AND NOT target_type STREQUAL "MODULE_LIBRARY"
        )
        message(FATAL_ERROR "Target '${target}' must be an executable or shared/module library to bundle dependencies")
    endif()

    find_target_dlls(TARGET ${target} OUT_VAR dlls FILTER_IMPORTED)

    set(dst "")
    if (NOVAPHY_PACKAGE_TYPE STREQUAL "CMake")
        set(dst ${NOVAPHY_BUNDLED_DST})
    elseif(NOVAPHY_PACKAGE_TYPE STREQUAL "Wheel")
        set(dst ${NOVAPHY_WHL_BUNEDLED_DST})
    else()
        message(FATAL_ERROR "Unexpected package type: ${NOVAPHY_PACKAGE_TYPE}")
    endif()

    if (dlls)
        install(FILES ${dlls}
            DESTINATION ${dst}
            COMPONENT bundled
        )
    endif()
endfunction()

# Generate and install CMake configuration files, only
# for CMake package. The contents installed by the function
# belong to the dev component, which is not needed for wheel
# package.
# Finalize install/export artifacts after all targets are registered.
#
# This currently applies only to CMake packages and generates:
# - Per-component exported target files
# - novaphyConfig.cmake / novaphyConfigVersion.cmake
#
# Files generated here are installed as the "dev" component.
function(novaphy_post_install)
    if (NOVAPHY_PACKAGE_TYPE STREQUAL "CMake")
        include(CMakePackageConfigHelpers)
        # Create targets file for each component
        get_property(components GLOBAL PROPERTY NOVAPHY_PACKAGE_COMPONENTS)
        foreach(component IN LISTS components)
            install(EXPORT novaphy-${component}-targets
                FILE novaphy-${component}-targets.cmake
                NAMESPACE novaphy::
                DESTINATION ${NOVAPHY_CMAKE_CONFIG_DST}
                COMPONENT dev
            )
        endforeach()
        _novaphy_log("Installed export target files for components: ${components}")

        write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/novaphyConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
        )

        # Create configure file
        get_property(dependencies GLOBAL PROPERTY NOVAPHY_PACKAGE_DEPENDENCIES)
        
        configure_package_config_file(
            "cmake/novaphyConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/novaphyConfig.cmake"
            INSTALL_DESTINATION ${NOVAPHY_CMAKE_CONFIG_DST}
        )
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/novaphyConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/novaphyConfigVersion.cmake"
            DESTINATION ${NOVAPHY_CMAKE_CONFIG_DST}
            COMPONENT dev
        )
        _novaphy_log("Installed CMake config files to '${NOVAPHY_CMAKE_CONFIG_DST}'")
    endif()
endfunction()

setup_novaphy_packager()