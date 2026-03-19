# Helper functions for install script

set(WINDOWS_IGNORE_REGEX
    "api-ms-.*" 
    "ext-ms-.*" 
    "kernel32\\.dll" 
    "user32\\.dll" 
    "gdi32\\.dll"
    "advapi32\\.dll"
    "shell32\\.dll"
    "msvcrt\\.dll"
    "ucrtbase\\.dll"
    "ntdll\\.dll" 
    "imm32\\.dll" 
    "ws2_32\\.dll" 
    "comctl32\\.dll"
    "ole32\\.dll" 
    "shlwapi\\.dll" 
    "msvcp_win\\.dll"
)

set(LINUX_IGNORE_REGEX
    "^libc\\.so.*" 
    "^libpthread\\.so.*" 
    "^librt\\.so.*" 
    "^libdl\\.so.*"
    "^libm\\.so.*"
    "^libgcc_s\\.so.*" 
    "^libX11\\.so.*" 
    "^libglib-.*"
    "^libstdc\\+\\+\\.so.*"
    "^ld-linux.*"
    "^libasound\\.so.*"
)

function(novaphy_install_dependencies target_type target_path dst)
    message(STATUS "Installing dependencies for ${target_path} to ${dst}")
    if (target_type STREQUAL "EXECUTABLE")
        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
            EXECUTABLES ${target_path}
            PRE_EXTRACT_REGEXES 
                ${WINDOWS_IGNORE_REGEX}
                ${LINUX_IGNORE_REGEX}
        )
    elseif (target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
            LIBRARIES ${target_path}
            PRE_EXTRACT_REGEXES 
                ${WINDOWS_IGNORE_REGEX}
                ${LINUX_IGNORE_REGEX}
        )
    else()
        message(FATAL_ERROR "Target type '${target_type}' is not supported for dependency installation")
    endif()
    if (unresolved_deps)
        message(WARNING "Unresolved dependencies for ${target_path}: ${unresolved_deps}")
    endif()
    foreach(dep IN LISTS resolved_deps)
        message(STATUS "Copying dependency ${dep} for ${target_path}")
        file(INSTALL ${dep} DESTINATION ${dst})
    endforeach()
endfunction()