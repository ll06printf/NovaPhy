# The module detects the environment and applying some patch.

# TODO

if (NOT SKBUILD) 
    list(APPEND VCPKG_MANIFEST_FEATURES "cmake-standalone")
endif()