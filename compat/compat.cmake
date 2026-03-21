# The module detects the environment and applying some patch.

set(NOVAPHY_OVERLAY_PORTS "")

if (CMAKE_HOST_WIN32)
    # Windows-specific overlay ports 
    # (more specifically, for MSVC, but MSVC can't be detected before project command)
    list(APPEND NOVAPHY_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/port/eigen3")
endif()

set(VCPKG_OVERLAY_PORTS "${NOVAPHY_OVERLAY_PORTS}" CACHE STRING "Overlay ports for vcpkg" FORCE)
message(STATUS "NovaPhy: VCPKG_OVERLAY_PORTS set to ${VCPKG_OVERLAY_PORTS}")