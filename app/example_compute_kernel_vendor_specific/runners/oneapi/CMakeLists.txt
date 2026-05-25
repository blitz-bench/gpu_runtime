# Level Zero loader is shipped by oneAPI but distros also ship it
# standalone (`libze-dev` / `liblevel-zero-dev`). No stock CMake config
# exists; we use pkg-config first, then a plain library/header fallback.

set(_have FALSE)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LEVEL_ZERO QUIET IMPORTED_TARGET libze_loader)
    if(LEVEL_ZERO_FOUND)
        set(_have TRUE)
        set(_link PkgConfig::LEVEL_ZERO)
    endif()
endif()

if(NOT _have)
    find_path(LEVEL_ZERO_INCLUDE_DIR ze_api.h
              PATH_SUFFIXES level_zero)
    find_library(LEVEL_ZERO_LIBRARY NAMES ze_loader)
    if(LEVEL_ZERO_INCLUDE_DIR AND LEVEL_ZERO_LIBRARY)
        set(_have TRUE)
        add_library(_level_zero_imp INTERFACE)
        target_include_directories(_level_zero_imp INTERFACE ${LEVEL_ZERO_INCLUDE_DIR})
        target_link_libraries(_level_zero_imp INTERFACE ${LEVEL_ZERO_LIBRARY})
        set(_link _level_zero_imp)
    endif()
endif()

if(NOT _have)
    set(EXAMPLE_VS_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()
if(NOT EXAMPLE_VS_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(EXAMPLE_VS_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

add_library(example_vs_oneapi STATIC oneapi_runner.cpp)
add_dependencies(example_vs_oneapi example_vs_shader)
target_link_libraries(example_vs_oneapi PUBLIC gpu_runtime ${_link})
target_include_directories(example_vs_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${EXAMPLE_VS_SHADER_INCDIR}
)
target_compile_features(example_vs_oneapi PUBLIC cxx_std_20)
set_target_properties(example_vs_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_oneapi INTERFACE EXAMPLE_VS_HAVE_ONEAPI)
set(EXAMPLE_VS_HAVE_ONEAPI TRUE PARENT_SCOPE)
