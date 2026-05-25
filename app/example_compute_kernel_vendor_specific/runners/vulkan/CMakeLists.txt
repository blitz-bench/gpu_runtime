# Vulkan runner CMake. Requires the Vulkan loader + headers AND the shared
# shader build (driven by app/.../shaders/CMakeLists.txt, processed before
# this subdir at the parent CMakeLists level).

find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(EXAMPLE_VS_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()
if(NOT EXAMPLE_VS_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(EXAMPLE_VS_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(example_vs_vulkan STATIC vulkan_runner.cpp)
add_dependencies(example_vs_vulkan example_vs_shader)
target_link_libraries(example_vs_vulkan PUBLIC gpu_runtime Vulkan::Vulkan)
target_include_directories(example_vs_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${EXAMPLE_VS_SHADER_INCDIR}
)
target_compile_features(example_vs_vulkan PUBLIC cxx_std_20)
set_target_properties(example_vs_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_vulkan INTERFACE EXAMPLE_VS_HAVE_VULKAN)
set(EXAMPLE_VS_HAVE_VULKAN TRUE PARENT_SCOPE)
