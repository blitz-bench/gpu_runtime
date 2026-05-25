# ROCm/HIP runner CMake. Uses HIP language support introduced in CMake 3.21.
# We try the modern HIP CONFIG package first; if missing, we silently skip.

find_package(hip CONFIG QUIET)
if(NOT hip_FOUND)
    set(EXAMPLE_VS_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif()

include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
    set(EXAMPLE_VS_HAVE_ROCM FALSE PARENT_SCOPE)
    return()
endif()
enable_language(HIP)

add_library(example_vs_rocm STATIC rocm_runner.hip)
target_link_libraries(example_vs_rocm PUBLIC gpu_runtime hip::host)
target_include_directories(example_vs_rocm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(example_vs_rocm PUBLIC cxx_std_20)
set_target_properties(example_vs_rocm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    HIP_STANDARD              20
    HIP_STANDARD_REQUIRED     ON
)
target_compile_definitions(example_vs_rocm INTERFACE EXAMPLE_VS_HAVE_ROCM)
set(EXAMPLE_VS_HAVE_ROCM TRUE PARENT_SCOPE)
