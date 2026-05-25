cmake_minimum_required(VERSION 3.20)

project(gpu_runtime LANGUAGES CXX VERSION 0.1.0)

if(NOT DEFINED GPGPU_BUILD_MODE)
    set(GPGPU_BUILD_MODE STATIC)
endif()

set(GPGPU_SOURCES
    src/backend.cpp
    src/device.cpp
    src/preference.cpp
    src/report.cpp
    src/runtime.cpp
    src/search_paths.cpp
    src/version.cpp
    src/backends/common.cpp
    src/backends/cuda.cpp
    src/backends/oneapi.cpp
    src/backends/opencl.cpp
    src/backends/rocm.cpp
    src/backends/vulkan.cpp
    src/backends/metal.cpp
)

if(WIN32)
    list(APPEND GPGPU_SOURCES src/platform/lib_loader_win.cpp)
else()
    list(APPEND GPGPU_SOURCES src/platform/lib_loader_unix.cpp)
endif()

if(APPLE)
    list(APPEND GPGPU_SOURCES src/backends/metal.mm)
    enable_language(OBJCXX)
endif()

add_library(gpu_runtime ${GPGPU_BUILD_MODE} ${GPGPU_SOURCES})
add_library(gpgpu::gpu_runtime ALIAS gpu_runtime)

set(GPGPU_OPENCL_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers)
set(GPGPU_VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/Vulkan-Headers/include)
set(GPGPU_LEVEL_ZERO_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/external/level-zero/include)

foreach(_dir IN ITEMS
        "${GPGPU_OPENCL_HEADERS_DIR}/CL/cl.h"
        "${GPGPU_VULKAN_HEADERS_DIR}/vulkan/vulkan_core.h"
        "${GPGPU_LEVEL_ZERO_DIR}/ze_api.h")
    if(NOT EXISTS "${_dir}")
        message(FATAL_ERROR
            "gpu_runtime: missing vendored header '${_dir}'. "
            "The OpenCL-Headers, Vulkan-Headers, and level-zero submodules "
            "under external/ must be checked out. Run "
            "'git submodule update --init --recursive' from this repo's root.")
    endif()
endforeach()

target_include_directories(gpu_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # Vendored third-party headers, sourced via git submodules under
        # external/<name>/. See README.md > "Third-party headers".
        ${GPGPU_OPENCL_HEADERS_DIR}
        ${GPGPU_VULKAN_HEADERS_DIR}
        ${GPGPU_LEVEL_ZERO_DIR}
)

# OpenCL 3.0 target API level; opt in to deprecated symbols we still query
# (CL_DEVICE_HOST_UNIFIED_MEMORY is a CL 1.1 attribute deprecated in CL 2.0).
target_compile_definitions(gpu_runtime PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_USE_DEPRECATED_OPENCL_1_1_APIS
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
    CL_USE_DEPRECATED_OPENCL_2_0_APIS
    # We resolve every Vulkan entry point via vkGetInstanceProcAddr, so skip
    # the function prototypes vulkan_core.h would otherwise emit.
    VK_NO_PROTOTYPES
)

target_compile_features(gpu_runtime PUBLIC cxx_std_17)

set_target_properties(gpu_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET   hidden
    POSITION_INDEPENDENT_CODE ON
)

if(NOT WIN32)
    # dlopen lives in libdl on glibc; Apple has it in libc.
    if(NOT APPLE)
        target_link_libraries(gpu_runtime PRIVATE dl)
    endif()
endif()

if(APPLE)
    target_link_libraries(gpu_runtime PRIVATE
        "-framework Foundation"
        "-framework Metal"
    )
endif()

if(BUILD_SAMPLE_APP)
    add_subdirectory(app/gpu_runtime_query)
    add_subdirectory(app/example_compute_kernel)
    add_subdirectory(app/example_compute_kernel_vendor_specific)
endif()
