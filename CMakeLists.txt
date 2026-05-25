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

target_include_directories(gpu_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
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
endif()
