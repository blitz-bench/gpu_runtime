set(APP example_compute_kernel_vendor_specific)

option(FAIL_ON_MISSING_BACKEND
       "${APP}: promote any missing vendor SDK to a configure-time fatal error" OFF)

set(_backends CUDA ROCM ONEAPI OPENCL VULKAN METAL)

# Shared shader build: detects glslangValidator and emits the SPIR-V .inl
# both Vulkan and Level Zero consume. Must run before those subdirs.
add_subdirectory(shaders)

# Each subdirectory does its own find_package and, on success, creates a
# static lib target named example_vs_<backend> with INTERFACE compile
# definition EXAMPLE_VS_HAVE_<BACKEND> and sets EXAMPLE_VS_HAVE_<BACKEND>=TRUE
# in PARENT_SCOPE.
add_subdirectory(runners/cuda)
add_subdirectory(runners/rocm)
add_subdirectory(runners/oneapi)
add_subdirectory(runners/opencl)
add_subdirectory(runners/vulkan)
if(APPLE)
    add_subdirectory(runners/metal)
else()
    set(EXAMPLE_VS_HAVE_METAL FALSE)
endif()

message(STATUS "${APP} runners:")
foreach(b IN LISTS _backends)
    if(EXAMPLE_VS_HAVE_${b})
        message(STATUS "    ${b}: YES")
    else()
        message(STATUS "    ${b}: NO  (SDK not found at build time)")
        if(FAIL_ON_MISSING_BACKEND)
            message(FATAL_ERROR "${APP}: backend ${b} required but its SDK was not found "
                                "(disable with -DFAIL_ON_MISSING_BACKEND=OFF)")
        endif()
    endif()
endforeach()

add_executable(${APP} main.cpp)
target_link_libraries(${APP} PRIVATE gpu_runtime)
target_compile_features(${APP} PRIVATE cxx_std_20)
target_include_directories(${APP} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${APP} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

foreach(b IN LISTS _backends)
    string(TOLOWER ${b} _bl)
    if(EXAMPLE_VS_HAVE_${b})
        target_link_libraries(${APP} PRIVATE example_vs_${_bl})
    endif()
endforeach()
