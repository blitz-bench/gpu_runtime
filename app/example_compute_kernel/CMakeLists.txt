set(APP example_compute_kernel)

set(SOURCES
    main.cpp
    runners/cuda_runner.cpp
    runners/rocm_runner.cpp
    runners/oneapi_runner.cpp
    runners/opencl_runner.cpp
    runners/vulkan_runner.cpp
)

# Metal runner is .mm and only enters the build on Apple. On other
# platforms a stub translation unit provides the run_vector_add_metal
# symbol so main.cpp links cleanly.
if(APPLE)
    list(APPEND SOURCES runners/metal_runner.mm)
else()
    list(APPEND SOURCES runners/metal_runner_stub.cpp)
endif()

add_executable(${APP} ${SOURCES})

target_link_libraries(${APP} PRIVATE gpu_runtime)
target_compile_features(${APP} PRIVATE cxx_std_20)   # std::span
target_include_directories(${APP} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Reuse the vendored header submodules from the parent project so each
    # runner sees the real Khronos / Intel types and calling-convention
    # macros (CL_API_CALL, VKAPI_PTR, ZE_APICALL).
    ${CMAKE_SOURCE_DIR}/external/OpenCL-Headers
    ${CMAKE_SOURCE_DIR}/external/Vulkan-Headers/include
    ${CMAKE_SOURCE_DIR}/external/level-zero/include
)
target_compile_definitions(${APP} PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
    VK_NO_PROTOTYPES
)

set_target_properties(${APP} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

if(APPLE)
    target_link_libraries(${APP} PRIVATE
        "-framework Foundation"
        "-framework Metal")
    enable_language(OBJCXX)
elseif(NOT WIN32)
    # dlopen / dlsym
    target_link_libraries(${APP} PRIVATE dl)
endif()
