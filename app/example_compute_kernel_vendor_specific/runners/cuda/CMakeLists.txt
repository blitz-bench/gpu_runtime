find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    set(EXAMPLE_VS_HAVE_CUDA FALSE PARENT_SCOPE)
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    set(EXAMPLE_VS_HAVE_CUDA FALSE PARENT_SCOPE)
    return()
endif()
enable_language(CUDA)

add_library(example_vs_cuda STATIC cuda_runner.cu)
target_link_libraries(example_vs_cuda PUBLIC gpu_runtime CUDA::cudart)
target_include_directories(example_vs_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..   # for result.hpp / timings.hpp
)
target_compile_features(example_vs_cuda PUBLIC cxx_std_20)
set_target_properties(example_vs_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD             20
    CUDA_STANDARD_REQUIRED    ON
    CUDA_ARCHITECTURES        "50;60;70;80;90"
)
target_compile_definitions(example_vs_cuda INTERFACE EXAMPLE_VS_HAVE_CUDA)
set(EXAMPLE_VS_HAVE_CUDA TRUE PARENT_SCOPE)
