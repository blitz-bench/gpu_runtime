# Metal runner CMake. Apple-only. The parent guards this subdir with if(APPLE).

enable_language(OBJCXX)

add_library(example_vs_metal STATIC metal_runner.mm)
target_link_libraries(example_vs_metal PUBLIC
    "-framework Foundation"
    "-framework Metal"
)
target_include_directories(example_vs_metal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(example_vs_metal PUBLIC cxx_std_20)
set_target_properties(example_vs_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_metal INTERFACE EXAMPLE_VS_HAVE_METAL)
set(EXAMPLE_VS_HAVE_METAL TRUE PARENT_SCOPE)
