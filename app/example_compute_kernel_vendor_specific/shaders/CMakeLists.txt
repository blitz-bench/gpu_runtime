# Shared shader build for Vulkan + Level Zero (both consume SPIR-V).
#
# Detects glslangValidator. On success, emits a custom command that compiles
# vector_add.comp -> vector_add.spv -> embedded into a constexpr C array
# (vector_add.spv.inl) in the build tree. Exposes the include directory of
# the .inl + a custom target other runners depend on.
#
# Variables set in PARENT_SCOPE on success:
#   EXAMPLE_VS_HAVE_SHADER    TRUE / FALSE
#   EXAMPLE_VS_SHADER_INCDIR  absolute path containing vector_add.spv.inl

find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator glslang
    DOC "GLSL -> SPIR-V compiler (part of the Vulkan SDK / glslang-tools)"
)
if(NOT GLSLANG_VALIDATOR)
    message(STATUS "    SHADER:  NO  (glslangValidator not found; Vulkan & OneAPI runners disabled)")
    set(EXAMPLE_VS_HAVE_SHADER FALSE PARENT_SCOPE)
    return()
endif()

set(_src ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.comp)
set(_spv ${CMAKE_CURRENT_BINARY_DIR}/vector_add.spv)
set(_inl ${CMAKE_CURRENT_BINARY_DIR}/vector_add.spv.inl)

add_custom_command(
    OUTPUT  ${_spv}
    COMMAND ${GLSLANG_VALIDATOR} -V ${_src} -o ${_spv}
    DEPENDS ${_src}
    COMMENT "example_vs/shaders: GLSL -> SPIR-V"
    VERBATIM
)
add_custom_command(
    OUTPUT  ${_inl}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_IN=${_spv}
            -DSPV_OUT=${_inl}
            -DSYMBOL=k_vector_add_spv
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${_spv}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    COMMENT "example_vs/shaders: embed SPIR-V -> vector_add.spv.inl"
    VERBATIM
)
add_custom_target(example_vs_shader DEPENDS ${_inl})

set(EXAMPLE_VS_HAVE_SHADER   TRUE                          PARENT_SCOPE)
set(EXAMPLE_VS_SHADER_INCDIR ${CMAKE_CURRENT_BINARY_DIR}   PARENT_SCOPE)
