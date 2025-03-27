add_library(tinympcstatic STATIC
    admm.cpp
    tiny_api.cpp
    codegen.cpp
    rho_benchmark.cpp
)

set_property(TARGET tinympcstatic PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_link_libraries(tinympcstatic PUBLIC Eigen)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Eigen)



if(USING_CODEGEN) # Defined in top-level CMakeLists.txt

# Files that are needed for embedded code generation
list( APPEND EMBEDDED_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/admm.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/admm.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/types.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/tiny_api_constants.hpp" )


foreach( f ${EMBEDDED_FILES} )
    get_filename_component( fname ${f} NAME )

    set( dest_file "${EMBEDDED_BUILD_TINYMPC_DIR}/${fname}" )
    list( APPEND EMBEDDED_BUILD_TINYMPC_FILES "${dest_file}" )

    add_custom_command(OUTPUT ${dest_file}
                        COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                        DEPENDS ${f}
                        COMMENT "Copying ${fname}")
endforeach()

add_custom_target( copy_codegen_tinympc_files DEPENDS ${EMBEDDED_BUILD_TINYMPC_FILES} )
add_dependencies( copy_codegen_files copy_codegen_tinympc_files )

endif(USING_CODEGEN)
