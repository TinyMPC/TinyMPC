cmake_minimum_required(VERSION 3.15)
project(tinympc VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include/Eigen)

# find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})
# message("${EIGEN3_INCLUDE_DIR}")

<<<<<<< HEAD
set(USING_CODEGEN OFF) # Turn this on when building Python wrapper

if(USING_CODEGEN)

=======
>>>>>>> feature/conic-constraints
# Create codegen directories
set( EMBEDDED_BUILD_ROOT_DIR "${CMAKE_BINARY_DIR}/codegen_src" )
set( EMBEDDED_BUILD_TINYMPC_DIR "${EMBEDDED_BUILD_ROOT_DIR}/tinympc" )
set( EMBEDDED_BUILD_INCLUDE_DIR "${EMBEDDED_BUILD_ROOT_DIR}/include" )
file( MAKE_DIRECTORY ${EMBEDDED_BUILD_ROOT_DIR} )
file( MAKE_DIRECTORY ${EMBEDDED_BUILD_TINYMPC_DIR} )
file( MAKE_DIRECTORY ${EMBEDDED_BUILD_INCLUDE_DIR} )

# Create target for copying files
add_custom_target(copy_codegen_files ALL
                COMMENT "Copying source files needed for code generation" )

# Copy Eigen from include/* to codegen_src/include/
add_custom_target(copy_codegen_include_files)
add_custom_command(TARGET copy_codegen_include_files PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
                    ${EMBEDDED_BUILD_INCLUDE_DIR})
add_dependencies(copy_codegen_files copy_codegen_include_files)

# Set codegen install directory
set(TINY_CODEGEN_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/tinympc/codegen_files" CACHE PATH "Location of codegen install")
install(DIRECTORY
        "${EMBEDDED_BUILD_ROOT_DIR}/"
        DESTINATION "${TINY_CODEGEN_INSTALL_DIR}" COMPONENT codegen)

endif(USING_CODEGEN)

# Add tinympcstatic library
add_subdirectory(src/tinympc)

# Add example executables
add_subdirectory(examples)
