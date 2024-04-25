cmake_minimum_required(VERSION 3.15)
project(tinympc VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include/Eigen)


add_subdirectory(examples)
add_subdirectory(src/tinympc)

# # ----------------------------------------------
# # TINYMPCLIB
# # ----------------------------------------------
# add_library(TINYMPCLIB OBJECT "")

# add_subdirectory(src/tinympc)

# get_property(
#   tinympclib_includes
#   TARGET TINYMPCLIB
#   PROPERTY INCLUDE_DIRECTORIES)
# get_property(
#   tinympclib_sources
#   TARGET TINYMPCLIB
#   PROPERTY SOURCES)

# get_property(
#   tinympclib_link_libs
#   TARGET TINYMPCLIB
#   PROPERTY LINK_LIBRARIES)



# # ----------------------------------------------
# # Shared tinympc library
# # ----------------------------------------------
# add_library(tinympc SHARED $<TARGET_OBJECTS:TINYMPCLIB>)
# target_include_directories(tinympc PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tinympc;${CMAKE_CURRENT_BINARY_DIR}/src/tinympc>"
#                                        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tinympc>")

# # Declare that we are building the shared library to get proper symbol exports.
# # Shared library consumers should also define TINYMPC_SHARED_LIB to get the library
# # exports properly, so we do it for them in the CMake interface by defining it as
# # a PUBLIC compile definition.
# target_compile_definitions(osqp PRIVATE BUILDING_OSQP)
# target_compile_definitions(osqp PUBLIC  OSQP_SHARED_LIB)


# add_subdirectory(examples)

