add_library(tinympcstatic STATIC
    admm.cpp
    codegen.cpp
)

# target_include_directories(
#     osqpstatic PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/public;${CMAKE_CURRENT_BINARY_DIR}/include/public>"
#                       "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/osqp>")


target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# add_library(tinympcShared SHARED
#     admm.cpp
#     codegen.cpp
# )

# target_include_directories(tinympcShared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of tinympc/


# target_sources(TINYMPCLIB PUBLIC ${osqp_headers})
# target_sources(TINYMPCLIB PRIVATE ${osqp_headers_private})
# target_include_directories(TINYMPCLIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/public")
# target_include_directories(TINYMPCLIB PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/public")
# target_include_directories(TINYMPCLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/private")
# target_include_directories(TINYMPCLIB PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/private")
