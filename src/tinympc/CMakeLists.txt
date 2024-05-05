add_library(tinympcstatic STATIC
    admm.cpp
    tiny_api.cpp
    codegen.cpp
)

set_property(TARGET tinympcstatic PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_link_libraries(tinympcstatic PUBLIC Eigen)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Eigen)

# add_library(tinympcShared SHARED
#     admm.cpp
#     codegen.cpp
# )

# target_include_directories(tinympcShared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of tinympc/