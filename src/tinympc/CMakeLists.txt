add_library(tinympcstatic STATIC
    admm.cpp
    codegen.cpp
)


target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# add_library(tinympcShared SHARED
#     admm.cpp
#     codegen.cpp
# )

# target_include_directories(tinympcShared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of tinympc/