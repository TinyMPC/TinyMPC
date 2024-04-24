add_library(tinympc STATIC
    admm.cpp
    codegen.cpp
)

target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) 

add_library(tinympcShared SHARED
    admm.cpp
    codegen.cpp
)

target_include_directories(tinympcShared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of tinympc/