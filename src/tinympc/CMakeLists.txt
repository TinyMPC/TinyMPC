add_library (tinympc
    admm.cpp
    )

target_include_directories (tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of tinympc/