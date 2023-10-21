# add_executable(quadrotor_track_example quadrotor_track_traj.cpp)
add_executable(test_codegen test_codegen.cpp)

# target_link_libraries(quadrotor_track_example LINK_PUBLIC tinympc)
target_link_libraries(test_codegen LINK_PUBLIC tinympc)