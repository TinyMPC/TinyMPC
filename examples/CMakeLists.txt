# add_executable(example_quadrotor_tracking example_quadrotor_tracking.cpp)
# add_executable(example_quadrotor_hovering example_quadrotor_hovering.cpp)
# add_executable(example_codegen example_codegen.cpp)
add_executable(example_cartpole example_cartpole.cpp)

# target_link_libraries(example_quadrotor_tracking LINK_PUBLIC tinympc)
# target_link_libraries(example_quadrotor_hovering LINK_PUBLIC tinympc)
# target_link_libraries(example_codegen LINK_PUBLIC tinympc)
target_link_libraries(example_cartpole LINK_PUBLIC tinympc)
