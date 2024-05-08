add_executable(quadrotor_tracking quadrotor_tracking.cpp)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympcstatic)

# add_executable(quadrotor_hovering quadrotor_hovering.cpp)
# target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympcstatic)

# add_executable(cartpole_example cartpole_example.cpp)
# target_link_libraries(cartpole_example LINK_PUBLIC tinympcstatic)

# add_executable(codegen_random codegen_random.cpp)
# target_link_libraries(codegen_random LINK_PUBLIC tinympcstatic)

# add_executable(codegen_cartpole codegen_cartpole.cpp)
# target_link_libraries(codegen_cartpole LINK_PUBLIC tinympcstatic)