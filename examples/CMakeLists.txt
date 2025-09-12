# Basic examples
add_executable(quadrotor_tracking quadrotor_tracking.cpp)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympcstatic)

add_executable(quadrotor_hovering quadrotor_hovering.cpp)
target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympcstatic)

add_executable(cartpole_example cartpole_example.cpp)
target_link_libraries(cartpole_example LINK_PUBLIC tinympcstatic)

# Linear constraint example
add_executable(quadrotor_linear_constraints quadrotor_linear_constraints.cpp)
target_link_libraries(quadrotor_linear_constraints LINK_PUBLIC tinympcstatic)

# Time-varying Linear constraint example
add_executable(quadrotor_tv_linear_constraints quadrotor_tv_linear_constraints.cpp)
target_link_libraries(quadrotor_tv_linear_constraints LINK_PUBLIC tinympcstatic)

# Conic example
add_executable(rocket_landing_mpc rocket_landing_mpc.cpp)
target_link_libraries(rocket_landing_mpc LINK_PUBLIC tinympcstatic)

# Codegen examples
add_executable(codegen_random codegen_random.cpp)
target_link_libraries(codegen_random LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_random LINK_PUBLIC "stdc++fs")
endif(MINGW)

add_executable(codegen_cartpole codegen_cartpole.cpp)
target_link_libraries(codegen_cartpole LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_cartpole LINK_PUBLIC "stdc++fs")
endif(MINGW)

add_compile_definitions(ENABLE_SDP_PROJECTION)

# Removed obstacle_avoidance_sdp.cpp - using obstacle_avoidance_sdp_lifted.cpp instead

# Removed obstacle_avoidance_augmented.cpp - file doesn't exist

add_executable(obstacle_avoidance_sdp_lifted obstacle_avoidance_sdp_lifted.cpp)
target_link_libraries(obstacle_avoidance_sdp_lifted LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(obstacle_avoidance_sdp_lifted LINK_PUBLIC "stdc++fs")
endif(MINGW)
