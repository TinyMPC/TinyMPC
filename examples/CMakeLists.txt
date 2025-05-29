# Basic examples
add_executable(quadrotor_tracking quadrotor_tracking.cpp)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympcstatic)

add_executable(quadrotor_hovering quadrotor_hovering.cpp)
target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympcstatic)

add_executable(cartpole_example cartpole_example.cpp)
target_link_libraries(cartpole_example LINK_PUBLIC tinympcstatic)

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
