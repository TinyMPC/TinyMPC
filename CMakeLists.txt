cmake_minimum_required(VERSION 3.0.0)
list(APPEND CMAKE_MESSAGE_CONTEXT slap)
project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

add_subdirectory(src/tinympc)
add_subdirectory(examples)
add_subdirectory(test)