cmake_minimum_required(VERSION 3.25)

project(mc_mini LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mc_common INTERFACE)

target_include_directories(mc_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(mc_history
    src/history/main.cpp
)

add_executable(mc_event
    src/event/main.cpp
)

target_link_libraries(mc_history PRIVATE mc_common)
target_link_libraries(mc_event PRIVATE mc_common)

target_compile_options(mc_history PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
)

target_compile_options(mc_event PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
)
