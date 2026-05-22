cmake_minimum_required(VERSION 3.25)

project(mc_mini LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mc_history
    src/history/main.cpp
)

add_executable(mc_event
    src/event/main.cpp
)
