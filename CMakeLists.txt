cmake_minimum_required(VERSION 3.10)
project(PendulumRL CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(SFML 2 REQUIRED COMPONENTS graphics window system)
add_executable(PendulumRL
    src/main.cpp
    src/app.cpp
    src/render/track.cpp
    src/render/cart.cpp
    src/render/pendulum.cpp
)
target_include_directories(PendulumRL PRIVATE src)
target_link_libraries(PendulumRL sfml-graphics sfml-window sfml-system)