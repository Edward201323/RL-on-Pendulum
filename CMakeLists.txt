cmake_minimum_required(VERSION 3.10)
project(SFMLTest CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(SFML 2 REQUIRED COMPONENTS graphics window system)
add_executable(SFMLTest src/main.cpp)
target_link_libraries(SFMLTest sfml-graphics sfml-window sfml-system)