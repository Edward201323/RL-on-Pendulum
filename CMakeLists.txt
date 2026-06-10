cmake_minimum_required(VERSION 3.14)  # FetchContent_MakeAvailable
project(PendulumRL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_APP "Build the SFML visual app" ON)
option(BUILD_PYTHON "Build the pybind11 physics module" ON)

# --- Shared headless physics core (no SFML) -------------------------------
# Single source of truth for the dynamics: linked into both the SFML app and
# the Python module so the demo and the trainer run identical physics.
add_library(cartpole_core STATIC src/physics/cartpole.cpp)
target_include_directories(cartpole_core PUBLIC src)
set_target_properties(cartpole_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- SFML visualizer ------------------------------------------------------
if(BUILD_APP)
    find_package(SFML 2 REQUIRED COMPONENTS graphics window system)
    add_executable(PendulumRL
        src/main.cpp
        src/app.cpp
        src/control/policy.cpp
        src/render/track.cpp
        src/render/cart.cpp
        src/render/pendulum.cpp
        src/render/hud.cpp
    )
    target_include_directories(PendulumRL PRIVATE src)
    target_link_libraries(PendulumRL PRIVATE cartpole_core sfml-graphics sfml-window sfml-system)
endif()

# --- Python module (cartpole_cpp) -----------------------------------------
if(BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(cartpole_cpp src/python/bindings.cpp)
    target_link_libraries(cartpole_cpp PRIVATE cartpole_core)
endif()
