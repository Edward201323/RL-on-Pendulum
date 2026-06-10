#pragma once

#include <random>

#include <SFML/Graphics.hpp>

#include "control/policy.hpp"
#include "physics/cartpole.hpp"
#include "render/cart.hpp"
#include "render/hud.hpp"
#include "render/pendulum.hpp"

class App {
public:
    App();
    void run();

    // Control input — any horizontal force on the cart, in physics units.
    // Set this from any source (keyboard, scripted test, RL policy) before
    // the next step; the value persists across frames until changed.
    void setControlForce(float force);
    float getControlForce() const;

    // State accessors (state = (x, xdot, theta, thetadot)) for external
    // observers like an RL agent. Delegate to the headless physics core.
    float getCartX() const;
    float getCartVelocity() const;
    float getPendulumAngle() const;
    float getPendulumAngularVelocity() const;

private:
    void processEvents();
    void update(float dt);
    void resetUpright();  // start a fresh balance attempt near the top
    void render();

    sf::RenderWindow window;
    sf::Clock clock;
    CartPole sim;
    Cart cart;
    Pendulum pendulum;
    Hud hud;              // on-screen x-axis + live data overlay
    Policy policy;        // trained RL policy that drives the cart
    std::mt19937 rng;     // small randomness for the reset angle
    float attemptTime;    // seconds the pole has balanced in the current attempt
    float bestTime;       // longest balance so far this session
};
