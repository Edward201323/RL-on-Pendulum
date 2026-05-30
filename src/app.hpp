#pragma once

#include <SFML/Graphics.hpp>

#include "render/cart.hpp"
#include "render/pendulum.hpp"

class App {
public:
    App();
    void run();

    // Control input — any horizontal force on the cart, in physics units.
    // Set this from any source (keyboard, scripted test, RL policy) before
    // the next physicsStep; the value persists across frames until changed.
    void setControlForce(float force);
    float getControlForce() const;

    // State accessors (state = (x, xdot, theta, thetadot)) for external
    // observers like an RL agent.
    float getCartX() const;
    float getCartVelocity() const;
    float getPendulumAngle() const;
    float getPendulumAngularVelocity() const;

private:
    void processEvents();
    void update(float dt);
    void physicsStep(float h);
    void pollKeyboardControl();
    void render();

    sf::RenderWindow window;
    sf::Clock clock;
    Cart cart;
    Pendulum pendulum;
    float controlForce;
};
