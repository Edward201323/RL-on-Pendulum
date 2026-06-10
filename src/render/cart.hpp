#pragma once

#include <SFML/Graphics.hpp>

// Pure renderer. Physics (position, velocity, mass, bounds) now lives in the
// headless CartPole core; App positions this each frame from the core's state.
class Cart {
public:
    Cart(float width = 100.f, float height = 50.f);

    void setPosition(float x, float y);
    sf::Vector2f getPivot() const; // attachment point for the pendulum (top-center)
    float getWidth() const;
    void draw(sf::RenderWindow &window) const;

private:
    float width;
    float height;
    float x;
    float y;
};
