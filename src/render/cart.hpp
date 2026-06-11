#pragma once

#include <SFML/Graphics.hpp>

// Pure renderer. Physics (position, velocity, mass, bounds) now lives in the
// headless CartPole core; App positions this each frame from the core's state.
class Cart {
public:
    Cart(float width = 60.f, float height = 30.f);

    void setPosition(float x, float y);
    sf::Vector2f getPivot() const; // attachment point for the pendulum (top-center)
    float getWidth() const;
    // alpha < 255 draws a translucent cart (used for the all-actors overlay).
    void draw(sf::RenderWindow &window, sf::Uint8 alpha = 255) const;

private:
    float width;
    float height;
    float x;
    float y;
};
