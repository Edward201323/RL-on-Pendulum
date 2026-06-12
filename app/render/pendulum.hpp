#pragma once

#include <SFML/Graphics.hpp>

// Pure renderer. The pole's dynamics (angle, angular velocity, mass, damping)
// now live in the headless CartPole core; App feeds this the pivot and angle.
class Pendulum {
public:
    Pendulum(float length = 130.f, float rodThickness = 6.f, float bobRadius = 14.f);

    void setPivot(sf::Vector2f pivot);
    void setAngle(float radians); // 0 = straight up, positive = clockwise tilt
    // alpha < 255 draws a translucent "ghost" pole (used for the all-actors overlay).
    void draw(sf::RenderWindow& window, sf::Uint8 alpha = 255) const;

private:
    float length;
    float rodThickness;
    float bobRadius;
    sf::Vector2f pivot;
    float angle;
};
