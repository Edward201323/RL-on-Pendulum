#pragma once

#include <SFML/Graphics.hpp>

class Pendulum {
public:
    Pendulum(float length = 220.f, float rodThickness = 10.f, float bobRadius = 22.f,
             float bobMass = 0.2f, float damping = 0.4f);

    void setPivot(sf::Vector2f pivot);
    void setAngle(float radians); // 0 = straight up, positive = clockwise tilt
    float getAngle() const;
    void setAngularVelocity(float omega);
    float getAngularVelocity() const;
    float getLength() const;
    float getBobMass() const;
    float getDamping() const;
    void draw(sf::RenderWindow& window) const;

private:
    float length;
    float rodThickness;
    float bobRadius;
    float bobMass;
    float damping;
    sf::Vector2f pivot;
    float angle;
    float angularVelocity;
};
