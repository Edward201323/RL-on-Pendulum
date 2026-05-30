#include "pendulum.hpp"

#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

Pendulum::Pendulum(float length, float rodThickness, float bobRadius,
                   float bobMass, float damping) {
    this->length = length;
    this->rodThickness = rodThickness;
    this->bobRadius = bobRadius;
    this->bobMass = bobMass;
    this->damping = damping;
    pivot = sf::Vector2f(0.f, 0.f);
    angle = 0;
    angularVelocity = 0.f;
}

void Pendulum::setPivot(sf::Vector2f p) {
    pivot = p;
}

void Pendulum::setAngle(float radians) {
    angle = radians;
}

float Pendulum::getAngle() const {
    return angle;
}

void Pendulum::setAngularVelocity(float omega) {
    angularVelocity = omega;
}

float Pendulum::getAngularVelocity() const {
    return angularVelocity;
}

float Pendulum::getLength() const {
    return length;
}

float Pendulum::getBobMass() const {
    return bobMass;
}

float Pendulum::getDamping() const {
    return damping;
}

void Pendulum::draw(sf::RenderWindow& window) const {
    const float strokeWidth = 4.f;
    const sf::Color accent(220, 80, 80);

    // Rod: rectangle with origin at bottom-center so it extends upward at angle 0.
    sf::RectangleShape rod(sf::Vector2f(rodThickness, length));
    rod.setOrigin(rodThickness / 2.f, length);
    rod.setPosition(pivot);
    rod.setRotation(angle * 180.f / kPi);
    rod.setFillColor(sf::Color::White);
    window.draw(rod);

    // Bob at the rod's free end: white border, colored fill.
    float bobX = pivot.x + length * std::sin(angle);
    float bobY = pivot.y - length * std::cos(angle);

    sf::CircleShape bobOuter(bobRadius);
    bobOuter.setOrigin(bobRadius, bobRadius);
    bobOuter.setPosition(bobX, bobY);
    bobOuter.setFillColor(sf::Color::White);
    window.draw(bobOuter);

    float innerR = bobRadius - strokeWidth;
    sf::CircleShape bobInner(innerR);
    bobInner.setOrigin(innerR, innerR);
    bobInner.setPosition(bobX, bobY);
    bobInner.setFillColor(accent);
    window.draw(bobInner);

    // Colored pivot dot marking the attachment point on the cart.
    float pivotR = 7.f;
    sf::CircleShape pivotDot(pivotR);
    pivotDot.setOrigin(pivotR, pivotR);
    pivotDot.setPosition(pivot);
    pivotDot.setFillColor(accent);
    window.draw(pivotDot);
}
