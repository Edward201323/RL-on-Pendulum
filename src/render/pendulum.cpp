#include "pendulum.hpp"

#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

Pendulum::Pendulum(float length, float rodThickness, float bobRadius) {
    this->length = length;
    this->rodThickness = rodThickness;
    this->bobRadius = bobRadius;
    this->pivot = sf::Vector2f(0.f, 0.f);
    this->angle = 0.f;
}

void Pendulum::setPivot(sf::Vector2f p) {
    this->pivot = p;
}

void Pendulum::setAngle(float radians) {
    this->angle = radians;
}

// Draw the rod from the pivot at the current angle, with the bob at its tip.
// alpha < 255 makes the whole pole translucent (the all-actors overlay ghosts).
void Pendulum::draw(sf::RenderWindow& window, sf::Uint8 alpha) const {
    const float strokeWidth = 4.f;
    const sf::Color white(255, 255, 255, alpha);
    const sf::Color accent(220, 80, 80, alpha);

    // Rod: rectangle with origin at bottom-center so it extends upward at angle 0.
    sf::RectangleShape rod(sf::Vector2f(this->rodThickness, this->length));
    rod.setOrigin(this->rodThickness / 2.f, this->length);
    rod.setPosition(this->pivot);
    rod.setRotation(this->angle * 180.f / kPi);
    rod.setFillColor(white);
    window.draw(rod);

    // Bob at the rod's free end: white border, colored fill.
    float bobX = this->pivot.x + this->length * std::sin(this->angle);
    float bobY = this->pivot.y - this->length * std::cos(this->angle);

    sf::CircleShape bobOuter(this->bobRadius);
    bobOuter.setOrigin(this->bobRadius, this->bobRadius);
    bobOuter.setPosition(bobX, bobY);
    bobOuter.setFillColor(white);
    window.draw(bobOuter);

    float innerR = this->bobRadius - strokeWidth;
    sf::CircleShape bobInner(innerR);
    bobInner.setOrigin(innerR, innerR);
    bobInner.setPosition(bobX, bobY);
    bobInner.setFillColor(accent);
    window.draw(bobInner);

    // Colored pivot dot marking the attachment point on the cart.
    float pivotR = 7.f;
    sf::CircleShape pivotDot(pivotR);
    pivotDot.setOrigin(pivotR, pivotR);
    pivotDot.setPosition(this->pivot);
    pivotDot.setFillColor(accent);
    window.draw(pivotDot);
}
