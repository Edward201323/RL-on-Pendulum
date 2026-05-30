#include "cart.hpp"

Cart::Cart(float width, float height, float mass, float friction) {
    this->width = width;
    this->height = height;
    this->mass = mass;
    this->friction = friction;
    x = 0.f;
    y = 0.f;
    velocity = 0.f;
    minX = 0.f;
    maxX = 0.f;
    placed = false;
}

void Cart::setBounds(float minX, float maxX, float y) {
    this->minX = minX;
    this->maxX = maxX;
    this->y = y;
    if (!placed) {
        x = (minX + maxX) / 2.f;
        placed = true;
    }
}

float Cart::getX() const { return x; }
float Cart::getVelocity() const { return velocity; }
float Cart::getMass() const { return mass; }
float Cart::getFriction() const { return this->friction; }
void Cart::setX(float v) { x = v; }
void Cart::setVelocity(float v) { velocity = v; }

bool Cart::clampToBounds() {
    float halfW = width / 2.f;
    if (x < minX + halfW) {
        x = minX + halfW;
        return true;
    }
    if (x > maxX - halfW) {
        x = maxX - halfW;
        return true;
    }
    return false;
}

int Cart::boundaryContact() const {
    const float halfW = this->width / 2.f;
    const float eps = 1e-4f;
    if (this->x <= this->minX + halfW + eps) return -1;
    if (this->x >= this->maxX - halfW - eps) return +1;
    return 0;
}

sf::Vector2f Cart::getPivot() const {
    return sf::Vector2f(x, y);
}

void Cart::draw(sf::RenderWindow& window) const {
    float strokeWidth = 4.f;
    sf::Color outerColor = sf::Color::White;
    sf::Color innerColor = sf::Color(100, 100, 100);

    sf::RectangleShape outer(sf::Vector2f(width, height));
    outer.setOrigin(width / 2.f, height / 2.f);
    outer.setPosition(x, y);
    outer.setFillColor(outerColor);
    window.draw(outer);

    float iw = width - 2.f * strokeWidth;
    float ih = height - 2.f * strokeWidth;
    sf::RectangleShape inner(sf::Vector2f(iw, ih));
    inner.setOrigin(iw / 2.f, ih / 2.f);
    inner.setPosition(x, y);
    inner.setFillColor(innerColor);
    window.draw(inner);
}
