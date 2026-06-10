#include "cart.hpp"

Cart::Cart(float width, float height) {
    this->width = width;
    this->height = height;
    this->x = 0.f;
    this->y = 0.f;
}

void Cart::setPosition(float x, float y) {
    this->x = x;
    this->y = y;
}

float Cart::getWidth() const { return this->width; }

// Point where the pendulum attaches (the cart's current screen position).
sf::Vector2f Cart::getPivot() const {
    return sf::Vector2f(this->x, this->y);
}

// Draw the cart as a white box with a darker inset (a simple stroked rectangle).
void Cart::draw(sf::RenderWindow& window) const {
    float strokeWidth = 4.f;
    sf::Color outerColor = sf::Color::White;
    sf::Color innerColor = sf::Color(100, 100, 100);

    sf::RectangleShape outer(sf::Vector2f(this->width, this->height));
    outer.setOrigin(this->width / 2.f, this->height / 2.f);
    outer.setPosition(this->x, this->y);
    outer.setFillColor(outerColor);
    window.draw(outer);

    float iw = this->width - 2.f * strokeWidth;
    float ih = this->height - 2.f * strokeWidth;
    sf::RectangleShape inner(sf::Vector2f(iw, ih));
    inner.setOrigin(iw / 2.f, ih / 2.f);
    inner.setPosition(this->x, this->y);
    inner.setFillColor(innerColor);
    window.draw(inner);
}
