#include "cart.hpp"

Cart::Cart(float width, float height, float speed) {
    this->width = width;
    this->height = height;
    this->speed = speed;
    x = 0.f;
    y = 0.f;
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

void Cart::update(float dt) {
    float dir = 0.f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
        dir -= 1.f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
        dir += 1.f;
    }
    x += dir * speed * dt;

    float halfW = width / 2.f;
    if (x < minX + halfW){
        x = minX + halfW;
    } 
    if (x > maxX - halfW){
        x = maxX - halfW;
    } 
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
