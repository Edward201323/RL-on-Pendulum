#pragma once

#include <SFML/Graphics.hpp>

class Cart {
public:
    Cart(float width = 100.f, float height = 50.f, float speed = 400.f);

    void setBounds(float minX, float maxX, float y);
    void update(float dt);
    void draw(sf::RenderWindow &window) const;
    sf::Vector2f getPivot() const; // top-center of cart

private:
    float width;
    float height;
    float speed;
    float x;
    float y;
    float minX;
    float maxX;
    bool placed;
};
