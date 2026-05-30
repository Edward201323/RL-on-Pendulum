#pragma once

#include <SFML/Graphics.hpp>

class Cart {
public:
    Cart(float width = 100.f, float height = 50.f, float mass = 1.0f,
         float friction = 1.0f);

    void setBounds(float minX, float maxX, float y);
    void draw(sf::RenderWindow &window) const;
    sf::Vector2f getPivot() const; // top-center of cart

    float getX() const;
    float getVelocity() const;
    float getMass() const;
    float getFriction() const; // viscous cart-track coefficient (force per velocity)
    void setX(float x);
    void setVelocity(float v);

    // Clamp x inside [minX+halfW, maxX-halfW]; returns true if it had to clamp.
    bool clampToBounds();

    // -1 if pressed against the left wall, +1 if against the right wall, 0 otherwise.
    int boundaryContact() const;

private:
    float width;
    float height;
    float mass;
    float friction;
    float x;
    float y;
    float velocity;
    float minX;
    float maxX;
    bool placed;
};
