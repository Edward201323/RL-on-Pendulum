#pragma once

#include <SFML/Graphics.hpp>

#include "render/cart.hpp"
#include "render/pendulum.hpp"

class App {
public:
    App();
    void run();

private:
    void processEvents();
    void update(float dt);
    void render();

    sf::RenderWindow window;
    sf::Clock clock;
    Cart cart;
    Pendulum pendulum;
};
