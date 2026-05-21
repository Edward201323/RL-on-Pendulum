#include <SFML/Graphics.hpp>

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Pendulum Balnacing");
    window.setFramerateLimit(60); // FPS

    sf::CircleShape circle(30.f); // Creates the circle
    circle.setFillColor(sf::Color::Cyan);
    circle.setPosition(100.f, 100.f);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) { // Closes if a window closes
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::Black);
        window.draw(circle);
        window.display();
    }

    return 0;
}