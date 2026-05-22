#include "app.hpp"

#include "render/track.hpp"

static sf::ContextSettings makeContextSettings() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    return settings;
}

App::App()
    : window(sf::VideoMode(1000, 600), "Pendulum Balnacing", sf::Style::Default,
             makeContextSettings()) {
    window.setFramerateLimit(60);
}

void App::run() {
    while (window.isOpen()) {
        processEvents();
        float dt = clock.restart().asSeconds();
        update(dt);
        render();
    }
}

void App::processEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window.close();
    }
}

void App::update(float dt) {
    TrackLayout layout = computeTrackLayout(window);
    cart.setBounds(layout.center.x - layout.width / 2.f,
                   layout.center.x + layout.width / 2.f,
                   layout.center.y);
    cart.update(dt);
    pendulum.setPivot(cart.getPivot());
}

void App::render() {
    window.clear(sf::Color(100, 100, 100));
    drawTrack(window);
    
    cart.draw(window);
    pendulum.draw(window);
    window.display();
}
