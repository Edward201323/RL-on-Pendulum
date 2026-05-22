#include "track.hpp"

static void drawCapsule(sf::RenderWindow &window, sf::Vector2f center,
                        float trackWidth, float thickness, sf::Color color) {
    float r = thickness / 2.f;

    // Draw the rail
    sf::RectangleShape rail(sf::Vector2f(trackWidth, thickness));
    rail.setFillColor(color);
    rail.setOrigin(trackWidth / 2.f, r);
    rail.setPosition(center);
    window.draw(rail);

    // Draw the left cap
    sf::CircleShape leftCap(r);
    leftCap.setFillColor(color);
    leftCap.setOrigin(r, r);
    leftCap.setPosition(center.x - trackWidth / 2.f, center.y);
    window.draw(leftCap);

    // Draw the right cap
    sf::CircleShape rightCap(r);
    rightCap.setFillColor(color);
    rightCap.setOrigin(r, r);
    rightCap.setPosition(center.x + trackWidth / 2.f, center.y);
    window.draw(rightCap);
}

TrackLayout computeTrackLayout(const sf::RenderWindow &window) {
    sf::Vector2u size = window.getSize();
    TrackLayout layout;
    layout.center = sf::Vector2f(size.x / 2.f, size.y * 0.78f);
    layout.width = size.x * 0.8f;
    layout.thickness = 25.f;
    return layout;
}

void drawTrack(sf::RenderWindow &window) {
    TrackLayout layout = computeTrackLayout(window);
    float strokeWidth = 4.f;
    float innerThickness = layout.thickness - 2.f * strokeWidth;
    sf::Color outerColor = sf::Color::White;
    sf::Color innerColor = sf::Color(100, 100, 100);

    drawCapsule(window, layout.center, layout.width, layout.thickness, outerColor);
    drawCapsule(window, layout.center, layout.width, innerThickness, innerColor);
}
