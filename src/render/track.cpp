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

// Work out where the track sits and how wide it is for the current window size.
TrackLayout computeTrackLayout(const sf::RenderWindow &window) {
    sf::Vector2u size = window.getSize();
    TrackLayout layout;
    // Keep the cart/track in the upper-middle (the main focus) so the lower
    // area is free for the smaller control-force time graph.
    layout.center = sf::Vector2f(size.x / 2.f, size.y * 0.40f);
    layout.width = size.x * 0.8f;
    layout.thickness = 25.f;
    return layout;
}

// Draw the rail as a white capsule with a darker inner capsule (the stroke look).
// Layout (center, width, thickness) is supplied by the caller in pixels.
void drawTrack(sf::RenderWindow &window, const TrackLayout &layout) {
    float strokeWidth = 4.f;
    float innerThickness = layout.thickness - 2.f * strokeWidth;
    sf::Color outerColor = sf::Color::White;
    sf::Color innerColor = sf::Color(100, 100, 100);

    drawCapsule(window, layout.center, layout.width, layout.thickness, outerColor);
    drawCapsule(window, layout.center, layout.width, innerThickness, innerColor);
}
