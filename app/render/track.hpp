#pragma once

#include <SFML/Graphics.hpp>

struct TrackLayout {
    sf::Vector2f center;
    float width;
    float thickness;
};

TrackLayout computeTrackLayout(const sf::RenderWindow &window);
void drawTrack(sf::RenderWindow &window, const TrackLayout &layout);
