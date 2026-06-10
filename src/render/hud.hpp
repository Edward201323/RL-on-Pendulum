#pragma once

#include <SFML/Graphics.hpp>

#include "render/track.hpp"

// Heads-up overlay for the SFML app: a position (x) axis under the track plus a
// small panel of live cart-pole readings. Tries to load a system monospace font
// at construction; if none is found it still draws the axis ticks but skips the
// text (labels and readout).
class Hud {
public:
    Hud();

    // Position scale drawn under the track, aligned to the current layout.
    // trackLimit is the cart's max |x| (physics px), so the axis spans the walls.
    void drawAxis(sf::RenderWindow& window, const TrackLayout& layout,
                  float trackLimit) const;

    // Live data panel (top-left): the full cart-pole state, whether the pole is
    // currently upright, and how long it has stayed up (current/best streak).
    void drawReadout(sf::RenderWindow& window, float x, float velocity,
                     float angle, float angularVelocity, float force,
                     bool upright, float uprightStreak, float bestUprightStreak) const;

private:
    sf::Font font;
    bool hasFont;
};
