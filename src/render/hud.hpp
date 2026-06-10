#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include <SFML/Graphics.hpp>

#include "render/track.hpp"

// Heads-up overlay for the SFML app: a position (x) axis under the track plus a
// small panel of live cart-pole readings. Tries to load a system monospace font
// at construction; if none is found it still draws the axis ticks but skips the
// text (labels and readout).
class Hud {
public:
    Hud();

    // Rounded frame around the play area (cart/track/pole). Drawn first, behind
    // the scene, so the cart and track read as the main focus.
    void drawPlayfield(sf::RenderWindow& window) const;

    // Position scale (in meters) drawn under the track, aligned to the layout.
    // trackLimit is the cart's max |x| in meters; pixelsPerMeter maps it to screen.
    void drawAxis(sf::RenderWindow& window, const TrackLayout& layout,
                  float trackLimit, float pixelsPerMeter) const;

    // Small boxed status panel showing the given (possibly multi-line) text.
    // bottom=false anchors it top-left; bottom=true anchors it bottom-left.
    void drawTextBox(sf::RenderWindow& window, const std::string& text, bool bottom) const;

    // Scrolling time graph (bottom-right): the last `capacity` samples, newest at
    // the right, on a fixed +-yRange vertical scale. Used for the control-force trace.
    void drawGraph(sf::RenderWindow& window, const std::deque<float>& samples,
                   std::size_t capacity, float yRange, const char* label) const;

    // Learning-curve graph (bottom-left): score (ys) vs. attempts (xs), the full
    // history from 0 to current, with both axes auto-scaled to fit the data.
    void drawScoreGraph(sf::RenderWindow& window, const std::vector<float>& xs,
                        const std::vector<float>& ys, const char* label) const;

private:
    sf::Font font;
    bool hasFont;
};
