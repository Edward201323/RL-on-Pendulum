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
    // the scene, so the cart and track read as the main focus. Outline defaults to
    // coral; pass green for the all-actors (training) view.
    void drawPlayfield(sf::RenderWindow& window,
                       sf::Color outline = sf::Color(225, 130, 95)) const;

    // Position scale (in meters) drawn under the track, aligned to the layout.
    // trackLimit is the cart's max |x| in meters; pixelsPerMeter maps it to screen.
    void drawAxis(sf::RenderWindow& window, const TrackLayout& layout,
                  float trackLimit, float pixelsPerMeter) const;

    // Small boxed status panel showing the given (possibly multi-line) text, with
    // its top-left at (x, y) and the given outline color. Returns the box width so
    // callers can place another panel flush beside it.
    float drawTextBox(sf::RenderWindow& window, const std::string& text,
                      float x, float y, sf::Color outline) const;

    // Scrolling time graph (bottom-right): the last `capacity` samples, newest at
    // the right, on a fixed +-yRange vertical scale. Used for the control-force trace.
    void drawGraph(sf::RenderWindow& window, const std::deque<float>& samples,
                   std::size_t capacity, float yRange, const char* label) const;

    // Learning-curve graph: score (ys) vs. attempts (xs), the full history from 0
    // to current, drawn into the rectangle (x0,y0)-(x1,y1). Caller picks the rect:
    // a bottom slot for the small graph, or the whole play area for the training view.
    void drawScoreGraph(sf::RenderWindow& window, const std::vector<float>& xs,
                        const std::vector<float>& ys, const char* label,
                        float x0, float y0, float x1, float y1) const;

private:
    sf::Font font;
    bool hasFont;
};
