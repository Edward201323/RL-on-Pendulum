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

    // Small, unboxed, faint right-aligned text in the top-right corner (controls hint).
    void drawCornerHint(sf::RenderWindow& window, const std::string& text) const;

    // Scrolling time graph: the last `capacity` samples, newest at the right, on a
    // fixed +-yRange vertical scale. Used for the control-force trace. xLeft sets the
    // panel's left edge; < 0 keeps the default bottom-right slot.
    void drawGraph(sf::RenderWindow& window, const std::deque<float>& samples,
                   std::size_t capacity, float yRange, const char* label,
                   float xLeft = -1.f) const;

    // Learning-curve graph: score (ys) vs. attempts (xs), the full history from 0
    // to current, drawn into the rectangle (x0,y0)-(x1,y1). Caller picks the rect:
    // a bottom slot for the small graph, or the whole play area for the training view.
    void drawScoreGraph(sf::RenderWindow& window, const std::vector<float>& xs,
                        const std::vector<float>& ys, const char* label,
                        float x0, float y0, float x1, float y1) const;

    // Device-pixels-per-point of the render target (2 on a Retina display, 1
    // otherwise). Glyphs are rasterized at this multiple of their point size so
    // text stays sharp when the view is scaled up to native resolution.
    void setRenderScale(float scale);

private:
    // Build a text whose glyphs are rasterized at native pixel density: the font
    // size is multiplied by renderScale and the text counter-scaled by 1/renderScale,
    // so it draws at its requested point size but with a high-resolution glyph atlas.
    // getLocalBounds() then reports bounds in the (scaled-up) glyph space, which is
    // exactly what setOrigin expects; callers needing point-space dimensions divide
    // the bounds by renderScale.
    sf::Text makeText(const std::string& str, unsigned pointSize) const;

    sf::Font font;
    bool hasFont;
    float renderScale = 1.f;
};
