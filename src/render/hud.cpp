#include "hud.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {
// Candidate system fonts, in preference order (Menlo is monospace, so the
// readout columns line up). First one that loads wins.
const char* kFontPaths[] = {
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
};

// Fraction of window height where the play area ends. The bottom panels sit
// below this, so they never overlap the track regardless of window size.
constexpr float kPlayBottomFrac = 0.72f;

// A w x h rounded rectangle with origin at its top-left, as a convex polygon.
// The caller sets position / fill / outline. Used for all the framed panels.
sf::ConvexShape roundedRect(float w, float h, float r) {
    const int cps = 6;  // points per corner
    const float pi = 3.14159265358979323846f;
    const float cx[4] = {w - r, w - r, r, r};
    const float cy[4] = {r, h - r, h - r, r};
    const float a0[4] = {-90.f, 0.f, 90.f, 180.f};  // arc start angle per corner
    sf::ConvexShape s;
    s.setPointCount(cps * 4);
    int idx = 0;
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < cps; ++i) {
            const float a = (a0[c] + 90.f * i / (cps - 1)) * pi / 180.f;
            s.setPoint(idx++, sf::Vector2f(cx[c] + r * std::cos(a), cy[c] + r * std::sin(a)));
        }
    }
    return s;
}
}  // namespace

Hud::Hud() : hasFont(false) {
    for (const char* path : kFontPaths) {
        if (this->font.loadFromFile(path)) {
            this->hasFont = true;
            // Nearest-neighbour glyph sampling keeps text edges crisp (SFML's
            // default linear smoothing softens them, which is worse once macOS
            // upscales the whole window on a Retina display).
            this->font.setSmooth(false);
            break;
        }
    }
}

void Hud::drawPlayfield(sf::RenderWindow& window) const {
    const sf::Vector2u ws = window.getSize();
    const float x = 40.f;
    const float y = static_cast<float>(ws.y) * 0.23f;  // below the top-left status box
    const float w = static_cast<float>(ws.x) - 2.f * x;
    const float h = static_cast<float>(ws.y) * kPlayBottomFrac - y;
    sf::ConvexShape box = roundedRect(w, h, 20.f);
    box.setPosition(x, y);
    box.setFillColor(sf::Color(72, 72, 72));
    box.setOutlineThickness(3.f);
    box.setOutlineColor(sf::Color(225, 130, 95));  // coral, like the reference UI
    window.draw(box);
}

void Hud::drawAxis(sf::RenderWindow& window, const TrackLayout& layout,
                   float trackLimit, float pixelsPerMeter) const {
    const float axisY = layout.center.y + 45.f;
    const sf::Color axisColor(210, 210, 210);
    const float halfPx = trackLimit * pixelsPerMeter;

    // Baseline spanning the cart's reachable range [-trackLimit, +trackLimit] (m).
    sf::Vertex baseline[] = {
        sf::Vertex(sf::Vector2f(layout.center.x - halfPx, axisY), axisColor),
        sf::Vertex(sf::Vector2f(layout.center.x + halfPx, axisY), axisColor),
    };
    window.draw(baseline, 2, sf::Lines);

    // Fine ruler: minor ticks every 0.05 m; major (taller, labelled) every
    // 0.25 m. x = 0 is centered. Denser ticks = higher position resolution.
    const int steps = static_cast<int>(trackLimit / 0.05f + 0.5f);
    for (int k = -steps; k <= steps; ++k) {
        const float meters = k * 0.05f;
        const float sx = layout.center.x + meters * pixelsPerMeter;
        const bool major = (k % 5 == 0);   // every 0.25 m
        const float h = major ? 12.f : 5.f;
        sf::Vertex tick[] = {
            sf::Vertex(sf::Vector2f(sx, axisY - h), axisColor),
            sf::Vertex(sf::Vector2f(sx, axisY + h), axisColor),
        };
        window.draw(tick, 2, sf::Lines);

        if (major && this->hasFont) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2f", meters);
            sf::Text label(buf, this->font, 16);
            label.setFillColor(axisColor);
            const sf::FloatRect b = label.getLocalBounds();
            // Snap origin and position to whole pixels so glyphs stay crisp.
            label.setOrigin(std::round(b.left + b.width / 2.f), 0.f);
            label.setPosition(std::round(sx), std::round(axisY + 16.f));
            window.draw(label);
        }
    }
}

void Hud::drawTextBox(sf::RenderWindow& window, const std::string& str, bool bottom) const {
    if (!this->hasFont) return;
    const sf::Vector2u ws = window.getSize();

    sf::Text text(str, this->font, 20);
    text.setFillColor(sf::Color::White);

    const sf::FloatRect b = text.getLocalBounds();
    const float boxW = b.width + 30.f, boxH = b.height + 30.f;
    const float bx = 40.f;
    const float by = bottom ? (static_cast<float>(ws.y) - boxH - 28.f) : 22.f;

    sf::ConvexShape box = roundedRect(boxW, boxH, 10.f);
    box.setPosition(bx, by);
    box.setFillColor(sf::Color(20, 20, 20, 215));
    box.setOutlineThickness(2.f);
    box.setOutlineColor(sf::Color(95, 190, 180));  // teal, like the reference UI
    window.draw(box);

    text.setPosition(std::round(bx + 14.f), std::round(by + 12.f));
    window.draw(text);
}

void Hud::drawScoreGraph(sf::RenderWindow& window, const std::vector<float>& xs,
                         const std::vector<float>& ys, const char* label) const {
    const sf::Vector2u ws = window.getSize();
    const float gW = 540.f;                              // bottom-LEFT, mirrors force graph
    const float x0 = 40.f, x1 = x0 + gW;
    const float y0 = static_cast<float>(ws.y) * kPlayBottomFrac + 16.f;
    const float y1 = static_cast<float>(ws.y) - 28.f;
    const sf::Color border(120, 200, 150), trace(130, 225, 150), grid(150, 150, 150);

    sf::ConvexShape panel = roundedRect(x1 - x0, y1 - y0, 10.f);
    panel.setPosition(x0, y0);
    panel.setFillColor(sf::Color(25, 25, 25, 190));
    panel.setOutlineThickness(2.f);
    panel.setOutlineColor(border);
    window.draw(panel);

    auto drawLabel = [&](const char* s, unsigned size, sf::Color c, float ax, float ay,
                         int align) {  // align: 0 left, 1 right
        if (!this->hasFont) return;
        sf::Text t(s, this->font, size);
        t.setFillColor(c);
        const sf::FloatRect b = t.getLocalBounds();
        const float ox = (align == 1) ? (b.left + b.width) : 0.f;
        t.setOrigin(std::round(ox), 0.f);
        t.setPosition(std::round(ax), std::round(ay));
        window.draw(t);
    };

    if (xs.size() < 2) {
        drawLabel(this->hasFont ? "Score  (collecting...)" : "", 16, trace, x0 + 12.f, y0 + 8.f, 0);
        return;
    }

    // Plot area inset for labels (title top, x labels bottom, y labels right).
    const float px0 = x0 + 14.f, px1 = x1 - 40.f;
    const float py0 = y0 + 28.f, py1 = y1 - 22.f;

    const float xmax = xs.back() > 0.f ? xs.back() : 1.f;  // attempts: 0 .. current
    const float ymin = 0.f, ymax = 100.f;  // score is a fixed 0..100 "% upright"

    sf::VertexArray curve(sf::LineStrip, xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const float fx = px0 + (xs[i] / xmax) * (px1 - px0);
        const float fy = py1 - ((ys[i] - ymin) / (ymax - ymin)) * (py1 - py0);
        curve[i].position = sf::Vector2f(fx, fy);
        curve[i].color = trace;
    }
    window.draw(curve);

    float best = ys[0];
    for (float v : ys) { if (v > best) best = v; }
    char buf[48];
    std::snprintf(buf, sizeof(buf), "Best  %.0f", best);
    drawLabel(buf, 16, sf::Color(220, 220, 220), x0 + 12.f, y0 + 6.f, 0);
    std::snprintf(buf, sizeof(buf), "%.0f", ymax);
    drawLabel(buf, 12, grid, x1 - 6.f, py0 - 6.f, 1);
    std::snprintf(buf, sizeof(buf), "%.0f", ymin);
    drawLabel(buf, 12, grid, x1 - 6.f, py1 - 12.f, 1);
    std::snprintf(buf, sizeof(buf), "%.0f attempts", xmax);
    drawLabel(buf, 12, grid, x1 - 6.f, y1 - 18.f, 1);
    drawLabel("0", 12, grid, px0, y1 - 18.f, 0);
}

void Hud::drawGraph(sf::RenderWindow& window, const std::deque<float>& samples,
                    std::size_t capacity, float yRange, const char* label) const {
    const sf::Vector2u ws = window.getSize();
    // A modest, secondary panel in the bottom-right corner (the cart/track is the
    // main focus, so the graph is deliberately small).
    const float gW = 540.f;
    const float x1 = static_cast<float>(ws.x) - 40.f;
    const float x0 = x1 - gW;
    // Sit just below the play area so it can never overlap the track.
    const float y0 = static_cast<float>(ws.y) * kPlayBottomFrac + 16.f;
    const float y1 = static_cast<float>(ws.y) - 28.f;
    const float graphH = y1 - y0;
    const float ymid = 0.5f * (y0 + y1);
    const float halfH = 0.5f * graphH - 10.f;  // leave a little headroom
    const sf::Color border(230, 150, 90);      // warm accent, like the reference UI
    const sf::Color trace(235, 200, 90);
    const sf::Color grid(150, 150, 150);

    // Rounded panel with a translucent fill and an accent outline.
    sf::ConvexShape panel = roundedRect(x1 - x0, y1 - y0, 10.f);
    panel.setPosition(x0, y0);
    panel.setFillColor(sf::Color(25, 25, 25, 190));
    panel.setOutlineThickness(2.f);
    panel.setOutlineColor(border);
    window.draw(panel);

    // Zero line.
    sf::Vertex zero[] = {
        sf::Vertex(sf::Vector2f(x0, ymid), grid),
        sf::Vertex(sf::Vector2f(x1, ymid), grid),
    };
    window.draw(zero, 2, sf::Lines);

    // Trace: oldest sample on the left, newest at the right edge.
    const std::size_t n = samples.size();
    if (n >= 2 && capacity > 1) {
        sf::VertexArray va(sf::LineStrip, n);
        for (std::size_t i = 0; i < n; ++i) {
            const float frac = static_cast<float>(capacity - n + i) /
                               static_cast<float>(capacity - 1);
            float v = samples[i];
            if (v > yRange) v = yRange;
            if (v < -yRange) v = -yRange;
            va[i].position = sf::Vector2f(x0 + frac * (x1 - x0), ymid - (v / yRange) * halfH);
            va[i].color = trace;
        }
        window.draw(va);
    }

    if (this->hasFont) {
        // Title + live value, top-left inside the panel.
        char head[64];
        std::snprintf(head, sizeof(head), "%s   %+6.2f", label, n ? samples.back() : 0.f);
        sf::Text title(head, this->font, 16);
        title.setFillColor(sf::Color(220, 220, 220));
        title.setPosition(std::round(x0 + 10.f), std::round(y0 + 6.f));
        window.draw(title);

        // Vertical scale labels (+yRange / 0 / -yRange) at the right edge.
        const float ys[3] = {y0 + 12.f, ymid, y1 - 12.f};
        const float vals[3] = {yRange, 0.f, -yRange};
        for (int i = 0; i < 3; ++i) {
            char vb[16];
            std::snprintf(vb, sizeof(vb), "%+.0f", vals[i]);
            sf::Text t(vb, this->font, 14);
            t.setFillColor(grid);
            const sf::FloatRect b = t.getLocalBounds();
            // Snap to whole pixels for crisp glyphs.
            t.setOrigin(std::round(b.left + b.width), std::round(b.top + b.height / 2.f));
            t.setPosition(std::round(x1 - 6.f), std::round(ys[i]));
            window.draw(t);
        }
    }
}
