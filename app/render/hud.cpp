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
            break;
        }
    }
}

// Device-pixel density of the render target (see header). Glyphs are rasterized at
// this multiple of their point size so text stays sharp at native resolution.
void Hud::setRenderScale(float scale) {
    this->renderScale = scale > 0.f ? scale : 1.f;
}

// A text drawn at its point size but with a glyph atlas rasterized at native pixel
// density: oversize the font, then counter-scale so the on-screen size is unchanged.
sf::Text Hud::makeText(const std::string& str, unsigned pointSize) const {
    sf::Text t(str, this->font,
               static_cast<unsigned>(std::round(pointSize * this->renderScale)));
    const float inv = 1.f / this->renderScale;
    t.setScale(inv, inv);
    return t;
}

void Hud::drawPlayfield(sf::RenderWindow& window, sf::Color outline) const {
    const sf::Vector2u ws = window.getSize();
    const float x = 40.f;
    const float y = static_cast<float>(ws.y) * 0.23f;  // below the top-left status box
    const float w = static_cast<float>(ws.x) - 2.f * x;
    const float h = static_cast<float>(ws.y) * kPlayBottomFrac - y;
    sf::ConvexShape box = roundedRect(w, h, 20.f);
    box.setPosition(x, y);
    box.setFillColor(sf::Color(72, 72, 72));
    box.setOutlineThickness(3.f);
    box.setOutlineColor(outline);
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
            sf::Text label = makeText(buf, 16);
            label.setFillColor(axisColor);
            const sf::FloatRect b = label.getLocalBounds();
            // Snap origin and position to whole pixels so glyphs stay crisp.
            label.setOrigin(std::round(b.left + b.width / 2.f), 0.f);
            label.setPosition(std::round(sx), std::round(axisY + 16.f));
            window.draw(label);
        }
    }
}

float Hud::drawTextBox(sf::RenderWindow& window, const std::string& str,
                       float x, float y, sf::Color outline) const {
    if (!this->hasFont) return 0.f;

    sf::Text text = makeText(str, 20);
    text.setFillColor(sf::Color::White);

    // getLocalBounds is in the oversized glyph space; scale back to points to size
    // the panel (the text itself is counter-scaled, so it draws at its point size).
    const sf::FloatRect b = text.getLocalBounds();
    const float boxW = b.width / this->renderScale + 30.f,
                boxH = b.height / this->renderScale + 30.f;

    sf::ConvexShape box = roundedRect(boxW, boxH, 10.f);
    box.setPosition(x, y);
    box.setFillColor(sf::Color(20, 20, 20, 215));
    box.setOutlineThickness(2.f);
    box.setOutlineColor(outline);
    window.draw(box);

    text.setPosition(std::round(x + 14.f), std::round(y + 12.f));
    window.draw(text);
    return boxW;
}

void Hud::drawCornerHint(sf::RenderWindow& window, const std::string& str) const {
    if (!this->hasFont) return;
    sf::Text t = makeText(str, 14);
    t.setFillColor(sf::Color(165, 165, 165));
    const sf::FloatRect b = t.getLocalBounds();
    t.setOrigin(std::round(b.left + b.width), 0.f);  // anchor the right edge
    t.setPosition(static_cast<float>(window.getSize().x) - 24.f, 20.f);
    window.draw(t);
}

void Hud::drawScoreGraph(sf::RenderWindow& window, const std::vector<float>& xs,
                         const std::vector<float>& ys, const char* label,
                         float x0, float y0, float x1, float y1) const {
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
        sf::Text t = makeText(s, size);
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
    auto sx = [&](float x) { return px0 + (x / xmax) * (px1 - px0); };
    auto sy = [&](float y) { return py1 - ((y - ymin) / (ymax - ymin)) * (py1 - py0); };

    // Vertical gridlines at "nice" round attempt counts (e.g. every 1000), so as the
    // run grows the axis extends and more lines appear -- the plot grows over time.
    auto niceStep = [](float rough) {
        if (rough <= 0.f) return 1.f;
        const float mag = std::pow(10.f, std::floor(std::log10(rough)));
        const float n = rough / mag;
        const float f = (n < 1.5f) ? 1.f : (n < 3.f) ? 2.f : (n < 7.f) ? 5.f : 10.f;
        return f * mag;
    };
    const float xstep = niceStep(xmax / 6.f);  // aim for ~6 vertical divisions

    // Faint gridlines drawn first so the fill and curve sit on top (clean area chart).
    const sf::Color gridLine(150, 165, 160, 55);
    for (int g = 1; g <= 4; ++g) {  // horizontal: 25/50/75/100
        const float gy = sy(ymax * g / 4.f);
        sf::Vertex h[] = {{{px0, gy}, gridLine}, {{px1, gy}, gridLine}};
        window.draw(h, 2, sf::Lines);
    }
    for (float xt = 0.f; xt <= xmax + 0.5f; xt += xstep) {  // vertical: at each tick
        const float gx = sx(xt);
        sf::Vertex v[] = {{{gx, py0}, gridLine}, {{gx, py1}, gridLine}};
        window.draw(v, 2, sf::Lines);
    }

    // Filled area under the curve (translucent green), then the bright curve on top.
    const sf::Color fillColor(120, 200, 150, 70);
    sf::VertexArray fill(sf::TriangleStrip, xs.size() * 2);
    sf::VertexArray curve(sf::LineStrip, xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const float fx = sx(xs[i]), fy = sy(ys[i]);
        fill[2 * i] = sf::Vertex(sf::Vector2f(fx, py1), fillColor);  // baseline
        fill[2 * i + 1] = sf::Vertex(sf::Vector2f(fx, fy), fillColor);  // curve top
        curve[i] = sf::Vertex(sf::Vector2f(fx, fy), trace);
    }
    window.draw(fill);
    window.draw(curve);

    // Labels: title (top-left), latest value (top-right), y gridline values (right),
    // and the attempt range along the bottom.
    char buf[48];
    drawLabel(label, 16, sf::Color(220, 220, 220), x0 + 12.f, y0 + 6.f, 0);
    std::snprintf(buf, sizeof(buf), "%.1f", ys.back());
    drawLabel(buf, 16, trace, x1 - 6.f, y0 + 6.f, 1);
    for (int g = 1; g <= 3; ++g) {
        std::snprintf(buf, sizeof(buf), "%d", 25 * g);
        drawLabel(buf, 12, grid, x1 - 6.f, sy(ymax * g / 4.f) - 8.f, 1);
    }
    // Attempt number under each vertical gridline (0, then every nice step).
    for (float xt = 0.f; xt <= xmax + 0.5f; xt += xstep) {
        std::snprintf(buf, sizeof(buf), "%.0f", xt);
        drawLabel(buf, 12, grid, sx(xt), y1 - 18.f, 0);
    }
}

void Hud::drawGraph(sf::RenderWindow& window, const std::deque<float>& samples,
                    std::size_t capacity, float yRange, const char* label,
                    float xLeft) const {
    const sf::Vector2u ws = window.getSize();
    // A modest, secondary panel (the cart/track is the main focus, so it's small).
    const float gW = 540.f;
    const float x0 = (xLeft >= 0.f) ? xLeft : static_cast<float>(ws.x) - 40.f - gW;
    const float x1 = x0 + gW;
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
        sf::Text title = makeText(head, 16);
        title.setFillColor(sf::Color(220, 220, 220));
        title.setPosition(std::round(x0 + 10.f), std::round(y0 + 6.f));
        window.draw(title);

        // Vertical scale labels (+yRange / 0 / -yRange) at the right edge.
        const float ys[3] = {y0 + 12.f, ymid, y1 - 12.f};
        const float vals[3] = {yRange, 0.f, -yRange};
        for (int i = 0; i < 3; ++i) {
            char vb[16];
            std::snprintf(vb, sizeof(vb), "%+.0f", vals[i]);
            sf::Text t = makeText(vb, 14);
            t.setFillColor(grid);
            const sf::FloatRect b = t.getLocalBounds();
            // Snap to whole pixels for crisp glyphs.
            t.setOrigin(std::round(b.left + b.width), std::round(b.top + b.height / 2.f));
            t.setPosition(std::round(x1 - 6.f), std::round(ys[i]));
            window.draw(t);
        }
    }
}

