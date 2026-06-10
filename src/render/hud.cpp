#include "hud.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {
constexpr float kPi = 3.14159265358979323846f;

// Candidate system fonts, in preference order (Menlo is monospace, so the
// readout columns line up). First one that loads wins.
const char* kFontPaths[] = {
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
};
}  // namespace

Hud::Hud() : hasFont(false) {
    for (const char* path : kFontPaths) {
        if (this->font.loadFromFile(path)) {
            this->hasFont = true;
            break;
        }
    }
}

void Hud::drawAxis(sf::RenderWindow& window, const TrackLayout& layout,
                   float trackLimit) const {
    const float axisY = layout.center.y + 45.f;
    const sf::Color axisColor(210, 210, 210);

    // Baseline spanning the cart's reachable range [-trackLimit, +trackLimit].
    sf::Vertex baseline[] = {
        sf::Vertex(sf::Vector2f(layout.center.x - trackLimit, axisY), axisColor),
        sf::Vertex(sf::Vector2f(layout.center.x + trackLimit, axisY), axisColor),
    };
    window.draw(baseline, 2, sf::Lines);

    // Ticks every 50 px; major (taller, labelled) every 100. x = 0 is centered.
    for (int p = -300; p <= 300; p += 50) {
        if (static_cast<float>(std::abs(p)) > trackLimit) continue;
        const float sx = layout.center.x + p;
        const bool major = (p % 100 == 0);
        const float h = major ? 12.f : 6.f;
        sf::Vertex tick[] = {
            sf::Vertex(sf::Vector2f(sx, axisY - h), axisColor),
            sf::Vertex(sf::Vector2f(sx, axisY + h), axisColor),
        };
        window.draw(tick, 2, sf::Lines);

        if (major && this->hasFont) {
            sf::Text label(std::to_string(p), this->font, 12);
            label.setFillColor(axisColor);
            const sf::FloatRect b = label.getLocalBounds();
            label.setOrigin(b.left + b.width / 2.f, 0.f);
            label.setPosition(sx, axisY + 14.f);
            window.draw(label);
        }
    }

    if (this->hasFont) {
        sf::Text caption("cart position x (px)", this->font, 12);
        caption.setFillColor(axisColor);
        caption.setPosition(layout.center.x - trackLimit, axisY + 32.f);
        window.draw(caption);
    }
}

void Hud::drawReadout(sf::RenderWindow& window, float x, float velocity,
                      float angle, float angularVelocity, float force,
                      float attemptTime, float bestTime) const {
    if (!this->hasFont) return;

    // angle: 0 = straight up; show degrees for readability.
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "x         : %+7.1f px\n"
                  "x_dot     : %+7.1f px/s\n"
                  "theta     : %+6.1f deg\n"
                  "theta_dot : %+6.2f rad/s\n"
                  "force     : %+7.0f\n"
                  "balance   : %6.2f s\n"
                  "best      : %6.2f s",
                  x, velocity, angle * 180.f / kPi, angularVelocity, force,
                  attemptTime, bestTime);

    sf::Text text(buf, this->font, 15);
    text.setFillColor(sf::Color::White);
    text.setPosition(16.f, 14.f);

    // Translucent backdrop so the text stays readable over the scene.
    const sf::FloatRect b = text.getLocalBounds();
    sf::RectangleShape bg(sf::Vector2f(b.width + 22.f, b.height + 22.f));
    bg.setPosition(8.f, 8.f);
    bg.setFillColor(sf::Color(0, 0, 0, 130));
    window.draw(bg);
    window.draw(text);
}
