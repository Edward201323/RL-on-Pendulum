#include "app.hpp"

#include <cmath>
#include <cstdio>

#include "render/track.hpp"

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kMaxInputForce = 5.f;      // Newtons; MUST match F_MAX in pendulum_env.py
constexpr float kEpisodeSeconds = 12.f;    // restart a swing-up attempt after this long
constexpr float kUprightCos = 0.95f;       // cos(theta) above this counts as "upright"

// Physics runs in SI (meters); rendering is in pixels. This is the one scale
// that converts between them for drawing. Tuned so the scene fills the window.
constexpr float kPixelsPerMeter = 300.f;
constexpr float kCartWidthMeters = 0.2f;
constexpr float kCartHeightMeters = 0.1f;
}

static sf::ContextSettings makeContextSettings() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    return settings;
}

App::App()
    : window(sf::VideoMode(1000, 600), "Pendulum Balnacing", sf::Style::Default,
             makeContextSettings()),
      sim(),
      cart(kCartWidthMeters * kPixelsPerMeter, kCartHeightMeters * kPixelsPerMeter),
      pendulum(sim.config().length * kPixelsPerMeter),
      rng(1234),
      episodeTime(0.f),
      uprightStreak(0.f),
      bestUprightStreak(0.f) {
    window.setFramerateLimit(60);

    // The trained RL policy drives the cart. Run the app from the repo root so
    // this relative path resolves. (See python/export_policy.py.)
    if (this->policy.load("python/policy.txt")) {
        std::printf("Loaded RL policy from python/policy.txt -- it will swing the pole up "
                    "and balance it. Press Space to restart an attempt.\n");
        this->resetEpisode();  // begin a swing-up attempt from the bottom
    } else {
        std::printf("No RL policy found (python/policy.txt). Train + export_policy.py first, "
                    "then run from the repo root.\n");
    }
}

// Main loop: handle input, step the physics by the real frame time, redraw.
void App::run() {
    while (window.isOpen()) {
        processEvents(); // lib
        float dt = clock.restart().asSeconds();
        update(dt);
        render();
    }
}

// Drain the window's event queue; close the window when asked to quit.
void App::processEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window.close();

        // Space starts a fresh swing-up attempt from the bottom.
        if (event.type == sf::Event::KeyPressed &&
            event.key.code == sf::Keyboard::Space) {
            this->resetEpisode();
        }
    }
}

// Drop the cart back to center with the pole hanging at the bottom (theta = pi)
// plus a little noise, so the policy has a fresh swing-up attempt (it was trained
// starting at the bottom).
void App::resetEpisode() {
    this->episodeTime = 0.f;
    this->uprightStreak = 0.f;
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    this->sim.setState(0.f, 0.f, kPi + dist(this->rng), 0.f);
}

void App::setControlForce(float force) { this->sim.setControlForce(force); }
float App::getControlForce() const { return this->sim.getControlForce(); }

float App::getCartX() const { return this->sim.getX(); }
float App::getCartVelocity() const { return this->sim.getVelocity(); }
float App::getPendulumAngle() const { return this->sim.getAngle(); }
float App::getPendulumAngularVelocity() const { return this->sim.getAngularVelocity(); }

// Advance one frame: sync the track limits, let the policy act, step the
// physics, then position the renderers from the new physics state.
void App::update(float dt) {
    // The track length is a fixed physical property (config.trackLimit, in
    // meters) -- not derived from the window -- so the simulated walls are real.
    TrackLayout layout = computeTrackLayout(window);

    // The policy outputs a normalized force in [-1, 1] each frame; scale it to
    // the cart's force command (it now chooses direction AND magnitude).
    const float action = this->policy.act(this->sim.getX(), this->sim.getVelocity(),
                                          this->sim.getAngle(), this->sim.getAngularVelocity());
    this->sim.setControlForce(action * kMaxInputForce);
    this->sim.advance(dt);

    // Track how long the pole stays upright (the swing-up goal), and run a fixed
    // episode length before restarting from the bottom -- mirrors training.
    this->episodeTime += dt;
    if (std::cos(this->sim.getAngle()) > kUprightCos) {
        this->uprightStreak += dt;
        if (this->uprightStreak > this->bestUprightStreak) {
            this->bestUprightStreak = this->uprightStreak;
        }
    } else {
        this->uprightStreak = 0.f;
    }
    if (this->episodeTime > kEpisodeSeconds) {
        this->resetEpisode();
    }

    // Map the centered physics x (meters) onto the screen and position renderers.
    const float screenX = layout.center.x + this->sim.getX() * kPixelsPerMeter;
    this->cart.setPosition(screenX, layout.center.y);
    this->pendulum.setPivot(this->cart.getPivot());
    this->pendulum.setAngle(this->sim.getAngle());
}

// Draw the current frame: background, track + position axis, cart, pendulum,
// then the live data overlay on top.
void App::render() {
    window.clear(sf::Color(100, 100, 100));

    // Draw the track to match the physical limits: cart-center range (+-trackLimit
    // meters) scaled to pixels, widened by half a cart so the body fits the rail.
    TrackLayout layout = computeTrackLayout(window);
    layout.width = 2.f * this->sim.config().trackLimit * kPixelsPerMeter + this->cart.getWidth();
    drawTrack(window, layout);
    this->hud.drawAxis(window, layout, this->sim.config().trackLimit, kPixelsPerMeter);

    cart.draw(window);
    pendulum.draw(window);

    this->hud.drawReadout(window, this->sim.getX(), this->sim.getVelocity(),
                          this->sim.getAngle(), this->sim.getAngularVelocity(),
                          this->sim.getControlForce(),
                          std::cos(this->sim.getAngle()) > kUprightCos,
                          this->uprightStreak, this->bestUprightStreak);
    window.display();
}
