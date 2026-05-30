#include "app.hpp"

#include <cmath>

#include "render/track.hpp"

namespace {
constexpr float kGravity = 1500.f;        // px/s^2
constexpr float kMaxInputForce = 4000.f;  // magnitude the keyboard maps to
constexpr float kInitialAngle = 0.15f;    // small tilt so the pendulum falls
constexpr int kPhysicsSubsteps = 4;       // substeps per frame for stability
}

static sf::ContextSettings makeContextSettings() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    return settings;
}

App::App()
    : window(sf::VideoMode(1000, 600), "Pendulum Balnacing", sf::Style::Default,
             makeContextSettings()),
      controlForce(0.f) {
    window.setFramerateLimit(60);
    pendulum.setAngle(kInitialAngle);
}

void App::run() {
    while (window.isOpen()) {
        processEvents(); // lib
        float dt = clock.restart().asSeconds();
        update(dt);
        render(); 
    }
}

void App::processEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window.close();
    }
}

void App::setControlForce(float force) { controlForce = force; }
float App::getControlForce() const { return controlForce; }

float App::getCartX() const { return cart.getX(); }
float App::getCartVelocity() const { return cart.getVelocity(); }
float App::getPendulumAngle() const { return pendulum.getAngle(); }
float App::getPendulumAngularVelocity() const { return pendulum.getAngularVelocity(); }

void App::pollKeyboardControl() {
    float dir = 0.f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)){
        dir -= 1.f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)){
        dir += 1.f;
    }
    setControlForce(dir * kMaxInputForce);
}

void App::physicsStep(float h) {
    // Coupled cart-pole: point-mass bob at length L, massless rod, frictionless cart.
    //   xddot   = (F + m L w^2 sin(t) - m g sin(t) cos(t)) / (M + m sin^2(t))
    //   thddot  = (g sin(t) - xddot cos(t)) / L  - c * w
    // F is the externally-supplied control input — set via setControlForce
    // (keyboard, scripted test, RL policy). The cart's velocity then emerges
    // from integration; it is not commanded directly.
    const float F = controlForce;
    const float t = pendulum.getAngle();
    const float w = pendulum.getAngularVelocity();
    const float s = std::sin(t);
    const float c = std::cos(t);
    const float M = cart.getMass();
    const float m = pendulum.getBobMass();
    const float L = pendulum.getLength();
    const float damping = pendulum.getDamping();

    const float xddot = (F + m * L * w * w * s - m * kGravity * s * c) / (M + m * s * s);
    const float thddot = (kGravity * s - xddot * c) / L - damping * w;

    // Semi-implicit Euler: update velocities first, then positions from new velocities.
    const float vNew = cart.getVelocity() + xddot * h;
    cart.setVelocity(vNew);
    cart.setX(cart.getX() + vNew * h);
    if (cart.clampToBounds()) {
        cart.setVelocity(0.f);
    }

    const float wNew = w + thddot * h;
    pendulum.setAngularVelocity(wNew);
    pendulum.setAngle(t + wNew * h);
}

void App::update(float dt) {
    TrackLayout layout = computeTrackLayout(window);
    cart.setBounds(layout.center.x - layout.width / 2.f, layout.center.x + layout.width / 2.f, layout.center.y);

    pollKeyboardControl();

    // Cap dt so a long pause can't blow up the integrator, then substep.
    if (dt > 0.1f) dt = 0.1f;
    const float h = dt / kPhysicsSubsteps;
    for (int i = 0; i < kPhysicsSubsteps; ++i) {
        physicsStep(h);
    }

    pendulum.setPivot(cart.getPivot());
}

void App::render() {
    window.clear(sf::Color(100, 100, 100));
    drawTrack(window);

    cart.draw(window);
    pendulum.draw(window);
    window.display();
}
