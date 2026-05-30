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
    const float F = this->controlForce;
    const float t = this->pendulum.getAngle();
    const float w = this->pendulum.getAngularVelocity();
    const float v = this->cart.getVelocity();
    const float s = std::sin(t);
    const float c = std::cos(t);
    const float M = this->cart.getMass();
    const float m = this->pendulum.getBobMass();
    const float L = this->pendulum.getLength();
    const float damping = this->pendulum.getDamping();
    const float mu = this->cart.getFriction();

    // Viscous cart-track friction: a force -mu*v added to the cart's net horizontal force.
    const float Fnet = F - mu * v;
    float xddot = (Fnet + m * L * w * w * s - m * kGravity * s * c) / (M + m * s * s);

    // Wall contact: if the cart is pinned against a wall and the net force
    // would push it further into the wall, the wall's normal force cancels
    // it — the cart's true acceleration is 0. Propagate that to thddot so the
    // pendulum doesn't feel a pivot acceleration that physically can't occur.
    const int contact = this->cart.boundaryContact();
    if ((contact < 0 && xddot < 0.f) || (contact > 0 && xddot > 0.f)) {
        xddot = 0.f;
        this->cart.setVelocity(0.f);
    }

    const float thddot = (kGravity * s - xddot * c) / L - damping * w;

    // Semi-implicit Euler: update velocities first, then positions from new velocities.
    const float vNew = this->cart.getVelocity() + xddot * h;
    this->cart.setVelocity(vNew);
    this->cart.setX(this->cart.getX() + vNew * h);
    if (this->cart.clampToBounds()) {
        this->cart.setVelocity(0.f);
    }

    const float wNew = w + thddot * h;
    this->pendulum.setAngularVelocity(wNew);
    this->pendulum.setAngle(t + wNew * h);
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
