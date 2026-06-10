#include "physics/cartpole.hpp"

#include <cmath>

// Build the sim from a config and drop it into its start-of-episode pose.
CartPole::CartPole(const CartPoleConfig& config) : cfg(config) {
    this->reset();
}

// Put the cart back in the center, at rest, with the pole hanging at the bottom.
void CartPole::reset() {
    this->x = 0.f;
    this->velocity = 0.f;
    this->theta = this->cfg.initialAngle;
    this->angularVelocity = 0.f;
    this->controlForce = 0.f;
}

// Force the whole state to chosen values (used for randomized resets / demo replay).
void CartPole::setState(float x, float velocity, float theta, float angularVelocity) {
    this->x = x;
    this->velocity = velocity;
    this->theta = theta;
    this->angularVelocity = angularVelocity;
}

void CartPole::setControlForce(float force) { this->controlForce = force; }
float CartPole::getControlForce() const { return this->controlForce; }

float CartPole::getX() const { return this->x; }
float CartPole::getVelocity() const { return this->velocity; }
float CartPole::getAngle() const { return this->theta; }
float CartPole::getAngularVelocity() const { return this->angularVelocity; }

CartPoleConfig& CartPole::config() { return this->cfg; }

// Report which wall (if any) the cart is currently touching: -1 left, +1 right, 0 free.
int CartPole::boundaryContact() const {
    const float eps = 1e-4f;
    if (this->x <= -this->cfg.trackLimit + eps) return -1;
    if (this->x >= this->cfg.trackLimit - eps) return +1;
    return 0;
}

// Keep the cart inside the track; returns true if it had to be pushed back in.
bool CartPole::clampToBounds() {
    if (this->x < -this->cfg.trackLimit) {
        this->x = -this->cfg.trackLimit;
        return true;
    }
    if (this->x > this->cfg.trackLimit) {
        this->x = this->cfg.trackLimit;
        return true;
    }
    return false;
}

// Advance the state by one small time slice h. Computes the cart and pole
// accelerations from the current state + control force, then integrates them.
void CartPole::step(float h) {
    // Coupled cart-pole: point-mass bob at length L, massless rod.
    //   xddot  = (F + m L w^2 sin(t) - m g sin(t) cos(t)) / (M + m sin^2(t))
    //   thddot = (g sin(t) - xddot cos(t)) / L - c * w
    const float F = this->controlForce;
    const float t = this->theta;
    const float w = this->angularVelocity;
    const float v = this->velocity;
    const float s = std::sin(t);
    const float c = std::cos(t);
    const float M = this->cfg.cartMass;
    const float m = this->cfg.bobMass;
    const float L = this->cfg.length;
    const float g = this->cfg.gravity;
    const float mu = this->cfg.cartFriction;
    const float damping = this->cfg.poleDamping;

    // Viscous cart-track friction: a force -mu*v added to the net horizontal force.
    const float Fnet = F - mu * v;
    float xddot = (Fnet + m * L * w * w * s - m * g * s * c) / (M + m * s * s);

    // Wall contact: if the cart is pinned against a wall and the net force would
    // push it further into the wall, the wall's normal force cancels it — the
    // cart's true acceleration is 0. Propagate that to thddot so the pendulum
    // doesn't feel a pivot acceleration that physically can't occur.
    const int contact = this->boundaryContact();
    if ((contact < 0 && xddot < 0.f) || (contact > 0 && xddot > 0.f)) {
        xddot = 0.f;
        this->velocity = 0.f;
    }

    const float thddot = (g * s - xddot * c) / L - damping * w;

    // Semi-implicit Euler: update velocities first, then positions from them.
    const float vNew = this->velocity + xddot * h;
    this->velocity = vNew;
    this->x += vNew * h;
    if (this->clampToBounds()) {
        this->velocity = 0.f;
    }

    const float wNew = w + thddot * h;
    this->angularVelocity = wNew;
    this->theta = t + wNew * h;
}

// Move the simulation forward by dt seconds. dt is capped (so a long frame
// can't explode the integrator) and split into a few substeps for stability.
void CartPole::advance(float dt) {
    if (dt > this->cfg.maxDt) dt = this->cfg.maxDt;
    const float h = dt / this->cfg.substeps;
    for (int i = 0; i < this->cfg.substeps; ++i) {
        this->step(h);
    }
}

// Has the pole tipped past the fail angle? Note: with the default bottom start
// the pole begins past this, so a swing-up task drives termination from the
// Python wrapper (time limit / wall) instead of relying on this.
bool CartPole::isDone() const {
    return std::fabs(this->theta) >= this->cfg.failAngle;
}
