#pragma once

// Headless cart-pole physics. No SFML, no rendering — cheap to instantiate
// thousands of times for neuroevolution. This is the single source of truth
// for the dynamics; both the SFML visualizer and the Python trainer drive it,
// so a balanced genome behaves identically in training and in the demo.
//
// Units are pixels and seconds (matching the SFML visualizer).
// State = (x, xdot, theta, thetadot):
//   x      cart-center offset from track center (px), 0 = centered
//   theta  pole angle (rad), 0 = straight up, +-pi = hanging down, positive = clockwise tilt
struct CartPoleConfig {
    float gravity = 1500.f;            // px/s^2
    float cartMass = 1.0f;             // M
    float bobMass = 0.2f;              // m, point mass at the rod tip
    float length = 130.f;              // L, rod length (px)
    float poleDamping = 0.4f;          // viscous angular damping (per second)
    float cartFriction = 1.0f;         // viscous cart-track coefficient (force per velocity)
    float trackLimit = 370.f;          // max |x| the cart center may reach (px)
    float initialAngle = 3.14159265f;  // pose set by reset(): pi rad = hanging straight down
    float failAngle = 1.5708f;         // |theta| past which isDone() reports a fall (rad, ~90 deg)
    float maxDt = 0.1f;                // dt cap so a long frame can't blow up the integrator
    int   substeps = 4;                // integrator substeps per advance()
};

class CartPole {
public:
    explicit CartPole(const CartPoleConfig& config = CartPoleConfig());

    // Restore the start-of-episode state (cart centered, pole at initialAngle).
    void reset();

    // Overwrite the full state directly (randomized resets, demo replay).
    void setState(float x, float velocity, float theta, float angularVelocity);

    // Control input: a horizontal force on the cart (physics units), persisted
    // until changed. Set it before each advance().
    void setControlForce(float force);
    float getControlForce() const;

    // Integrate forward by dt seconds (dt is capped at maxDt, then substepped).
    void advance(float dt);

    // State accessors: state = (x, xdot, theta, thetadot).
    float getX() const;
    float getVelocity() const;
    float getAngle() const;
    float getAngularVelocity() const;

    // True once the pole has fallen past failAngle. Out-of-track is handled by
    // clamping (matching the app), not termination — the wrapper decides whether
    // hitting a wall ends the episode.
    bool isDone() const;

    // Mutable config for curriculum learning (raise gravity, lower friction...).
    CartPoleConfig& config();

    // Which wall (if any) the cart is currently touching: -1 left, +1 right,
    // 0 free. Public so the RL wrapper can apply a wall-contact penalty.
    int boundaryContact() const;

private:
    void step(float h);           // one semi-implicit Euler substep
    bool clampToBounds();         // pin x inside [-trackLimit, trackLimit]

    CartPoleConfig cfg;
    float x;
    float velocity;
    float theta;
    float angularVelocity;
    float controlForce;
};
