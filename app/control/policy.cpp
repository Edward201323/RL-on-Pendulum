#include "control/policy.hpp"

#include <cmath>
#include <fstream>

bool Policy::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    if (!(in >> this->inputs >> this->hidden >> this->outputs)) return false;

    // Read exactly n floats into v; fail if the file runs short.
    auto readN = [&in](std::vector<float>& v, int n) {
        v.resize(n);
        for (int i = 0; i < n; ++i) {
            if (!(in >> v[i])) return false;
        }
        return true;
    };
    if (!readN(this->obsScale, this->inputs)) return false;
    if (!readN(this->w1, this->hidden * this->inputs)) return false;
    if (!readN(this->b1, this->hidden)) return false;
    if (!readN(this->w2, this->outputs * this->hidden)) return false;
    if (!readN(this->b2, this->outputs)) return false;

    this->loaded = true;
    return true;
}

bool Policy::ready() const { return this->loaded; }

float Policy::act(float x, float velocity, float theta, float angularVelocity) const {
    if (!this->loaded) return 0.f;  // coast

    // Build the observation exactly as trainer/pendulum_env.py _obs() does:
    //   [x, x_dot, sin(theta), cos(theta), theta_dot], each divided by obsScale.
    // sin/cos (not raw theta) so the swing-up angle is continuous across +-pi.
    const float input[5] = {
        x / this->obsScale[0],
        velocity / this->obsScale[1],
        std::sin(theta) / this->obsScale[2],
        std::cos(theta) / this->obsScale[3],
        angularVelocity / this->obsScale[4],
    };

    // Hidden layer: h = tanh(W1 * input + b1).
    std::vector<float> h(this->hidden);
    for (int o = 0; o < this->hidden; ++o) {
        float sum = this->b1[o];
        for (int i = 0; i < this->inputs; ++i) {
            sum += this->w1[o * this->inputs + i] * input[i];
        }
        h[o] = std::tanh(sum);
    }

    // Output layer: a single value = the mean action (the policy's force command,
    // in normalized units). Clamp to the [-1, 1] range training clipped to.
    float out = this->b2[0];
    for (int i = 0; i < this->hidden; ++i) {
        out += this->w2[i] * h[i];
    }
    if (out < -1.f) out = -1.f;
    if (out > 1.f) out = 1.f;
    return out;
}
