#pragma once

#include <string>
#include <vector>

// Tiny feed-forward policy (inputs -> hidden -> outputs) loaded from the text
// file written by python/export_policy.py. It runs the exact same forward pass
// as the trained PyTorch net, so the SFML app can play the RL policy with no
// Python at runtime. Pure math + file IO; no SFML dependency.
class Policy {
public:
    // Load weights from a policy.txt file. Returns false if the file is missing
    // or malformed (the app then just stays in manual control).
    bool load(const std::string& path);
    bool ready() const;

    // Greedy normalized force in [-1, 1] for a cart-pole state -- the policy's
    // mean action (deterministic for playback). Multiply by the max force to get
    // the cart command. Returns 0 (coast) if no policy is loaded.
    float act(float x, float velocity, float theta, float angularVelocity) const;

private:
    bool loaded = false;
    int inputs = 0;
    int hidden = 0;
    int outputs = 0;
    std::vector<float> obsScale;  // inputs            (input normalization)
    std::vector<float> w1;        // hidden * inputs   ([out][in], row-major)
    std::vector<float> b1;        // hidden
    std::vector<float> w2;        // outputs * hidden
    std::vector<float> b2;        // outputs
};
