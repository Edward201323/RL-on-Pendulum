#pragma once

#include <sys/types.h>  // pid_t

#include <deque>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <SFML/Graphics.hpp>

#include "control/policy.hpp"
#include "physics/cartpole.hpp"
#include "render/cart.hpp"
#include "render/hud.hpp"
#include "render/pendulum.hpp"

class App {
public:
    App(int argc, char** argv);
    ~App();
    void run();

    // Control input — any horizontal force on the cart, in physics units.
    // Set this from any source (keyboard, scripted test, RL policy) before
    // the next step; the value persists across frames until changed.
    void setControlForce(float force);
    float getControlForce() const;

    // State accessors (state = (x, xdot, theta, thetadot)) for external
    // observers like an RL agent. Delegate to the headless physics core.
    float getCartX() const;
    float getCartVelocity() const;
    float getPendulumAngle() const;
    float getPendulumAngularVelocity() const;

private:
    void processEvents();
    void update(float dt);
    void resetEpisode();  // start a fresh swing-up attempt from the bottom
    void render();

    void launchTraining();      // spawn the Python trainer as a child process
    void snapshotPolicy();      // load the latest policy + capture its attempt count
    void writeAgentCount() const;      // tell the trainer how many agents to use
    void maybeReloadScores();   // reload the score-vs-attempts history when it changes
    std::string displayText() const;   // top-left consolidated status

    sf::RenderWindow window;
    sf::Clock clock;
    CartPole sim;
    Cart cart;
    Pendulum pendulum;
    Hud hud;                  // on-screen x-axis + live data overlay
    Policy policy;            // trained RL policy that drives the cart
    std::mt19937 rng;         // small randomness for the reset angle
    float episodeTime;        // seconds since the current episode started
    float uprightStreak;      // seconds the pole has been continuously upright
    float bestUprightStreak;  // longest upright streak this session
    int shownAttempts;        // training-attempt count behind the on-screen policy
    std::deque<float> forceHistory;  // recent control forces for the time graph

    std::string projectRoot;  // repo root (holds python/), found at startup
    std::string policyPath;   // projectRoot + "/python/policy.txt"
    bool trainingMode;        // true = launch training and show its progress
    pid_t trainingPid;        // child trainer pid (0 = none), killed on exit
    bool trainingPaused;      // true while training is paused (SIGSTOP) via P
    int agentCount;           // parallel training episodes per update (arrow keys)

    std::vector<float> scoreXs;  // learning curve: attempts ...
    std::vector<float> scoreYs;  // ... and the matching scores
    std::filesystem::file_time_type scoresMtime;  // last-seen history file write time
    bool haveScoresMtime;
};
