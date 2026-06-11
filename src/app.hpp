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
    void resetActors();        // restart the actor-overlay replay from frame 0
    void updateActors(float dt);  // advance the actor-overlay replay one frame
    void maybeReloadActors();  // reload the exported rollout batch when it changes
    void render();

    void launchTraining();      // spawn the Python trainer as a child process
    void snapshotPolicy();      // load the latest policy + capture its attempt count
    void writeAgentCount() const;      // tell the trainer how many episodes to run in parallel
    void maybeReloadScores();   // reload the score-vs-attempts history when it changes
    std::string displayText() const;   // single status box (--watch mode)
    std::string controlsText() const;  // top-right corner key hints
    std::string orangeText() const;    // orange box: this displayed run (policy/time/score)
    std::string greenText() const;     // green box: overall training (state/episodes/avg)
    float displayedScore() const;      // two-component score of the on-screen run so far
    float avgRecentScore() const;      // mean of the last 10 logged training scores
    float maxScore() const;            // best logged training score so far

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
    // Live two-component score of the on-screen run, accumulated each frame and
    // reset on Space (mirrors the trainer's score; see displayedScore()).
    float liveUprightSum;     // sum of max(0, cos theta) over the displayed run
    float liveCenteredSum;    // same, weighted by centeredness (1 - |x|/limit)
    int liveScoreFrames;      // frames accumulated since the last reset
    int shownAttempts;        // training-attempt count behind the on-screen policy
    std::deque<float> forceHistory;  // recent control forces for the time graph

    std::string projectRoot;  // repo root (holds python/), found at startup
    std::string policyPath;   // projectRoot + "/python/policy.txt"
    bool trainingMode;        // true = launch training and show its progress
    pid_t trainingPid;        // child trainer pid (0 = none), killed on exit
    bool trainingPaused;      // true while training is paused (SIGSTOP) via Space
    int agentCount;           // episodes trained in parallel per update (Up/Down)
    // Left/Right toggle between these views.
    enum View { kViewSingle = 0, kViewActors, kViewCount };
    int view;                 // current View
    // All-actors overlay: a real rollout batch exported by the trainer, replayed.
    // x/theta are flat [steps * count], indexed frame*count + actor.
    std::vector<float> actorX, actorTheta;
    int actorCount;           // actors in the current rollout file
    int actorSteps;           // frames per actor (episode length)
    int actorFrame;           // current replay frame
    std::filesystem::file_time_type actorsMtime;
    bool haveActorsMtime;

    std::vector<float> scoreXs;  // learning curve: attempts ...
    std::vector<float> scoreYs;  // ... and the matching scores
    std::filesystem::file_time_type scoresMtime;  // last-seen history file write time
    bool haveScoresMtime;
};
