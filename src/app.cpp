#include "app.hpp"

#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <string>
#include <system_error>
#include <vector>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "render/track.hpp"

extern char** environ;  // for posix_spawn (launching the Python trainer)

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kMaxInputForce = 12.f;     // Newtons; MUST match F_MAX in pendulum_env.py
constexpr float kUprightCos = 0.95f;       // cos(theta) above this counts as "upright"

// Physics runs in SI (meters); rendering is in pixels. This is the one scale
// that converts between them for drawing.
constexpr float kPixelsPerMeter = 300.f;
constexpr float kCartWidthMeters = 0.14f;      // small relative to the track
constexpr float kCartHeightMeters = 0.07f;
constexpr float kRodThicknessMeters = 0.02f;   // pole rod (visual)
constexpr float kBobRadiusMeters = 0.05f;      // pole bob (visual)

// Control-force time graph: keep the last ~4 s at 60 FPS.
constexpr std::size_t kGraphSamples = 240;

// Parallel training episodes per update: default and max (max MUST match
// MAX_AGENTS in reinforce.py). The arrow keys vary it between 1 and kMaxAgents.
constexpr int kDefaultAgents = 8;
constexpr int kMaxAgents = 16;
}

static sf::ContextSettings makeContextSettings() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    return settings;
}

// Find the repo root (the directory containing python/reinforce.py), whether the
// app is run from the repo root or from elsewhere. Searches the working directory
// and the executable's directory and its ancestors.
static std::string findProjectRoot() {
    namespace fs = std::filesystem;
    std::vector<fs::path> roots;
    std::error_code ec;
    roots.push_back(fs::current_path(ec));

#ifdef __APPLE__
    char buf[4096];
    std::uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        fs::path exe = fs::weakly_canonical(fs::path(buf), ec);
        for (fs::path d = exe.parent_path(); !d.empty() && d != d.root_path();
             d = d.parent_path()) {
            roots.push_back(d);
        }
    }
#endif

    for (const fs::path& root : roots) {
        if (fs::exists(root / "python" / "reinforce.py", ec)) return root.string();
    }
    return ".";  // fall back to the current directory
}

App::App(int argc, char** argv)
    : window(sf::VideoMode(1200, 740), "Pendulum Balnacing",
             sf::Style::Titlebar | sf::Style::Close, makeContextSettings()),
      sim(),
      cart(kCartWidthMeters * kPixelsPerMeter, kCartHeightMeters * kPixelsPerMeter),
      pendulum(sim.config().length * kPixelsPerMeter,
               kRodThicknessMeters * kPixelsPerMeter,
               kBobRadiusMeters * kPixelsPerMeter),
      rng(1234),
      episodeTime(0.f),
      uprightStreak(0.f),
      bestUprightStreak(0.f),
      shownAttempts(0),
      trainingMode(true),
      trainingPid(0),
      trainingPaused(false),
      agentCount(kDefaultAgents),
      haveScoresMtime(false) {
    window.setFramerateLimit(60);

    // "--watch" skips training and just plays whatever policy.txt already exists.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--watch") this->trainingMode = false;
    }

    this->projectRoot = findProjectRoot();
    this->policyPath = this->projectRoot + "/python/policy.txt";

    this->resetEpisode();  // pole at the bottom
    if (this->trainingMode) {
        // Start "untrained" (Attempt 0, no policy -> coasts). Press Space to sample
        // the current policy as training progresses.
        this->launchTraining();
    } else {
        this->snapshotPolicy();  // watch mode: show the existing policy right away
        std::printf("Watch mode: playing the existing policy (no training).\n");
    }
}

App::~App() {
    // Don't leave the trainer running after the window closes. If it's paused
    // (SIGSTOP'd), continue it first so it can receive the terminate.
    if (this->trainingPid > 0) {
        if (this->trainingPaused) kill(this->trainingPid, SIGCONT);
        kill(this->trainingPid, SIGTERM);
    }
}

// Launch `python3.12 python/reinforce.py <updates>` from the repo root as a child
// process. It re-exports policy.txt periodically, which this app hot-reloads.
void App::launchTraining() {
    // Clear any stale status from a previous run so the box doesn't briefly show
    // an old "Training done" before the new run writes its first update.
    std::error_code ec;
    std::filesystem::remove(this->projectRoot + "/python/train_status.txt", ec);
    std::filesystem::remove(this->projectRoot + "/python/train_history.txt", ec);
    this->scoreXs.clear();
    this->scoreYs.clear();
    this->haveScoresMtime = false;
    this->writeAgentCount();  // start with the default agent count

    // No update count -> reinforce.py trains indefinitely until we kill it (S key).
    const std::string cmd = "cd '" + this->projectRoot +
                            "' && exec python3.12 python/reinforce.py";
    const char* argv[] = {"/bin/sh", "-c", cmd.c_str(), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr,
                    const_cast<char* const*>(argv), environ) == 0) {
        this->trainingPid = pid;
        std::printf("Training started -- runs until you press S. Watch it learn live.\n");
    } else {
        std::printf("Could not start the training process.\n");
    }
}

// Grab the latest exported policy and remember how many training attempts it
// represents. Called on Space (and once at startup in watch mode) -- so the
// on-screen policy is frozen between presses, labelled by its attempt count.
void App::snapshotPolicy() {
    Policy fresh;
    if (fresh.load(this->policyPath)) {
        this->policy = fresh;
    }
    std::ifstream in(this->projectRoot + "/python/train_status.txt");
    int a = 0, done = 0;
    float r = 0.f;
    this->shownAttempts = (in >> a >> r >> done) ? a : 0;
}

// Top-left box: consolidated status. Header (state + agent count), how many
// training episodes have run ("trained"), which policy vintage is on screen
// ("shown #N"), and the live demo-run time ("watch").
std::string App::displayText() const {
    char buf[192];
    if (!this->trainingMode) {  // --watch: no trainer
        std::snprintf(buf, sizeof(buf), "Watching\nPolicy: #%d\nElapsed: %5.2fs",
                      this->shownAttempts, this->episodeTime);
        return std::string(buf);
    }

    int trained = 0, done = 0;
    float ret = 0.f;
    { std::ifstream in(this->projectRoot + "/python/train_status.txt");
      in >> trained >> ret >> done; }

    const char* state = this->trainingPaused ? "Paused" : "Training";
    std::snprintf(buf, sizeof(buf),
                  "%s  %d agents\nEpisodes: %d\nPolicy: #%d\nElapsed: %5.2fs",
                  state, this->agentCount, trained, this->shownAttempts, this->episodeTime);
    return std::string(buf);
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

        if (event.type != sf::Event::KeyPressed) continue;

        // Space samples the current policy: load the latest weights, label them
        // with the training-attempt count, and restart from the bottom.
        if (event.key.code == sf::Keyboard::Space) {
            this->snapshotPolicy();
            this->resetEpisode();
        } else if (this->trainingMode && event.key.code == sf::Keyboard::Up) {
            if (this->agentCount < kMaxAgents) ++this->agentCount;  // more agents
            this->writeAgentCount();
        } else if (this->trainingMode && event.key.code == sf::Keyboard::Down) {
            if (this->agentCount > 1) --this->agentCount;  // fewer agents (lighter training)
            this->writeAgentCount();
        } else if (event.key.code == sf::Keyboard::P) {
            // Pause / resume training (freeze the child process; the window keeps
            // running so you can still replay the last policy with Space).
            if (this->trainingPid > 0) {
                this->trainingPaused = !this->trainingPaused;
                kill(this->trainingPid, this->trainingPaused ? SIGSTOP : SIGCONT);
                std::printf(this->trainingPaused ? "Training paused.\n"
                                                 : "Training resumed.\n");
            }
        }
    }
}

// Write the desired number of training agents to a file the trainer polls each
// update (the arrow keys change it live).
void App::writeAgentCount() const {
    const std::string path = this->projectRoot + "/python/train_agents.txt";
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp);
    if (!out) return;
    out << this->agentCount << "\n";
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);  // atomic; no torn reads
}

// Reload the score-vs-attempts learning curve from the trainer's history file
// when it changes on disk (appended every few updates).
void App::maybeReloadScores() {
    namespace fs = std::filesystem;
    const std::string path = this->projectRoot + "/python/train_history.txt";
    std::error_code ec;
    const fs::file_time_type t = fs::last_write_time(path, ec);
    if (ec) return;  // no history yet
    if (this->haveScoresMtime && t == this->scoresMtime) return;
    this->scoresMtime = t;
    this->haveScoresMtime = true;

    std::ifstream in(path);
    std::vector<float> xs, ys;
    float a = 0.f, s = 0.f;
    while (in >> a >> s) {  // skips a torn final line cleanly
        xs.push_back(a);
        ys.push_back(s);
    }
    this->scoreXs.swap(xs);
    this->scoreYs.swap(ys);
}

// Drop the cart back to center with the pole hanging at the bottom (theta = pi)
// plus a little noise, so the policy has a fresh swing-up attempt (it was trained
// starting at the bottom).
void App::resetEpisode() {
    this->episodeTime = 0.f;
    this->uprightStreak = 0.f;
    this->forceHistory.clear();  // start the force graph fresh each attempt
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
    this->maybeReloadScores();  // pick up new learning-curve points

    // The track length is a fixed physical property (config.trackLimit, in
    // meters) -- not derived from the window -- so the simulated walls are real.
    TrackLayout layout = computeTrackLayout(window);

    // The policy outputs a normalized force in [-1, 1] each frame; scale it to
    // the cart's force command (it now chooses direction AND magnitude).
    const float action = this->policy.act(this->sim.getX(), this->sim.getVelocity(),
                                          this->sim.getAngle(), this->sim.getAngularVelocity());
    this->sim.setControlForce(action * kMaxInputForce);
    this->sim.advance(dt);

    // Record the applied force for the scrolling time graph (keep the last window).
    this->forceHistory.push_back(this->sim.getControlForce());
    if (this->forceHistory.size() > kGraphSamples) {
        this->forceHistory.pop_front();
    }

    // Run continuously; the attempt only restarts when the user presses Space.
    this->episodeTime += dt;

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

    // Framed play area (cart/track) as the main focus.
    this->hud.drawPlayfield(window);

    // Draw the track to match the physical limits: cart-center range (+-trackLimit
    // meters) scaled to pixels, widened by half a cart so the body fits the rail.
    TrackLayout layout = computeTrackLayout(window);
    layout.width = 2.f * this->sim.config().trackLimit * kPixelsPerMeter + this->cart.getWidth();
    drawTrack(window, layout);
    this->hud.drawAxis(window, layout, this->sim.config().trackLimit, kPixelsPerMeter);

    cart.draw(window);
    pendulum.draw(window);

    // Bottom-left: score-vs-attempts learning curve. Bottom-right: control force.
    if (this->trainingMode) {
        this->hud.drawScoreGraph(window, this->scoreXs, this->scoreYs, "Score");
    }
    this->hud.drawGraph(window, this->forceHistory, kGraphSamples, kMaxInputForce,
                        "Control force (N)");
    // Consolidated status box, top-left.
    this->hud.drawTextBox(window, this->displayText(), /*bottom=*/false);
    window.display();
}
