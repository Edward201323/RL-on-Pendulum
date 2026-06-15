#include "app.hpp"

#include <algorithm>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <string>
#include <sys/wait.h>  // waitpid (reap the trainer before wiping its data)
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
// Mouse pushes in the demo: hover the cursor over the bob to shove the pole. Each
// contact fires a single impulse in the contact direction (cursor -> bob center)
// projected onto the bob's swing, then goes on a short cooldown so holding the
// cursor on the ball doesn't spam kicks. Up/Down change the strength. (Training
// applies its own random kicks; see pendulum_env.py.)
constexpr float kKickStrengthDefault = 3.0f;
constexpr float kKickStrengthStep = 1.0f;
constexpr float kKickStrengthMax = 12.0f;
constexpr float kKickCooldown = 1.0f;  // seconds between mouse-contact kicks
// Episodes trained in parallel per update: default + max. Max MUST match MAX_AGENTS
// in reinforce.py. Up/Down change it live; the trainer polls train_agents.txt.
constexpr int kDefaultAgents = 8;
constexpr int kMaxAgents = 100;
// Two-component live score, mirroring SCORE_W_* in reinforce.py: balance (time
// upright) is primary, centering (sitting at x=0 while upright) the secondary
// tie-breaker. Must sum to 1 so a perfect on-screen run reads 100.
constexpr float kScoreWBalance = 0.85f;
constexpr float kScoreWCenter = 0.15f;

// Physics runs in SI (meters); rendering is in pixels. This is the one scale
// that converts between them for drawing.
constexpr float kPixelsPerMeter = 300.f;
constexpr float kCartWidthMeters = 0.14f;      // small relative to the track
constexpr float kCartHeightMeters = 0.07f;
constexpr float kRodThicknessMeters = 0.02f;   // pole rod (visual)
constexpr float kBobRadiusMeters = 0.05f;      // pole bob (visual)

// Control-force time graph: keep the last ~4 s at 60 FPS.
constexpr std::size_t kGraphSamples = 240;
}

static sf::ContextSettings makeContextSettings() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    return settings;
}

// Find the repo root (the directory containing trainer/reinforce.py), whether the
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
        if (fs::exists(root / "trainer" / "reinforce.py", ec)) return root.string();
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
      liveUprightSum(0.f),
      liveCenteredSum(0.f),
      liveScoreFrames(0),
      shownAttempts(0),
      trainingMode(true),
      trainingPid(0),
      trainingPaused(false),
      agentCount(kDefaultAgents),
      kickStrength(kKickStrengthDefault),
      kickCooldown(0.f),
      view(kViewSingle),
      actorCount(0),
      actorSteps(0),
      actorFrame(0),
      haveActorsMtime(false),
      haveScoresMtime(false) {
    window.setFramerateLimit(60);

    // "--watch" skips training and just plays whatever policy.txt already exists.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--watch") this->trainingMode = false;
    }

    this->projectRoot = findProjectRoot();
    this->policyPath = this->projectRoot + "/trainer/policy.txt";

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

// Launch `python3.12 trainer/reinforce.py <updates>` from the repo root as a child
// process. It re-exports policy.txt periodically, which this app hot-reloads.
void App::launchTraining() {
    // NOTE: we deliberately DON'T delete the trainer's data files here -- a normal
    // quit leaves them in place and the trainer resumes from policy.pt + the history
    // on the next launch. Only Command+Shift+R (wipeTrainingData) discards them.
    // Drop our in-memory copies so the curve/overlay reload cleanly from disk.
    this->scoreXs.clear();
    this->scoreYs.clear();
    this->haveScoresMtime = false;
    this->actorCount = 0;
    this->actorSteps = 0;
    this->haveActorsMtime = false;
    this->writeAgentCount();  // publish the starting parallel-episode count

    // No update count -> reinforce.py trains indefinitely until we kill it (S key).
    const std::string cmd = "cd '" + this->projectRoot +
                            "' && exec python3.12 trainer/reinforce.py";
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

// Delete every file the trainer persists -- the saved checkpoint, the exported
// policy, the learning curve, the status, and the overlay rollout. After this the
// trainer has nothing to resume from, so it starts fresh. Only Command+Shift+R
// reaches here; a normal quit keeps all of it.
void App::wipeTrainingData() {
    std::error_code ec;
    const std::string dir = this->projectRoot + "/trainer/";
    for (const char* name : {"policy.pt", "policy.txt", "train_history.txt",
                             "train_status.txt", "actors.txt"}) {
        std::filesystem::remove(dir + name, ec);
    }
}

// Command+Shift+R: throw away the saved training and start over. Hard-kill the
// running trainer first (SIGKILL, not SIGTERM, so its save-on-exit can't recreate
// the files we're about to delete) and reap it before wiping.
void App::restartTrainingFresh() {
    if (this->trainingPid > 0) {
        if (this->trainingPaused) kill(this->trainingPid, SIGCONT);
        kill(this->trainingPid, SIGKILL);
        waitpid(this->trainingPid, nullptr, 0);  // ensure it's gone before wiping
        this->trainingPid = 0;
        this->trainingPaused = false;
    }
    this->wipeTrainingData();
    this->shownAttempts = 0;
    this->policy = Policy();      // drop the trained policy from the on-screen run
    this->resetEpisode();
    this->launchTraining();       // fresh run: no checkpoint left to resume from
    std::printf("Training data wiped -- starting over from scratch.\n");
}

// Grab the latest exported policy and remember how many training attempts it
// represents. Called on Space (and once at startup in watch mode) -- so the
// on-screen policy is frozen between presses, labelled by its attempt count.
void App::snapshotPolicy() {
    Policy fresh;
    if (fresh.load(this->policyPath)) {
        this->policy = fresh;
    }
    std::ifstream in(this->projectRoot + "/trainer/train_status.txt");
    int a = 0, done = 0;
    float r = 0.f;
    this->shownAttempts = (in >> a >> r >> done) ? a : 0;
}

// Two-component score (0..100) of the run currently on screen: balance (fraction
// of frames upright) scaled by a centering tie-breaker. Mirrors reinforce.py so
// the on-screen number is comparable to the trainer's learning curve.
float App::displayedScore() const {
    if (this->liveScoreFrames <= 0) return 0.f;
    const float balance = this->liveUprightSum / this->liveScoreFrames;
    const float centering = this->liveCenteredSum / std::max(this->liveUprightSum, 1e-6f);
    return 100.f * balance * (kScoreWBalance + kScoreWCenter * centering);
}

// Mean of the last (up to) 10 logged training scores -- a steadier "where is it
// now" number than the all-time best, which a single lucky run can inflate.
float App::avgRecentScore() const {
    if (this->scoreYs.empty()) return 0.f;
    const std::size_t n = std::min<std::size_t>(10, this->scoreYs.size());
    float sum = 0.f;
    for (std::size_t i = this->scoreYs.size() - n; i < this->scoreYs.size(); ++i)
        sum += this->scoreYs[i];
    return sum / static_cast<float>(n);
}

// Best logged training score so far (the all-time peak of the learning curve).
float App::maxScore() const {
    if (this->scoreYs.empty()) return 0.f;
    return *std::max_element(this->scoreYs.begin(), this->scoreYs.end());
}

// Top-right corner key hints (Up/Down only listed on the training slide).
std::string App::controlsText() const {
    if (this->view == kViewActors) {
        return "Left/Right: Switch view\nSpace: Pause Training"
               "\nUp/Down: Increase/Decrease actors"
               "\nCmd+Shift+R: Wipe & restart training";
    }
    return "Left/Right: Switch view\nSpace: Reset"
           "\nHover ball: Push pole\nUp/Down: Push strength"
           "\nCmd+Shift+R: Wipe & restart training";
}

// --watch single box (no trainer running): which policy is shown and its run time.
std::string App::displayText() const {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "Watching\nPolicy: #%d\nElapsed: %5.2fs\nScore: %5.1f",
                  this->shownAttempts, this->episodeTime, this->displayedScore());
    return std::string(buf);
}

// Orange box: everything about the run currently on screen.
std::string App::orangeText() const {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "Policy: #%d\nElapsed: %5.2fs\nScore: %5.1f\nKick: %4.1f",
                  this->shownAttempts, this->episodeTime, this->displayedScore(),
                  this->kickStrength);
    return std::string(buf);
}

// Green box: overall training progress (state + agent count, episodes, best score).
std::string App::greenText() const {
    int trained = 0, done = 0;
    float ret = 0.f;
    { std::ifstream in(this->projectRoot + "/trainer/train_status.txt");
      in >> trained >> ret >> done; }

    // While paused, drop the actor count (it isn't training); while training, show it.
    char head[32];
    if (this->trainingPaused) std::snprintf(head, sizeof(head), "Paused");
    else std::snprintf(head, sizeof(head), "Training  %d actors", this->agentCount);

    char buf[200];
    if (this->view == kViewActors) {
        // Replay runs one frame per displayed frame (~60 FPS), so seconds = frame/60.
        const float secs = this->actorFrame / 60.f;
        const float total = this->actorSteps / 60.f;
        std::snprintf(buf, sizeof(buf),
                      "%s\nEpisodes: %d\nMax score: %5.1f\nBatch: %4.1f / %4.1fs",
                      head, trained, this->maxScore(), secs, total);
    } else {
        std::snprintf(buf, sizeof(buf), "%s\nEpisodes: %d\nMax score: %5.1f",
                      head, trained, this->maxScore());
    }
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

        // Command+Shift+R (system = Cmd on macOS): the ONLY way to discard saved
        // training. Wipes the trainer's files and restarts from scratch. A normal
        // quit keeps everything and the next launch resumes it.
        if (this->trainingMode && event.key.code == sf::Keyboard::R &&
            event.key.shift && event.key.system) {
            this->restartTrainingFresh();
            continue;
        }

        // Space is context-sensitive: on the single-run view it restarts the run
        // with the latest policy; on the training view it pauses/resumes the trainer.
        if (event.key.code == sf::Keyboard::Space) {
            if (this->view == kViewSingle) {
                this->snapshotPolicy();   // fresh swing-up with the newest policy
                this->resetEpisode();
            } else if (this->trainingPid > 0) {
                this->trainingPaused = !this->trainingPaused;
                kill(this->trainingPid, this->trainingPaused ? SIGSTOP : SIGCONT);
                std::printf(this->trainingPaused ? "Training paused.\n"
                                                 : "Training resumed.\n");
            }
        } else if (this->trainingMode && this->view == kViewActors &&
                   event.key.code == sf::Keyboard::Up) {
            if (this->agentCount < kMaxAgents) ++this->agentCount;  // more parallel episodes
            this->writeAgentCount();
        } else if (this->trainingMode && this->view == kViewActors &&
                   event.key.code == sf::Keyboard::Down) {
            if (this->agentCount > 1) --this->agentCount;           // fewer parallel episodes
            this->writeAgentCount();
        } else if (this->view == kViewSingle && event.key.code == sf::Keyboard::Up) {
            this->kickStrength = std::min(kKickStrengthMax,
                                          this->kickStrength + kKickStrengthStep);
        } else if (this->view == kViewSingle && event.key.code == sf::Keyboard::Down) {
            this->kickStrength = std::max(0.f, this->kickStrength - kKickStrengthStep);
        } else if (this->trainingMode && (event.key.code == sf::Keyboard::Left ||
                                          event.key.code == sf::Keyboard::Right)) {
            // Left/Right cycle through the views, wrapping (Right forward, Left back).
            const int step = (event.key.code == sf::Keyboard::Right) ? 1 : kViewCount - 1;
            this->view = (this->view + step) % kViewCount;
            if (this->view == kViewSingle) {                           // single policy
                this->snapshotPolicy();   // show the most recent policy
                this->resetEpisode();
            } else if (this->view == kViewActors) {                    // all-actors overlay
                this->resetActors();      // load the most recent batch + policy
            }
        }
    }
}

// Write the desired parallel-episode count to a file the trainer polls each
// update (Up/Down change it live).
void App::writeAgentCount() const {
    const std::string path = this->projectRoot + "/trainer/train_agents.txt";
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
    const std::string path = this->projectRoot + "/trainer/train_history.txt";
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
        // Defensive: drop any point whose attempt count isn't strictly increasing.
        // A clean curve is monotonic; this collapses a torn/interleaved read into a
        // single tidy line instead of a tangle of back-tracking segments.
        if (!xs.empty() && a <= xs.back()) continue;
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
    this->liveUprightSum = 0.f;   // restart the on-screen score with the run
    this->liveCenteredSum = 0.f;
    this->liveScoreFrames = 0;
    this->forceHistory.clear();  // start the force graph fresh each attempt
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    this->sim.setState(0.f, 0.f, kPi + dist(this->rng), 0.f);
}

// Restart the actor-overlay replay and load the latest rollout batch right away.
void App::resetActors() {
    this->actorFrame = 0;
    this->haveActorsMtime = false;  // force a fresh read
    this->maybeReloadActors();
}

// Advance the replay one frame, looping. A newer rollout is only swapped in when
// the replay wraps back to the start, so each batch plays its full ~10 s first.
void App::updateActors(float /*dt*/) {
    if (this->actorSteps <= 0) { this->maybeReloadActors(); return; }  // need a first batch
    this->actorFrame = (this->actorFrame + 1) % this->actorSteps;
    if (this->actorFrame == 0) this->maybeReloadActors();  // swap only at the loop boundary
}

// Reload the exported rollout batch (real actor trajectories) when it changes.
// File format: "B T" header, then T lines of B "x theta" pairs (t-major).
void App::maybeReloadActors() {
    namespace fs = std::filesystem;
    const std::string path = this->projectRoot + "/trainer/actors.txt";
    std::error_code ec;
    const fs::file_time_type t = fs::last_write_time(path, ec);
    if (ec) return;  // no rollout exported yet
    if (this->haveActorsMtime && t == this->actorsMtime) return;
    this->actorsMtime = t;
    this->haveActorsMtime = true;

    std::ifstream in(path);
    int B = 0, T = 0;
    if (!(in >> B >> T) || B <= 0 || T <= 0) return;
    std::vector<float> xs, ths;
    xs.reserve(static_cast<std::size_t>(B) * T);
    ths.reserve(static_cast<std::size_t>(B) * T);
    float x = 0.f, th = 0.f;
    for (long i = 0; i < static_cast<long>(B) * T; ++i) {
        if (!(in >> x >> th)) break;
        xs.push_back(x);
        ths.push_back(th);
    }
    if (static_cast<int>(xs.size()) < B * T) return;  // torn/partial -- keep the old batch
    this->actorX.swap(xs);
    this->actorTheta.swap(ths);
    this->actorCount = B;
    this->actorSteps = T;
    if (this->actorFrame >= T) this->actorFrame = 0;
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

    // All-actors overlay: replay the exported rollout batch (skip the single demo).
    // Frozen while paused -- the training graphics stop with the trainer.
    if (this->view == kViewActors) {
        if (!this->trainingPaused) this->updateActors(dt);
        return;
    }

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

    // Accumulate the live two-component score for the displayed run: uprightness
    // (max(0,cos theta)) and how centered it is while upright. Mirrors reinforce.py.
    const float up = std::max(0.f, std::cos(this->sim.getAngle()));
    const float frac = std::fabs(this->sim.getX()) / this->sim.config().trackLimit;
    const float centerQ = std::max(0.f, std::min(1.f, 1.f - frac));
    this->liveUprightSum += up;
    this->liveCenteredSum += up * centerQ;
    this->liveScoreFrames += 1;

    this->episodeTime += dt;  // shown as "Elapsed"; the run continues until the user
                              // presses Space to restart it with the latest policy.

    // Map the centered physics x (meters) onto the screen and position renderers.
    const float screenX = layout.center.x + this->sim.getX() * kPixelsPerMeter;
    this->cart.setPosition(screenX, layout.center.y);
    this->pendulum.setPivot(this->cart.getPivot());
    this->pendulum.setAngle(this->sim.getAngle());

    // Let the cursor shove the pole when it's resting on the ball (uses the just-
    // positioned bob, so the hit test matches what's drawn this frame).
    this->nudgeFromMouse(dt);
}

// Mouse "contact" push: when the cursor touches the bob, fire one impulse. The push
// points from the cursor into the bob center (where it made contact); only the part
// of that along the bob's swing arc can move it, so we project onto the tangent. A
// cooldown gates repeats so resting the cursor on the ball gives one kick per second,
// not a per-frame torrent.
void App::nudgeFromMouse(float dt) {
    if (this->kickCooldown > 0.f) this->kickCooldown -= dt;  // tick toward ready

    const sf::Vector2i mp = sf::Mouse::getPosition(this->window);
    const float mx = static_cast<float>(mp.x);
    const float my = static_cast<float>(mp.y);

    const float theta = this->sim.getAngle();
    const float lengthPx = this->sim.config().length * kPixelsPerMeter;
    const sf::Vector2f pivot = this->cart.getPivot();
    const float bobX = pivot.x + lengthPx * std::sin(theta);
    const float bobY = pivot.y - lengthPx * std::cos(theta);

    // Only when the cursor is actually touching the ball (small grab margin so the
    // 15 px bob is reachable) and the cooldown has elapsed.
    const float grabR = kBobRadiusMeters * kPixelsPerMeter + 4.f;
    const float dx = bobX - mx, dy = bobY - my;
    const float dist = std::sqrt(dx * dx + dy * dy);
    if (this->kickCooldown > 0.f || dist > grabR || dist < 1e-3f) return;

    // Push direction = cursor -> bob center; tangent of the swing arc = (cos, sin).
    // The tangential component is the only part that changes the angle.
    const float pushX = dx / dist, pushY = dy / dist;
    const float tangential = pushX * std::cos(theta) + pushY * std::sin(theta);
    const float dOmega = tangential * this->kickStrength;
    this->sim.setState(this->sim.getX(), this->sim.getVelocity(), theta,
                       this->sim.getAngularVelocity() + dOmega);
    this->kickCooldown = kKickCooldown;  // start the cooldown after a kick lands
}

// Draw the current frame: background, track + position axis, cart, pendulum,
// then the live data overlay on top.
void App::render() {
    window.clear(sf::Color(100, 100, 100));

    const float W = static_cast<float>(window.getSize().x);
    const float H = static_cast<float>(window.getSize().y);
    // Bottom graph row (mirrors hud's kPlayBottomFrac = 0.72) and the main play area.
    const float rowY0 = H * 0.72f + 16.f, rowY1 = H - 28.f;
    char title[32];
    std::snprintf(title, sizeof(title), "Curr Average  %.1f", this->avgRecentScore());

    // Framed play area (cart/track) as the main focus. Green frame in the all-actors
    // view (it shows training), coral in the single-run view.
    this->hud.drawPlayfield(window, this->view == kViewActors
                                        ? sf::Color(120, 200, 150)   // green
                                        : sf::Color(225, 130, 95));  // coral

    // Draw the track to match the physical limits: cart-center range (+-trackLimit
    // meters) scaled to pixels, widened by half a cart so the body fits the rail.
    TrackLayout layout = computeTrackLayout(window);
    layout.width = 2.f * this->sim.config().trackLimit * kPixelsPerMeter + this->cart.getWidth();
    drawTrack(window, layout);
    this->hud.drawAxis(window, layout, this->sim.config().trackLimit, kPixelsPerMeter);

    if (this->view == kViewActors) {
        // TRAINING view. Overlay every actor's pole on the one track, translucent so
        // the spread of swing-ups reads at once (states from the replayed rollout).
        const sf::Uint8 alpha = static_cast<sf::Uint8>(
            std::max(40, 200 / std::max(1, this->actorCount)));
        const int base = this->actorFrame * this->actorCount;
        for (int i = 0; i < this->actorCount; ++i) {
            const float sx = layout.center.x + this->actorX[base + i] * kPixelsPerMeter;
            this->cart.setPosition(sx, layout.center.y);
            this->pendulum.setPivot(this->cart.getPivot());
            this->pendulum.setAngle(this->actorTheta[base + i]);
            this->cart.draw(window, alpha);
            this->pendulum.draw(window, alpha);
        }
        // Bottom: the learning curve, centered.
        const float cx = (W - 540.f) / 2.f;
        this->hud.drawScoreGraph(window, this->scoreXs, this->scoreYs, title,
                                 cx, rowY0, cx + 540.f, rowY1);
    } else {
        // SINGLE-RUN view. Just the cart/pole and its control-force trace, centered.
        cart.draw(window);
        pendulum.draw(window);
        const float cx = (W - 540.f) / 2.f;
        this->hud.drawGraph(window, this->forceHistory, kGraphSamples, kMaxInputForce,
                            "Control force (N)", cx);
    }

    // Status panels, top-left. While training, split into two: an orange box for
    // the displayed run and a green box (next to it) for overall training progress.
    if (this->trainingMode && this->view == kViewActors) {
        // Training view: the green training box (no single-run orange box).
        this->hud.drawTextBox(window, this->greenText(), 40.f, 22.f,
                              sf::Color(120, 200, 150));                  // green
    } else if (this->trainingMode) {
        // Single-run view: focus on the run -- just the orange box.
        this->hud.drawTextBox(window, this->orangeText(), 40.f, 22.f,
                              sf::Color(225, 130, 95));                   // coral
    } else {
        this->hud.drawTextBox(window, this->displayText(), 40.f, 22.f,
                              sf::Color(95, 190, 180));                   // teal
    }

    if (this->trainingMode) this->hud.drawCornerHint(window, this->controlsText());
    window.display();
}
