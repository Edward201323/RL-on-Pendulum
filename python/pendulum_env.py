"""Gymnasium environment wrapping the C++ cart-pole physics core.

This is the *swing-up* task: the pole starts hanging at the bottom and the agent
must pump energy in to swing it up and then balance it at the top. All dynamics
live in the compiled ``cartpole_cpp`` module; this file only adds the RL
contract (observation, reward, termination).

Run the smoke test from the repo root (so the built .so is importable)::

    python3.12 python/pendulum_env.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Make the compiled extension importable when running straight from the repo
# (the .so lands in ./build). Harmless if it is already on sys.path.
_here = os.path.dirname(os.path.abspath(__file__))
_build = os.path.normpath(os.path.join(_here, os.pardir, "build"))
if os.path.isdir(_build) and _build not in sys.path:
    sys.path.insert(0, _build)

import cartpole_cpp  # noqa: E402  (must follow the sys.path tweak above)

# Max push force in NEWTONS (the [-1,1] action is scaled by this). ~0.8x the
# cart's weight (M*g = 1.0*9.81 = 9.81 N) -> max accel ~0.82 g. A modest, realistic
# servo: it can't muscle the pole up, so swing-up needs real energy-pumping.
# MUST match kMaxInputForce in src/app.cpp (training vs. playback force scale).
F_MAX = 5.0
DT = 1.0 / 60.0         # physics step (s), matched to the 60 FPS SFML demo

# Continuous action: a single value in [-1, 1] scaled to a horizontal force on
# the cart (-1 = full push left, +1 = full push right, 0 = coast). The policy
# chooses both direction AND magnitude.


class PendulumSwingUpEnv(gym.Env):
    """Swing the pole up from the bottom and balance it. Fixed-length episode."""

    metadata = {"render_modes": []}

    def __init__(self, init_angle_noise: float = 0.05, max_steps: int = 500):
        super().__init__()
        self.sim = cartpole_cpp.CartPole()
        self.init_angle_noise = float(init_angle_noise)
        self.max_steps = int(max_steps)
        self.steps = 0

        # One continuous force in [-1, 1] (scaled by F_MAX); see top of file.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # Observation: (x, x_dot, sin theta, cos theta, theta_dot). We feed
        # sin/cos instead of the raw angle because the pole rotates through every
        # angle here -- raw theta wraps discontinuously at +-pi, which a network
        # can't represent; sin/cos are smooth everywhere.
        high = np.array([np.inf] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        # Start hanging at the bottom (theta = pi) with a little noise.
        theta = math.pi + float(self.np_random.uniform(-self.init_angle_noise,
                                                        self.init_angle_noise))
        self.sim.set_state(0.0, 0.0, theta, 0.0)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        # action may arrive as a scalar or a length-1 array; clip to [-1, 1] and
        # scale to a force. (The policy samples from an unbounded Gaussian, so
        # clipping here is what actually bounds the applied force.)
        a = float(np.asarray(action).reshape(-1)[0])
        force = max(-1.0, min(1.0, a)) * F_MAX
        self.sim.set_control_force(force)
        self.sim.advance(DT)
        self.steps += 1

        theta = self.sim.angle
        theta_dot = self.sim.angular_velocity
        limit = self.sim.config.track_limit

        # Swing-up reward. cos(theta) is +1 at the top, -1 hanging down, so the
        # agent is continuously rewarded for getting (and staying) upright --
        # wrap-safe, unlike theta**2. The small extra terms keep it centered,
        # discourage endless spinning, and put a tiny price on control.
        upright = math.cos(theta)
        reward = upright
        reward -= 0.1 * (self.sim.x / limit) ** 2
        reward -= 0.001 * theta_dot ** 2
        reward -= 0.001 * (force / F_MAX) ** 2
        # Bonus for being upright AND slow: rewards actually catching/balancing
        # at the top rather than just whirling through it.
        if upright > 0.95:
            reward += max(0.0, 1.0 - abs(theta_dot) / 2.0)
        if self.sim.boundary_contact() != 0:
            reward -= 5.0  # strong penalty: hitting a track edge is undesired
            # (5x the max per-step upright reward, so any wall contact clearly hurts)

        # No failure state -- the pole may legitimately be at any angle. The
        # episode simply runs for a fixed horizon (truncation).
        terminated = False
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    # -- helpers ------------------------------------------------------------

    def _obs(self):
        theta = self.sim.angle
        return np.array(
            [self.sim.x, self.sim.velocity, math.sin(theta), math.cos(theta),
             self.sim.angular_velocity],
            dtype=np.float32,
        )


def _smoke_test():
    """Random-policy rollout to confirm the env and C++ module work together."""
    env = PendulumSwingUpEnv()
    obs, _ = env.reset(seed=0)
    print("initial obs:", obs)
    total = 0.0
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    print(f"episode ended after {env.steps} steps, total reward {total:.2f}")


if __name__ == "__main__":
    _smoke_test()
