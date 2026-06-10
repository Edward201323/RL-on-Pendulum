"""Gymnasium environment wrapping the C++ cart-pole physics core.

This is the *balance* task: the pole starts near upright (theta = 0) and the
agent must keep it there. All dynamics live in the compiled ``cartpole_cpp``
module; this file only adds the RL contract (observation, reward, termination).

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

F_MAX = 4000.0          # magnitude of the push the discrete actions apply
DT = 1.0 / 60.0         # physics step, matched to the 60 FPS SFML demo

# Discrete action -> horizontal force on the cart.
#   0 = push left, 1 = coast (no force), 2 = push right
ACTION_FORCES = (-F_MAX, 0.0, +F_MAX)


class PendulumBalanceEnv(gym.Env):
    """Balance the pole upright. Episode ends when it tips past +-90 degrees."""

    metadata = {"render_modes": []}

    def __init__(self, init_angle_noise: float = 0.05, max_steps: int = 500):
        super().__init__()
        self.sim = cartpole_cpp.CartPole()
        self.init_angle_noise = float(init_angle_noise)
        self.max_steps = int(max_steps)
        self.steps = 0

        # Three discrete pushes (see ACTION_FORCES).
        self.action_space = spaces.Discrete(3)
        # Observation is the raw physical state (x, x_dot, theta, theta_dot).
        # Bounds are loose; only theta is truly bounded (we terminate past pi/2).
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        # Start near upright with a small random tilt so the policy can't just
        # memorize one direction. theta = 0 is straight up.
        theta = float(self.np_random.uniform(-self.init_angle_noise,
                                              self.init_angle_noise))
        self.sim.set_state(0.0, 0.0, theta, 0.0)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        force = ACTION_FORCES[int(action)]
        self.sim.set_control_force(force)
        self.sim.advance(DT)
        self.steps += 1

        obs = self._obs()
        theta = self.sim.angle
        theta_dot = self.sim.angular_velocity

        # Reward: stay upright (theta -> 0), stay calm (small theta_dot), and a
        # tiny penalty on control effort so the agent doesn't flail at full
        # force every step. F is normalized so this term is O(1).
        reward = -(theta ** 2) - 0.1 * (theta_dot ** 2) - 0.001 * (force / F_MAX) ** 2
        if self.sim.boundary_contact() != 0:
            reward -= 10.0  # slamming into a wall is a bad outcome

        # Terminate when the pole has fallen past horizontal; truncate on the
        # time limit (gymnasium distinguishes "failed" from "ran out of time").
        terminated = abs(theta) > math.pi / 2
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}

    # -- helpers ------------------------------------------------------------

    def _obs(self):
        return np.array(
            [self.sim.x, self.sim.velocity, self.sim.angle, self.sim.angular_velocity],
            dtype=np.float32,
        )


def _smoke_test():
    """Random-policy rollout to confirm the env and C++ module work together."""
    env = PendulumBalanceEnv()
    obs, _ = env.reset(seed=0)
    print("initial obs:", obs)
    total = 0.0
    for _ in range(500):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    print(f"episode ended after {env.steps} steps, total reward {total:.2f} "
          f"(terminated={terminated}, truncated={truncated})")


if __name__ == "__main__":
    _smoke_test()
