"""REINFORCE (vanilla policy gradient), written from scratch.

PyTorch only provides the neural net + optimizer (it's a math library); the RL
algorithm -- rollout, discounted returns, and the policy-gradient loss -- is all
hand-written below. No RL framework involved.

The idea in one line: run some episodes, then nudge the probability of each
action up or down in proportion to how good the outcome that followed it was.

Run from the repo root:
    python3.12 python/reinforce.py            # full run (200 updates)
    python3.12 python/reinforce.py 60         # shorter run
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from pendulum_env import PendulumBalanceEnv

GAMMA = 0.99        # discount: how much a future reward counts vs. an immediate one
LR = 1e-2
HIDDEN = 64
BATCH = 8           # episodes collected per gradient update -> averages out the noise

# Observations are raw physics units: x is in pixels (~hundreds) while theta is
# in radians (~1). Divide by a fixed scale so every input is roughly O(1) --
# otherwise the large x / velocity values would swamp the pole angle, which is
# the signal the policy needs most.
OBS_SCALE = torch.tensor([100.0, 100.0, 1.0, 5.0])


class Policy(nn.Module):
    """Maps an observation to logits over the 3 discrete actions (L / coast / R)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 3),
        )

    def forward(self, obs):
        return self.net(obs / OBS_SCALE)


def discounted_returns(rewards):
    """Reward-to-go: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Each step's return is the total *future* discounted reward from that step
    onward -- the "how good was it" weight the policy gradient assigns to the
    action taken at step t. Built back-to-front so each G reuses the next one.
    """
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        returns.insert(0, g)
    return returns


def run_episode(env, policy):
    """Play one episode; return its per-step log-probs and reward-to-go returns."""
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        logits = policy(torch.from_numpy(obs))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                     # sample (don't argmax) to explore
        log_probs.append(dist.log_prob(action))    # log pi(a|s): the gradient handle
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    return torch.stack(log_probs), discounted_returns(rewards), env.steps


def main():
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    env = PendulumBalanceEnv()
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    for update in range(updates):
        log_probs, returns, lengths = [], [], []
        for _ in range(BATCH):                     # collect a batch of episodes
            lp, ret, steps = run_episode(env, policy)
            log_probs.append(lp)
            returns.extend(ret)
            lengths.append(steps)

        log_probs = torch.cat(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Standardize returns across the whole batch. This subtracts a baseline
        # (the batch mean) so an action is rewarded only for beating average, and
        # rescales to unit variance -- the single biggest stabilizer for REINFORCE.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy-gradient loss: maximize sum_t log pi(a_t|s_t) * G_t, so minimize
        # its negative. Above-average returns push their action's log-prob up;
        # below-average ones push it down. Divide by BATCH for a stable step size.
        loss = -(log_probs * returns).sum() / BATCH

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (update + 1) % 10 == 0:
            print(f"update {update + 1:4d}   avg steps: {np.mean(lengths):6.1f}")

    torch.save(policy.state_dict(), "python/policy.pt")
    print("saved trained policy -> python/policy.pt")


if __name__ == "__main__":
    main()
