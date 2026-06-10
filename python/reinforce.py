"""REINFORCE (vanilla policy gradient), written from scratch.

PyTorch only provides the neural net + optimizer (it's a math library); the RL
algorithm -- rollout, discounted returns, and the policy-gradient loss -- is all
hand-written below. No RL framework involved.

The idea in one line: run some episodes, then nudge the probability of each
action up or down in proportion to how good the outcome that followed it was.

Run from the repo root:
    python3.12 python/reinforce.py            # full run (400 updates)
    python3.12 python/reinforce.py 60         # shorter run

While training, it re-exports python/policy.txt every few updates (plus a
python/train_status.txt progress file), so the SFML app can hot-reload and
visualize the policy improving live.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from pendulum_env import PendulumSwingUpEnv

EXPORT_EVERY = 5    # re-export policy.txt this often (updates) so the app can watch

GAMMA = 0.99        # discount: how much a future reward counts vs. an immediate one
LR = 1e-2
HIDDEN = 64
BATCH = 8           # episodes collected per gradient update -> averages out the noise

# Observations are SI mixed with sin/cos: x ~ +-1 m, x_dot ~ a few m/s, sin/cos
# are O(1), theta_dot ~ a few rad/s. Divide by a fixed per-component scale so
# every input is roughly O(1) (keeps the network's inputs balanced).
OBS_SCALE = torch.tensor([1.0, 2.0, 1.0, 1.0, 6.0])


class Policy(nn.Module):
    """Gaussian policy for a continuous force.

    The network outputs the MEAN action; a separate learned (state-independent)
    log-std sets how much we explore around it. The agent acts by *sampling* from
    Normal(mean, std) -- that sampling is the exploration.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )
        # log of the stddev, shared across states; exp() keeps it positive.
        # Starts at 0 -> std = 1 (broad exploration in normalized action units).
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        return self.net(obs / OBS_SCALE)  # the mean action (unbounded; env clips)


def write_policy_txt(policy, path="python/policy.txt"):
    """Dump the policy's mean network to the flat text format the C++ app reads.

    Writes to a temp file then atomically renames, so a reader (the live demo)
    never sees a half-written file.
    """
    lin1, lin2 = policy.net[0], policy.net[2]  # net = [Linear, Tanh, Linear]
    inputs, hidden, outputs = lin1.in_features, lin1.out_features, lin2.out_features

    nums = []
    nums += OBS_SCALE.tolist()
    nums += lin1.weight.detach().flatten().tolist()  # [hidden, inputs] row-major
    nums += lin1.bias.detach().tolist()
    nums += lin2.weight.detach().flatten().tolist()  # [outputs, hidden]
    nums += lin2.bias.detach().tolist()

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{inputs} {hidden} {outputs}\n")
        f.write(" ".join(repr(x) for x in nums))
        f.write("\n")
    os.replace(tmp, path)  # atomic


def write_status(attempts, avg_return, done, path="python/train_status.txt"):
    """Progress file the app reads for its on-screen training monitor.

    `attempts` = total practice episodes so far (updates * BATCH) -- a simpler,
    more intuitive count than gradient updates for the on-screen display.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{attempts} {avg_return:.2f} {1 if done else 0}\n")
    os.replace(tmp, path)


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
    """Play one episode; return per-step log-probs, reward-to-go, and total reward."""
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        mean = policy(torch.from_numpy(obs))
        dist = torch.distributions.Normal(mean, policy.log_std.exp())
        action = dist.sample()                       # sample a force (exploration)
        log_probs.append(dist.log_prob(action).sum())  # log pi(a|s) for continuous a
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    return torch.stack(log_probs), discounted_returns(rewards), sum(rewards)


def main():
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    env = PendulumSwingUpEnv()
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # Export the starting (random) policy so a live viewer shows the "before".
    write_policy_txt(policy)
    write_status(0, 0.0, done=False)

    avg = 0.0
    for update in range(updates):
        log_probs, returns, episode_returns = [], [], []
        for _ in range(BATCH):                     # collect a batch of episodes
            lp, ret, total = run_episode(env, policy)
            log_probs.append(lp)
            returns.extend(ret)
            episode_returns.append(total)

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

        avg = float(np.mean(episode_returns))
        if (update + 1) % 10 == 0:
            # Episodes are fixed-length now, so total reward (not length) is the
            # signal: it climbs as the agent learns to swing up and stay up.
            print(f"update {update + 1:4d}   avg return: {avg:8.1f}")

        # Re-export periodically so the SFML app can hot-reload and show progress.
        if (update + 1) % EXPORT_EVERY == 0:
            write_policy_txt(policy)
            write_status((update + 1) * BATCH, avg, done=False)

    write_policy_txt(policy)
    torch.save(policy.state_dict(), "python/policy.pt")
    write_status(updates * BATCH, avg, done=True)
    print("saved trained policy -> python/policy.pt")


if __name__ == "__main__":
    main()
