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
import time

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


def read_train_speed(path="python/train_speed.txt"):
    """Per-update sleep (seconds) the app uses to slow training down; 0 = full speed."""
    try:
        with open(path) as f:
            return max(0.0, float(f.read().split()[0]))
    except Exception:
        return 0.0


def collect_batch(envs, policy):
    """Roll out all BATCH episodes in lockstep (no gradients -- fast).

    Every episode is a fixed `max_steps` long, so the envs stay synchronized and
    we can act on all of them with ONE batched policy forward per timestep instead
    of one-per-episode-per-step. Returns flattened arrays over all steps:
    (observations [N,5], actions [N], reward-to-go returns [N], per-episode totals).
    """
    B = len(envs)
    T = envs[0].max_steps
    obs = np.stack([e.reset()[0] for e in envs]).astype(np.float32)  # [B, 5]
    std = policy.log_std.exp().item()  # constant during the rollout (no updates yet)

    obs_seq = np.empty((T, B, 5), dtype=np.float32)
    act_seq = np.empty((T, B), dtype=np.float32)
    rew_seq = np.empty((T, B), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            means = policy(torch.from_numpy(obs)).squeeze(-1).numpy()  # [B]
            actions = np.random.normal(means, std).astype(np.float32)  # sample [B]
            obs_seq[t] = obs
            act_seq[t] = actions
            for i, e in enumerate(envs):
                obs[i], rew_seq[t, i], _, _, _ = e.step(float(actions[i]))

    # Reward-to-go G_t = r_t + gamma*r_{t+1} + ... computed back-to-front, per env.
    returns = np.empty((T, B), dtype=np.float32)
    g = np.zeros(B, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        g = rew_seq[t] + GAMMA * g
        returns[t] = g

    episode_returns = rew_seq.sum(axis=0)  # total reward per episode [B]
    return (obs_seq.reshape(-1, 5), act_seq.reshape(-1),
            returns.reshape(-1), episode_returns)


def main():
    # Optional CLI cap (for standalone runs); None = train indefinitely (the app
    # launches it this way and stops it by killing the process).
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else None
    torch.set_num_threads(1)  # the net is tiny; threading overhead only slows it down
    envs = [PendulumSwingUpEnv() for _ in range(BATCH)]  # stepped in lockstep
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # Export the starting (random) policy so a live viewer shows the "before".
    write_policy_txt(policy)
    write_status(0, 0.0, done=False)

    avg = 0.0
    update = 0
    attempt = 0
    while updates is None or update < updates:
        obs_flat, act_flat, ret_flat, episode_returns = collect_batch(envs, policy)

        # One batched forward/backward over every collected step (the speed win).
        obs_t = torch.from_numpy(obs_flat)                  # [N, 5]
        act_t = torch.from_numpy(act_flat)                  # [N]
        returns = torch.from_numpy(ret_flat)                # [N]
        # Standardize returns across the whole batch. This subtracts a baseline
        # (the batch mean) so an action is rewarded only for beating average, and
        # rescales to unit variance -- the single biggest stabilizer for REINFORCE.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        means = policy(obs_t).squeeze(-1)  # [N], now WITH gradients
        dist = torch.distributions.Normal(means, policy.log_std.exp())
        log_probs = dist.log_prob(act_t)   # log pi(a_t|s_t) for every step

        # Policy-gradient loss: maximize sum_t log pi(a_t|s_t) * G_t, so minimize
        # its negative. Above-average returns push their action's log-prob up;
        # below-average ones push it down. Divide by BATCH for a stable step size.
        loss = -(log_probs * returns).sum() / BATCH

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every individual attempt's total reward (return).
        for r in episode_returns:
            attempt += 1
            print(f"attempt {attempt:6d}   return {float(r):8.1f}")
        avg = float(np.mean(episode_returns))  # still used for the on-screen status

        # Re-export periodically so the SFML app can hot-reload and show progress.
        # Also checkpoint policy.pt here -- with the infinite loop the post-loop
        # save never runs, and the trainer is stopped by being killed.
        if (update + 1) % EXPORT_EVERY == 0:
            write_policy_txt(policy)
            torch.save(policy.state_dict(), "python/policy.pt")
            write_status((update + 1) * BATCH, avg, done=False)

        # Optional slow-mo: the app writes a per-update delay via the arrow keys.
        delay = read_train_speed()
        if delay > 0.0:
            time.sleep(delay)

        update += 1

    # Reached only for a finite CLI cap; the app trains indefinitely.
    write_policy_txt(policy)
    torch.save(policy.state_dict(), "python/policy.pt")
    write_status(updates * BATCH, avg, done=True)
    print("saved trained policy -> python/policy.pt")


if __name__ == "__main__":
    main()
