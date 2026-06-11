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
LR = 3e-3           # lower than 1e-2 for long-run stability (REINFORCE diverges easily)
HIDDEN = 64
BATCH = 8           # episodes rolled out and averaged into each gradient update (fixed)
GRAD_CLIP = 1.0     # cap the gradient norm so one bad batch can't blow up the weights
LOG_STD_MIN = -2.0  # clamp the policy std to [~0.14, ~1.6] so exploration can't collapse
LOG_STD_MAX = 0.5   # (collapse -> stuck deterministic) or explode

# Score = two weighted components, must sum to 1.0 so a perfect run scores 100.
# Balance (time spent upright) is primary; centering (sitting at x=0 *while* upright)
# is the secondary tie-breaker. Centering is GATED on uprightness -- being centered
# while the pole hangs earns nothing -- so balancing is always worth more.
SCORE_W_BALANCE = 0.8   # max points from keeping the pole up
SCORE_W_CENTER = 0.2    # max extra points from doing it centered at x=0

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


def append_history(attempts, score, path="python/train_history.txt"):
    """Append one (attempts, score) point to the learning-curve history file.

    `score` is an EMA-smoothed average return -- a steadier signal than the raw
    noisy per-batch return. The app plots the whole file as a score-vs-attempts
    curve. Open/append/close each call so the line is flushed for the reader.
    """
    with open(path, "a") as f:
        f.write(f"{attempts} {score:.2f}\n")


def write_status(attempts, avg_return, done, path="python/train_status.txt"):
    """Progress file the app reads for its on-screen training monitor.

    `attempts` = total practice episodes so far (updates * BATCH) -- a simpler,
    more intuitive count than gradient updates for the on-screen display.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{attempts} {avg_return:.2f} {1 if done else 0}\n")
    os.replace(tmp, path)


def collect_batch(envs, policy):
    """Roll out all BATCH episodes in lockstep (no gradients -- fast).

    Every episode is a fixed `max_steps` long, so the envs stay synchronized and
    we can act on all of them with ONE batched policy forward per timestep instead
    of one-per-episode-per-step. Returns flattened arrays over all steps:
    (observations [N,5], actions [N], reward-to-go returns [N], per-episode totals).
    """
    B = len(envs)
    T = envs[0].max_steps
    limit = envs[0].sim.config.track_limit  # max |x|; turns position into a 0..1 fraction
    obs = np.stack([e.reset()[0] for e in envs]).astype(np.float32)  # [B, 5]
    std = policy.log_std.exp().item()  # constant during the rollout (no updates yet)

    obs_seq = np.empty((T, B, 5), dtype=np.float32)
    act_seq = np.empty((T, B), dtype=np.float32)
    rew_seq = np.empty((T, B), dtype=np.float32)
    upright_sum = np.zeros(B, dtype=np.float32)   # sum of max(0, cos theta): "how upright"
    centered_sum = np.zeros(B, dtype=np.float32)  # same, weighted by centeredness: "upright AND centered"

    with torch.no_grad():
        for t in range(T):
            means = policy(torch.from_numpy(obs)).squeeze(-1).numpy()  # [B]
            actions = np.random.normal(means, std).astype(np.float32)  # sample [B]
            obs_seq[t] = obs
            act_seq[t] = actions
            up = np.maximum(0.0, obs[:, 3])                      # obs[:,3] = cos(theta), >0 in upper half
            center_q = np.clip(1.0 - np.abs(obs[:, 0]) / limit, 0.0, 1.0)  # 1 at x=0, 0 at the edge
            upright_sum += up
            centered_sum += up * center_q                       # centering only counts while upright
            for i, e in enumerate(envs):
                obs[i], rew_seq[t, i], _, _, _ = e.step(float(actions[i]))

    # Reward-to-go G_t = r_t + gamma*r_{t+1} + ... computed back-to-front, per env.
    returns = np.empty((T, B), dtype=np.float32)
    g = np.zeros(B, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        g = rew_seq[t] + GAMMA * g
        returns[t] = g

    episode_returns = rew_seq.sum(axis=0)         # total reward per episode [B]
    # Two-component 0..100 score. balance = fraction of the episode spent upright.
    # centering = uprightness-weighted average centeredness (how centered it was *while*
    # balanced), so 0..1 and only earned when the pole is actually up. Balance dominates;
    # centering is the SCORE_W_CENTER-weighted tie-breaker on top.
    balance = upright_sum / T                                      # 0..1
    centering = centered_sum / np.maximum(upright_sum, 1e-8)       # 0..1, guarded
    episode_scores = 100.0 * balance * (SCORE_W_BALANCE + SCORE_W_CENTER * centering)
    return (obs_seq.reshape(-1, 5), act_seq.reshape(-1),
            returns.reshape(-1), episode_returns, episode_scores)


def main():
    # Optional CLI cap (for standalone runs); None = train indefinitely (the app
    # launches it this way and stops it by killing the process).
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else None
    torch.set_num_threads(1)  # the net is tiny; threading overhead only slows it down
    envs = [PendulumSwingUpEnv() for _ in range(BATCH)]  # fixed batch of episodes per update
    policy = Policy()
    # weight_decay keeps the network weights from growing unbounded (which is what
    # saturates the output into the +-max-force bang-bang chatter).
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR, weight_decay=1e-4)

    # Export the starting (random) policy so a live viewer shows the "before".
    write_policy_txt(policy)
    write_status(0, 0.0, done=False)
    open("python/train_history.txt", "w").close()  # fresh learning-curve log

    score_ema = None  # EMA-smoothed 0..100 "% upright" score = the plotted "score"
    update = 0
    attempt = 0
    while updates is None or update < updates:
        obs_flat, act_flat, ret_flat, episode_returns, episode_scores = collect_batch(envs, policy)

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
        # below-average ones push it down. Divide by the batch size for a stable step.
        loss = -(log_probs * returns).sum() / BATCH

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()
        # Keep the exploration std in a sane band so the policy can't lock into a
        # collapsed deterministic behavior (or blow up).
        with torch.no_grad():
            policy.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

        # Print every individual attempt's total reward (return).
        for r in episode_returns:
            attempt += 1
            print(f"attempt {attempt:6d}   return {float(r):8.1f}")
        # The plotted "score" is the 0..100 % upright, EMA-smoothed for stability.
        avg_score = float(np.mean(episode_scores))
        score_ema = avg_score if score_ema is None else 0.9 * score_ema + 0.1 * avg_score

        # Re-export periodically so the SFML app can hot-reload and show progress.
        # Also checkpoint policy.pt here -- with the infinite loop the post-loop
        # save never runs, and the trainer is stopped by being killed.
        if (update + 1) % EXPORT_EVERY == 0:
            write_policy_txt(policy)
            torch.save(policy.state_dict(), "python/policy.pt")
            write_status(attempt, score_ema, done=False)  # attempt = cumulative episodes
            append_history(attempt, score_ema)

        update += 1

    # Reached only for a finite CLI cap; the app trains indefinitely.
    write_policy_txt(policy)
    torch.save(policy.state_dict(), "python/policy.pt")
    write_status(attempt, score_ema if score_ema is not None else 0.0, done=True)
    print("saved trained policy -> python/policy.pt")


if __name__ == "__main__":
    main()
