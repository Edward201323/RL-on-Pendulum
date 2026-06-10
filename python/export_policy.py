"""Export the trained PyTorch policy to a flat text file the C++ app can read.

The SFML app re-runs the same tiny forward pass natively (see src/control/
policy.cpp), so it can play the RL policy with no Python at runtime. Run after
training:

    python3.12 python/export_policy.py        # writes python/policy.txt

File format (all whitespace-separated, matching PyTorch's [out, in] weight layout):
    line 1:  inputs hidden outputs            (4 64 3)
    then:    obs_scale(4), W1(hidden*inputs), b1(hidden), W2(outputs*hidden), b2(outputs)
"""

import torch

from reinforce import Policy, OBS_SCALE, HIDDEN


def main():
    policy = Policy()
    policy.load_state_dict(torch.load("python/policy.pt"))
    policy.eval()

    lin1, lin2 = policy.net[0], policy.net[2]  # net = [Linear, Tanh, Linear]
    nums = []
    nums += OBS_SCALE.tolist()
    nums += lin1.weight.detach().flatten().tolist()  # [hidden, inputs] row-major
    nums += lin1.bias.detach().tolist()
    nums += lin2.weight.detach().flatten().tolist()  # [outputs, hidden]
    nums += lin2.bias.detach().tolist()

    with open("python/policy.txt", "w") as f:
        f.write(f"4 {HIDDEN} 3\n")
        f.write(" ".join(repr(x) for x in nums))
        f.write("\n")
    print(f"wrote python/policy.txt ({len(nums)} parameters)")


if __name__ == "__main__":
    main()
