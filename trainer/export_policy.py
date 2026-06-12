"""Export a trained PyTorch policy to the flat text file the C++ app reads.

The SFML app re-runs the same tiny forward pass natively (see src/control/
policy.cpp), so it can play the RL policy with no Python at runtime. Run after
training:

    python3.12 trainer/export_policy.py        # writes trainer/policy.txt

(reinforce.py also exports automatically while training; this is for converting
a finished trainer/policy.pt on its own.)
"""

import torch

from reinforce import Policy, write_policy_txt


def main():
    policy = Policy()
    policy.load_state_dict(torch.load("trainer/policy.pt"))
    policy.eval()
    write_policy_txt(policy)
    print("wrote trainer/policy.txt")


if __name__ == "__main__":
    main()
