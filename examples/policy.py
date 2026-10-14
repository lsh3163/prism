"""CPU forward/backward example; synthetic data, not a benchmark experiment."""

import torch
from torch import nn

from prism_robot import PRISM


def main() -> None:
    torch.manual_seed(0)
    policy = nn.Sequential(PRISM(32, 256), nn.ELU(), nn.Linear(256, 12))
    observations = torch.randn(8, 32)
    actions = policy(observations)
    actions.square().mean().backward()
    assert actions.shape == (8, 12)
    assert policy[0].interaction_scales.grad is not None
    print(f"Actions: {tuple(actions.shape)}; polynomial coefficient receives gradients.")


if __name__ == "__main__":
    main()
