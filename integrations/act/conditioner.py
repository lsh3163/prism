"""Camera-ready ACT conditioners; preserve their checkpoint parameter names."""

import torch
from torch import nn


class BaseConditioner(nn.Module):
    """Native affine projection with the shared output RMS normalization."""

    def __init__(self, native_projection: nn.Linear) -> None:
        super().__init__()
        self.projection = native_projection
        self.output_norm = nn.RMSNorm(native_projection.out_features, eps=1e-6, elementwise_affine=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.projection(state))


class PrismConditioner(nn.Module):
    """Scalar quadratic interaction, identity projection, then native RMSNorm."""

    def __init__(self, native_projection: nn.Linear) -> None:
        super().__init__()
        width = native_projection.out_features
        self.u = nn.Linear(native_projection.in_features, width)
        self.v = nn.Linear(native_projection.in_features, width)
        self.projection = nn.Linear(width, width)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.output_norm = nn.RMSNorm(width, eps=1e-6, elementwise_affine=True)
        with torch.no_grad():
            self.u.weight.copy_(native_projection.weight)
            self.u.bias.copy_(native_projection.bias)
            self.projection.weight.copy_(torch.eye(width))
            self.projection.bias.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u, v = self.u(state), self.v(state)
        return self.output_norm(self.projection(u + self.alpha * u * v))
