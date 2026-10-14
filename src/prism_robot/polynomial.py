"""The polynomial input layer in the camera-ready paper, Eqs. (4)--(6)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PRISM(nn.Module):
    """Replace an affine input layer with learned polynomial features.

    ``PRISM(d, m)`` maps ``(..., d)`` to ``(..., m)``. By default it computes
    ``u + s * (u * v)``, where u and v are separate affine maps and s is a
    learned scalar initialized to 0.01. Setting s to zero recovers u exactly.

    Higher degrees repeat ``h = h + s * (h * affine(x))`` with one scalar
    per stage. Activation, projection, normalization, and the policy head
    belong downstream; they are not part of these polynomial features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        degree: int = 2,
        scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.factors = nn.ModuleList(nn.Linear(input_dim, output_dim) for _ in range(degree))
        if degree > 1:
            self.interaction_scales = nn.Parameter(torch.full((degree - 1,), float(scale_init)))
        else:
            self.register_parameter("interaction_scales", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final input dimension {self.input_dim}, got {x.shape[-1]}")
        features = self.factors[0](x)
        for k, factor in enumerate(self.factors[1:]):
            features = features + self.interaction_scales[k] * (features * factor(x))
        return features

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, degree={self.degree}"
