"""Gated Diffusion PRISM with explicit historical checkpoint compatibility."""

import math

import torch
from torch import Tensor, nn


class PolynomialKernelConditioner(nn.Module):
    """Transform normalized state history through two learned affine factors.

    ``gated_quadratic`` learns a per-feature alpha in ``left * (1 + alpha * right)``.
    ``latent_quadratic`` and ``raw`` preserve the historical ungated product,
    normalization, initialization, and checkpoint keys. Modes are never inferred
    from missing checkpoint parameters or silently converted while loading.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lift_mode: str,
        latent_dim: int,
        hidden_dim: int,
        gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if lift_mode not in {"raw", "latent_quadratic", "gated_quadratic"}:
            raise ValueError(f"Unsupported polynomial kernel lift mode: {lift_mode}")
        self.lift_mode = lift_mode
        self.input_norm = nn.LayerNorm(input_dim)
        factor_dim = input_dim if lift_mode == "raw" else latent_dim
        self.left_proj = nn.Linear(input_dim, factor_dim)
        self.right_proj = nn.Linear(input_dim, factor_dim)
        if lift_mode == "gated_quadratic":
            if not math.isfinite(gate_init):
                raise ValueError("gate_init must be finite")
            self.quadratic_scale = nn.Parameter(torch.full((factor_dim,), float(gate_init)))
            nn.init.zeros_(self.right_proj.bias)
        else:
            self.register_parameter("quadratic_scale", None)
        self.net = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, global_cond: Tensor) -> Tensor:
        features = self.input_norm(global_cond)
        left, right = self.left_proj(features), self.right_proj(features)
        factors = (
            left * right if self.quadratic_scale is None else left * (1.0 + self.quadratic_scale * right)
        )
        return self.net(factors)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # LeRobot's pretrained loader defaults to strict=False. A wrong actor
        # version must still fail instead of leaving a newly initialized gate or
        # discarding a learned gate from the incoming checkpoint.
        gate_key = prefix + "quadratic_scale"
        if (self.quadratic_scale is not None) != (gate_key in state_dict):
            error_msgs.append(
                f"{prefix}checkpoint gate schema does not match lift_mode={self.lift_mode!r}; "
                "load with the saved actor mode; automatic legacy/gated conversion is unsupported"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
