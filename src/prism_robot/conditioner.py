"""Learnable polynomial conditioning for proprioceptive robot state."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMS normalization over the final feature dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        inverse_rms = x_fp32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        normalized = x_fp32 * inverse_rms * self.weight.float()
        return normalized.to(dtype=x.dtype)


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    activations: dict[str, Callable[[], nn.Module]] = {
        "elu": nn.ELU,
        "mish": nn.Mish,
        "silu": nn.SiLU,
    }
    try:
        return activations[name]
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}; choose one of: {supported}") from exc


class _ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"post_mlp_layers must be at least 1, got {num_layers}")

        make_activation = _activation_factory(activation)
        if num_layers == 1:
            self.net = nn.Linear(input_dim, output_dim)
            return

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), make_activation()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), make_activation()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PRISMConditioner(nn.Module):
    """Map proprioception to a learned low-degree polynomial representation.

    In ``gated`` mode, the latent recurrence is

    ``h[0] = affine[0](x)``
    ``h[k] = h[k-1] * (1 + scale[k-1] * affine[k](x))``.

    The scales are learned independently for every latent feature. Initializing
    them near zero preserves a strong first-order path at the beginning of
    training while allowing higher-order interactions to emerge end-to-end.

    ``factorized`` mode instead multiplies affine factors directly and is
    provided for the pure factorized-polynomial ablations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        degree: int = 2,
        post_mlp_layers: int = 2,
        activation: str = "silu",
        interaction_mode: str = "gated",
        gate_init: float = 1e-2,
        use_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}")
        if interaction_mode not in {"gated", "factorized"}:
            raise ValueError(
                f"Unsupported interaction_mode={interaction_mode!r}; "
                "choose 'gated' or 'factorized'"
            )

        hidden_dim = hidden_dim or output_dim
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.degree = degree
        self.interaction_mode = interaction_mode

        self.factors = nn.ModuleList(
            nn.Linear(input_dim, hidden_dim) for _ in range(degree)
        )
        for factor in self.factors[1:]:
            nn.init.zeros_(factor.bias)

        if interaction_mode == "gated" and degree > 1:
            self.interaction_scales = nn.Parameter(
                torch.full((degree - 1, hidden_dim), float(gate_init))
            )
        else:
            self.register_parameter("interaction_scales", None)

        self.post_mlp = _ProjectionMLP(
            hidden_dim,
            output_dim,
            hidden_dim,
            post_mlp_layers,
            activation,
        )
        self.output_norm = (
            RMSNorm(output_dim, eps=rmsnorm_eps) if use_rmsnorm else nn.Identity()
        )

    def polynomial_features(self, x: Tensor) -> Tensor:
        """Return the latent polynomial features before downstream projection."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final input dimension {self.input_dim}, got {x.shape[-1]}"
            )

        features = self.factors[0](x)
        for index, factor_layer in enumerate(self.factors[1:]):
            factor = factor_layer(x)
            if self.interaction_mode == "gated":
                scale = self.interaction_scales[index]
                features = features * (1.0 + scale * factor)
            else:
                features = features * factor
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.polynomial_features(x)
        return self.output_norm(self.post_mlp(features))

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, degree={self.degree}, "
            f"interaction_mode={self.interaction_mode!r}"
        )
