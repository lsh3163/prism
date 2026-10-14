"""Camera-ready native mjlab actors; preserve original checkpoint keys and computation."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

ACTOR_DIM = 286
ROBOT_COMMAND_DIM = 99
HEIGHT_DIM = 187
ACTION_DIM = 29
NATIVE_HIDDEN = (512, 256, 128)
FACTOR_DIM = 256
ALPHA_INITIAL = 0.01
RMS_EPS = 1.0e-6
VARIANTS = ("standard", "matched", "wider", "deeper", "prism")
CAPACITY_HIDDEN = {
    "standard": NATIVE_HIDDEN,
    "matched": (590, 295, 148),
    "wider": (762, 381, 190),
    "deeper": (750, 375, 188, 94),
    "prism": NATIVE_HIDDEN,
}
PROTOCOL = "native_mjlab_shared_head_rms_actor_comparison_v1"


def dense_parameter_count(hidden: tuple[int, ...]) -> int:
    dims = (ACTOR_DIM, *hidden, ACTION_DIM)
    return sum((left + 1) * right for left, right in zip(dims, dims[1:]))


def actor_recipe(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown shared-RMS actor variant: {variant}")
    hidden = CAPACITY_HIDDEN[variant]
    count = dense_parameter_count(hidden) + hidden[-1]
    if variant == "prism":
        count += 2 * (ROBOT_COMMAND_DIM + 1) * FACTOR_DIM
        count += FACTOR_DIM + (FACTOR_DIM + 1) * ROBOT_COMMAND_DIM
    return {
        "protocol": PROTOCOL,
        "experiment": "shared_head_rms_new_scratch_comparison",
        "variant": variant,
        "actor_class": "actors_mjlab_shared_rms:NativeSharedRmsActor",
        "actor_input_dim": ACTOR_DIM,
        "critic_input_dim": 298,
        "action_dim": ACTION_DIM,
        "native_head_hidden_dims": list(NATIVE_HIDDEN),
        "mean_hidden_dims": list(hidden),
        "mean_network_parameters": count,
        "distribution_parameters": ACTION_DIM,
        "total_trainable_parameters": count + ACTION_DIM,
        "shared_rms_width": hidden[-1],
        "shared_rms_parameters": hidden[-1],
        "shared_rms_placement": "after final hidden ELU, before final 29-action Linear",
        "shared_rms_eps": RMS_EPS,
        "shared_rms_initial_gain": 1.0,
        "shared_rms_accumulation_dtype": "float64",
        "normalization": "unchanged native EmpiricalNormalization over all 286 coordinates before the mean network",
        "gaussian_distribution": "unchanged native GaussianDistribution init_std=1.0 std_type=scalar",
        "external_actor_rng_consumption": "identical to native MLPModel initialization",
        "robot_command_coordinates": [0, ROBOT_COMMAND_DIM],
        "bypass_height_coordinates": [ROBOT_COMMAND_DIM, ACTOR_DIM],
        "factor_dim": FACTOR_DIM if variant == "prism" else None,
        "factor_degree": 2 if variant == "prism" else None,
        "initial_alpha": ALPHA_INITIAL if variant == "prism" else None,
        "pre_product_or_pre_projection_rms": False,
        "graph": "shared_rms_native_head(concat(projection(u*(1+alpha*v)), height))" if variant == "prism" else "MLP with final-hidden shared RMS",
        "historical_humanoid_graph_reproduction": False,
        "training_stability_or_performance_proven": False,
    }


class StableSharedRMSNorm(nn.Module):
    """Learnable RMS gain; FP64 moments avoid squaring finite FP32 overflow.

    This module cannot recover nonfinite features produced before its input.
    It normalizes the feature magnitude; learned gains remain unconstrained.
    """

    def __init__(self, width: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features64 = features.to(dtype=torch.float64)
        inverse_rms64 = torch.rsqrt(features64.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (features64 * inverse_rms64).to(dtype=features.dtype)
        return normalized * self.weight


class SharedRmsHead(nn.Sequential):
    """Insert one RMS layer without renaming any existing dense-layer keys."""

    def __init__(self, initialized_head: nn.Sequential):
        # _modules retains repeated ELU references at every position. In
        # contrast, named_children() deduplicates them and would lose ELUs.
        entries = list(initialized_head._modules.items())
        if len(entries) < 3 or not isinstance(entries[-1][1], nn.Linear):
            raise ValueError("Expected native MLP with final Linear")
        if not isinstance(entries[-2][1], nn.ELU):
            raise ValueError("Expected native final hidden ELU")
        final_name, final_layer = entries[-1]
        layers = OrderedDict(entries[:-1])
        layers["shared_rms"] = StableSharedRMSNorm(final_layer.in_features)
        layers[final_name] = final_layer
        super().__init__(layers)


class NativeSharedRmsPrismMean(nn.Module):
    def __init__(self, native_head: nn.Module):
        super().__init__()
        self.factor_u = nn.Linear(ROBOT_COMMAND_DIM, FACTOR_DIM)
        self.factor_v = nn.Linear(ROBOT_COMMAND_DIM, FACTOR_DIM)
        self.alpha = nn.Parameter(torch.full((FACTOR_DIM,), ALPHA_INITIAL))
        self.projection = nn.Linear(FACTOR_DIM, ROBOT_COMMAND_DIM)
        self.native_head = SharedRmsHead(native_head)

    def transformed_input(self, normalized: torch.Tensor) -> torch.Tensor:
        robot_command = normalized[..., :99]
        height = normalized[..., 99:]
        u = self.factor_u(robot_command)
        v = self.factor_v(robot_command)
        projected = self.projection(u * (1.0 + self.alpha * v))
        return torch.cat((projected, height), dim=-1)

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.native_head(self.transformed_input(normalized))


class NativeSharedRmsActor(MLPModel):
    """All five actors use the native API plus the same final-hidden RMS rule."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=NATIVE_HIDDEN,
                 activation="elu", obs_normalization=False, distribution_cfg=None,
                 variant="prism"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown shared-RMS actor variant: {variant}")
        if tuple(hidden_dims) != NATIVE_HIDDEN or activation != "elu":
            raise ValueError("Keep the registered native 512/256/128 ELU actor configuration")
        if obs_set != "actor" or output_dim != ACTION_DIM or not obs_normalization:
            raise ValueError("The shared-RMS actor requires native normalized 286-to-29 input/output")
        if distribution_cfg != {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}:
            raise ValueError("The native Gaussian configuration must remain unchanged")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims,
                         activation=activation, obs_normalization=obs_normalization,
                         distribution_cfg=distribution_cfg)
        if self.obs_dim != ACTOR_DIM or self.distribution.input_dim != ACTION_DIM:
            raise ValueError("Pinned native actor/distribution dimensions changed")
        self.variant = variant
        # Consume the native actor's global RNG first; additions/alternatives
        # must not change the subsequently initialized native critic.
        with torch.random.fork_rng(devices=[]):
            if variant == "prism":
                self.mlp = NativeSharedRmsPrismMean(self.mlp)
            elif variant == "standard":
                self.mlp = SharedRmsHead(self.mlp)
            else:
                initialized = MLP(ACTOR_DIM, ACTION_DIM, CAPACITY_HIDDEN[variant], activation)
                self.distribution.init_mlp_weights(initialized)
                self.mlp = SharedRmsHead(initialized)
        expected = actor_recipe(variant)["mean_network_parameters"]
        if sum(parameter.numel() for parameter in self.mlp.parameters()) != expected:
            raise AssertionError("Shared-RMS recipe and constructed actor differ")
