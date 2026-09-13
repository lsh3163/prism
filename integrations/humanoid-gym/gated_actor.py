# SPDX-License-Identifier: BSD-3-Clause
# Uses the retained RSL-RL policy interface; see LICENSE.rsl-rl.
"""Learned per-feature gated G1 actor, independent of Isaac Gym (Python 3.8)."""

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from actor import ActorCritic

ACTOR_VARIANT = "g1_gated_poly_v2"


class GatedPolyEncoder(nn.Module):
    """Map (..., input_dim) observations to gated, ELU-projected latent features.

    The interaction stage is polynomial in the input; the final ELU is not.
    Degree one has one affine factor and no interaction gate. At higher degrees,
    every feature of every additional factor has its own PPO-trained alpha.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, degree: int = 2, gate_init: float = 0.01):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or degree < 1:
            raise ValueError("Input dimension, latent dimension and polynomial degree must be positive")
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.degree = degree
        self.factors = nn.ModuleList(nn.Linear(input_dim, hidden_dim) for _ in range(degree))
        if degree > 1:
            self.interaction_scales = nn.Parameter(torch.full((degree - 1, hidden_dim), float(gate_init)))
            for factor in self.factors[1:]:
                nn.init.zeros_(factor.bias)
        else:
            self.register_parameter("interaction_scales", None)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ELU()

    def polynomial_features(self, observations: Tensor) -> Tensor:
        features = self.factors[0](observations)
        for index in range(1, self.degree):
            features = features * (
                1.0 + self.interaction_scales[index - 1] * self.factors[index](observations)
            )
        return features

    def forward(self, observations: Tensor) -> Tensor:
        return self.activation(self.projection(self.polynomial_features(observations)))


class GatedPolyActorCritic(ActorCritic):
    """PPO policy with a gated actor history encoder and an independent critic.

    Default G1 interfaces are 705 actor inputs, 219 privileged critic inputs and
    12 Gaussian action coordinates. The marker and distinct parameter names
    prevent historical residual checkpoints from being loaded as this actor.
    """

    actor_variant = ACTOR_VARIANT

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (768, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        poly_hidden_dim: int = 256,
        poly_degree: int = 2,
        gate_init: float = 0.01,
        actor_variant: str = ACTOR_VARIANT,
    ):
        if actor_variant != ACTOR_VARIANT or activation != "elu":
            raise ValueError("g1_gated_poly_v2 requires its explicit variant ID and ELU activation")
        super().__init__(
            num_actor_obs=poly_hidden_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        self.actor_encoder = GatedPolyEncoder(num_actor_obs, poly_hidden_dim, poly_degree, gate_init)
        self.register_buffer("actor_variant_version", torch.tensor(2, dtype=torch.long))

    def update_distribution(self, observations: Tensor) -> None:
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act_inference(self, observations: Tensor) -> Tensor:
        return self.actor(self.actor_encoder(observations))

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True, **kwargs):
        marker = state_dict.get("actor_variant_version")
        if not strict:
            raise ValueError("g1_gated_poly_v2 requires strict checkpoint loading")
        if not isinstance(marker, Tensor) or marker.numel() != 1 or marker.item() != 2:
            raise ValueError("Expected g1_gated_poly_v2 checkpoint; select an explicit legacy recipe for v1")
        return super().load_state_dict(state_dict, strict=True, **kwargs)
