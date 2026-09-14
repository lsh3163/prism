# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE.rsl-rl for the retained upstream policy interface and notice.
"""Alpha-only control for the legacy G1 residual actor (Python 3.8, CPU importable).

This diagnostic preserves the legacy raw skip, affine modules, ELU, and scalar
warmup. Its learned gates multiply residual interaction terms; it does not use
the default PRISM recurrence implemented in gated_actor.py.
"""

import math
from typing import Mapping, Sequence

import torch
from actor import PolyActorCritic, ResidualPolyEncoder
from torch import Tensor, nn

ACTOR_VARIANT = "g1_residual_learned_gate_v1"
ACTOR_VARIANT_VERSION = 3
WARMUP_UPDATES = 500


class ResidualLearnedGateEncoder(ResidualPolyEncoder):
    """Attach unit-initialized gates to already initialized residual modules.

    The source encoder is transferred without initializing or copying weights.
    This preserves the RNG state after the *whole* legacy actor and critic have
    been constructed. Gates have shape (degree - 1, hidden_dim), broadcast over
    the leading dimensions of observations shaped (..., input_dim).
    """

    def __init__(self, source: ResidualPolyEncoder):
        # Calling ResidualPolyEncoder.__init__ would draw new random weights.
        nn.Module.__init__(self)
        self.input_layer_norm = source.input_layer_norm
        self.output_layer_norm = source.output_layer_norm
        self.activation = source.activation
        self.raw_proj = source.raw_proj
        self.poly_in_proj = source.poly_in_proj
        self.A = source.A
        self.S = source.S
        self.poly_out_proj = source.poly_out_proj
        self.degree = source.degree
        self.register_buffer("poly_scale", source.poly_scale)
        self.interaction_scales = nn.Parameter(
            source.raw_proj.weight.new_ones((source.degree - 1, source.raw_proj.out_features))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer_norm(x)
        raw = self.raw_proj(x)

        h = self.poly_in_proj(x)
        poly = self.A[0](h)
        for i in range(1, self.degree):
            # Preserve the legacy product's arithmetic order when alpha == 1.
            interaction = self.A[i](h) * self.S[i - 1](poly)
            poly = poly + self.interaction_scales[i - 1] * interaction
        poly = self.poly_out_proj(poly)

        out = raw + self.poly_scale * poly
        out = self.output_layer_norm(out)
        return self.activation(out)


class ResidualLearnedGateActorCritic(PolyActorCritic):
    """Legacy G1 PPO policy with learned residual interaction amplitudes only.

    Main interfaces are 705 actor inputs, 219 critic inputs, and 12 Gaussian
    actions. The actor keeps its 256-wide encoder and 512/256/128 MLP; the critic
    keeps its 768/256/128 MLP. Every existing tensor is initialized in the legacy
    order before adding deterministic gates and the checkpoint version marker.
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
        actor_use_poly: bool = True,
        critic_use_poly: bool = False,
        actor_poly_mode: str = "residual",
        critic_poly_mode: str = "replace",
        actor_use_input_layer_norm: bool = False,
        critic_use_input_layer_norm: bool = False,
        actor_use_hidden_layer_norm: bool = False,
        critic_use_hidden_layer_norm: bool = False,
        actor_output_tanh: bool = False,
        actor_poly_warmup_updates: int = WARMUP_UPDATES,
        gate_init: float = 1.0,
        actor_variant: str = ACTOR_VARIANT,
    ):
        if actor_variant != ACTOR_VARIANT:
            raise ValueError("Expected the explicit g1_residual_learned_gate_v1 actor variant")
        if gate_init != 1.0:
            raise ValueError("The residual learned-gate control requires gate_init=1.0")
        if not actor_use_poly or actor_poly_mode != "residual" or critic_use_poly:
            raise ValueError("The residual learned-gate control requires a residual actor and MLP critic")
        if (
            activation != "elu"
            or actor_output_tanh
            or any(
                (
                    actor_use_input_layer_norm,
                    critic_use_input_layer_norm,
                    actor_use_hidden_layer_norm,
                    critic_use_hidden_layer_norm,
                )
            )
        ):
            raise ValueError(
                "The residual learned-gate control requires ELU without LayerNorm or output tanh"
            )
        if actor_poly_warmup_updates != WARMUP_UPDATES:
            raise ValueError("The residual learned-gate control requires the legacy 500-update warmup")
        if poly_degree < 2 or poly_hidden_dim < 1:
            raise ValueError("The residual learned-gate control requires degree >= 2 and positive width")

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            poly_hidden_dim=poly_hidden_dim,
            poly_degree=poly_degree,
            actor_use_poly=actor_use_poly,
            critic_use_poly=critic_use_poly,
            actor_poly_mode=actor_poly_mode,
            critic_poly_mode=critic_poly_mode,
            actor_use_input_layer_norm=actor_use_input_layer_norm,
            critic_use_input_layer_norm=critic_use_input_layer_norm,
            actor_use_hidden_layer_norm=actor_use_hidden_layer_norm,
            critic_use_hidden_layer_norm=critic_use_hidden_layer_norm,
            actor_output_tanh=actor_output_tanh,
            actor_poly_warmup_updates=actor_poly_warmup_updates,
        )
        self.actor_encoder = ResidualLearnedGateEncoder(self.actor_encoder)
        self.register_buffer("actor_variant_version", torch.tensor(ACTOR_VARIANT_VERSION, dtype=torch.long))

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True, **kwargs):
        if not strict:
            raise ValueError("g1_residual_learned_gate_v1 requires strict checkpoint loading")
        marker = state_dict.get("actor_variant_version")
        if (
            not isinstance(marker, Tensor)
            or marker.shape != torch.Size([])
            or marker.dtype != torch.long
            or marker.item() != ACTOR_VARIANT_VERSION
        ):
            raise ValueError(
                "Expected a g1_residual_learned_gate_v1 checkpoint, not a legacy or default actor"
            )
        scales = state_dict.get("actor_encoder.interaction_scales")
        if (
            not isinstance(scales, Tensor)
            or scales.shape != self.actor_encoder.interaction_scales.shape
            or not torch.is_floating_point(scales)
            or not torch.isfinite(scales).all().item()
        ):
            raise ValueError(
                "Checkpoint residual gates must be finite and match the selected degree and width"
            )

        step = state_dict.get("actor_poly_warmup_step")
        scale = state_dict.get("actor_encoder.poly_scale")
        if (
            not isinstance(step, Tensor)
            or step.shape != torch.Size([])
            or step.dtype != torch.long
            or not 0 <= step.item() <= WARMUP_UPDATES
            or not isinstance(scale, Tensor)
            or scale.shape != torch.Size([])
            or not torch.is_floating_point(scale)
            or not math.isclose(scale.item(), step.item() / float(WARMUP_UPDATES), rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError("Checkpoint warmup buffers must agree with the legacy 500-update schedule")
        return super().load_state_dict(state_dict, strict=True, **kwargs)
