# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE.rsl-rl for the upstream notice and SOURCE_MANIFEST.json for provenance.
# ruff: noqa: E741
"""Checkpoint-compatible actors used by the G1 Humanoid-Gym experiments.

This module intentionally retains the experimental residual polynomial actor.
It is independent of Isaac Gym and supports its Python 3.8 environment.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "identity":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None


class LayerNormMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation,
        use_input_layer_norm=False,
        use_hidden_layer_norm=False,
        final_tanh=False,
    ):
        super().__init__()

        self.input_layer_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_hidden_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if final_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.input_layer_norm(x))


class PolyEncoder(nn.Module):
    """
    Feed-forward polynomial encoder inspired by factorized Pi-net style interactions.

    This block is intentionally lightweight enough to slot in before a standard MLP
    actor/critic while still exposing higher-order multiplicative terms.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        degree=2,
        activation="elu",
        use_input_layer_norm=False,
        use_output_layer_norm=False,
    ):
        super().__init__()
        if degree < 1:
            raise ValueError(f"Expected degree >= 1, got {degree}.")

        self.input_layer_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()
        self.output_layer_norm = nn.LayerNorm(hidden_dim) if use_output_layer_norm else nn.Identity()

        self.degree = degree
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.A = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(degree)])
        self.S = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(degree - 1)])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        h = self.in_proj(self.input_layer_norm(x))
        out = self.A[0](h)

        for i in range(1, self.degree):
            out = out + self.A[i](h) * self.S[i - 1](out)

        out = self.out_proj(out)
        out = self.output_layer_norm(out)
        out = self.activation(out)
        return out


class ResidualPolyEncoder(nn.Module):
    """
    Residual Poly encoder that preserves a direct raw-observation path.

    This is a safer drop-in for strong locomotion baselines because the policy can
    still rely on first-order features while Poly contributes higher-order terms.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        degree=2,
        activation="elu",
        use_input_layer_norm=False,
        use_output_layer_norm=False,
    ):
        super().__init__()
        if degree < 1:
            raise ValueError(f"Expected degree >= 1, got {degree}.")

        self.input_layer_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()
        self.output_layer_norm = nn.LayerNorm(hidden_dim) if use_output_layer_norm else nn.Identity()
        self.activation = get_activation(activation)

        self.raw_proj = nn.Linear(input_dim, hidden_dim)
        self.poly_in_proj = nn.Linear(input_dim, hidden_dim)
        self.A = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(degree)])
        self.S = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(degree - 1)])
        self.poly_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.degree = degree
        self.register_buffer("poly_scale", torch.tensor(1.0))

    def forward(self, x):
        x = self.input_layer_norm(x)
        raw = self.raw_proj(x)

        h = self.poly_in_proj(x)
        poly = self.A[0](h)
        for i in range(1, self.degree):
            poly = poly + self.A[i](h) * self.S[i - 1](poly)
        poly = self.poly_out_proj(poly)

        out = raw + self.poly_scale * poly
        out = self.output_layer_norm(out)
        return self.activation(out)

    def set_poly_scale(self, scale):
        self.poly_scale.fill_(float(scale))

    def get_poly_scale(self):
        return float(self.poly_scale.item())


class PolyActorCritic(nn.Module):
    """
    Feed-forward polynomial actor-critic intended as a PPO-compatible proxy for
    testing Poly encoders against stronger modern feed-forward humanoid baselines.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        poly_hidden_dim=256,
        poly_degree=2,
        actor_use_poly=True,
        critic_use_poly=True,
        actor_poly_mode="replace",
        critic_poly_mode="replace",
        actor_use_input_layer_norm=True,
        critic_use_input_layer_norm=True,
        actor_use_hidden_layer_norm=True,
        critic_use_hidden_layer_norm=True,
        actor_output_tanh=False,
        actor_poly_warmup_updates=0,
        **kwargs,
    ):
        if kwargs:
            print(
                "PolyActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor_poly_warmup_updates = max(0, int(actor_poly_warmup_updates))
        self.register_buffer("actor_poly_warmup_step", torch.zeros((), dtype=torch.long))

        if actor_use_poly:
            if actor_poly_mode == "residual":
                self.actor_encoder = ResidualPolyEncoder(
                    input_dim=num_actor_obs,
                    hidden_dim=poly_hidden_dim,
                    degree=poly_degree,
                    activation=activation,
                    use_input_layer_norm=actor_use_input_layer_norm,
                    use_output_layer_norm=actor_use_hidden_layer_norm,
                )
            else:
                self.actor_encoder = PolyEncoder(
                    input_dim=num_actor_obs,
                    hidden_dim=poly_hidden_dim,
                    degree=poly_degree,
                    activation=activation,
                    use_input_layer_norm=actor_use_input_layer_norm,
                    use_output_layer_norm=actor_use_hidden_layer_norm,
                )
            actor_input_dim = poly_hidden_dim
            actor_input_layer_norm = False
        else:
            self.actor_encoder = nn.Identity()
            actor_input_dim = num_actor_obs
            actor_input_layer_norm = actor_use_input_layer_norm

        if critic_use_poly:
            if critic_poly_mode == "residual":
                self.critic_encoder = ResidualPolyEncoder(
                    input_dim=num_critic_obs,
                    hidden_dim=poly_hidden_dim,
                    degree=poly_degree,
                    activation=activation,
                    use_input_layer_norm=critic_use_input_layer_norm,
                    use_output_layer_norm=critic_use_hidden_layer_norm,
                )
            else:
                self.critic_encoder = PolyEncoder(
                    input_dim=num_critic_obs,
                    hidden_dim=poly_hidden_dim,
                    degree=poly_degree,
                    activation=activation,
                    use_input_layer_norm=critic_use_input_layer_norm,
                    use_output_layer_norm=critic_use_hidden_layer_norm,
                )
            critic_input_dim = poly_hidden_dim
            critic_input_layer_norm = False
        else:
            self.critic_encoder = nn.Identity()
            critic_input_dim = num_critic_obs
            critic_input_layer_norm = critic_use_input_layer_norm

        self.actor = LayerNormMLP(
            input_dim=actor_input_dim,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_input_layer_norm=actor_input_layer_norm,
            use_hidden_layer_norm=actor_use_hidden_layer_norm,
            final_tanh=actor_output_tanh,
        )
        self.critic = LayerNormMLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            use_input_layer_norm=critic_input_layer_norm,
            use_hidden_layer_norm=critic_use_hidden_layer_norm,
            final_tanh=False,
        )

        print(f"Actor encoder: {self.actor_encoder}")
        print(f"Critic encoder: {self.critic_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        if self.actor_poly_warmup_updates > 0 and not isinstance(self.actor_encoder, ResidualPolyEncoder):
            print(
                "PolyActorCritic: actor_poly_warmup_updates is only supported with "
                "actor_poly_mode='residual'. Warm-up will be disabled."
            )
            self.actor_poly_warmup_updates = 0

        self._sync_actor_poly_warmup()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        encoded_obs = self.actor_encoder(observations)
        mean = self.actor(encoded_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def _sync_actor_poly_warmup(self):
        if not isinstance(self.actor_encoder, ResidualPolyEncoder):
            return

        if self.actor_poly_warmup_updates <= 0:
            self.actor_encoder.set_poly_scale(1.0)
            return

        warmup_step = int(self.actor_poly_warmup_step.item())
        scale = min(1.0, warmup_step / float(self.actor_poly_warmup_updates))
        self.actor_encoder.set_poly_scale(scale)

    def advance_poly_warmup(self):
        if self.actor_poly_warmup_updates <= 0:
            return
        if not isinstance(self.actor_encoder, ResidualPolyEncoder):
            return

        next_step = min(int(self.actor_poly_warmup_step.item()) + 1, self.actor_poly_warmup_updates)
        self.actor_poly_warmup_step.fill_(next_step)
        self._sync_actor_poly_warmup()

    def get_actor_poly_scale(self):
        if not isinstance(self.actor_encoder, ResidualPolyEncoder):
            return None
        return self.actor_encoder.get_poly_scale()

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        encoded_obs = self.actor_encoder(observations)
        return self.actor(encoded_obs)

    def evaluate(self, critic_observations, **kwargs):
        encoded_obs = self.critic_encoder(critic_observations)
        return self.critic(encoded_obs)
