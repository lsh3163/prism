# SPDX-License-Identifier: BSD-3-Clause
# Derived from Unitree RL Gym and ETH Zurich / NVIDIA legged-gym code.
# See LICENSE.unitree and LICENSE.rsl-rl for retained upstream notices.
# See SOURCE_MANIFEST.json for exact source provenance and extraction scope.
"""The G1 Humanoid-Gym observation, history, gait and reward implementation."""

import numpy as np
import torch
from isaacgym import gymtorch
from legacy_base import HumanoidLeggedRobot


class G1Robot(HumanoidLeggedRobot):
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
        [NOTE]: Must be adapted when changing the observations structure
        """
        noise_vec = torch.zeros(
            int(getattr(self.cfg.env, "num_single_obs", self.cfg.env.raw_observation_dim)),
            dtype=torch.float,
            device=self.device,
        )
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:5] = 0.0
        noise_vec[5 : 5 + self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[5 + self.num_actions : 5 + 2 * self.num_actions] = (
            noise_scales.dof_vel * self.obs_scales.dof_vel
        )
        noise_vec[5 + 2 * self.num_actions : 5 + 3 * self.num_actions] = 0.0
        noise_vec[41:44] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[44:47] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        knee_names = getattr(self.cfg.asset, "knee_names", ["left_knee_link", "right_knee_link"])
        self.knee_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[0]),
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[1]),
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.knee_state = self.rigid_body_states_view[:, self.knee_indices, :]
        self.knee_pos = self.knee_state[:, :, :3]
        self.last_feet_z = self.feet_pos[:, :, 2].clone()
        self.feet_height = torch.zeros((self.num_envs, self.feet_num), device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        history_len = int(getattr(self.cfg.env, "actor_history_length", 1))
        critic_history_len = int(getattr(self.cfg.env, "critic_history_length", 1))
        actor_frame_dim = self.cfg.env.raw_observation_dim
        if self.cfg.env.use_feature_lifting:
            actor_frame_dim += self.cfg.env.lifted_observation_dim
        self.actor_history_length = history_len
        self.actor_frame_dim = actor_frame_dim
        if history_len > 1:
            self.actor_obs_history = torch.zeros(
                self.num_envs, history_len, actor_frame_dim, dtype=torch.float, device=self.device
            )
        else:
            self.actor_obs_history = None
        self.critic_history_length = critic_history_len
        critic_frame_dim = int(
            getattr(self.cfg.env, "single_num_privileged_obs", self.num_privileged_obs or 0)
        )
        if critic_history_len > 1 and critic_frame_dim > 0:
            self.critic_obs_history = torch.zeros(
                self.num_envs, critic_history_len, critic_frame_dim, dtype=torch.float, device=self.device
            )
        else:
            self.critic_obs_history = None
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_left = self.phase.clone()
        self.phase_right = torch.full((self.num_envs,), 0.5, dtype=torch.float, device=self.device)
        self.leg_phase = torch.stack((self.phase_left, self.phase_right), dim=-1)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.ref_action = torch.zeros_like(self.dof_pos)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.knee_state = self.rigid_body_states_view[:, self.knee_indices, :]
        self.knee_pos = self.knee_state[:, :, :3]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and self.actor_obs_history is not None:
            self.actor_obs_history[env_ids] = 0.0
        if len(env_ids) > 0 and self.critic_obs_history is not None:
            self.critic_obs_history[env_ids] = 0.0
        if len(env_ids) > 0:
            self.feet_height[env_ids] = 0.0
            self.last_feet_z[env_ids] = self.feet_pos[env_ids, :, 2]

    def _post_physics_step_callback(self):
        self.update_feet_state()
        period = float(getattr(self.cfg.rewards, "cycle_time", 0.8))
        offset = 0.5
        self.phase = self.episode_length_buf * self.dt / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self._compute_humanoid_gym_ref_state()
        return super()._post_physics_step_callback()

    def _get_humanoid_gym_gait_phase(self):
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        stance_mask = torch.zeros((self.num_envs, self.feet_num), device=self.device, dtype=torch.bool)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = True
        return stance_mask

    def _compute_humanoid_gym_ref_state(self):
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos.zero_()
        scale_1 = float(getattr(self.cfg.rewards, "target_joint_pos_scale", 0.17))
        scale_2 = 2.0 * scale_1
        sin_pos_l[sin_pos_l > 0.0] = 0.0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        sin_pos_r[sin_pos_r < 0.0] = 0.0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.0
        self.ref_action = 2.0 * self.ref_dof_pos

    def _build_humanoid_gym_actor_observations(self, command_input, dof_pos_scaled, dof_vel_scaled):
        quat_scale = float(getattr(self.obs_scales, "quat", 1.0))
        return torch.cat(
            (
                command_input,
                dof_pos_scaled,
                dof_vel_scaled,
                self.actions,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.rpy * quat_scale,
            ),
            dim=-1,
        )

    def _build_humanoid_gym_critic_observations(self, command_input, dof_pos_scaled, dof_vel_scaled):
        quat_scale = float(getattr(self.obs_scales, "quat", 1.0))
        stance_mask = self._get_humanoid_gym_gait_phase().float()
        contact_mask = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()
        diff = self.dof_pos - self.ref_dof_pos
        return torch.cat(
            (
                command_input,
                dof_pos_scaled,
                dof_vel_scaled,
                self.actions,
                diff,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.rpy * quat_scale,
                self.rand_push_force[:, :2],
                self.rand_push_torque,
                self.friction_coeffs,
                self.base_masses / 30.0,
                stance_mask,
                contact_mask,
            ),
            dim=-1,
        )

    def compute_observations(self):
        """Computes observations."""
        self._compute_humanoid_gym_ref_state()
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        command_input = torch.cat((sin_phase, cos_phase, self.commands[:, :3] * self.commands_scale), dim=1)
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_scaled = self.dof_vel * self.obs_scales.dof_vel
        actor_obs = self._build_humanoid_gym_actor_observations(
            command_input=command_input, dof_pos_scaled=dof_pos_scaled, dof_vel_scaled=dof_vel_scaled
        )
        privileged_obs = self._build_humanoid_gym_critic_observations(
            command_input=command_input, dof_pos_scaled=dof_pos_scaled, dof_vel_scaled=dof_vel_scaled
        )
        if self.add_noise:
            actor_obs = (
                actor_obs + torch.randn_like(actor_obs) * self.noise_scale_vec * self.cfg.noise.noise_level
            )
        if self.actor_obs_history is not None:
            self.actor_obs_history = torch.roll(self.actor_obs_history, shifts=-1, dims=1)
            self.actor_obs_history[:, -1, :] = actor_obs
            actor_history_obs = self.actor_obs_history.reshape(self.num_envs, -1)
        else:
            actor_history_obs = actor_obs
        self.obs_buf = actor_history_obs
        if self.critic_obs_history is not None:
            self.critic_obs_history = torch.roll(self.critic_obs_history, shifts=-1, dims=1)
            self.critic_obs_history[:, -1, :] = privileged_obs
            self.privileged_obs_buf = self.critic_obs_history.reshape(self.num_envs, -1)
        else:
            self.privileged_obs_buf = privileged_obs
        return

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.rpy[:, :2]), dim=1) * 10.0)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20.0)
        return (quat_mismatch + orientation) / 2.0

    def _reward_joint_pos(self):
        diff = self.dof_pos - self.ref_dof_pos
        diff_norm = torch.norm(diff, dim=1)
        return torch.exp(-2.0 * diff_norm) - 0.2 * diff_norm.clamp(0.0, 0.5)

    def _reward_feet_distance(self):
        foot_dist = torch.norm(self.feet_pos[:, 0, :2] - self.feet_pos[:, 1, :2], dim=1)
        min_dist = float(getattr(self.cfg.rewards, "min_dist", 0.2))
        max_dist = float(getattr(self.cfg.rewards, "max_dist", 0.5))
        d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_dist, 0.0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0

    def _reward_knee_distance(self):
        knee_dist = torch.norm(self.knee_pos[:, 0, :2] - self.knee_pos[:, 1, :2], dim=1)
        min_dist = float(getattr(self.cfg.rewards, "min_dist", 0.2))
        max_dist = float(getattr(self.cfg.rewards, "max_dist", 0.5)) / 2.0
        d_min = torch.clamp(knee_dist - min_dist, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - max_dist, 0.0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.feet_vel[:, :, :2], dim=2)
        reward = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(reward, dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_humanoid_gym_gait_phase()
        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0.0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        return torch.sum(air_time, dim=1)

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_humanoid_gym_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0.0, 50.0)
        return torch.exp(-yaw_roll * 100.0) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        stance_mask = self._get_humanoid_gym_gait_phase().float()
        measured_heights = torch.sum(self.feet_pos[:, :, 2] * stance_mask, dim=1) / torch.clamp(
            torch.sum(stance_mask, dim=1), min=1.0
        )
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100.0)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        return torch.exp(-torch.norm(root_acc, dim=1) * 3.0)

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10.0)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)
        return (lin_mismatch + ang_mismatch) / 2.0

    def _reward_track_vel_hard(self):
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10.0)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10.0)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        feet_z = self.feet_pos[:, :, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z.clone()
        swing_mask = (~self._get_humanoid_gym_gait_phase()).float()
        target_feet_height = float(getattr(self.cfg.rewards, "target_feet_height", 0.06))
        reward = (torch.abs(self.feet_height - target_feet_height) < 0.01).float()
        reward = torch.sum(reward * swing_mask, dim=1)
        self.feet_height *= (~contact).float()
        return reward

    def _reward_low_speed(self):
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2.0 * self.last_actions), dim=1
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
