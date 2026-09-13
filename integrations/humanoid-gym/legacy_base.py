# SPDX-License-Identifier: BSD-3-Clause
# Derived from Unitree RL Gym and ETH Zurich / NVIDIA legged-gym code.
# See LICENSE.unitree and LICENSE.rsl-rl for retained upstream notices.
# See SOURCE_MANIFEST.json for exact source provenance and extraction scope.
"""Only base-environment overrides needed by the G1 experiment.

The remaining physics implementation and robot assets come from pinned Unitree RL Gym.
"""

import time
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, to_torch, get_axis_params
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class HumanoidLeggedRobot(LeggedRobot):
    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        clipped_actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.actions = self._apply_action_delay(clipped_actions)
        action_noise = float(getattr(self.cfg.domain_rand, "action_noise", 0.0))
        if action_noise > 0.0:
            self.actions = self.actions + action_noise * torch.randn_like(self.actions) * self.actions
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras)

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
        self.compute_observations()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_action_delay(env_ids)
        self._resample_pd_gains(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if hasattr(self, "rand_push_force"):
            self.rand_push_force[env_ids] = 0.0
        if hasattr(self, "rand_push_torque"):
            self.rand_push_torque[env_ids] = 0.0
        if hasattr(self, "action_history"):
            self.action_history[env_ids] = 0.0
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,), dtype=torch.long)
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = float(self.friction_coeffs[env_id, 0])
        elif env_id == 0 and (not hasattr(self, "friction_coeffs")):
            self.friction_coeffs = torch.full((self.num_envs, 1), float(props[0].friction), device="cpu")
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            self._base_masses_cpu = np.zeros((self.num_envs, 1), dtype=np.float32)
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if False and hasattr(props[0], "com"):
            rng = self.cfg.domain_rand.com_displacement_range
            props[0].com.x += np.random.uniform(rng[0][0], rng[0][1])
            props[0].com.y += np.random.uniform(rng[1][0], rng[1][1])
            props[0].com.z += np.random.uniform(rng[2][0], rng[2][1])
        self._base_masses_cpu[env_id, 0] = props[0].mass
        return props

    def _apply_action_delay(self, actions):
        if not getattr(self.cfg.domain_rand, "action_delay", False):
            return actions
        delay_mode = getattr(self.cfg.domain_rand, "action_delay_mode", "blend")
        if delay_mode == "discrete":
            if self.action_history.shape[1] == 1:
                return actions
            self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
            self.action_history[:, 0, :] = actions
            env_ids = torch.arange(self.num_envs, device=self.device)
            return self.action_history[env_ids, self.action_delay_steps.squeeze(-1)]
        delay = self.action_delay_fractions
        return (1.0 - delay) * actions + delay * self.last_actions

    def _resample_action_delay(self, env_ids):
        if len(env_ids) == 0:
            return
        if not getattr(self.cfg.domain_rand, "action_delay", False):
            self.action_delay_fractions[env_ids] = 0.0
            if hasattr(self, "action_delay_steps"):
                self.action_delay_steps[env_ids] = 0
            return
        delay_mode = getattr(self.cfg.domain_rand, "action_delay_mode", "blend")
        if delay_mode == "discrete":
            step_range = getattr(self.cfg.domain_rand, "action_delay_step_range", [0, 0])
            self.action_delay_steps[env_ids] = torch.randint(
                int(step_range[0]),
                int(step_range[1]) + 1,
                (len(env_ids), 1),
                device=self.device,
                dtype=torch.long,
            )
            self.action_delay_fractions[env_ids] = 0.0
            return
        delay_range = self.cfg.domain_rand.action_delay_range
        self.action_delay_fractions[env_ids] = torch_rand_float(
            delay_range[0], delay_range[1], (len(env_ids), 1), device=self.device
        )
        if hasattr(self, "action_delay_steps"):
            self.action_delay_steps[env_ids] = 0

    def _resample_pd_gains(self, env_ids):
        if len(env_ids) == 0:
            return
        self.p_gains[env_ids] = self.base_p_gains[env_ids]
        self.d_gains[env_ids] = self.base_d_gains[env_ids]
        if not getattr(self.cfg.domain_rand, "randomize_pd_gains", False):
            return
        kp_range = self.cfg.domain_rand.stiffness_multiplier_range
        kd_range = self.cfg.domain_rand.damping_multiplier_range
        kp_scale = torch_rand_float(kp_range[0], kp_range[1], (len(env_ids), 1), device=self.device)
        kd_scale = torch_rand_float(kd_range[0], kd_range[1], (len(env_ids), 1), device=self.device)
        self.p_gains[env_ids] *= kp_scale
        self.d_gains[env_ids] *= kd_scale

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        target_dof_pos = self.default_dof_pos.expand_as(actions_scaled)
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + target_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[
            self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0
        ]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_ang_vel = float(getattr(self.cfg.domain_rand, "max_push_ang_vel", 0.0))
        self.rand_push_force[:] = 0.0
        self.rand_push_torque[:] = 0.0
        self.rand_push_force[push_env_ids, :2] = torch_rand_float(
            -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
        )
        self.root_states[push_env_ids, 7:9] = self.rand_push_force[push_env_ids, :2]
        if max_ang_vel > 0.0:
            self.rand_push_torque[push_env_ids] = torch_rand_float(
                -max_ang_vel, max_ang_vel, (len(push_env_ids), 3), device=self.device
            )
            self.root_states[push_env_ids, 10:13] = self.rand_push_torque[push_env_ids]
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[: self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.action_delay_fractions = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.action_delay_steps = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        max_delay_step = int(max(getattr(self.cfg.domain_rand, "action_delay_step_range", [0, 0])))
        self.action_history = torch.zeros(
            self.num_envs,
            max_delay_step + 1,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.rand_push_force = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.rand_push_torque = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if hasattr(self, "friction_coeffs"):
            self.friction_coeffs = self.friction_coeffs.to(self.device, dtype=torch.float)
        else:
            default_friction = float(getattr(self.cfg.terrain, "static_friction", 1.0))
            self.friction_coeffs = torch.full(
                (self.num_envs, 1), default_friction, dtype=torch.float, device=self.device
            )
        if hasattr(self, "_base_masses_cpu"):
            self.base_masses = torch.tensor(self._base_masses_cpu, dtype=torch.float, device=self.device)
        else:
            self.base_masses = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.base_p_gains = self.p_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.base_d_gains = self.d_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.p_gains = self.base_p_gains.clone()
        self.d_gains = self.base_d_gains.clone()
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_action_delay(all_env_ids)
        self._resample_pd_gains(all_env_ids)
