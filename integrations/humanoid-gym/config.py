# SPDX-License-Identifier: BSD-3-Clause
# Derived from Unitree RL Gym and ETH Zurich / NVIDIA legged-gym code.
# See LICENSE.unitree and LICENSE.rsl-rl for retained upstream notices.
# See SOURCE_MANIFEST.json for exact source provenance and extraction scope.
"""Paper G1 configurations; simulator-only, with inactive sensor settings removed."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0,
            "torso_joint": 0.0,
        }

    class env(LeggedRobotCfg.env):
        raw_observation_dim = 47
        lifted_observation_dim = 24
        use_feature_lifting = False
        num_observations = raw_observation_dim
        num_privileged_obs = 50
        num_actions = 12

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40}
        damping = {"hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf"
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-07
            dof_vel = -0.001
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18


class G1HumanoidGymCfg(G1RoughCfg):
    class env(G1RoughCfg.env):
        humanoid_gym_style = True
        actor_history_length = 15
        critic_history_length = 3
        num_single_obs = 47
        single_num_privileged_obs = 73
        raw_observation_dim = num_single_obs
        lifted_observation_dim = 0
        use_feature_lifting = False
        num_observations = actor_history_length * num_single_obs
        num_privileged_obs = critic_history_length * single_num_privileged_obs
        num_actions = 12
        episode_length_s = 24
        use_ref_actions = False

    class asset(G1RoughCfg.asset):
        knee_names = ["left_knee_link", "right_knee_link"]
        penalize_contacts_on = ["pelvis"]
        terminate_after_contacts_on = ["pelvis"]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1]
        restitution = 0.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.0
        heading_command = True

        class ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class control(G1RoughCfg.control):
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        action_delay = True
        action_delay_mode = "blend"
        action_delay_range = [0.0, 0.5]
        action_delay_step_range = [0, 0]
        action_noise = 0.02

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06
        cycle_time = 0.64
        only_positive_rewards = True
        tracking_sigma = 5.0
        max_contact_force = 700.0

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 1.0
            torques = -1e-05
            dof_vel = -0.0005
            dof_acc = -1e-07
            base_height = 0.2
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0
            alive = 0.0
            hip_pos = 0.0
            contact_no_vel = 0.0
            feet_swing_height = 0.0
            contact = 0.0
            feet_contact_forces = -0.01
            joint_pos = 1.6
            feet_clearance = 1.0
            feet_contact_number = 1.2
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5
            default_joint_pos = 0.5
            base_acc = 0.2
            action_smoothness = -0.002

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.6

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 0.5
            lin_vel = 0.05
            ang_vel = 0.1
            quat = 0.03
            gravity = 0.0
            height_measurements = 0.1


class G1HumanoidGymCfgPPO(LeggedRobotCfgPPO):
    seed = 5

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-05
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60
        max_iterations = 3001
        save_interval = 100
        experiment_name = "g1_humanoidgym_ppo"
        run_name = ""


class G1HumanoidGymCfgPPOParamMatched(G1HumanoidGymCfgPPO):
    class policy(G1HumanoidGymCfgPPO.policy):
        actor_hidden_dims = [816, 352, 160]

    class runner(G1HumanoidGymCfgPPO.runner):
        policy_class_name = "ActorCritic"
        experiment_name = "g1_humanoidgym_ppo_parammatch"


class G1HumanoidGymCfgPPOPoly(G1HumanoidGymCfgPPO):
    class policy(G1HumanoidGymCfgPPO.policy):
        poly_hidden_dim = 256
        poly_degree = 2
        actor_use_poly = True
        critic_use_poly = False
        actor_poly_mode = "residual"
        actor_use_input_layer_norm = False
        critic_use_input_layer_norm = False
        actor_use_hidden_layer_norm = False
        critic_use_hidden_layer_norm = False
        actor_output_tanh = False

    class runner(G1HumanoidGymCfgPPO.runner):
        policy_class_name = "PolyActorCritic"
        experiment_name = "g1_humanoidgym_ppo_poly"


class G1HumanoidGymCfgPPOPolyD1(G1HumanoidGymCfgPPOPoly):
    class policy(G1HumanoidGymCfgPPOPoly.policy):
        poly_degree = 1

    class runner(G1HumanoidGymCfgPPOPoly.runner):
        policy_class_name = "PolyActorCritic"
        experiment_name = "g1_humanoidgym_ppo_poly_d1"


class G1HumanoidGymCfgPPOPolyD3(G1HumanoidGymCfgPPOPoly):
    class policy(G1HumanoidGymCfgPPOPoly.policy):
        poly_degree = 3

    class runner(G1HumanoidGymCfgPPOPoly.runner):
        policy_class_name = "PolyActorCritic"
        experiment_name = "g1_humanoidgym_ppo_poly_d3"


class G1HumanoidGymCfgPPOPolyWarmup(G1HumanoidGymCfgPPOPoly):
    class policy(G1HumanoidGymCfgPPOPoly.policy):
        actor_poly_warmup_updates = 500

    class runner(G1HumanoidGymCfgPPOPoly.runner):
        policy_class_name = "PolyActorCritic"
        experiment_name = "g1_humanoidgym_ppo_poly_warmup"
