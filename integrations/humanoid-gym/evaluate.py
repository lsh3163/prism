# SPDX-License-Identifier: BSD-3-Clause
# Derived from the local Unitree RL Gym experiment scripts; see SOURCE_MANIFEST.json.
# Upstream notices are retained in LICENSE.unitree and LICENSE.rsl-rl.
"""Historical nominal/plane evaluation; protocol mapping to paper tables is unresolved."""

import copy
import json
import os

import isaacgym  # noqa: F401  # Required before torch imports.
import numpy as np
import torch
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from runtime import register_tasks

task_registry = register_tasks()
# Upstream env registration must precede importing utils to avoid its circular import.
from legged_gym.utils import get_load_path  # noqa: E402


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1_humanoidgym_ppo"},
        {"name": "--model_path", "type": str, "default": None},
        {"name": "--experiment_name", "type": str, "default": None},
        {"name": "--variant_name", "type": str, "default": None},
        {"name": "--load_run", "type": str, "default": None},
        {"name": "--checkpoint", "type": int, "default": None},
        {"name": "--episodes", "type": int, "default": 200},
        {"name": "--num_envs", "type": int, "default": 100},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--condition_name", "type": str, "default": "match_nopush"},
        {"name": "--terrain_mode", "type": str, "default": "match"},
        {"name": "--push_mode", "type": str, "default": "off"},
        {"name": "--push_interval_s", "type": float, "default": None},
        {"name": "--max_push_vel_xy", "type": float, "default": None},
        {"name": "--save_path", "type": str, "default": None},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate a trained Humanoid-Gym policy.", custom_parameters=custom_parameters
    )

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"

    args.max_iterations = None
    args.resume = False
    args.run_name = None

    return args


def override_eval_cfg(env_cfg, num_envs):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.action_noise = 0.0
    env_cfg.env.test = True

    return env_cfg


def apply_eval_condition_overrides(env_cfg, args):
    terrain_mode = args.terrain_mode.lower()
    if terrain_mode == "plane":
        env_cfg.terrain.mesh_type = "plane"
    elif terrain_mode == "rough":
        env_cfg.terrain.mesh_type = "trimesh"
        if hasattr(env_cfg.terrain, "flat_terrain_probability"):
            env_cfg.terrain.flat_terrain_probability = 0.0
    elif terrain_mode != "match":
        raise ValueError(f"Unsupported terrain_mode: {args.terrain_mode}")

    push_mode = args.push_mode.lower()
    if push_mode == "off":
        env_cfg.domain_rand.push_robots = False
    elif push_mode == "train":
        env_cfg.domain_rand.push_robots = True
    elif push_mode == "strong":
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.push_interval_s = 2.0
    else:
        raise ValueError(f"Unsupported push_mode: {args.push_mode}")

    if args.push_interval_s is not None:
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.push_interval_s = args.push_interval_s
    if args.max_push_vel_xy is not None:
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.max_push_vel_xy = args.max_push_vel_xy

    return env_cfg


def resolve_checkpoint_path(
    task, train_cfg, model_path=None, experiment_name=None, load_run=None, checkpoint=None
):
    if model_path is not None:
        return os.path.abspath(model_path)

    experiment_name = experiment_name or train_cfg.runner.experiment_name
    load_run = -1 if load_run is None else load_run
    checkpoint = -1 if checkpoint is None else checkpoint
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def load_eval_bundle(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)

    env_cfg = override_eval_cfg(env_cfg, args.num_envs)
    env_cfg = apply_eval_condition_overrides(env_cfg, args)
    if env_cfg.terrain.mesh_type != "plane":
        raise ValueError("This extracted experiment currently supports the paper plane task only.")
    if getattr(args, "enable_camera_sensors", False):
        env_cfg.env.enable_camera_sensors = True
    env_cfg.seed = train_cfg.seed = args.seed
    checkpoint_path = resolve_checkpoint_path(
        task=args.task,
        train_cfg=train_cfg,
        model_path=args.model_path,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    return {
        "task": args.task,
        "checkpoint_path": checkpoint_path,
        "env": env,
        "runner": runner,
        "policy": policy,
    }


def evaluate_bundle(bundle, num_episodes):
    env = bundle["env"]
    policy = bundle["policy"]
    obs = env.get_observations()

    cur_return = torch.zeros(env.num_envs, device=env.device)
    cur_length = torch.zeros(env.num_envs, device=env.device)
    cur_lin_error_sum = torch.zeros(env.num_envs, device=env.device)
    cur_yaw_error_sum = torch.zeros(env.num_envs, device=env.device)

    episode_returns = []
    episode_lengths = []
    episode_lin_errors = []
    episode_yaw_errors = []
    episode_success = []

    while len(episode_returns) < num_episodes:
        with torch.no_grad():
            actions = policy(obs.detach())
            obs, _, rewards, dones, _ = env.step(actions.detach())

        lin_error = torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1)
        yaw_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])

        cur_return += rewards
        cur_length += 1
        cur_lin_error_sum += lin_error
        cur_yaw_error_sum += yaw_error

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            continue

        for idx in done_ids.tolist():
            episode_length = cur_length[idx].item()
            episode_returns.append(cur_return[idx].item())
            episode_lengths.append(episode_length)
            episode_lin_errors.append((cur_lin_error_sum[idx] / max(cur_length[idx], 1)).item())
            episode_yaw_errors.append((cur_yaw_error_sum[idx] / max(cur_length[idx], 1)).item())
            episode_success.append(float(episode_length >= env.max_episode_length - 1))
            if len(episode_returns) >= num_episodes:
                break

        cur_return[done_ids] = 0
        cur_length[done_ids] = 0
        cur_lin_error_sum[done_ids] = 0
        cur_yaw_error_sum[done_ids] = 0

    return {
        "task": bundle["task"],
        "checkpoint_path": bundle["checkpoint_path"],
        "num_episodes": len(episode_returns),
        "avg_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "avg_lin_vel_error": float(np.mean(episode_lin_errors)),
        "std_lin_vel_error": float(np.std(episode_lin_errors)),
        "avg_yaw_vel_error": float(np.mean(episode_yaw_errors)),
        "std_yaw_vel_error": float(np.std(episode_yaw_errors)),
        "success_rate": float(np.mean(episode_success)),
    }


def print_summary(args, result):
    print("\n========== Evaluation Results ==========")
    print(f"Variant:                {args.variant_name or 'unnamed'}")
    print(f"Task:                   {result['task']}")
    print(f"Checkpoint:             {result['checkpoint_path']}")
    print(f"Condition:              {args.condition_name}")
    print(f"Episodes:               {result['num_episodes']}")
    print(f"Average return:         {result['avg_return']:.4f}")
    print(f"Average episode length: {result['avg_episode_length']:.4f}")
    print(f"Average lin vel error:  {result['avg_lin_vel_error']:.4f}")
    print(f"Average yaw vel error:  {result['avg_yaw_vel_error']:.4f}")
    print(f"Success rate (%):       {100.0 * result['success_rate']:.2f}")
    print("========================================")


def maybe_save_results(args, result):
    if args.save_path is None:
        return

    payload = {
        "metadata": {
            "variant_name": args.variant_name,
            "condition_name": args.condition_name,
            "terrain_mode": args.terrain_mode,
            "push_mode": args.push_mode,
            "push_interval_s": args.push_interval_s,
            "max_push_vel_xy": args.max_push_vel_xy,
            "seed": args.seed,
            "episodes": args.episodes,
            "num_envs": args.num_envs,
        },
        "result": result,
    }

    save_path = os.path.abspath(args.save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved evaluation results to: {save_path}")


def close_bundle(bundle):
    env = bundle.get("env")
    if env is None:
        return

    viewer = getattr(env, "viewer", None)
    sim = getattr(env, "sim", None)
    gym = getattr(env, "gym", None)

    if gym is not None and viewer is not None:
        gym.destroy_viewer(viewer)
    if gym is not None and sim is not None:
        gym.destroy_sim(sim)


def main(args):
    bundle = load_eval_bundle(args)
    result = evaluate_bundle(bundle, args.episodes)
    print_summary(args, result)
    maybe_save_results(args, result)
    close_bundle(bundle)


if __name__ == "__main__":
    main(get_eval_args())
