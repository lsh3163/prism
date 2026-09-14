# SPDX-License-Identifier: BSD-3-Clause
# Derived from the local Unitree RL Gym experiment scripts; see SOURCE_MANIFEST.json.
# Upstream notices are retained in LICENSE.unitree and LICENSE.rsl-rl.
"""G1 evaluation with equal episode quotas and explicit historical compatibility."""

import copy
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import isaacgym  # noqa: F401  # Required before torch imports.
import numpy as np
import torch
from checkpoints import actor_variant, validate_checkpoint
from evaluation_protocol import PROTOCOL_IDS, BeforeResetCapture, EpisodeCollector, episode_quotas
from isaacgym import gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from provenance import file_identity, source_snapshot
from runtime import VARIANTS, register_tasks

task_registry = register_tasks()
# Upstream env registration must precede importing utils to avoid its circular import.
from legged_gym.utils import class_to_dict, get_load_path  # noqa: E402


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": None},
        {"name": "--variant", "type": str, "default": None},
        {"name": "--model_path", "type": str, "default": None},
        {"name": "--experiment_name", "type": str, "default": None},
        {"name": "--variant_name", "type": str, "default": None},
        {"name": "--load_run", "type": str, "default": None},
        {"name": "--checkpoint", "type": int, "default": None},
        {"name": "--episodes", "type": int, "default": 200},
        {"name": "--num_envs", "type": int, "default": 100},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--evaluation_protocol", "type": str, "default": "balanced", "choices": list(PROTOCOL_IDS)},
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
    episode_quotas(args.episodes, args.num_envs, args.evaluation_protocol)
    if args.seed < 0:
        raise ValueError("Evaluation requires an explicit nonnegative seed")
    if args.variant is not None and args.variant not in VARIANTS:
        raise ValueError("Unknown recipe --variant=" + args.variant)
    if args.task is None:
        args.variant = args.variant or "prism"
        args.task = VARIANTS[args.variant][0]
    elif args.task not in {item[0] for item in VARIANTS.values()}:
        raise ValueError("Unknown released G1 task: " + args.task)
    elif args.variant is not None and VARIANTS[args.variant][0] != args.task:
        raise ValueError("--task and --variant select different G1 actor recipes")
    elif args.variant is None:
        args.variant = next(name for name, item in VARIANTS.items() if item[0] == args.task)

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
    if args.save_path is not None and (Path(args.save_path).exists() or Path(args.save_path).is_symlink()):
        raise FileExistsError("Choose a fresh evaluation --save_path: " + args.save_path)
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
    checkpoint_identity = file_identity(checkpoint_path)
    checkpoint_path = checkpoint_identity["path"]
    manifest_path = Path(checkpoint_path).parent / "prism_run_manifest.json"
    training_manifest = None
    if manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        manifest_payload = json.loads(manifest_bytes)
        if (
            not isinstance(manifest_payload, dict)
            or manifest_payload.get("schema_version") != 1
            or manifest_payload.get("record_kind") != "training_start"
        ):
            raise ValueError("Unrecognized sibling training manifest: " + str(manifest_path))
        training_seed = manifest_payload.get("seed")
        if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
            raise ValueError("Training manifest has no explicit nonnegative training seed")
        manifest_actor = manifest_payload.get("actor", {})
        if (
            not isinstance(manifest_actor, dict)
            or manifest_payload.get("task") != args.task
            or manifest_actor.get("class") != train_cfg.runner.policy_class_name
            or manifest_actor.get("actor_variant") != actor_variant(train_cfg.runner.policy_class_name)
        ):
            raise ValueError("Training manifest task/actor does not match the selected evaluation recipe")
        training_manifest = {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "payload": manifest_payload,
        }
    validate_checkpoint(Path(checkpoint_path), class_to_dict(train_cfg))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    if file_identity(checkpoint_path) != checkpoint_identity:
        raise RuntimeError("Checkpoint changed while it was being loaded")
    model = runner.alg.actor_critic
    actor_identity = {
        "class": type(model).__name__,
        "actor_variant": actor_variant(train_cfg.runner.policy_class_name),
        "actor_mean_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(("actor_encoder.", "actor."))
        ),
        "total_actor_critic_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    provenance = {
        "training_manifest": training_manifest,
        "training_seed": training_manifest["payload"].get("seed") if training_manifest else None,
        "checkpoint": checkpoint_identity,
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "evaluation_seed": args.seed,
        "evaluation_config": class_to_dict(env.cfg),
        "evaluation_arguments": {
            key: str(value) if key == "physics_engine" else value for key, value in vars(args).items()
        },
        "actor": actor_identity,
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "isaacgym": str(getattr(isaacgym, "__version__", "not exposed")),
            "torch_cuda": torch.version.cuda,
            "device": str(env.device),
        },
        "source": source_snapshot(Path(__file__).parent, dict(sys.modules)),
    }

    return {
        "task": args.task,
        "checkpoint_path": checkpoint_path,
        "env": env,
        "runner": runner,
        "policy": policy,
        "actor_variant": actor_variant(train_cfg.runner.policy_class_name),
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "provenance": provenance,
    }


def evaluate_bundle(bundle, num_episodes, evaluation_protocol="balanced"):
    env = bundle["env"]
    policy = bundle["policy"]
    obs = env.get_observations()
    collector = EpisodeCollector(
        env.num_envs, num_episodes, int(env.max_episode_length), evaluation_protocol, float(env.dt)
    )
    initial_episode_steps = env.episode_length_buf.detach().cpu().tolist()

    cur_return = torch.zeros(env.num_envs, device=env.device)
    cur_length = torch.zeros(env.num_envs, device=env.device)
    cur_lin_error_sum = torch.zeros(env.num_envs, device=env.device)
    cur_yaw_error_sum = torch.zeros(env.num_envs, device=env.device)

    def capture_step_metrics():
        # Called for all environments after reward/termination computation and
        # before reset_idx changes commands, velocities, or episode bookkeeping.
        return (
            torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1),
            torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2]),
            env.time_out_buf.clone(),
        )

    step = 0
    max_steps = (max(collector.quotas) if collector.quotas else num_episodes) * (
        int(env.max_episode_length) + 1
    )
    with BeforeResetCapture(env, capture_step_metrics) as capture:
        while not collector.complete:
            if step >= max_steps:
                raise RuntimeError("Environment did not complete the requested episodes within its horizon")
            step += 1
            capture.begin_step()
            with torch.no_grad():
                actions = policy(obs.detach())
                obs, _, rewards, dones, _ = env.step(actions.detach())
            lin_error, yaw_error, timeouts = capture.snapshot()
            if evaluation_protocol == "legacy-pooled":
                lin_error = torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1)
                yaw_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])

            cur_return += rewards
            cur_length += 1
            cur_lin_error_sum += lin_error
            cur_yaw_error_sum += yaw_error
            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
            # Process the complete batch, including terminal events after the
            # legacy pool fills; the collector records whether each was counted.
            for idx in done_ids.tolist():
                episode_length = int(cur_length[idx].item())
                collector.add_episode(
                    env_id=idx,
                    completion_step=step,
                    length=episode_length,
                    episode_return=float(cur_return[idx].item()),
                    avg_lin_vel_error=float((cur_lin_error_sum[idx] / episode_length).item()),
                    avg_yaw_vel_error=float((cur_yaw_error_sum[idx] / episode_length).item()),
                    timeout=bool(timeouts[idx].item()),
                )
            cur_return[done_ids] = 0
            cur_length[done_ids] = 0
            cur_lin_error_sum[done_ids] = 0
            cur_yaw_error_sum[done_ids] = 0

    result = collector.summary()
    result.update(
        {
            "task": bundle["task"],
            "checkpoint_path": bundle["checkpoint_path"],
            "actor_variant": bundle.get("actor_variant"),
            "checkpoint_sha256": bundle.get("checkpoint_sha256"),
            "checkpoint_iteration": bundle.get("checkpoint_iteration"),
            "evaluation_steps": step,
            "initial_env_episode_steps": initial_episode_steps,
        }
    )
    return result


def print_summary(args, result):
    print("\n========== Evaluation Results ==========")
    print(f"Variant:                {args.variant_name or 'unnamed'}")
    print(f"Task:                   {result['task']}")
    print(f"Checkpoint:             {result['checkpoint_path']}")
    print(f"Condition:              {args.condition_name}")
    print(f"Protocol:               {result['protocol_id']}")
    print(f"Episodes:               {result['num_episodes']}")
    print(f"Average return:         {result['avg_return']:.4f}")
    print(f"Average episode length: {result['avg_episode_length']:.4f}")
    print(f"Average lin vel error:  {result['avg_lin_vel_error']:.4f}")
    print(f"Average yaw vel error:  {result['avg_yaw_vel_error']:.4f}")
    print(f"Success rate (%):       {100.0 * result['success_rate']:.2f}")
    print("========================================")


def maybe_save_results(args, result, bundle=None):
    if args.save_path is None:
        return

    payload = {
        "schema_version": 2,
        "metadata": {
            **(bundle.get("provenance", {}) if bundle else {}),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "variant_name": args.variant_name,
            "recipe_variant": getattr(args, "variant", None),
            "actor_variant": result.get("actor_variant"),
            "condition_name": args.condition_name,
            "terrain_mode": args.terrain_mode,
            "push_mode": args.push_mode,
            "push_interval_s": args.push_interval_s,
            "max_push_vel_xy": args.max_push_vel_xy,
            "seed": args.seed,
            "evaluation_seed": args.seed,
            "episodes": args.episodes,
            "num_envs": result["num_envs"],
            "protocol_id": result["protocol_id"],
            "evaluation_protocol": result["protocol"],
        },
        "result": result,
    }

    save_path = os.path.abspath(args.save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
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
    try:
        result = evaluate_bundle(bundle, args.episodes, args.evaluation_protocol)
        print_summary(args, result)
        maybe_save_results(args, result, bundle=bundle)
    finally:
        close_bundle(bundle)


if __name__ == "__main__":
    main(get_eval_args())
