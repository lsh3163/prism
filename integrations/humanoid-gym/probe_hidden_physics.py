# SPDX-License-Identifier: BSD-3-Clause
# Derived from the local Unitree RL Gym experiment scripts; see SOURCE_MANIFEST.json.
# Upstream notices are retained in LICENSE.unitree and LICENSE.rsl-rl.
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from runtime import VARIANTS, register_tasks
from actor import ActorCritic, PolyActorCritic
from gated_actor import GatedPolyActorCritic
from checkpoints import validate_checkpoint

task_registry = register_tasks()
from legged_gym.utils import class_to_dict  # noqa: E402


METHODS = {
    "MLP": {
        "kind": "mlp",
        "path": None,
    },
    "PRISM": {
        "kind": "gated",
        "path": None,
    },
}


def make_args(num_envs, seed, sim_device, rl_device, headless=True):
    device_type, _, device_id = sim_device.partition(":")
    compute_device_id = int(device_id or 0) if device_type == "cuda" else 0
    return SimpleNamespace(
        seed=seed,
        num_envs=num_envs,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        physics_engine=gymapi.SIM_PHYSX,
        device=sim_device,
        sim_device=sim_device,
        sim_device_type=device_type,
        sim_device_id=compute_device_id,
        compute_device_id=compute_device_id,
        use_gpu=(device_type == "cuda"),
        use_gpu_pipeline=(device_type == "cuda"),
        subscenes=0,
        num_threads=0,
        headless=headless,
        rl_device=rl_device,
    )


def override_probe_cfg(env_cfg, num_envs):
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 1.0
    env_cfg.domain_rand.max_push_vel_xy = 0.2
    env_cfg.domain_rand.max_push_ang_vel = 0.4
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.action_noise = 0.0
    env_cfg.env.test = False
    return env_cfg


def load_model(method, root, device):
    if method["kind"] == "mlp":
        model = ActorCritic(
            num_actor_obs=705,
            num_critic_obs=219,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[768, 256, 128],
            activation="elu",
            init_noise_std=1.0,
        ).to(device)
    elif method["kind"] == "gated":
        model = GatedPolyActorCritic(705, 219, 12).to(device)
    elif method["kind"] == "respoly":
        model = PolyActorCritic(
            num_actor_obs=705,
            num_critic_obs=219,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[768, 256, 128],
            activation="elu",
            init_noise_std=1.0,
            poly_hidden_dim=256,
            poly_degree=2,
            actor_use_poly=True,
            critic_use_poly=False,
            actor_poly_mode="residual",
            actor_use_input_layer_norm=False,
            critic_use_input_layer_norm=False,
            actor_use_hidden_layer_norm=False,
            critic_use_hidden_layer_norm=False,
            actor_output_tanh=False,
            actor_poly_warmup_updates=500,
        ).to(device)
    else:
        raise ValueError(method["kind"])

    ckpt = torch.load(root / method["path"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    if method["kind"] == "respoly":
        model.actor_encoder.set_poly_scale(1.0)
    return model


def actor_action(model, obs):
    if hasattr(model, "actor_encoder"):
        return model.actor(model.actor_encoder(obs))
    return model.actor(obs)


PRISM_FEATURE = "combined"


def actor_latent(model, obs):
    if hasattr(model, "actor_encoder"):
        enc = model.actor_encoder
        if PRISM_FEATURE == "combined":
            return enc(obs)
        if PRISM_FEATURE == "final_hidden":
            encoded = enc(obs)
            layers = model.actor.net if hasattr(model.actor, "net") else model.actor
            return layers[:-1](encoded)
        if isinstance(model, GatedPolyActorCritic):
            return enc(obs) if PRISM_FEATURE == "encoder" else enc.polynomial_features(obs)
        x = enc.input_layer_norm(obs)
        if PRISM_FEATURE == "encoder":
            return enc(obs)
        h = enc.poly_in_proj(x)
        poly = enc.A[0](h)
        for i in range(1, enc.degree):
            poly = poly + enc.A[i](h) * enc.S[i - 1](poly)
        poly = enc.poly_out_proj(poly)
        return enc.activation(poly)
    # Final hidden activation of the MLP actor before the action layer.
    return model.actor[:-1](obs)


def target_snapshot(env):
    friction = env.friction_coeffs.to(env.device).float()
    mass = (env.base_masses / 30.0).float()
    push = torch.norm(env.rand_push_force[:, :2], dim=1, keepdim=True).float()
    contact = (env.contact_forces[:, env.feet_indices, 2] > 5.0).float()
    return {
        "friction": friction,
        "push force": push,
        "base mass": mass,
        "foot contact": contact,
    }


def stable_snapshot(env, min_height, max_roll, max_pitch, min_episode_steps):
    """Mask out collapse/reset states so probes cannot exploit obvious falls."""
    height_ok = env.root_states[:, 2] > min_height
    roll_ok = torch.abs(env.rpy[:, 0]) < max_roll
    pitch_ok = torch.abs(env.rpy[:, 1]) < max_pitch
    age_ok = env.episode_length_buf >= min_episode_steps
    alive_ok = env.reset_buf == 0
    return (height_ok & roll_ok & pitch_ok & age_ok & alive_ok).detach()


def collect_dataset(
    env,
    collector_model,
    samples,
    warmup_steps,
    stable_only=False,
    min_height=0.55,
    max_roll=0.35,
    max_pitch=0.35,
    min_episode_steps=10,
):
    obs = env.get_observations()
    obs_blocks = []
    mask_blocks = []
    targets = {name: [] for name in ["friction", "push force", "base mass", "foot contact"]}
    total = 0
    stable_total = 0
    max_raw_samples = max(samples * 20, samples + env.num_envs * 100)
    with torch.no_grad():
        for _ in range(warmup_steps):
            actions = actor_action(collector_model, obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

        while (stable_total if stable_only else total) < samples and total < max_raw_samples:
            stable_mask = stable_snapshot(env, min_height, max_roll, max_pitch, min_episode_steps)
            obs_blocks.append(obs.detach().cpu())
            mask_blocks.append(stable_mask.detach().cpu())
            snap = target_snapshot(env)
            for name, value in snap.items():
                targets[name].append(value.detach().cpu())
            total += obs.shape[0]
            stable_total += int(stable_mask.sum().item())
            actions = actor_action(collector_model, obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

    observations_t = torch.cat(obs_blocks, dim=0)
    out_targets_t = {name: torch.cat(parts, dim=0) for name, parts in targets.items()}
    stable_mask_t = torch.cat(mask_blocks, dim=0)
    if stable_only:
        keep = stable_mask_t.nonzero(as_tuple=False).flatten()[:samples]
        if keep.numel() < samples:
            raise RuntimeError(f"Only collected {keep.numel()} stable samples out of requested {samples}.")
        observations_t = observations_t[keep]
        out_targets_t = {name: values[keep] for name, values in out_targets_t.items()}
    else:
        observations_t = observations_t[:samples]
        out_targets_t = {name: values[:samples] for name, values in out_targets_t.items()}

    observations = observations_t.numpy().astype(np.float32)
    out_targets = {name: values.numpy().astype(np.float32) for name, values in out_targets_t.items()}
    return observations, out_targets


def featurize_observations(model, observations, device, batch_size=4096):
    feats = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            batch = torch.from_numpy(observations[start : start + batch_size]).to(device)
            feats.append(actor_latent(model, batch).detach().cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


def standardize(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + eps
    return (train - mean) / std, (test - mean) / std


def ridge_probe(x, y, seed, ridge=1e-3):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test = standardize(x_train, x_test)
    y_train, y_test = standardize(y_train, y_test)

    x_train_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    x_test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=x_test.dtype)], axis=1)
    eye = np.eye(x_train_aug.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(x_train_aug.T @ x_train_aug + ridge * eye, x_train_aug.T @ y_train)
    pred = x_test_aug @ weights
    mse = float(np.mean((pred - y_test) ** 2))

    corrs = []
    for dim in range(y_test.shape[1]):
        if np.std(y_test[:, dim]) < 1e-6 or np.std(pred[:, dim]) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(pred[:, dim], y_test[:, dim])[0, 1]))
    corr = float(np.mean(corrs)) if corrs else 0.0
    return {"mse": mse, "r": corr}


def main():
    global PRISM_FEATURE
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="g1_humanoidgym_ppo")
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/humanoid/hidden_physics_probe.json"))
    parser.add_argument("--mlp-path", type=Path, required=True)
    parser.add_argument("--prism-path", type=Path, required=True)
    parser.add_argument("--prism-variant", choices=("prism", "legacy-prism"), default="prism")
    parser.add_argument(
        "--stable-only", action="store_true", help="Probe only upright, non-reset walking states."
    )
    parser.add_argument("--stable-min-height", type=float, default=0.55)
    parser.add_argument("--stable-max-roll", type=float, default=0.35)
    parser.add_argument("--stable-max-pitch", type=float, default=0.35)
    parser.add_argument("--stable-min-episode-steps", type=int, default=10)
    parser.add_argument(
        "--prism-feature",
        choices=["combined", "poly", "encoder", "final_hidden"],
        default="combined",
        help="Representation used for the PRISM probe.",
    )
    args = parser.parse_args()
    METHODS["PRISM"]["kind"] = "gated" if args.prism_variant == "prism" else "respoly"
    _, prism_cfg = task_registry.get_cfgs(VARIANTS[args.prism_variant][0])
    selected_actor = validate_checkpoint(args.prism_path, class_to_dict(prism_cfg))
    PRISM_FEATURE = args.prism_feature
    if args.mlp_path is not None:
        METHODS["MLP"]["path"] = args.mlp_path
    if args.prism_path is not None:
        METHODS["PRISM"]["path"] = args.prism_path

    root = Path.cwd()
    device = torch.device(args.rl_device if torch.cuda.is_available() else "cpu")
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = override_probe_cfg(copy.deepcopy(env_cfg), args.num_envs)
    env_cfg.seed = args.seed
    env_args = make_args(args.num_envs, args.seed, args.sim_device, args.rl_device, headless=True)

    results = {}
    metadata = {
        "task": args.task,
        "actor_variant": selected_actor,
        "prism_recipe": args.prism_variant,
        "samples": args.samples,
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "targets": ["friction", "push force", "base mass", "foot contact"],
        "note": "Linear probes from frozen actor latents. Actor observations do not include privileged physical targets.",
        "prism_feature": args.prism_feature,
        "stable_only": args.stable_only,
        "stable_filter": {
            "min_height": args.stable_min_height,
            "max_roll": args.stable_max_roll,
            "max_pitch": args.stable_max_pitch,
            "min_episode_steps": args.stable_min_episode_steps,
        },
    }

    env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=copy.deepcopy(env_cfg))
    collector = load_model(METHODS["PRISM"], root, device)
    print("\n=== Collecting shared rollout dataset with PRISM policy ===")
    env.reset()
    observations, ys = collect_dataset(
        env,
        collector,
        args.samples,
        args.warmup_steps,
        stable_only=args.stable_only,
        min_height=args.stable_min_height,
        max_roll=args.stable_max_roll,
        max_pitch=args.stable_max_pitch,
        min_episode_steps=args.stable_min_episode_steps,
    )
    del collector

    for i, (name, method) in enumerate(METHODS.items()):
        print(f"\n=== Probing {name} on shared observations ===")
        model = load_model(method, root, device)
        x = featurize_observations(model, observations, device)
        method_results = {}
        for target_name, y in ys.items():
            method_results[target_name] = ridge_probe(x, y, seed=args.seed + 100 * i)
            print(name, target_name, method_results[target_name])
        results[name] = method_results
        del model
        torch.cuda.empty_cache()

    payload = {"metadata": metadata, "results": results}
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
