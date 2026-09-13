# SPDX-License-Identifier: BSD-3-Clause
# Derived from the local Unitree RL Gym experiment scripts; see SOURCE_MANIFEST.json.
# Upstream notices are retained in LICENSE.unitree and LICENSE.rsl-rl.
"""Probe future locomotion response quantities from frozen Humanoid-Gym actor latents."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from runtime import register_tasks
from actor import ActorCritic, PolyActorCritic

task_registry = register_tasks()


METHODS = {
    "MLP Actor Latent": {
        "kind": "mlp",
        "path": None,
    },
    "PRISM Actor Latent": {
        "kind": "respoly",
        "path": None,
    },
}
TARGET_NAMES = [
    "Future Tracking Error",
    "Future Foot Impulse",
    "Future Foot Slip",
    "Future Mechanical Power",
    "Future Orientation Error",
]


def make_args(num_envs: int, seed: int, sim_device: str, rl_device: str, headless: bool = True):
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


def override_probe_cfg(env_cfg, num_envs: int):
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


def load_model(method: dict, root: Path, device: torch.device):
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
    ckpt = torch.load(root / method["path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if hasattr(model, "actor_encoder"):
        model.actor_encoder.set_poly_scale(1.0)
    return model


def actor_action(model, obs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "actor_encoder"):
        return model.actor(model.actor_encoder(obs))
    return model.actor(obs)


def actor_latent(model, obs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "actor_encoder"):
        return model.actor_encoder(obs)
    return model.actor[:-1](obs)


def stable_mask(
    env, min_height: float, max_roll: float, max_pitch: float, min_episode_steps: int
) -> torch.Tensor:
    return (
        (env.root_states[:, 2] > min_height)
        & (torch.abs(env.rpy[:, 0]) < max_roll)
        & (torch.abs(env.rpy[:, 1]) < max_pitch)
        & (env.episode_length_buf >= min_episode_steps)
        & (env.reset_buf == 0)
    )


def snapshot(env) -> dict[str, torch.Tensor]:
    contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    foot_force = torch.norm(env.contact_forces[:, env.feet_indices, :3], dim=-1).sum(dim=-1)
    foot_slip = (torch.norm(env.feet_vel[:, :, :2], dim=-1) * contact.float()).sum(dim=-1)
    power_dims = min(env.torques.shape[1], env.dof_vel.shape[1])
    mech_power = torch.sum(torch.abs(env.torques[:, :power_dims] * env.dof_vel[:, :power_dims]), dim=-1)
    lin_err = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=-1)
    yaw_err = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    orient_err = torch.sum(torch.square(env.rpy[:, :2]), dim=-1)
    push = torch.norm(env.rand_push_force[:, :2], dim=-1)
    return {
        "tracking_error": lin_err + yaw_err,
        "foot_force": foot_force,
        "foot_slip": foot_slip,
        "mechanical_power": mech_power,
        "orientation_error": orient_err,
        "push": push,
    }


def collect_future_response_dataset(
    env,
    collector_model,
    samples: int,
    warmup_steps: int,
    horizon: int,
    min_height: float,
    max_roll: float,
    max_pitch: float,
    min_episode_steps: int,
    push_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obs = env.get_observations()
    obs_history: list[torch.Tensor] = []
    mask_history: list[torch.Tensor] = []
    state_history: dict[str, list[torch.Tensor]] = {
        key: []
        for key in [
            "tracking_error",
            "foot_force",
            "foot_slip",
            "mechanical_power",
            "orientation_error",
            "push",
        ]
    }
    total = 0
    kept = 0
    max_raw_samples = max(samples * 50, samples + env.num_envs * (horizon + 100))
    with torch.no_grad():
        for _ in range(warmup_steps):
            actions = actor_action(collector_model, obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

        while kept < samples and total < max_raw_samples:
            obs_history.append(obs.detach().cpu())
            mask_history.append(
                stable_mask(env, min_height, max_roll, max_pitch, min_episode_steps).detach().cpu()
            )
            snap = snapshot(env)
            for key, value in snap.items():
                state_history[key].append(value.detach().cpu())
            total += obs.shape[0]
            if len(obs_history) > horizon:
                future_push = torch.stack(state_history["push"][-horizon:], dim=0).amax(dim=0)
                candidate = mask_history[-horizon - 1] & (future_push > push_threshold)
                kept += int(candidate.sum().item())
            actions = actor_action(collector_model, obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

    obs_t = torch.stack(obs_history, dim=0)
    mask_t = torch.stack(mask_history, dim=0)
    states = {key: torch.stack(values, dim=0) for key, values in state_history.items()}
    obs_rows = []
    target_rows = {name: [] for name in TARGET_NAMES}
    for t in range(0, obs_t.shape[0] - horizon):
        future_push = states["push"][t + 1 : t + horizon + 1].amax(dim=0)
        keep = mask_t[t] & (future_push > push_threshold)
        if not torch.any(keep):
            continue
        obs_rows.append(obs_t[t][keep])
        target_rows["Future Tracking Error"].append(
            states["tracking_error"][t + 1 : t + horizon + 1, keep].mean(dim=0)
        )
        target_rows["Future Foot Impulse"].append(
            torch.log1p(states["foot_force"][t + 1 : t + horizon + 1, keep].sum(dim=0))
        )
        target_rows["Future Foot Slip"].append(states["foot_slip"][t + 1 : t + horizon + 1, keep].sum(dim=0))
        target_rows["Future Mechanical Power"].append(
            torch.log1p(states["mechanical_power"][t + 1 : t + horizon + 1, keep].sum(dim=0))
        )
        target_rows["Future Orientation Error"].append(
            states["orientation_error"][t + 1 : t + horizon + 1, keep].mean(dim=0)
        )

    observations = torch.cat(obs_rows, dim=0)[:samples]
    if observations.shape[0] < samples:
        raise RuntimeError(f"Only collected {observations.shape[0]} perturbation-conditioned samples.")
    targets = {name: torch.cat(rows, dim=0)[:samples] for name, rows in target_rows.items()}
    return observations.numpy().astype(np.float32), {
        name: value.numpy().astype(np.float32) for name, value in targets.items()
    }


def featurize(model, observations: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            batch = torch.from_numpy(observations[start : start + batch_size]).to(device)
            feats.append(actor_latent(model, batch).detach().cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray, eps: float = 1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + eps
    return (train - mean) / std, (test - mean) / std


def ridge_probe(x: np.ndarray, y: np.ndarray, seed: int, ridge: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = y.reshape(-1, 1)
    n = min(x.shape[0], y.shape[0])
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    x_train, x_test = standardize(x[train_idx], x[test_idx])
    y_train, y_test = standardize(y[train_idx], y[test_idx])
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=x_test.dtype)], axis=1)
    eye = np.eye(x_train.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)
    pred = x_test @ weights
    mse = float(np.mean((pred - y_test) ** 2))
    corr = float(np.corrcoef(pred[:, 0], y_test[:, 0])[0, 1])
    return {"mse": mse, "r": corr}


def rank_text(value: float, values: list[float], higher: bool) -> str:
    order = sorted(values, reverse=higher)
    text = f"{value:.3f}"
    if value == order[0]:
        return rf"\textbf{{{text}}}"
    if len(order) > 1 and value == order[1]:
        return rf"\underline{{{text}}}"
    return text


def write_tex(payload: dict, output_tex: Path) -> None:
    results = payload["results"]
    proxies = {
        "Future Tracking Error": r"$\frac{1}{H}\sum_{h \le H}\|v_{t+h}-v^{cmd}\|^2$",
        "Future Foot Impulse": r"$\log(1+\sum_{h \le H}\|f^{foot}_{t+h}\|)$",
        "Future Foot Slip": r"$\sum_{h \le H}\mathbbm{1}_{c}\|v^{foot}_{xy,t+h}\|$",
        "Future Mechanical Power": r"$\log(1+\sum_{h \le H}|\tau_{t+h}^{\top}\dot q_{t+h}|)$",
        "Future Orientation Error": r"$\frac{1}{H}\sum_{h \le H}\|\mathrm{roll,pitch}_{t+h}\|^2$",
    }
    labels = {
        "Future Tracking Error": r"\shortstack{Future\\Tracking Error}",
        "Future Foot Impulse": r"\shortstack{Future\\Foot Impulse}",
        "Future Foot Slip": r"\shortstack{Future\\Foot Slip}",
        "Future Mechanical Power": r"\shortstack{Future\\Mechanical Power}",
        "Future Orientation Error": r"\shortstack{Future\\Orientation Error}",
    }
    methods = list(METHODS.keys())
    best_mse_counts = {method: 0 for method in methods}
    best_r_counts = {method: 0 for method in methods}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"\textbf{PRISM actor latents make future locomotion responses more linearly predictable.}",
        rf"We probe perturbation-conditioned Humanoid-Gym walking windows with horizon $H={payload['metadata']['horizon']}$.",
        r"Targets are computed from privileged simulator quantities, but probes use only frozen actor representations from deployable observations.",
        r"MSE is computed after standardizing each target.",
        r"}",
        r"\label{tab:locomotion_future_response_probe}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cclcc}",
        r"\toprule",
        r"Target & Proxy & Representation & MSE $\downarrow$ & $r$ $\uparrow$ \\",
        r"\midrule",
    ]
    for qi, target in enumerate(TARGET_NAMES):
        mse_values = [results[target][method]["mse"] for method in methods]
        r_values = [results[target][method]["r"] for method in methods]
        best_mse_counts[methods[int(np.argmin(mse_values))]] += 1
        best_r_counts[methods[int(np.argmax(r_values))]] += 1
        for mi, method in enumerate(methods):
            row = results[target][method]
            target_cell = f"\\multirow{{2}}{{*}}{{{labels[target]}}}" if mi == 0 else ""
            proxy_cell = f"\\multirow{{2}}{{*}}{{{proxies[target]}}}" if mi == 0 else ""
            lines.append(
                f"{target_cell} & {proxy_cell} & {method} & "
                f"{rank_text(row['mse'], mse_values, False)} & {rank_text(row['r'], r_values, True)} \\\\"
            )
        if qi != len(TARGET_NAMES) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{2}}{{c}}{{\textit{{Best count}}}} & {methods[0]} & {best_mse_counts[methods[0]]}/5 MSE & {best_r_counts[methods[0]]}/5 $r$ \\",
            rf"\multicolumn{{2}}{{c}}{{}} & {methods[1]} & \textbf{{{best_mse_counts[methods[1]]}/5 MSE}} & \textbf{{{best_r_counts[methods[1]]}/5 $r$}} \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    output_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="g1_humanoidgym_ppo")
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--push-threshold", type=float, default=1e-6)
    parser.add_argument("--stable-min-height", type=float, default=0.55)
    parser.add_argument("--stable-max-roll", type=float, default=0.35)
    parser.add_argument("--stable-max-pitch", type=float, default=0.35)
    parser.add_argument("--stable-min-episode-steps", type=int, default=10)
    parser.add_argument("--mlp-path", type=Path, required=True)
    parser.add_argument("--prism-path", type=Path, required=True)
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/humanoid/future_response_probe.json")
    )
    parser.add_argument("--output-tex", type=Path, default=Path("outputs/humanoid/future_response_probe.tex"))
    args = parser.parse_args()
    if args.mlp_path is not None:
        METHODS["MLP Actor Latent"]["path"] = args.mlp_path
    if args.prism_path is not None:
        METHODS["PRISM Actor Latent"]["path"] = args.prism_path

    root = Path.cwd()
    device = torch.device(args.rl_device if torch.cuda.is_available() else "cpu")
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = override_probe_cfg(copy.deepcopy(env_cfg), args.num_envs)
    env_cfg.seed = args.seed
    env_args = make_args(args.num_envs, args.seed, args.sim_device, args.rl_device, headless=True)
    env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=copy.deepcopy(env_cfg))
    collector = load_model(METHODS["PRISM Actor Latent"], root, device)
    env.reset()
    observations, targets = collect_future_response_dataset(
        env,
        collector,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        horizon=args.horizon,
        min_height=args.stable_min_height,
        max_roll=args.stable_max_roll,
        max_pitch=args.stable_max_pitch,
        min_episode_steps=args.stable_min_episode_steps,
        push_threshold=args.push_threshold,
    )
    del collector

    results = {}
    for i, (method_name, method) in enumerate(METHODS.items()):
        print(f"\n=== Probing {method_name} ===")
        model = load_model(method, root, device)
        x = featurize(model, observations, device)
        method_results = {}
        for target_name, y in targets.items():
            method_results[target_name] = ridge_probe(x, y, seed=args.seed + 100 * i, ridge=args.ridge)
            print(method_name, target_name, method_results[target_name])
        results[method_name] = method_results
        del model
        torch.cuda.empty_cache()

    payload = {
        "metadata": {
            "task": args.task,
            "samples": int(observations.shape[0]),
            "num_envs": args.num_envs,
            "horizon": args.horizon,
            "push_threshold": args.push_threshold,
            "ridge": args.ridge,
            "targets": TARGET_NAMES,
        },
        "results": {
            target: {method: results[method][target] for method in METHODS} for target in TARGET_NAMES
        },
    }
    out_json = root / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    write_tex(payload, args.output_tex)
    print(f"\nSaved {out_json}")
    print(f"Saved {args.output_tex}")
    print(args.output_tex.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
