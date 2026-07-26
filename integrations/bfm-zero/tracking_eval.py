# Derived for use with BFM-Zero. This integration file is subject to the
# upstream BFM-Zero CC BY-NC 4.0 license; see ../../NOTICE.md.

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from humanoidverse.agents.evaluations.humanoidverse_isaac import _calc_metrics
from humanoidverse.agents.evaluations.humanoidverse_isaac import group_assign_motions_to_envs_with_map
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.evaluations.humanoidverse_isaac import HumanoidVerseIsaacTrackingEvaluationConfig
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.train import TrainConfig
from torch.utils._pytree import tree_map


def _normalize_train_config_dict(cfg_data: dict[str, Any]) -> dict[str, Any]:
    cfg_data = json.loads(json.dumps(cfg_data))

    agent_model = cfg_data.get("agent", {}).get("model", {})
    model_device = agent_model.get("device")
    if isinstance(model_device, str) and model_device.startswith("cuda"):
        agent_model["device"] = "cuda"

    return cfg_data


def _parse_hydra_overrides(hydra_overrides_json: str | None) -> list[str]:
    if not hydra_overrides_json:
        return []
    parsed = json.loads(hydra_overrides_json)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("hydra_overrides_json must decode to a JSON list of strings.")
    return parsed


def _parse_motion_ids(motion_ids_json: str | None) -> list[int] | None:
    if not motion_ids_json:
        return None
    parsed = json.loads(motion_ids_json)
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError("motion_ids_json must decode to a JSON list of integers.")
    return parsed


def _first_tracking_eval_config(cfg: TrainConfig) -> HumanoidVerseIsaacTrackingEvaluationConfig | None:
    evaluations = cfg.evaluations
    if isinstance(evaluations, list):
        for evaluation in evaluations:
            if isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluationConfig):
                return evaluation
        return None

    for evaluation in evaluations.values():
        if isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluationConfig):
            return evaluation
    return None


def _ensure_checkpoint_link(output_dir: Path, model_folder: Path) -> None:
    checkpoint_link = output_dir / "checkpoint"
    checkpoint_target = model_folder / "checkpoint"
    if not checkpoint_target.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_target}")

    if checkpoint_link.exists() or checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != checkpoint_target.resolve():
            raise RuntimeError(
                f"{checkpoint_link} already exists and does not point to {checkpoint_target}"
            )
        return

    checkpoint_link.symlink_to(checkpoint_target, target_is_directory=True)


def _build_eval_cfg(
    *,
    config_path: Path,
    output_dir: Path,
    eval_log_name: str,
    data_path: Path | None,
    headless: bool,
    device: str,
    simulator: str | None,
    disable_dr: bool,
    disable_obs_noise: bool,
    hydra_overrides: list[str],
    num_envs: int,
    disable_tqdm: bool,
) -> TrainConfig:
    cfg_json = json.loads(config_path.read_text())
    cfg = TrainConfig.model_validate(_normalize_train_config_dict(cfg_json))

    eval_template = _first_tracking_eval_config(cfg)
    if eval_template is None:
        eval_template = HumanoidVerseIsaacTrackingEvaluationConfig(
            name="HumanoidVerseIsaacTrackingEvaluationConfig",
            name_in_logs=eval_log_name,
            env=None,
            num_envs=num_envs,
            n_episodes_per_motion=1,
            disable_tqdm=disable_tqdm,
        )
    else:
        eval_template = eval_template.model_copy(
            deep=True,
            update={
                "name_in_logs": eval_log_name,
                "env": None,
                "num_envs": num_envs,
                "disable_tqdm": disable_tqdm,
                "generate_videos": False,
            },
        )

    env_hydra_overrides = list(cfg.env.hydra_overrides)
    env_hydra_overrides.extend(
        [
            "env.config.max_episode_length_s=10000",
            f"env.config.headless={headless}",
        ]
    )
    if simulator:
        env_hydra_overrides.append(f"simulator={simulator}")
    env_hydra_overrides.extend(hydra_overrides)

    env_update: dict[str, Any] = {
        "device": device,
        "enable_cameras": False,
        "disable_domain_randomization": disable_dr,
        "disable_obs_noise": disable_obs_noise,
        "hydra_overrides": env_hydra_overrides,
    }
    if data_path is not None:
        env_update["lafan_tail_path"] = str(data_path.resolve())

    return cfg.model_copy(
        deep=True,
        update={
            "work_dir": str(output_dir),
            "use_wandb": False,
            "disable_tqdm": disable_tqdm,
            "prioritization": False,
            "online_parallel_envs": num_envs,
            "env": cfg.env.model_copy(deep=True, update=env_update),
            "agent": cfg.agent.model_copy(
                deep=True,
                update={
                    "model": cfg.agent.model.model_copy(
                        deep=True,
                        update={"device": device},
                    ),
                },
            ),
            "evaluations": [eval_template],
        },
    )


def _summarize_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise RuntimeError(f"No rows found in evaluation CSV: {csv_path}")

    numeric_fields = [
        "distance",
        "emd",
        "proximity",
        "success",
        "mpjpe_l",
        "obs_state_distance",
        "obs_state_emd",
        "obs_state_proximity",
        "obs_state_success",
        "vel_dist",
        "accel_dist",
    ]
    means = {}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field, "") not in {"", None}]
        if values:
            means[field] = sum(values) / len(values)

    timestep_values = [int(float(row["timestep"])) for row in rows if row.get("timestep", "") not in {"", None}]
    return {
        "num_rows": len(rows),
        "timestep": timestep_values[0] if timestep_values else None,
        "means": means,
    }


def _take_or_pad_tensor(seq: torch.Tensor, length: int, start: int = 0) -> torch.Tensor:
    seq = seq[start:]
    if seq.shape[0] >= length:
        return seq[:length].clone()
    if seq.shape[0] == 0:
        raise ValueError("Cannot pad an empty tensor sequence.")
    pad = seq[-1:].repeat(length - seq.shape[0], *([1] * (seq.dim() - 1)))
    return torch.cat([seq, pad], dim=0)


def _take_or_pad_tree(tree: dict[str, torch.Tensor], length: int, start: int = 0) -> dict[str, torch.Tensor]:
    return {key: _take_or_pad_tensor(value, length=length, start=start) for key, value in tree.items()}


def _prepare_motion_payload(env, model, motion_id: int) -> dict[str, Any]:
    tracking_target, tracking_target_dict = get_backward_observation(env._env, motion_id)
    z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    ctx = model.project_z(z)

    ref_body_rots = tracking_target_dict["ref_body_rots"][0, 0]
    if env._env.simulator.__class__.__name__ == "IsaacSim":
        ref_body_rots = ref_body_rots[[3, 0, 1, 2]]

    ref_root_init_state = torch.cat(
        [
            tracking_target_dict["ref_body_pos"][0, 0],
            ref_body_rots,
            tracking_target_dict["ref_body_vels"][0, 0],
            tracking_target_dict["ref_body_angular_vels"][0, 0],
        ]
    )
    return {
        "ctx": ctx,
        "tracking_target": tree_map(lambda x: x.cpu(), tracking_target),
        "joint_target": tracking_target_dict["dof_pos"].clone().cpu(),
        "root_init_state": ref_root_init_state,
        "dof_pos_init": tracking_target_dict["dof_pos"][0].clone(),
        "dof_vel_init": tracking_target_dict["ref_dof_vel"][0].clone(),
        "motion_file": env._env._motion_lib.curr_motion_keys[motion_id],
    }


def _build_switch_targets(
    *,
    initial_payload: dict[str, Any],
    switch_payload: dict[str, Any] | None,
    max_steps: int,
    command_switch_step: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    target_len = max_steps + 1
    if switch_payload is None or command_switch_step < 0 or command_switch_step >= max_steps:
        return (
            _take_or_pad_tree(initial_payload["tracking_target"], length=target_len),
            _take_or_pad_tensor(initial_payload["joint_target"], length=target_len),
        )

    prefix_len = min(command_switch_step + 1, target_len)
    suffix_len = target_len - prefix_len

    tracking_target = {}
    for key, value in initial_payload["tracking_target"].items():
        prefix = _take_or_pad_tensor(value, length=prefix_len)
        suffix = _take_or_pad_tensor(switch_payload["tracking_target"][key], length=suffix_len, start=1)
        tracking_target[key] = torch.cat([prefix, suffix], dim=0)

    joint_target = torch.cat(
        [
            _take_or_pad_tensor(initial_payload["joint_target"], length=prefix_len),
            _take_or_pad_tensor(switch_payload["joint_target"], length=suffix_len, start=1),
        ],
        dim=0,
    )
    return tracking_target, joint_target


def _run_custom_tracking_eval(
    *,
    cfg: TrainConfig,
    model_folder: Path,
    csv_path: Path,
    timestep: int,
    num_envs: int,
    motion_ids: list[int] | None,
    max_steps: int,
    push_every_steps: int,
    push_start_step: int,
    command_switch_step: int,
    switch_motion_id: int | None,
) -> None:
    env, _ = cfg.env.build(num_envs=num_envs)
    isaac_env = env._env
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=cfg.agent.model.device)
    model.to(cfg.agent.model.device)
    model.eval()

    if motion_ids is None:
        motion_ids = list(range(isaac_env._motion_lib._num_unique_motions))
    if switch_motion_id is not None and switch_motion_id not in motion_ids:
        motion_ids = sorted(set(motion_ids + [switch_motion_id]))

    if not isaac_env._motion_lib.all_motions_loaded:
        isaac_env._motion_lib.all_motions_loaded = True
        isaac_env._motion_lib.load_motions(
            random_sample=False,
            num_motions_to_load=isaac_env._motion_lib._num_unique_motions,
            start_idx=0,
        )

    fieldnames: list[str] | None = None
    with csv_path.open("w", newline="") as f:
        writer = None
        for motion_chunk_start in range(0, len(motion_ids), num_envs):
            chunk = motion_ids[motion_chunk_start : motion_chunk_start + num_envs]
            assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(
                chunk,
                num_envs,
                device=isaac_env.device,
            )
            payloads = {motion_id: _prepare_motion_payload(env, model, motion_id) for motion_id in set(chunk)}
            if switch_motion_id is not None and switch_motion_id not in payloads:
                payloads[switch_motion_id] = _prepare_motion_payload(env, model, switch_motion_id)

            dof_states = []
            root_states = []
            for env_id in range(num_envs):
                motion_id = assigned_motions[env_id].item()
                payload = payloads[motion_id]
                dof_state = torch.zeros_like(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[0])
                dof_state[..., 0] = payload["dof_pos_init"]
                dof_state[..., 1] = payload["dof_vel_init"]
                dof_states.append(dof_state)
                root_states.append(payload["root_init_state"])

            target_states = {
                "dof_states": torch.stack(dof_states),
                "root_states": torch.stack(root_states),
            }

            with torch.no_grad():
                observation, _ = env.reset(target_states=target_states, to_numpy=False)
                current_motion_by_env = assigned_motions.clone()
                motion_local_step = torch.zeros(num_envs, dtype=torch.long, device=isaac_env.device)
                observation_states = [observation["state"].cpu()]
                joint_pos = [isaac_env.simulator.dof_state[..., 0].cpu()]

                for step in range(max_steps):
                    if (
                        command_switch_step >= 0
                        and switch_motion_id is not None
                        and step == command_switch_step
                    ):
                        current_motion_by_env[:] = switch_motion_id
                        motion_local_step.zero_()

                    if (
                        push_every_steps > 0
                        and step >= push_start_step
                        and (step - push_start_step) % push_every_steps == 0
                    ):
                        isaac_env._push_robots(torch.arange(num_envs, device=isaac_env.device))

                    ctx_batch = []
                    for env_id in range(num_envs):
                        motion_id = current_motion_by_env[env_id].item()
                        ctx = payloads[motion_id]["ctx"]
                        ctx_batch.append(ctx[motion_local_step[env_id].item() % ctx.shape[0]])
                    ctx_batch = torch.stack(ctx_batch)
                    action = model.act(observation, ctx_batch, mean=True)
                    observation, _, _, _, _ = env.step(action, to_numpy=False)
                    motion_local_step += 1
                    observation_states.append(observation["state"].cpu())
                    joint_pos.append(isaac_env.simulator.dof_state[..., 0].cpu())

                observation_states = torch.stack(observation_states)
                joint_pos = torch.stack(joint_pos)

                for motion_id, env_ids in motion_to_envs.items():
                    for motion_repetition, env_id in enumerate(env_ids):
                        switch_payload = payloads.get(switch_motion_id) if switch_motion_id is not None else None
                        tracking_target, target_joint_pos = _build_switch_targets(
                            initial_payload=payloads[motion_id],
                            switch_payload=switch_payload,
                            max_steps=max_steps,
                            command_switch_step=command_switch_step,
                        )
                        metric_motion_file = payloads[motion_id]["motion_file"]
                        if switch_motion_id is not None and command_switch_step >= 0:
                            metric_motion_file = f"{metric_motion_file}->switch#{switch_motion_id}"
                        metrics = _calc_metrics(
                            {
                                "tracking_target": tracking_target,
                                "motion_id": motion_id,
                                "motion_file": metric_motion_file,
                                "observation": {"state": observation_states[:, env_id]},
                                "joint_pos": joint_pos[:, env_id],
                                "target_joint_pos": target_joint_pos,
                            }
                        )
                        row = list(metrics.values())[0]
                        row["timestep"] = timestep
                        if fieldnames is None:
                            fieldnames = list(row.keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                        assert writer is not None
                        writer.writerow(row)
                        break

    env.close()


def main(
    model_folder: Path,
    data_path: Path | None = None,
    output_subdir: str = "tracking_eval_nominal",
    eval_log_name: str = "humanoidverse_tracking_eval_nominal",
    headless: bool = True,
    device: str = "cuda:0",
    simulator: str | None = None,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    hydra_overrides_json: str | None = None,
    num_envs: int = 256,
    motion_ids_json: str | None = None,
    max_steps: int | None = None,
    push_every_steps: int = 0,
    push_start_step: int = 0,
    command_switch_step: int = -1,
    switch_motion_id: int | None = None,
    disable_tqdm: bool = True,
):
    model_folder = Path(model_folder).resolve()
    if not model_folder.is_dir():
        raise FileNotFoundError(f"Missing model folder: {model_folder}")

    output_dir = model_folder / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_link(output_dir, model_folder)

    hydra_overrides = _parse_hydra_overrides(hydra_overrides_json)
    motion_ids = _parse_motion_ids(motion_ids_json)
    scenario_metadata = {
        "model_folder": str(model_folder),
        "output_subdir": output_subdir,
        "eval_log_name": eval_log_name,
        "data_path": str(data_path.resolve()) if data_path is not None else None,
        "headless": headless,
        "device": device,
        "simulator": simulator,
        "disable_dr": disable_dr,
        "disable_obs_noise": disable_obs_noise,
        "hydra_overrides": hydra_overrides,
        "num_envs": num_envs,
        "motion_ids": motion_ids,
        "max_steps": max_steps,
        "push_every_steps": push_every_steps,
        "push_start_step": push_start_step,
        "command_switch_step": command_switch_step,
        "switch_motion_id": switch_motion_id,
    }
    (output_dir / "scenario_metadata.json").write_text(json.dumps(scenario_metadata, indent=2))

    csv_path = output_dir / f"{eval_log_name}.csv"
    summary_path = output_dir / "summary.json"
    if csv_path.exists():
        csv_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    cfg = _build_eval_cfg(
        config_path=model_folder / "config.json",
        output_dir=output_dir,
        eval_log_name=eval_log_name,
        data_path=data_path,
        headless=headless,
        device=device,
        simulator=simulator,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        hydra_overrides=hydra_overrides,
        num_envs=num_envs,
        disable_tqdm=disable_tqdm,
    )

    use_custom_rollout = (
        motion_ids is not None
        or max_steps is not None
        or push_every_steps > 0
        or command_switch_step >= 0
    )

    if use_custom_rollout:
        timestep_cfg = json.loads((model_folder / "checkpoint" / "train_status.json").read_text())
        timestep = int(timestep_cfg["time"])
        rollout_steps = max_steps if max_steps is not None else 100
        print(f"[eval-start] timestep={timestep} output_dir={output_dir} custom_rollout=true")
        _run_custom_tracking_eval(
            cfg=cfg,
            model_folder=model_folder,
            csv_path=csv_path,
            timestep=timestep,
            num_envs=num_envs,
            motion_ids=motion_ids,
            max_steps=rollout_steps,
            push_every_steps=push_every_steps,
            push_start_step=push_start_step,
            command_switch_step=command_switch_step,
            switch_motion_id=switch_motion_id,
        )
        summary = _summarize_csv(csv_path)
        summary["scenario"] = scenario_metadata
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"[eval-done] {summary_path}")
        return

    workspace = cfg.build()
    try:
        timestep = workspace._checkpoint_time
        print(f"[eval-start] timestep={timestep} output_dir={output_dir}")
        workspace.eval(timestep, replay_buffer={})
        summary = _summarize_csv(csv_path)
        summary["scenario"] = scenario_metadata
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"[eval-done] {summary_path}")
    finally:
        if hasattr(workspace, "train_env"):
            workspace.train_env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
