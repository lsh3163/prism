"""Train gated Diffusion PRISM, matched controls, or explicit historical recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    INTEGRATION_ROOT,
    SUITES,
    Run,
    add_execution_args,
    command_prefix,
    csv_values,
    nonnegative_int,
    positive_int,
    run_commands,
)
from profile_catalog import DIFFUSION_PROFILE_NAMES, resolve_diffusion_profile


def build_command(args: argparse.Namespace, task: dict) -> tuple[list[str], Path]:
    profile = resolve_diffusion_profile(args.profile, task["profiles"])
    specification = profile.specification
    output = args.output_root / args.profile / task["suite"] / f"task_{task['task_id']}"
    command = [
        *command_prefix(args, "train"),
        f"--policy.device={getattr(args, 'device', 'cuda')}",
        "--policy.type=diffusion",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.num_train_timesteps=100",
        "--policy.num_inference_steps=10",
        "--policy.resize_shape=[128,128]",
        "--policy.down_dims=[128,256,512]",
        "--policy.use_separate_rgb_encoder_per_camera=false",
        "--policy.optimizer_lr=0.0001",
        "--policy.compile_model=false",
        "--policy.use_amp=true",
        f"--policy.use_poly_kernel_conditioning={str(specification.conditioning_enabled).lower()}",
        "--policy.poly_kernel_source=state",
        f"--policy.poly_kernel_lift_mode={specification.lift_mode}",
        "--policy.poly_kernel_latent_dim=256",
        "--policy.poly_kernel_hidden_dim=256",
        "--policy.poly_kernel_gate_scale_init=0.01",
        "--dataset.repo_id=HuggingFaceVLA/libero",
        f"--dataset.episodes={json.dumps(task['episodes'], separators=(',', ':'))}",
        "--env.type=libero",
        f"--env.task={task['suite']}",
        f"--env.task_ids=[{task['task_id']}]",
        f"--steps={args.steps}",
        "--save_freq=10000",
        "--log_freq=200",
        "--eval_freq=0",
        f"--batch_size={profile.batch_size}",
        f"--num_workers={profile.num_workers}",
        "--prefetch_factor=4",
        "--persistent_workers=true",
        f"--seed={args.seed}",
        "--wandb.enable=false",
        f"--output_dir={output}",
    ]
    if args.dataset_revision:
        command.append(f"--dataset.revision={args.dataset_revision}")
    return command, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_execution_args(parser)
    parser.add_argument(
        "--profile",
        choices=DIFFUSION_PROFILE_NAMES,
        default="prism",
        help="Default: gated PRISM. baseline is its matched control; legacy profiles preserve archived recipes",
    )
    parser.add_argument("--suites", type=csv_values, default=",".join(SUITES))
    parser.add_argument("--task-ids", type=csv_values, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--steps", type=positive_int, default=20000)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/prism_diffusion"))
    args = parser.parse_args()
    suites = args.suites
    if not set(suites) <= set(SUITES) or not set(args.task_ids) <= {str(index) for index in range(10)}:
        parser.error("Use the four LIBERO suites and task IDs 0 through 9")
    task_ids = [int(value) for value in args.task_ids]
    tasks = json.loads((INTEGRATION_ROOT / "recipes/diffusion_tasks.json").read_text())["tasks"]
    runs = []
    for task in tasks:
        if task["suite"] in suites and task["task_id"] in task_ids:
            command, output = build_command(args, task)
            profile = resolve_diffusion_profile(args.profile, task["profiles"])
            runs.append(
                Run(
                    command,
                    output,
                    {
                        "policy": "diffusion",
                        "profile": args.profile,
                        "canonical_profile": profile.specification.canonical_name,
                        "profile_settings": profile.as_dict(),
                        "actor_variant": profile.specification.actor_variant,
                        "architecture": profile.specification.architecture,
                        "suite": task["suite"],
                        "task_id": task["task_id"],
                        "seed": args.seed,
                        "dataset_revision": args.dataset_revision,
                        "action": "train",
                    },
                )
            )
    try:
        run_commands(runs, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
