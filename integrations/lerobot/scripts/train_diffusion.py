"""Print or execute the recorded per-task LIBERO Diffusion training recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    ACTOR_IDS,
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


def build_command(args: argparse.Namespace, task: dict) -> tuple[list[str], Path]:
    profile = "historical-baseline" if args.profile == "matched-baseline" else args.profile
    settings = task["profiles"][profile]
    batch_size = 64 if args.profile == "matched-baseline" else settings["batch_size"]
    workers = 8 if args.profile == "matched-baseline" else settings["num_workers"]
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
        f"--policy.use_poly_kernel_conditioning={str(args.profile == 'prism').lower()}",
        "--policy.poly_kernel_source=state",
        "--policy.poly_kernel_lift_mode=latent_quadratic",
        "--policy.poly_kernel_latent_dim=256",
        "--policy.poly_kernel_hidden_dim=256",
        "--dataset.repo_id=HuggingFaceVLA/libero",
        f"--dataset.episodes={json.dumps(task['episodes'], separators=(',', ':'))}",
        "--env.type=libero",
        f"--env.task={task['suite']}",
        f"--env.task_ids=[{task['task_id']}]",
        f"--steps={args.steps}",
        "--save_freq=10000",
        "--log_freq=200",
        "--eval_freq=0",
        f"--batch_size={batch_size}",
        f"--num_workers={workers}",
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
        "--profile", choices=("prism", "historical-baseline", "matched-baseline"), default="prism"
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
            runs.append(
                Run(
                    command,
                    output,
                    {
                        "policy": "diffusion",
                        "profile": args.profile,
                        "actor_variant": ACTOR_IDS["diffusion"] if args.profile == "prism" else None,
                        "architecture": "factorized_state" if args.profile == "prism" else "nominal",
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
