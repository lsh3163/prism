"""Evaluate baseline, MCC-Sensorless, MCC-Oracle, and representation PRISM."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    SUITES,
    Run,
    add_execution_args,
    bool_arg,
    command_prefix,
    csv_values,
    nonnegative_float,
    nonnegative_int,
    positive_int,
    probability,
    run_commands,
)

ROW_NAMES = {
    "baseline": "diffusion",
    "mcc": "diffusion_mcc_sensorless",
    "oracle": "diffusion_mcc_oracle",
    "prism": "prism",
}


def build_command(args: argparse.Namespace, row: str) -> tuple[list[str], Path, dict[str, str]]:
    checkpoint = args.prism if row == "prism" else args.baseline
    if checkpoint is None:
        raise ValueError(f"A {'PRISM' if row == 'prism' else 'baseline'} checkpoint is required for {row}")
    output = args.output_root / ROW_NAMES[row]
    command = [
        *command_prefix(args, "eval"),
        f"--policy.path={checkpoint}",
        "--policy.n_action_steps=8",
        "--policy.num_inference_steps=10",
        f"--policy.device={args.device}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--env.task_ids=[{args.task_id}]",
        "--env.terminate_on_success=false",
        "--env.max_parallel_tasks=1",
        "--env.excess_contact_force_threshold=20",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--eval.max_episodes_rendered={args.render_episodes}",
        f"--seed={args.seed}",
        f"--output_dir={output}",
    ]
    if args.controller == "shared" or row in {"mcc", "oracle"}:
        command.extend(["--env.controller_kp=50", "--env.controller_damping_ratio=1"])
    enabled = row in {"mcc", "oracle"}
    command.append(f"--env.mcc_noisy_enabled={bool_arg(enabled)}")
    if enabled:
        oracle = row == "oracle"
        command.extend(
            [
                f"--env.mcc_force_source={'contact' if oracle else 'sensorless'}",
                f"--env.mcc_force_gain={args.mcc_gain}",
                f"--env.mcc_noise_std={0.0 if oracle else args.mcc_noise}",
                f"--env.mcc_delay={0 if oracle else args.mcc_delay}",
                f"--env.mcc_force_ema_alpha={args.mcc_ema}",
                "--env.mcc_wrench_regularization=0.0001",
            ]
        )
        if args.mcc_clip is not None:
            command.append(f"--env.mcc_correction_clip={args.mcc_clip}")
    environment = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LEROBOT_EVAL_ACTION_DELAY_STEPS": str(args.action_delay),
        "LEROBOT_EVAL_PROPRIO_NOISE_STD": str(args.proprio_noise),
        "LEROBOT_EVAL_IMAGE_NOISE_STD": str(args.image_noise),
        "LEROBOT_SAVE_PROBE_TRACES": "1" if args.save_probe_traces else "0",
    }
    return command, output, environment


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_execution_args(parser)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prism", type=Path)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--rows", type=csv_values, default="baseline,mcc,oracle,prism")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--render-episodes", type=nonnegative_int, default=0)
    # The original shells set SEED for paths but did not override EvalPipelineConfig.seed.
    parser.add_argument("--seed", type=nonnegative_int, default=1000)
    parser.add_argument("--controller", choices=("shared", "historical"), default="shared")
    parser.add_argument("--mcc-gain", type=nonnegative_float, default=0.015)
    parser.add_argument("--mcc-noise", type=nonnegative_float, default=0.15)
    parser.add_argument("--mcc-delay", type=nonnegative_int, default=2)
    parser.add_argument("--mcc-ema", type=probability, default=0.9)
    parser.add_argument("--mcc-clip", type=nonnegative_float)
    parser.add_argument("--action-delay", type=nonnegative_int, default=0)
    parser.add_argument("--proprio-noise", type=nonnegative_float, default=0.0)
    parser.add_argument("--image-noise", type=nonnegative_float, default=0.0)
    parser.add_argument("--save-probe-traces", action="store_true")
    return parser


def build_run(args: argparse.Namespace, row: str) -> Run:
    command, output, environment = build_command(args, row)
    return Run(
        command,
        output,
        {
            "policy": "diffusion",
            "profile": "prism" if row == "prism" else "baseline",
            # Execution derives the exact gated/legacy variant from config.json.
            "actor_variant": None,
            "architecture": "from_checkpoint" if row == "prism" else "nominal",
            "action": "eval",
            "method": row,
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "controller": args.controller,
        },
        environment,
        args.prism if row == "prism" else args.baseline,
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    rows = args.rows
    if not set(rows) <= ROW_NAMES.keys():
        parser.error(f"--rows must contain only {','.join(ROW_NAMES)}")
    try:
        run_commands([build_run(args, row) for row in rows], args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
