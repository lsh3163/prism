"""Run the nominal LIBERO matrix, MCC validation, or published noise/delay sweep."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import SUITES, add_execution_args, nonnegative_int, positive_int, run_commands
from eval_diffusion import build_run, make_parser

VALIDATION_TASKS = ("libero_spatial:1", "libero_object:3", "libero_goal:7", "libero_10:1")
PERTURBATIONS = {
    "clean": (0, 0.0, 0.0),
    "action_delay": (1, 0.0, 0.0),
    "proprio_noise": (0, 0.01, 0.0),
    "image_corrupt": (0, 0.0, 0.05),
    "combined": (1, 0.01, 0.03),
}
# gain, delay, EMA, noise standard deviation, correction clip; original c00-c08 order.
MCC_CONFIGS = [
    (0.005, 0, 0.70, 0.05, 0.05),
    (0.005, 1, 0.90, 0.05, 0.05),
    (0.010, 0, 0.70, 0.05, 0.05),
    (0.010, 1, 0.90, 0.05, 0.05),
    (0.015, 1, 0.90, 0.15, 0.05),
    (0.015, 2, 0.90, 0.15, 0.05),
    (0.020, 1, 0.90, 0.15, 0.05),
    (0.010, 1, 0.70, 0.15, 0.10),
    (0.015, 1, 0.70, 0.05, 0.10),
]


def build_plan(args: argparse.Namespace) -> tuple[list, dict[Path, str]]:
    checkpoints = json.loads(args.checkpoint_map.read_text())
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("Checkpoint map must be a nonempty JSON object")
    allowed = {f"{suite}:{task_id}" for suite in SUITES for task_id in range(10)}
    if not set(checkpoints) <= allowed:
        raise ValueError("Checkpoint map keys must be LIBERO_SUITE:TASK_ID (0 through 9)")
    tasks = sorted(checkpoints) if args.kind == "nominal" else VALIDATION_TASKS
    required = ("baseline",) if args.kind == "mcc" else ("baseline", "prism")
    for key in tasks:
        entry = checkpoints.get(key)
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(name), str) or not entry[name].strip() for name in required
        ):
            raise ValueError(f"Checkpoint map requires {', '.join(required)} paths for {key}")
    extra_files = {}
    if args.kind == "mcc":
        config_lines = [
            f"c{index:02d} " + " ".join(map(str, values)) for index, values in enumerate(MCC_CONFIGS)
        ]
        extra_files[args.output_root / "configs.txt"] = "\n".join(config_lines) + "\n"
        extra_files[args.output_root / "tasks.txt"] = " ".join(tasks) + "\n"
    runs = []
    for key in tasks:
        suite, task_id = key.split(":")
        entry = checkpoints[key]
        options = make_parser().parse_args(
            [
                "--lerobot-root",
                str(args.lerobot_root),
                "--suite",
                suite,
                "--task-id",
                task_id,
                "--output-root",
                str(args.output_root),
            ]
        )
        options.baseline = Path(entry["baseline"])
        options.prism = Path(entry["prism"]) if args.kind != "mcc" else None
        options.seed = args.seed
        options.python = args.python
        options.device = args.device
        options.egl_vendor_file = args.egl_vendor_file
        options.controller = args.controller
        options.episodes = args.episodes or (10 if args.kind == "nominal" else 5)
        options.batch_size = 5 if args.kind == "robustness" else 1
        options.save_probe_traces = args.kind == "mcc"
        options.mcc_clip = 0.05 if args.kind == "robustness" else None
        variants = (
            list(PERTURBATIONS)
            if args.kind == "robustness"
            else (
                [f"c{index:02d}" for index in range(len(MCC_CONFIGS))] if args.kind == "mcc" else ["nominal"]
            )
        )
        for index, variant in enumerate(variants):
            current = copy.copy(options)
            current.output_root = args.output_root / variant / f"{suite}_task_{task_id}"
            rows = ("baseline", "mcc", "oracle", "prism")
            if args.kind == "robustness":
                current.action_delay, current.proprio_noise, current.image_noise = PERTURBATIONS[variant]
                rows = ("baseline", "mcc", "prism")
            elif args.kind == "mcc":
                current.mcc_gain, current.mcc_delay, current.mcc_ema, current.mcc_noise, current.mcc_clip = (
                    MCC_CONFIGS[index]
                )
                rows = ("mcc",)
            for row in rows:
                run = build_run(current, row)
                run.metadata.update(sweep=args.kind, variant=variant)
                runs.append(run)
    return runs, extra_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_execution_args(parser)
    parser.add_argument("--checkpoint-map", type=Path, required=True)
    parser.add_argument("--kind", choices=("nominal", "robustness", "mcc"), default="nominal")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=nonnegative_int, default=1000)
    parser.add_argument(
        "--episodes", type=positive_int, help="Default: 10 nominal; 5 per task for MCC/robustness"
    )
    parser.add_argument("--controller", choices=("shared", "historical"), default="shared")
    args = parser.parse_args()
    try:
        runs, extra_files = build_plan(args)
        run_commands(runs, args, extra_files=extra_files)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
