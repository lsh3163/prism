"""Run the nominal LIBERO matrix, MCC validation, or published noise/delay sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import SUITES, add_execution_args, nonnegative_int, positive_int, run_commands
from eval_diffusion import build_run_from_request
from evaluation_specs import (
    EVALUATION_PROTOCOL,
    MCC_SWEEP_SPECS,
    ROBUSTNESS_PERTURBATIONS,
    VALIDATION_TASKS,
    EvaluationRequest,
    mcc_configs_text,
)

# Public compatibility views. The immutable definitions above own their order.
PERTURBATIONS = {specification.name: specification.as_tuple() for specification in ROBUSTNESS_PERTURBATIONS}
MCC_CONFIGS = [specification.as_tuple() for specification in MCC_SWEEP_SPECS]


def evaluation_request(
    args: argparse.Namespace,
    *,
    entry: dict,
    suite: str,
    task_id: int,
    output_root: Path,
    episodes: int,
    batch_size: int,
    save_probe_traces: bool,
    mcc_gain: float = EVALUATION_PROTOCOL.mcc_gain,
    mcc_delay: int = EVALUATION_PROTOCOL.mcc_delay,
    mcc_ema: float = EVALUATION_PROTOCOL.mcc_ema,
    mcc_noise: float = EVALUATION_PROTOCOL.mcc_noise,
    mcc_clip: float | None = None,
    action_delay: int = 0,
    proprio_noise: float = 0.0,
    image_noise: float = 0.0,
) -> EvaluationRequest:
    """Construct a sweep request explicitly, without reparsing or mutating CLI state."""

    return EvaluationRequest(
        baseline=Path(entry["baseline"]),
        prism=Path(entry["prism"]) if args.kind != "mcc" else None,
        suite=suite,
        task_id=task_id,
        output_root=output_root,
        episodes=episodes,
        batch_size=batch_size,
        render_episodes=EVALUATION_PROTOCOL.render_episodes,
        seed=args.seed,
        controller=args.controller,
        mcc_gain=mcc_gain,
        mcc_noise=mcc_noise,
        mcc_delay=mcc_delay,
        mcc_ema=mcc_ema,
        mcc_clip=mcc_clip,
        action_delay=action_delay,
        proprio_noise=proprio_noise,
        image_noise=image_noise,
        save_probe_traces=save_probe_traces,
        python=args.python,
        device=args.device,
    )


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
        extra_files[args.output_root / "configs.txt"] = mcc_configs_text()
        extra_files[args.output_root / "tasks.txt"] = " ".join(tasks) + "\n"
    runs = []
    for key in tasks:
        suite, task_id = key.split(":")
        entry = checkpoints[key]
        episodes = args.episodes or (10 if args.kind == "nominal" else 5)
        if args.kind == "robustness":
            variants = ((specification.name, specification) for specification in ROBUSTNESS_PERTURBATIONS)
        elif args.kind == "mcc":
            variants = ((specification.name, specification) for specification in MCC_SWEEP_SPECS)
        else:
            variants = (("nominal", None),)
        for variant, specification in variants:
            output_root = args.output_root / variant / f"{suite}_task_{task_id}"
            if args.kind == "robustness":
                request = evaluation_request(
                    args,
                    entry=entry,
                    suite=suite,
                    task_id=int(task_id),
                    output_root=output_root,
                    episodes=episodes,
                    batch_size=5,
                    save_probe_traces=False,
                    mcc_clip=0.05,
                    action_delay=specification.action_delay,
                    proprio_noise=specification.proprio_noise,
                    image_noise=specification.image_noise,
                )
                rows = ("baseline", "mcc", "prism")
            elif args.kind == "mcc":
                request = evaluation_request(
                    args,
                    entry=entry,
                    suite=suite,
                    task_id=int(task_id),
                    output_root=output_root,
                    episodes=episodes,
                    batch_size=1,
                    save_probe_traces=True,
                    mcc_gain=specification.gain,
                    mcc_delay=specification.delay,
                    mcc_ema=specification.ema,
                    mcc_noise=specification.noise,
                    mcc_clip=specification.correction_clip,
                )
                rows = ("mcc",)
            else:
                request = evaluation_request(
                    args,
                    entry=entry,
                    suite=suite,
                    task_id=int(task_id),
                    output_root=output_root,
                    episodes=episodes,
                    batch_size=1,
                    save_probe_traces=False,
                )
                rows = ("baseline", "mcc", "oracle", "prism")
            for row in rows:
                run = build_run_from_request(request, row)
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
