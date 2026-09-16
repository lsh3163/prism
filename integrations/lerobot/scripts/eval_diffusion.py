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
from evaluation_specs import (
    EVALUATION_PROTOCOL,
    EVALUATION_ROW_NAMES,
    EVALUATION_ROW_SPECS,
    EvaluationRequest,
    evaluation_row,
)

# Public compatibility mapping retained for callers that only need the output
# directory name. The typed specification owns the values.
ROW_NAMES = {specification.name: specification.output_name for specification in EVALUATION_ROW_SPECS}


def build_command_from_request(
    request: EvaluationRequest, row_name: str
) -> tuple[list[str], Path, dict[str, str]]:
    """Render one immutable evaluation request as the released LeRobot command."""

    row = evaluation_row(row_name)
    checkpoint = request.checkpoint_for(row)
    if checkpoint is None:
        raise ValueError(
            f"A {'PRISM' if row.checkpoint_role == 'prism' else 'baseline'} checkpoint is required for {row.name}"
        )
    output = request.output_root / row.output_name
    command = [
        *command_prefix(request, "eval"),
        f"--policy.path={checkpoint}",
        f"--policy.n_action_steps={EVALUATION_PROTOCOL.n_action_steps}",
        f"--policy.num_inference_steps={EVALUATION_PROTOCOL.inference_steps}",
        f"--policy.device={request.device}",
        "--env.type=libero",
        f"--env.task={request.suite}",
        f"--env.task_ids=[{request.task_id}]",
        f"--env.terminate_on_success={bool_arg(EVALUATION_PROTOCOL.terminate_on_success)}",
        f"--env.max_parallel_tasks={EVALUATION_PROTOCOL.max_parallel_tasks}",
        f"--env.excess_contact_force_threshold={EVALUATION_PROTOCOL.excess_contact_force_threshold}",
        f"--eval.n_episodes={request.episodes}",
        f"--eval.batch_size={request.batch_size}",
        f"--eval.max_episodes_rendered={request.render_episodes}",
        f"--seed={request.seed}",
        f"--output_dir={output}",
    ]
    if request.controller == "shared" or row.requires_controller_override:
        command.extend(["--env.controller_kp=50", "--env.controller_damping_ratio=1"])
    command.append(f"--env.mcc_noisy_enabled={bool_arg(row.mcc_enabled)}")
    if row.mcc_enabled:
        oracle = row.mcc_force_source == "contact"
        command.extend(
            [
                f"--env.mcc_force_source={row.mcc_force_source}",
                f"--env.mcc_force_gain={request.mcc_gain}",
                f"--env.mcc_noise_std={0.0 if oracle else request.mcc_noise}",
                f"--env.mcc_delay={0 if oracle else request.mcc_delay}",
                f"--env.mcc_force_ema_alpha={request.mcc_ema}",
                f"--env.mcc_wrench_regularization={EVALUATION_PROTOCOL.mcc_wrench_regularization}",
            ]
        )
        if request.mcc_clip is not None:
            command.append(f"--env.mcc_correction_clip={request.mcc_clip}")
    return command, output, request.environment()


def build_command(args: argparse.Namespace, row: str) -> tuple[list[str], Path, dict[str, str]]:
    """Compatibility wrapper for the public argparse-based command builder."""

    return build_command_from_request(EvaluationRequest.from_args(args), row)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_execution_args(parser)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prism", type=Path)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--rows", type=csv_values, default="baseline,mcc,oracle,prism")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, default=EVALUATION_PROTOCOL.episodes)
    parser.add_argument("--batch-size", type=positive_int, default=EVALUATION_PROTOCOL.batch_size)
    parser.add_argument(
        "--render-episodes", type=nonnegative_int, default=EVALUATION_PROTOCOL.render_episodes
    )
    # The original shells set SEED for paths but did not override EvalPipelineConfig.seed.
    parser.add_argument("--seed", type=nonnegative_int, default=EVALUATION_PROTOCOL.seed)
    parser.add_argument(
        "--controller", choices=("shared", "historical"), default=EVALUATION_PROTOCOL.controller
    )
    parser.add_argument("--mcc-gain", type=nonnegative_float, default=EVALUATION_PROTOCOL.mcc_gain)
    parser.add_argument("--mcc-noise", type=nonnegative_float, default=EVALUATION_PROTOCOL.mcc_noise)
    parser.add_argument("--mcc-delay", type=nonnegative_int, default=EVALUATION_PROTOCOL.mcc_delay)
    parser.add_argument("--mcc-ema", type=probability, default=EVALUATION_PROTOCOL.mcc_ema)
    parser.add_argument("--mcc-clip", type=nonnegative_float)
    parser.add_argument("--action-delay", type=nonnegative_int, default=0)
    parser.add_argument("--proprio-noise", type=nonnegative_float, default=0.0)
    parser.add_argument("--image-noise", type=nonnegative_float, default=0.0)
    parser.add_argument("--save-probe-traces", action="store_true")
    return parser


def build_run_from_request(request: EvaluationRequest, row_name: str) -> Run:
    """Build one run record from an immutable, already-resolved request."""

    row = evaluation_row(row_name)
    command, output, environment = build_command_from_request(request, row_name)
    return Run(
        command,
        output,
        {
            "policy": "diffusion",
            "profile": row.profile,
            # Execution derives the exact gated/legacy variant from config.json.
            "actor_variant": None,
            "architecture": row.architecture,
            "action": "eval",
            "method": row.name,
            "suite": request.suite,
            "task_id": request.task_id,
            "seed": request.seed,
            "controller": request.controller,
            "evaluation_row": row.as_dict(),
            "protocol": request.protocol_for(row),
        },
        environment,
        request.checkpoint_for(row),
    )


def build_run(args: argparse.Namespace, row: str) -> Run:
    """Compatibility wrapper for direct CLI and existing runner callers."""

    return build_run_from_request(EvaluationRequest.from_args(args), row)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    rows = args.rows
    if not set(rows) <= set(EVALUATION_ROW_NAMES):
        parser.error(f"--rows must contain only {','.join(ROW_NAMES)}")
    try:
        run_commands([build_run(args, row) for row in rows], args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
