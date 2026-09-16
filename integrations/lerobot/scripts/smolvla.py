"""Print or run the released SmolVLA state-projection training/evaluation recipes."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    SUITES,
    Run,
    add_execution_args,
    command_prefix,
    nonnegative_int,
    positive_int,
    run_commands,
)
from profile_catalog import SMOLVLA_PROFILE_NAMES, smolvla_profile


def build_run(args: argparse.Namespace) -> Run:
    command = command_prefix(args, args.action)
    profile = smolvla_profile(args.profile) if args.profile is not None else None
    if args.action == "train":
        if profile is None:
            raise ValueError("train requires a SmolVLA profile")
        command.extend(
            [
                "--policy.type=smolvla",
                "--policy.push_to_hub=false",
                "--policy.load_vlm_weights=true",
                "--policy.freeze_vision_encoder=true",
                "--policy.train_expert_only=true",
                "--policy.train_state_proj=true",
                f"--policy.state_conditioner_type={profile.conditioner_type}",
                f"--policy.state_conditioner_num_layers={profile.num_layers}",
                f"--policy.state_conditioner_product_mode={profile.product_mode}",
                f"--policy.state_conditioner_gate_scale_init={profile.gate_scale_init}",
                f"--policy.state_conditioner_use_rmsnorm={str(profile.use_rmsnorm).lower()}",
                "--policy.scheduler_warmup_steps=100",
                "--policy.scheduler_decay_steps=100000",
                "--dataset.repo_id=HuggingFaceVLA/libero",
                "--batch_size=64",
                "--num_workers=8",
                "--prefetch_factor=4",
                "--persistent_workers=true",
                "--eval_freq=0",
                "--save_freq=20000",
                f"--steps={args.steps or 100000}",
                "--wandb.enable=false",
            ]
        )
        if profile.hidden_dim is not None:
            command.append(f"--policy.state_conditioner_hidden_dim={profile.hidden_dim}")
    else:
        command.extend(
            [
                f"--policy.path={args.checkpoint}",
                "--eval.n_episodes=50",
                "--eval.batch_size=1",
                "--env.max_parallel_tasks=1",
            ]
        )
    command.extend(
        [
            "--env.type=libero",
            f"--env.task={','.join(SUITES)}",
            f"--policy.device={args.device}",
            f"--seed={args.seed}",
            f"--output_dir={args.output}",
        ]
    )
    return Run(
        command,
        args.output,
        {
            "policy": "smolvla",
            "profile": args.profile,
            "canonical_profile": profile.name if profile is not None else None,
            "profile_settings": profile.as_dict() if profile is not None else None,
            "actor_variant": profile.actor_variant if profile is not None else None,
            "architecture": profile.conditioner_type if profile is not None else None,
            "action": args.action,
            "seed": args.seed,
            "product_mode": profile.product_mode
            if args.action == "train" and profile.has_learned_gate
            else None,
            "gate_scale_init": profile.gate_scale_init
            if args.action == "train" and profile.has_learned_gate
            else None,
        },
        {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
        args.checkpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "eval"))
    add_execution_args(parser)
    parser.add_argument(
        "--profile",
        choices=SMOLVLA_PROFILE_NAMES,
        help="Train default: prism. For eval, optionally require this checkpoint architecture",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=positive_int, help="Training only; default: 100000")
    args = parser.parse_args()
    if args.action == "eval":
        if args.checkpoint is None:
            parser.error("eval requires --checkpoint (the reported checkpoint is 080000)")
        if args.steps is not None:
            parser.error("--steps applies only to train")
    else:
        if args.checkpoint is not None:
            parser.error(
                "--checkpoint applies only to eval; this recipe initializes from pretrained VLM weights"
            )
        args.profile = args.profile or "prism"
    try:
        run_commands([build_run(args)], args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
