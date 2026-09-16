"""Immutable Diffusion evaluation rows, perturbations, and MCC sweep settings.

These declarations deliberately separate experiment protocol from the LeRobot
command renderer.  Their order is part of the release contract: downstream
analysis reads the row directory names and the MCC ``c00``--``c08`` sequence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationProtocolSpec:
    """Shared direct-evaluation defaults and fixed LeRobot command values."""

    n_action_steps: int = 8
    inference_steps: int = 10
    terminate_on_success: bool = False
    max_parallel_tasks: int = 1
    excess_contact_force_threshold: int = 20
    episodes: int = 10
    batch_size: int = 1
    render_episodes: int = 0
    seed: int = 1000
    controller: str = "shared"
    mcc_gain: float = 0.015
    mcc_noise: float = 0.15
    mcc_delay: int = 2
    mcc_ema: float = 0.9
    mcc_wrench_regularization: float = 0.0001


EVALUATION_PROTOCOL = EvaluationProtocolSpec()


@dataclass(frozen=True)
class EvaluationRowSpec:
    """One named method row in a direct or swept Diffusion evaluation."""

    name: str
    output_name: str
    checkpoint_role: str
    profile: str
    architecture: str
    mcc_force_source: str | None = None

    @property
    def mcc_enabled(self) -> bool:
        return self.mcc_force_source is not None

    @property
    def requires_controller_override(self) -> bool:
        return self.mcc_enabled

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.name,
            "output_name": self.output_name,
            "checkpoint_role": self.checkpoint_role,
            "profile": self.profile,
            "architecture": self.architecture,
            "mcc_enabled": self.mcc_enabled,
            "mcc_force_source": self.mcc_force_source,
            "requires_controller_override": self.requires_controller_override,
        }


EVALUATION_ROW_SPECS = (
    EvaluationRowSpec(
        name="baseline",
        output_name="diffusion",
        checkpoint_role="baseline",
        profile="baseline",
        architecture="nominal",
    ),
    EvaluationRowSpec(
        name="mcc",
        output_name="diffusion_mcc_sensorless",
        checkpoint_role="baseline",
        profile="baseline",
        architecture="nominal",
        mcc_force_source="sensorless",
    ),
    EvaluationRowSpec(
        name="oracle",
        output_name="diffusion_mcc_oracle",
        checkpoint_role="baseline",
        profile="baseline",
        architecture="nominal",
        mcc_force_source="contact",
    ),
    EvaluationRowSpec(
        name="prism",
        output_name="prism",
        checkpoint_role="prism",
        profile="prism",
        architecture="from_checkpoint",
    ),
)
EVALUATION_ROWS = {specification.name: specification for specification in EVALUATION_ROW_SPECS}
EVALUATION_ROW_NAMES = tuple(EVALUATION_ROWS)

if len(EVALUATION_ROWS) != len(EVALUATION_ROW_SPECS):
    raise RuntimeError("Duplicate Diffusion evaluation row name")


@dataclass(frozen=True)
class PerturbationSpec:
    """One fixed robustness condition, in published condition order."""

    name: str
    action_delay: int
    proprio_noise: float
    image_noise: float

    def as_tuple(self) -> tuple[int, float, float]:
        return (self.action_delay, self.proprio_noise, self.image_noise)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_delay": self.action_delay,
            "proprio_noise": self.proprio_noise,
            "image_noise": self.image_noise,
        }


ROBUSTNESS_PERTURBATIONS = (
    PerturbationSpec("clean", 0, 0.0, 0.0),
    PerturbationSpec("action_delay", 1, 0.0, 0.0),
    PerturbationSpec("proprio_noise", 0, 0.01, 0.0),
    PerturbationSpec("image_corrupt", 0, 0.0, 0.05),
    PerturbationSpec("combined", 1, 0.01, 0.03),
)


@dataclass(frozen=True)
class MCCSweepSpec:
    """One sensorless MCC validation configuration in c00--c08 order."""

    name: str
    gain: float
    delay: int
    ema: float
    noise: float
    correction_clip: float

    def as_tuple(self) -> tuple[float, int, float, float, float]:
        return (self.gain, self.delay, self.ema, self.noise, self.correction_clip)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gain": self.gain,
            "delay": self.delay,
            "ema": self.ema,
            "noise": self.noise,
            "correction_clip": self.correction_clip,
        }


MCC_SWEEP_SPECS = (
    MCCSweepSpec("c00", 0.005, 0, 0.70, 0.05, 0.05),
    MCCSweepSpec("c01", 0.005, 1, 0.90, 0.05, 0.05),
    MCCSweepSpec("c02", 0.010, 0, 0.70, 0.05, 0.05),
    MCCSweepSpec("c03", 0.010, 1, 0.90, 0.05, 0.05),
    MCCSweepSpec("c04", 0.015, 1, 0.90, 0.15, 0.05),
    MCCSweepSpec("c05", 0.015, 2, 0.90, 0.15, 0.05),
    MCCSweepSpec("c06", 0.020, 1, 0.90, 0.15, 0.05),
    MCCSweepSpec("c07", 0.010, 1, 0.70, 0.15, 0.10),
    MCCSweepSpec("c08", 0.015, 1, 0.70, 0.05, 0.10),
)


# These exact task keys are part of both appendix sweep protocols.
VALIDATION_TASKS = ("libero_spatial:1", "libero_object:3", "libero_goal:7", "libero_10:1")


@dataclass(frozen=True)
class EvaluationRequest:
    """Fully resolved input to the dependency-free Diffusion command renderer."""

    baseline: Path | None
    prism: Path | None
    suite: str
    task_id: int
    output_root: Path
    episodes: int
    batch_size: int
    render_episodes: int
    seed: int
    controller: str
    mcc_gain: float
    mcc_noise: float
    mcc_delay: int
    mcc_ema: float
    mcc_clip: float | None
    action_delay: int
    proprio_noise: float
    image_noise: float
    save_probe_traces: bool
    python: Path | None
    device: str

    @classmethod
    def from_args(cls, args: Any) -> EvaluationRequest:
        """Adapt the existing CLI namespace without mutating it."""

        return cls(
            baseline=args.baseline,
            prism=args.prism,
            suite=args.suite,
            task_id=args.task_id,
            output_root=args.output_root,
            episodes=args.episodes,
            batch_size=args.batch_size,
            render_episodes=args.render_episodes,
            seed=args.seed,
            controller=args.controller,
            mcc_gain=args.mcc_gain,
            mcc_noise=args.mcc_noise,
            mcc_delay=args.mcc_delay,
            mcc_ema=args.mcc_ema,
            mcc_clip=args.mcc_clip,
            action_delay=args.action_delay,
            proprio_noise=args.proprio_noise,
            image_noise=args.image_noise,
            save_probe_traces=args.save_probe_traces,
            python=args.python,
            device=args.device,
        )

    def checkpoint_for(self, row: EvaluationRowSpec) -> Path | None:
        return self.prism if row.checkpoint_role == "prism" else self.baseline

    def environment(self) -> dict[str, str]:
        return {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LEROBOT_EVAL_ACTION_DELAY_STEPS": str(self.action_delay),
            "LEROBOT_EVAL_PROPRIO_NOISE_STD": str(self.proprio_noise),
            "LEROBOT_EVAL_IMAGE_NOISE_STD": str(self.image_noise),
            "LEROBOT_SAVE_PROBE_TRACES": "1" if self.save_probe_traces else "0",
        }

    def protocol_for(self, row: EvaluationRowSpec) -> dict[str, Any]:
        """Record all resolved control settings without changing the command."""

        mcc = {
            "enabled": row.mcc_enabled,
            "force_source": row.mcc_force_source,
            "gain": None,
            "noise": None,
            "delay": None,
            "ema": None,
            "correction_clip": None,
            "wrench_regularization": None,
        }
        if row.mcc_enabled:
            oracle = row.mcc_force_source == "contact"
            mcc.update(
                gain=self.mcc_gain,
                noise=0.0 if oracle else self.mcc_noise,
                delay=0 if oracle else self.mcc_delay,
                ema=self.mcc_ema,
                correction_clip=self.mcc_clip,
                wrench_regularization=EVALUATION_PROTOCOL.mcc_wrench_regularization,
            )
        return {
            "episodes": self.episodes,
            "batch_size": self.batch_size,
            "render_episodes": self.render_episodes,
            "seed": self.seed,
            "controller": self.controller,
            "continue_after_success": not EVALUATION_PROTOCOL.terminate_on_success,
            "action_delay": self.action_delay,
            "proprio_noise": self.proprio_noise,
            "image_noise": self.image_noise,
            "save_probe_traces": self.save_probe_traces,
            "mcc": mcc,
        }


def evaluation_row(name: str) -> EvaluationRowSpec:
    """Resolve one stable public evaluation row name."""

    try:
        return EVALUATION_ROWS[name]
    except KeyError as error:
        raise ValueError("Unknown Diffusion evaluation row: " + name) from error


def mcc_configs_text() -> str:
    """Render the historical configs.txt content byte-for-byte predictably."""

    return "".join(
        specification.name + " " + " ".join(map(str, specification.as_tuple())) + "\n"
        for specification in MCC_SWEEP_SPECS
    )


def main() -> None:
    """Print the complete released evaluation protocol without optional imports."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", "rows", "robustness", "mcc"), default="all")
    args = parser.parse_args()
    specification = {}
    if args.kind in {"all", "rows"}:
        specification["protocol"] = EVALUATION_PROTOCOL.__dict__
        specification["rows"] = [row.as_dict() for row in EVALUATION_ROW_SPECS]
    if args.kind in {"all", "robustness"}:
        specification["robustness"] = [condition.as_dict() for condition in ROBUSTNESS_PERTURBATIONS]
    if args.kind in {"all", "mcc"}:
        specification["mcc"] = [condition.as_dict() for condition in MCC_SWEEP_SPECS]
        specification["validation_tasks"] = list(VALIDATION_TASKS)
    print(json.dumps(specification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
