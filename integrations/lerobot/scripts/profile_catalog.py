"""Canonical, dependency-free training profiles for the LeRobot integrations.

The profile names, output-directory components, and historical task JSON stay
publicly stable.  This module only resolves the settings that were previously
spread across command-building branches, so a recipe can be inspected without
importing LeRobot, LIBERO, or a simulator.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Keep these checkpoint identities close to the recipes that select them.  The
# common runner re-exports its historical constants for callers that use it.
DIFFUSION_GATED_ACTOR_ID = "diffusion_gated_state_v2"
DIFFUSION_LEGACY_ACTOR_ID = "diffusion_factorized_state_v1"
SMOLVLA_GATED_ACTOR_ID = "smolvla_gated_quadratic_v1"


@dataclass(frozen=True)
class DiffusionProfileSpec:
    """One accepted Diffusion training profile, including historical routing."""

    name: str
    canonical_name: str
    task_settings_profile: str
    batch_size_override: int | None
    num_workers_override: int | None
    conditioning_enabled: bool
    lift_mode: str
    actor_variant: str | None
    architecture: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        """Return only JSON-friendly, effective recipe fields."""

        return {
            "requested_profile": self.name,
            "canonical_profile": self.canonical_name,
            "task_settings_profile": self.task_settings_profile,
            "batch_size_override": self.batch_size_override,
            "num_workers_override": self.num_workers_override,
            "conditioning_enabled": self.conditioning_enabled,
            "lift_mode": self.lift_mode,
            "actor_variant": self.actor_variant,
            "architecture": self.architecture,
            "description": self.description,
            "inactive_conditioning_flags_emitted": not self.conditioning_enabled,
        }


@dataclass(frozen=True)
class ResolvedDiffusionProfile:
    """A Diffusion profile after its immutable task-level evidence is resolved."""

    specification: DiffusionProfileSpec
    batch_size: int
    num_workers: int
    historical_settings: tuple[tuple[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        """Expose both the selected command values and preserved source evidence."""

        return {
            **self.specification.as_dict(),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "historical_settings": dict(self.historical_settings),
        }


# Preserve every accepted public spelling.  Aliases remain individual entries
# because their requested name is part of output paths and run records.
DIFFUSION_PROFILE_SPECS = (
    DiffusionProfileSpec(
        name="prism",
        canonical_name="prism",
        task_settings_profile="legacy-prism",
        batch_size_override=None,
        num_workers_override=None,
        conditioning_enabled=True,
        lift_mode="gated_quadratic",
        actor_variant=DIFFUSION_GATED_ACTOR_ID,
        architecture="gated_state",
        description="New learned-gate PRISM, using the recorded PRISM task split and matched batch.",
    ),
    DiffusionProfileSpec(
        name="baseline",
        canonical_name="baseline",
        task_settings_profile="historical-baseline",
        batch_size_override=64,
        num_workers_override=8,
        conditioning_enabled=False,
        lift_mode="gated_quadratic",
        actor_variant=None,
        architecture="nominal",
        description="New nominal control matched to the gated PRISM batch and worker counts.",
    ),
    DiffusionProfileSpec(
        name="legacy-prism",
        canonical_name="legacy-prism",
        task_settings_profile="legacy-prism",
        batch_size_override=None,
        num_workers_override=None,
        conditioning_enabled=True,
        lift_mode="latent_quadratic",
        actor_variant=DIFFUSION_LEGACY_ACTOR_ID,
        architecture="factorized_state",
        description="Archived ungated factorized-state PRISM recipe.",
    ),
    DiffusionProfileSpec(
        name="legacy-baseline",
        canonical_name="legacy-baseline",
        task_settings_profile="historical-baseline",
        batch_size_override=None,
        num_workers_override=None,
        conditioning_enabled=False,
        lift_mode="latent_quadratic",
        actor_variant=None,
        architecture="nominal",
        description="Archived nominal recipe with its recorded batch and worker counts.",
    ),
    DiffusionProfileSpec(
        name="historical-baseline",
        canonical_name="legacy-baseline",
        task_settings_profile="historical-baseline",
        batch_size_override=None,
        num_workers_override=None,
        conditioning_enabled=False,
        lift_mode="latent_quadratic",
        actor_variant=None,
        architecture="nominal",
        description="Historical-name alias of legacy-baseline; output and metadata retain this spelling.",
    ),
    DiffusionProfileSpec(
        name="matched-baseline",
        canonical_name="baseline",
        task_settings_profile="historical-baseline",
        batch_size_override=64,
        num_workers_override=8,
        conditioning_enabled=False,
        lift_mode="gated_quadratic",
        actor_variant=None,
        architecture="nominal",
        description="Matched-control alias of baseline; output and metadata retain this spelling.",
    ),
)


@dataclass(frozen=True)
class SmolVLAProfileSpec:
    """One released SmolVLA state-conditioner training profile."""

    name: str
    conditioner_type: str
    num_layers: int
    hidden_dim: int | None
    product_mode: str
    gate_scale_init: float
    use_rmsnorm: bool
    actor_variant: str | None
    description: str

    @property
    def has_learned_gate(self) -> bool:
        return self.conditioner_type == "prism"

    def as_dict(self) -> dict[str, Any]:
        """Return the exact state-conditioner flags emitted by the runner."""

        return {
            "profile": self.name,
            "conditioner_type": self.conditioner_type,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "product_mode": self.product_mode,
            "gate_scale_init": self.gate_scale_init,
            "use_rmsnorm": self.use_rmsnorm,
            "actor_variant": self.actor_variant,
            "has_learned_gate": self.has_learned_gate,
            "description": self.description,
            # The upstream config accepts these flags for all conditioner types.
            # Keep emitting them for controls so historical command/config shape
            # stays fixed, while recording that they are inactive there.
            "inactive_product_settings_emitted": not self.has_learned_gate,
        }


SMOLVLA_PROFILE_SPECS = (
    SmolVLAProfileSpec(
        name="baseline",
        conditioner_type="linear",
        num_layers=2,
        hidden_dim=None,
        product_mode="gated_quadratic",
        gate_scale_init=0.01,
        use_rmsnorm=True,
        actor_variant=None,
        description="Linear state-projection control with the shared runner config.",
    ),
    SmolVLAProfileSpec(
        name="larger",
        conditioner_type="mlp",
        num_layers=3,
        hidden_dim=2048,
        product_mode="gated_quadratic",
        gate_scale_init=0.01,
        use_rmsnorm=True,
        actor_variant=None,
        description="2048-wide three-layer MLP state-projection capacity control.",
    ),
    SmolVLAProfileSpec(
        name="prism",
        conditioner_type="prism",
        num_layers=2,
        hidden_dim=None,
        product_mode="gated_quadratic",
        gate_scale_init=0.01,
        use_rmsnorm=True,
        actor_variant=SMOLVLA_GATED_ACTOR_ID,
        description="Two-layer RMSNorm gated-quadratic PRISM state conditioner.",
    ),
)


def _index_by_name(specifications: tuple[Any, ...], kind: str) -> dict[str, Any]:
    indexed = {specification.name: specification for specification in specifications}
    if len(indexed) != len(specifications):
        raise RuntimeError("Duplicate " + kind + " profile name")
    return indexed


DIFFUSION_PROFILES = _index_by_name(DIFFUSION_PROFILE_SPECS, "Diffusion")
SMOLVLA_PROFILES = _index_by_name(SMOLVLA_PROFILE_SPECS, "SmolVLA")
DIFFUSION_PROFILE_NAMES = tuple(DIFFUSION_PROFILES)
SMOLVLA_PROFILE_NAMES = tuple(SMOLVLA_PROFILES)


def diffusion_profile(name: str) -> DiffusionProfileSpec:
    """Resolve an accepted Diffusion public profile name."""

    try:
        return DIFFUSION_PROFILES[name]
    except KeyError as error:
        raise ValueError("Unknown Diffusion profile: " + name) from error


def resolve_diffusion_profile(
    name: str, task_profiles: Mapping[str, Mapping[str, Any]]
) -> ResolvedDiffusionProfile:
    """Resolve task-level historical defaults without mutating source evidence."""

    specification = diffusion_profile(name)
    try:
        historical = task_profiles[specification.task_settings_profile]
    except KeyError as error:
        raise ValueError(
            "Task does not provide the required historical profile "
            + repr(specification.task_settings_profile)
        ) from error
    if not isinstance(historical, Mapping):
        raise ValueError("Historical task profile must be a JSON object")
    try:
        historical_batch_size = historical["batch_size"]
        historical_num_workers = historical["num_workers"]
    except KeyError as error:
        raise ValueError("Historical task profile must declare batch_size and num_workers") from error
    if not isinstance(historical_batch_size, int) or not isinstance(historical_num_workers, int):
        raise ValueError("Historical batch_size and num_workers must be integers")
    return ResolvedDiffusionProfile(
        specification=specification,
        batch_size=(
            specification.batch_size_override
            if specification.batch_size_override is not None
            else historical_batch_size
        ),
        num_workers=(
            specification.num_workers_override
            if specification.num_workers_override is not None
            else historical_num_workers
        ),
        historical_settings=tuple(sorted(historical.items())),
    )


def smolvla_profile(name: str) -> SmolVLAProfileSpec:
    """Resolve an accepted SmolVLA public profile name."""

    try:
        return SMOLVLA_PROFILES[name]
    except KeyError as error:
        raise ValueError("Unknown SmolVLA profile: " + name) from error


def main() -> None:
    """Print the stable, fully declared profile catalog for human review."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", "diffusion", "smolvla"), default="all")
    args = parser.parse_args()
    catalog = {}
    if args.kind in {"all", "diffusion"}:
        catalog["diffusion"] = [specification.as_dict() for specification in DIFFUSION_PROFILE_SPECS]
    if args.kind in {"all", "smolvla"}:
        catalog["smolvla"] = [specification.as_dict() for specification in SMOLVLA_PROFILE_SPECS]
    print(json.dumps(catalog, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
