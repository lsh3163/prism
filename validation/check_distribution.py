"""Check that built distributions contain the core and a usable source release."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

CORE_FILES = {"prism_robot/__init__.py", "prism_robot/conditioner.py"}
SOURCE_FILES = {
    "pyproject.toml",
    "Makefile",
    "NOTICE.md",
    "README.md",
    "CONTRIBUTING.md",
    "GATED_MIGRATION.md",
    "configs/actor_contracts.json",
    "integrations/humanoid-gym/actor.py",
    "integrations/humanoid-gym/gated_actor.py",
    "integrations/humanoid-gym/checkpoints.py",
    "integrations/humanoid-gym/config.py",
    "integrations/humanoid-gym/runtime.py",
    "integrations/humanoid-gym/provenance.py",
    "integrations/humanoid-gym/train.py",
    "integrations/humanoid-gym/evaluate.py",
    "integrations/humanoid-gym/run_suite.py",
    "integrations/humanoid-gym/evaluation_protocol.py",
    "integrations/humanoid-gym/build_main_table.py",
    "integrations/humanoid-gym/run_main_table.py",
    "integrations/humanoid-gym/residual_gated_actor.py",
    "integrations/humanoid-gym/run_residual_gate_controls.py",
    "integrations/humanoid-gym/run_capacity_controls.py",
    "integrations/humanoid-gym/run_robustness.py",
    "integrations/humanoid-gym/robustness.py",
    "integrations/humanoid-gym/ROBUSTNESS.md",
    "integrations/humanoid-gym/MAIN_TABLE.md",
    "integrations/humanoid-gym/RESIDUAL_GATE_CONTROL.md",
    "integrations/humanoid-gym/CAPACITY_COMPARISON.md",
    "integrations/humanoid-gym/g1_env.py",
    "integrations/humanoid-gym/legacy_base.py",
    "integrations/humanoid-gym/SOURCE_MANIFEST.json",
    "integrations/humanoid-gym/LICENSE.rsl-rl",
    "integrations/humanoid-gym/LICENSE.unitree",
    "integrations/lerobot/install.py",
    "integrations/lerobot/source_manifest.json",
    "integrations/lerobot/diffusion_libero.patch",
    "integrations/lerobot/diffusion_conditioner.py",
    "integrations/lerobot/recipes/gated_prism_task0_train_config.json",
    "integrations/lerobot/libero_episode.py",
    "integrations/lerobot/scripts/common.py",
    "integrations/lerobot/scripts/train_diffusion.py",
    "integrations/lerobot/scripts/eval_diffusion.py",
    "integrations/lerobot/scripts/sweep_diffusion.py",
    "integrations/lerobot/scripts/smolvla.py",
    "integrations/lerobot-smolvla.patch",
    "integrations/bfm-zero.patch",
    "integrations/bfm-zero-evaluation.patch",
    "integrations/bfm-zero/evaluate_scenarios.py",
    "integrations/bfm-zero/tracking_eval.py",
    "analysis/paper/aggregate_training_seeds.py",
    "validation/checkpoint_inventory.py",
    "validation/checkpoint_sources.example.json",
    "tests/test_actor_contracts.py",
    *{"src/" + name for name in CORE_FILES},
}
EXCLUDED_DIRECTORIES = {"outputs", "checkpoints", "datasets", "results", "wandb", ".venv", "__pycache__"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pyc"}


def check_distributions(directory: Path) -> None:
    """Reject incomplete bundles, research artifacts, or mismatched core files."""
    wheels = sorted(directory.glob("prism_robot-*.whl"))
    sources = sorted(directory.glob("prism_robot-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected one PRISM wheel and one source distribution; use a fresh build directory")
    with zipfile.ZipFile(wheels[0]) as wheel, tarfile.open(sources[0], "r:gz") as source:
        wheel_names = set(wheel.namelist())
        if missing := CORE_FILES - wheel_names:
            raise ValueError(f"Wheel is missing core modules: {sorted(missing)}")
        source_root = sources[0].name.removesuffix(".tar.gz")
        members = {}
        for item in source.getmembers():
            path = PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != source_root:
                raise ValueError(f"Invalid source distribution path: {item.name}")
            if not (item.isfile() or item.isdir()):
                raise ValueError(f"Unsupported source distribution entry: {item.name}")
            if item.isfile():
                name = str(path.relative_to(source_root))
                if name in members:
                    raise ValueError(f"Duplicate source distribution entry: {item.name}")
                members[name] = item
        if missing := SOURCE_FILES - members.keys():
            raise ValueError(f"Source distribution is missing release files: {sorted(missing)}")
        for name in [*members, *wheel_names]:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid distribution path: {name}")
            if EXCLUDED_DIRECTORIES.intersection(path.parts) or path.suffix in EXCLUDED_SUFFIXES:
                raise ValueError(f"Research/runtime artifact was bundled: {name}")
        for name in CORE_FILES:
            member = source.extractfile(members["src/" + name])
            if member is None or wheel.read(name) != member.read():
                raise ValueError(f"Wheel and source distribution disagree: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the wheel and source distribution")
    args = parser.parse_args()
    try:
        check_distributions(args.directory)
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print("Distribution contents and core source consistency verified.")


if __name__ == "__main__":
    main()
