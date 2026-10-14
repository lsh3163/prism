"""Check that built distributions contain the core and a usable source release."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

CORE_FILES = {"prism_robot/__init__.py", "prism_robot/conditioner.py", "prism_robot/polynomial.py"}
CORE_TEST_FILES = {"tests/test_prism.py", "tests/test_conditioner.py"}
SOURCE_FILES = {
    "pyproject.toml",
    "NOTICE.md",
    "LICENSE",
    "README.md",
    "examples/policy.py",
    "docs/ADAPTERS.md",
    "docs/camera-ready-sources.json",
    "tests/test_prism.py",
    "tests/test_conditioner.py",
    "validation/check_distribution.py",
    *{"src/" + name for name in CORE_FILES},
}
EXCLUDED_DIRECTORIES = {
    "outputs",
    "checkpoints",
    "datasets",
    "results",
    "wandb",
    ".venv",
    "__pycache__",
    "integrations",
    "analysis",
    "configs",
    "assets",
}
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
            if path.parts[0] == "tests" and name not in CORE_TEST_FILES:
                raise ValueError(f"Non-core test was bundled: {name}")
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
