"""Verify or idempotently install the simulation bundle on pinned LeRobot source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(target: Path, *, with_smolvla: bool = False, apply: bool = False) -> None:
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    components = [(ROOT / manifest["patch"], manifest["patch_sha256"], manifest["files"])]
    if with_smolvla:
        smolvla = manifest["smolvla"]
        components.append((ROOT / smolvla["patch"], smolvla["sha256"], smolvla["files"]))
    # Verify the shipped bundle before interpreting target state or writing files.
    for patch, expected_hash, _ in components:
        if file_hash(patch) != expected_hash:
            raise ValueError(f"Bundle patch hash mismatch: {patch}")
    for module in manifest["copied_modules"]:
        if file_hash(ROOT / module["source"]) != module["sha256"]:
            raise ValueError(f"Bundle adapter hash mismatch: {module['source']}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if revision != manifest["upstream_commit"]:
        raise ValueError(f"Expected LeRobot {manifest['upstream_commit']}; found {revision}")
    patches = []
    for patch, _, files in components:
        actual = {item["upstream_path"]: file_hash(target / item["upstream_path"]) for item in files}
        if all(actual[item["upstream_path"]] == item["release_sha256"] for item in files):
            continue
        if not all(actual[item["upstream_path"]] == item["upstream_sha256"] for item in files):
            raise ValueError(f"Mixed or modified target files for {patch.name}; use a fresh pinned checkout")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=target, check=True)
        patches.append(patch)
    modules = []
    for module in manifest["copied_modules"]:
        destination = target / module["target"]
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or file_hash(destination) != module["sha256"]
            ):
                raise ValueError(f"Refusing to overwrite modified adapter: {destination}")
        else:
            modules.append(module)
    print(f"Verified {revision}: {len(patches)} pending patch(es), {len(modules)} pending module(s)")
    if not patches and not modules:
        print("Integration already installed; no changes needed.")
        return
    if not apply:
        print("Preflight only. Add --apply to install the reviewed integration.")
        return
    for patch in patches:
        subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
    for module in modules:
        destination = target / module["target"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((ROOT / module["source"]).read_bytes())
    for _, _, files in components:
        for item in files:
            if file_hash(target / item["upstream_path"]) != item["release_sha256"]:
                raise ValueError(f"Post-install source hash mismatch: {item['upstream_path']}")
    print("Installed simulation adapters; next install the selected LeRobot dependency extras.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lerobot_root", type=Path)
    parser.add_argument("--with-smolvla", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply after all preflight checks pass")
    args = parser.parse_args()
    try:
        install(args.lerobot_root.resolve(), with_smolvla=args.with_smolvla, apply=args.apply)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
