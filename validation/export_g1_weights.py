"""Extract tensor-only actors from the registered camera-ready checkpoint map."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_map", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.checkpoint_map.read_text())
    manifest = {
        "schema": "prism_g1_actor_release_v1",
        "license": "MIT",
        "mjlab_commit": "033ae22a2c7a30a25a6fa77b16c113ed88dd1b55",
        "observation_dim": 286,
        "action_dim": 29,
        "control_hz": 50,
        "normalization": "Native learned EmpiricalNormalization, epsilon 0.01; frozen during play",
        "observation_order": [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "height_scan",
        ],
        "scope": "Deterministic trained actor with normalization and distribution tensors; no critic or optimizer",
        "actors": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, variant in (("prism", "prism"), ("mlp", "standard")):
        entry = source["actors"][variant]
        path = Path(entry["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
            raise ValueError(f"Source checkpoint checksum mismatch: {variant}")
        original = torch.load(path, map_location="cpu", weights_only=True)
        state = original["actor_state_dict"]
        if not all(
            isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()
        ):
            raise ValueError("Actor state must contain only finite tensors")
        target = args.output / f"{name}.pt"
        torch.save(state, target)
        released = torch.load(target, map_location="cpu", weights_only=True)
        if state.keys() != released.keys() or not all(
            torch.equal(state[key], released[key]) for key in state
        ):
            raise ValueError("Released tensors differ from the trained actor")
        manifest["actors"][name] = {
            "file": target.name,
            "variant": variant,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
            "source_checkpoint_sha256": entry["checkpoint_sha256"],
            "training_seed": entry["training_seed"],
            "completed_updates": entry["completed_updates"],
            "completed_control_steps": entry["completed_control_steps"],
            "trainable_parameters": entry["actor_recipe"]["total_trainable_parameters"],
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["actors"], indent=2))


if __name__ == "__main__":
    main()
