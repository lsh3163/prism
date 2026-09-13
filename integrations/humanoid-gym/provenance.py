"""Portable training provenance; independent of Isaac Gym and model weights."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def repository_identity(path):
    """Report git identity when present; exported source trees remain supported."""

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(Path(path).parent), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=10,
        ).stdout.strip()

    try:
        return {
            "root": git("rev-parse", "--show-toplevel"),
            "commit": git("rev-parse", "HEAD"),
            "tracked_files_modified": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"root": None, "commit": None, "tracked_files_modified": None}


def source_snapshot(integration_dir, modules):
    """Hash the release and actually imported upstream Python implementations."""
    integration_dir = Path(integration_dir).resolve()
    sources = set(integration_dir.glob("*.py"))
    sources.add(integration_dir / "SOURCE_MANIFEST.json")
    upstream_roots = {}
    for name, module in modules.items():
        if name.split(".")[0] not in {"legged_gym", "rsl_rl"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py":
            sources.add(Path(filename).resolve())
            if name in {"legged_gym", "rsl_rl"}:
                upstream_roots[name] = filename
    repositories = {"release": repository_identity(integration_dir / "train.py")}
    repositories.update({name: repository_identity(path) for name, path in upstream_roots.items()})
    return {
        "files": [file_identity(path) for path in sorted(sources)],
        "repositories": repositories,
        "note": "File hashes cover release and imported upstream Python files. Git status excludes untracked files; this is not a complete dependency lock or asset archive.",
    }


def write_training_manifest(log_dir, payload):
    """Reserve a fresh run directory and record metadata without overwriting it."""
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    log_dir = Path(log_dir).resolve()
    # Fail even for an empty existing directory: a run must have one owner.
    log_dir.mkdir(parents=True, exist_ok=False)
    path = log_dir / "prism_run_manifest.json"
    with path.open("x", encoding="utf-8") as target:
        target.write(encoded)
    return path


def training_manifest(variant, task, command, args, env_config, train_config, runner, modules):
    policy = train_config["policy"]
    policy_class = train_config["runner"]["policy_class_name"]
    is_gated = policy_class == "GatedPolyActorCritic"
    is_polynomial = is_gated or policy_class == "PolyActorCritic"
    actor_identity = {
        "actor_variant": "g1_gated_poly_v2"
        if is_gated
        else ("g1_residual_poly_v1" if is_polynomial else None),
        "policy_type": "gated_polynomial"
        if is_gated
        else ("residual_polynomial" if is_polynomial else "mlp"),
        "recipe_variant": variant,
        "class": policy_class,
        "actor_hidden_dims": policy["actor_hidden_dims"],
        "polynomial_degree": policy.get("poly_degree") if is_polynomial else None,
        "warmup_ppo_updates": policy.get("actor_poly_warmup_updates", 0) if is_polynomial else None,
        "gate_semantics": "learned_per_feature"
        if is_gated
        else ("legacy_scalar_scale" if is_polynomial else None),
        "gate_init": policy.get("gate_init", 0.01) if is_gated else None,
    }
    model = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if model is not None:
        actor_identity["actor_mean_parameters"] = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(("actor_encoder.", "actor."))
        )
        actor_identity["total_actor_critic_parameters"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
    versions = {"python": platform.python_version(), "executable": sys.executable}
    for name in ("torch", "numpy", "isaacgym", "rsl_rl"):
        module = modules.get(name)
        versions[name] = str(getattr(module, "__version__", "not exposed"))
    torch_module = modules.get("torch")
    versions["torch_cuda"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    return {
        "schema_version": 1,
        "record_kind": "training_start",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "actor": actor_identity,
        "seed": train_config["seed"],
        "command": command,
        "cwd": str(Path.cwd()),
        # Isaac Gym uses a pybind SimType enum here rather than an int.
        "arguments": {
            key: str(value) if key == "physics_engine" else value for key, value in vars(args).items()
        },
        "environment_config": env_config,
        "training_config": train_config,
        "initial_learning_iteration": runner.current_learning_iteration,
        "runtime": versions,
        "source": source_snapshot(Path(__file__).parent, modules),
        "note": "Created before learn(); it records inputs and provenance, not training completion or paper performance. Metadata does not mutate model state; actor schema is recorded explicitly.",
    }
