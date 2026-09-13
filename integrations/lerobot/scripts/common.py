"""Portable, preflighted simulation commands and machine-readable run records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ACTOR_IDS = {"diffusion": "diffusion_gated_state_v2", "smolvla": "smolvla_gated_quadratic_v1"}
DIFFUSION_LEGACY_ACTOR_ID = "diffusion_factorized_state_v1"


class Run:
    def __init__(
        self,
        command: list[str],
        output: Path,
        metadata: dict,
        environment: dict[str, str] | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        self.command = command
        self.output = output
        self.metadata = metadata
        self.environment = environment or {}
        self.checkpoint = checkpoint


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def probability(value: str) -> float:
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if not all(values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("use nonempty, unique comma-separated values")
    return values


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, help="Use this interpreter via uv --no-project")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--egl-vendor-file", type=Path, help="Optional installed EGL vendor JSON")
    parser.add_argument("--execute", action="store_true", help="Execute; otherwise print commands only")


def command_prefix(args: argparse.Namespace, action: str) -> list[str]:
    interpreter = getattr(args, "python", None)
    prefix = ["uv", "run", "--no-sync"]
    if interpreter is not None:
        prefix = ["uv", "run", "--no-project", "--python", str(interpreter.resolve())]
    return [*prefix, "python", "-m", f"lerobot.scripts.lerobot_{action}"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_installed(root: Path, *, with_smolvla: bool) -> dict:
    manifest_path = INTEGRATION_ROOT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Expected a pinned LeRobot Git checkout: {root}") from error
    if revision != manifest["upstream_commit"]:
        raise ValueError(f"Expected LeRobot {manifest['upstream_commit']}; found {revision}")
    files = list(manifest["files"])
    if with_smolvla:
        files.extend(manifest["smolvla"]["files"])
    expected = {item["upstream_path"]: item["release_sha256"] for item in files}
    expected.update({item["target"]: item["sha256"] for item in manifest["copied_modules"]})
    for path, digest in expected.items():
        if not (root / path).is_file() or sha256(root / path) != digest:
            raise ValueError(f"Installed source differs from the release: {root / path}; run install.py")
    return {
        "upstream_commit": manifest["upstream_commit"],
        "git_revision": revision,
        "manifest_sha256": sha256(manifest_path),
        "files": expected,
    }


def checkpoint_metadata(path: Path, policy: str, profile: str | None) -> dict:
    config_path = path / "config.json"
    weights = path / "model.safetensors"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config must be a JSON object: {config_path}")
    if config.get("type") != policy:
        raise ValueError(f"Expected a {policy} checkpoint: {config_path}")
    if policy == "diffusion":
        if config.get("use_poly_compliance", False):
            raise ValueError(f"Action-residual checkpoints are excluded: {config_path}")
        prism = config.get("use_poly_kernel_conditioning", False)
        actor_variant = None
        if prism:
            expected = {
                "poly_kernel_source": "state",
                "poly_kernel_latent_dim": 256,
                "poly_kernel_hidden_dim": 256,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Expected the released state-conditioning Diffusion actor: {config_path}")
            # A missing mode remains legacy, matching the pinned dataclass. New
            # gated runs always save an explicit mode; evaluation never overrides it.
            mode = config.get("poly_kernel_lift_mode", "latent_quadratic")
            if mode not in {"latent_quadratic", "gated_quadratic"}:
                raise ValueError(f"Unsupported Diffusion checkpoint lift mode {mode!r}: {config_path}")
            actor_variant = ACTOR_IDS[policy] if mode == "gated_quadratic" else DIFFUSION_LEGACY_ACTOR_ID
            architecture = "gated_state" if mode == "gated_quadratic" else "factorized_state"
        else:
            architecture = "nominal"
    else:
        if any(key.startswith("poly_interaction") or key == "use_poly_interaction" for key in config):
            raise ValueError(f"Legacy SmolVLA interaction schema is not supported: {config_path}")
        architecture = config.get("state_conditioner_type", "linear")
        prism = architecture == "prism"
        if architecture not in {"linear", "mlp", "prism"}:
            raise ValueError(f"Unknown SmolVLA conditioner: {config_path}")
        if prism:
            expected = {
                "state_conditioner_product_mode": "gated_quadratic",
                "state_conditioner_num_layers": 2,
                "state_conditioner_use_rmsnorm": True,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    f"Expected the two-layer, RMSNorm gated-quadratic SmolVLA actor: {config_path}"
                )
        actor_variant = ACTOR_IDS[policy] if prism else None
    if profile is not None:
        expected_architecture = {"baseline": "linear", "larger": "mlp", "prism": "prism"}
        mismatch = (profile == "prism") != prism
        if policy == "smolvla":
            mismatch = expected_architecture[profile] != architecture
        if mismatch:
            raise ValueError(f"Checkpoint does not match requested profile {profile}: {config_path}")
    return {
        "path": str(path),
        "config_sha256": sha256(config_path),
        "weights_sha256": sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "actor_variant": actor_variant,
        "architecture": architecture,
        "auxiliary_files": {
            str(item.relative_to(path)): sha256(item)
            for item in sorted(path.rglob("*"))
            if item.is_file() and item not in {config_path, weights}
        },
    }


def resolved_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def run_commands(
    runs: list[Run], args: argparse.Namespace, *, extra_files: dict[Path, str] | None = None
) -> None:
    """Preflight the whole plan before launching any run or creating any files."""
    root = args.lerobot_root.resolve()
    extra_files = extra_files or {}
    outputs = [resolved_path(run.output, root) for run in runs]
    records = [path.with_name(path.name + ".command.json") for path in outputs]
    destinations = [*outputs, *records, *(resolved_path(path, root) for path in extra_files)]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Run plan contains duplicate output or record paths")
    source = None
    checkpoints = {}
    if args.execute:
        for path in destinations:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Choose a fresh output path; refusing to overwrite {path}")
        if args.python is not None and not args.python.resolve().is_file():
            raise ValueError(f"Interpreter does not exist: {args.python}")
        if args.egl_vendor_file is not None and not args.egl_vendor_file.resolve().is_file():
            raise ValueError(f"EGL vendor JSON does not exist: {args.egl_vendor_file}")
        source = verify_installed(root, with_smolvla=any(run.metadata["policy"] == "smolvla" for run in runs))
        for run in runs:
            if run.checkpoint is not None:
                key = (
                    resolved_path(run.checkpoint, root),
                    run.metadata["policy"],
                    run.metadata.get("profile"),
                )
                if key not in checkpoints:
                    checkpoints[key] = checkpoint_metadata(*key)
        for path, content in extra_files.items():
            destination = resolved_path(path, root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x") as stream:
                stream.write(content)
    for run, record in zip(runs, records, strict=True):
        inherited_keys = (
            "CUDA_VISIBLE_DEVICES",
            "__EGL_VENDOR_LIBRARY_DIRS",
            "__EGL_VENDOR_LIBRARY_FILENAMES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
        environment = {key: os.environ[key] for key in inherited_keys if key in os.environ}
        environment.update(run.environment)
        # Explicit module execution plus this path selects the reviewed checkout even
        # when the interpreter has another editable LeRobot installation.
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")])
        )
        if args.egl_vendor_file is not None:
            environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(args.egl_vendor_file.resolve())
        printed = ["env", *(f"{key}={value}" for key, value in environment.items()), *run.command]
        print(f"cd {shlex.quote(str(root))} && {shlex.join(printed)}", flush=True)
        if not args.execute:
            continue
        metadata = dict(run.metadata)
        checkpoint = None
        if run.checkpoint is not None:
            key = (resolved_path(run.checkpoint, root), metadata["policy"], metadata.get("profile"))
            checkpoint = checkpoints[key]
            metadata["actor_variant"] = checkpoint["actor_variant"]
            metadata["architecture"] = checkpoint["architecture"]
        payload = {
            "schema_version": 1,
            "command": run.command,
            "cwd": str(root),
            "environment": environment,
            "run": metadata,
            "source": source,
            "checkpoint": checkpoint,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.parent.mkdir(parents=True, exist_ok=True)
        with record.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        try:
            result = subprocess.run(run.command, cwd=root, env={**os.environ, **environment}, check=False)
            payload.update(
                exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed"
            )
            if result.returncode:
                raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
        except (OSError, KeyboardInterrupt):
            payload["status"] = "interrupted_or_failed_to_start"
            raise
        finally:
            payload["finished_at"] = datetime.now(timezone.utc).isoformat()
            record.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def bool_arg(value: bool) -> str:
    return str(value).lower()
