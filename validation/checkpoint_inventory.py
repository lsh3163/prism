#!/usr/bin/env python3
"""Inventory simulation checkpoints/configs without inferring seeds from folders.

Roots and selection globs are explicit. Large checkpoint contents are not hashed
by default; safetensors headers provide a cheap schema fingerprint instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_from_saved_config(config: dict) -> dict:
    """Only the saved top-level training configuration establishes this field."""
    value = config.get("seed")
    if isinstance(value, int) and not isinstance(value, bool):
        return {"value": value, "evidence": "saved training config /seed"}
    return {"value": None, "evidence": "training seed absent or invalid in saved config"}


def directory_seed_hint(path: Path) -> int | None:
    match = re.search(r"(?:^|[_/])seed(\d+)(?:[_/]|$)", path.as_posix())
    return int(match.group(1)) if match else None


def normalize_nonfinite_values(document: dict) -> dict:
    """Retain unavailable historical metrics as null, recording each replacement."""
    replacements = []

    def visit(value, pointer: str):
        if isinstance(value, float) and not math.isfinite(value):
            replacements.append(
                {
                    "json_pointer": pointer,
                    "original_nonfinite_value": "NaN"
                    if math.isnan(value)
                    else ("Infinity" if value > 0 else "-Infinity"),
                }
            )
            return None
        if isinstance(value, dict):
            return {
                key: visit(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [visit(child, f"{pointer}/{index}") for index, child in enumerate(value)]
        return value

    result = visit(document, "")
    result["nonfinite_values"] = {
        "policy": "Nonfinite source metrics and derived values become null (unavailable); source files are unchanged. Aggregates containing a nonfinite value remain unavailable; no values are imputed or omitted.",
        "replacements": replacements,
    }
    return result


class Inventory:
    def __init__(self, roots: dict[str, Path], max_hash_bytes: int):
        self.roots = {name: path.resolve() for name, path in roots.items()}
        self.max_hash_bytes = max_hash_bytes
        self.fingerprints: dict[Path, dict] = {}

    def reference(self, path: Path) -> dict:
        resolved = path.resolve()
        # Most-specific root wins if, for example, BFM lies inside LeRobot.
        for name, root in sorted(self.roots.items(), key=lambda item: -len(item[1].parts)):
            if resolved.is_relative_to(root):
                return {"root": name, "path": resolved.relative_to(root).as_posix()}
        return {"root": None, "path": None, "status": "outside supplied source roots", "basename": path.name}

    def fingerprint(self, path: Path) -> dict:
        resolved = path.resolve()
        if resolved in self.fingerprints:
            return self.fingerprints[resolved]
        out = {**self.reference(path), "exists": path.is_file()}
        if path.is_file():
            out["size_bytes"] = path.stat().st_size
            if path.stat().st_size <= self.max_hash_bytes:
                out["sha256"] = sha256(path)
            else:
                out["sha256"] = None
                out["hash_status"] = "not hashed: exceeds configured whole-file byte limit"
        self.fingerprints[resolved] = out
        return out

    def safetensors_schema(self, path: Path) -> dict:
        if not path.is_file():
            return {"status": "checkpoint file missing"}
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            size = struct.unpack("<Q", size_bytes)[0]
            if size > 16 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large safetensors header: {path}")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"Incomplete safetensors header: {path}")
        header = json.loads(raw)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        selected = {
            key: {"shape": value["shape"], "dtype": value["dtype"]}
            for key, value in tensors.items()
            if any(token in key for token in ("poly", "state_proj", "state_output_norm", "conditioner"))
        }
        return {
            "status": "header inspected; full inference not verified",
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "n_tensors": len(tensors),
            "n_stored_elements": sum(math.prod(value["shape"]) for value in tensors.values()),
            "selected_tensors": selected,
        }

    def inspect_lerobot(self, path: Path) -> dict:
        config = json.loads(path.read_text())
        policy = config["policy"]
        checkpoint = path.parent / "model.safetensors"
        selected = {key: value for key, value in policy.items() if "poly" in key or "conditioner" in key}
        policy_kind = policy.get("type")
        identity = None
        identity_status = "unidentified representation"
        if policy_kind == "diffusion":
            if not policy.get("use_poly_kernel_conditioning", False):
                identity_status = (
                    "nominal Diffusion; historical feature ordering requires source verification"
                )
            elif policy.get("poly_kernel_source") == "state" and policy.get("poly_kernel_lift_mode") in {
                "latent_quadratic",
                "gated_quadratic",
            }:
                identity = (
                    "diffusion_gated_state_v2"
                    if policy["poly_kernel_lift_mode"] == "gated_quadratic"
                    else "diffusion_factorized_state_v1"
                )
                identity_status = (
                    "saved config selects representation; verify weight schema and source separately"
                )
        elif policy_kind == "smolvla":
            if (
                policy.get("state_conditioner_type") == "prism"
                and policy.get("state_conditioner_product_mode") == "gated_quadratic"
            ):
                identity = "smolvla_gated_quadratic_v1"
                identity_status = "saved config selects released gated state conditioner"
            elif policy.get("use_poly_interaction"):
                identity_status = (
                    "legacy use_poly_interaction schema; not the released gated state conditioner"
                )
            else:
                identity_status = "nominal SmolVLA configuration"
        preprocessing = path.parent / "policy_preprocessor.json"
        normalizers = []
        if preprocessing.exists():
            pipeline = json.loads(preprocessing.read_text())
            for step in pipeline.get("steps", []):
                if step.get("registry_name") == "normalizer_processor":
                    normalizers.append(
                        {
                            "norm_map": step["config"].get("norm_map"),
                            "eps": step["config"].get("eps"),
                            "state_file": self.fingerprint(path.parent / step["state_file"]),
                        }
                    )
        return {
            "training_seed": seed_from_saved_config(config),
            "directory_seed_hint": directory_seed_hint(path),
            "training_config": self.fingerprint(path),
            "policy_config": self.fingerprint(path.parent / "config.json"),
            "checkpoint": self.fingerprint(checkpoint),
            "checkpoint_step_label": path.parent.parent.name,
            "configured_training_steps": config.get("steps"),
            "batch_size": config.get("batch_size"),
            "num_workers": config.get("num_workers"),
            "dataset_repo_id": config.get("dataset", {}).get("repo_id"),
            "dataset_revision": config.get("dataset", {}).get("revision"),
            "selected_episode_count": len(config.get("dataset", {}).get("episodes") or []),
            "actor": {
                "variant_id": identity,
                "status": identity_status,
                "policy_type": policy_kind,
                "fields": selected,
            },
            "observation_features": policy.get("input_features"),
            "n_obs_steps": policy.get("n_obs_steps"),
            "policy_normalization": policy.get("normalization_mapping"),
            "saved_normalizers": normalizers,
            "checkpoint_schema": self.safetensors_schema(checkpoint),
            "unknown": [
                "historical training-source commit",
                "evaluation-to-checkpoint mapping unless an explicit evaluation reference is found",
            ],
        }

    def inspect_g1(self, path: Path) -> dict:
        import torch

        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        weights = payload["model_state_dict"]
        buffers = {
            key: weights[key].item()
            for key in ("actor_poly_warmup_step", "actor_encoder.poly_scale")
            if key in weights
        }
        has_residual = (
            "actor_encoder.raw_proj.weight" in weights and "actor_encoder.poly_in_proj.weight" in weights
        )
        has_gated = (
            "actor_variant_version" in weights
            and weights["actor_variant_version"].item() == 2
            and "actor_encoder.factors.0.weight" in weights
        )
        degree = sum(bool(re.fullmatch(r"actor_encoder\.A\.\d+\.weight", key)) for key in weights)
        gated_degree = sum(bool(re.fullmatch(r"actor_encoder\.factors\.\d+\.weight", key)) for key in weights)
        shapes = {key: list(value.shape) for key, value in weights.items()}
        infos = payload.get("infos")
        seed = (
            seed_from_saved_config(infos)
            if isinstance(infos, dict)
            else {
                "value": None,
                "evidence": "checkpoint has no saved training config/seed; infos is absent or null",
            }
        )
        return {
            "training_seed": seed,
            "directory_seed_hint": directory_seed_hint(path),
            "checkpoint": self.fingerprint(path),
            "checkpoint_iteration": payload.get("iter"),
            "checkpoint_top_level_keys": list(payload),
            "actor": {
                "variant_id": "g1_gated_poly_v2"
                if has_gated
                else ("g1_residual_poly_v1" if has_residual else None),
                "status": "gated encoder schema; activations and source must be established separately"
                if has_gated
                else (
                    "compatible residual encoder schema; activations and source must be established separately"
                    if has_residual
                    else "MLP tensor schema"
                ),
                "degree_from_A_matrices": degree if has_residual else None,
                "degree_from_gated_factors": gated_degree if has_gated else None,
                "gate_shape": shapes.get("actor_encoder.interaction_scales"),
                "checkpoint_buffers": buffers,
                "normalization_parameter_keys": [key for key in weights if "layer_norm" in key],
                "schema_shapes": shapes,
                "warmup_schedule_updates": 0 if has_gated else None,
                "warmup_note": "gated v2 has no scalar warmup"
                if has_gated
                else "saved current step/scale do not determine the original configured ramp duration",
            },
            "unknown": [
                "independently recorded training seed",
                "historical training-source commit",
                "training config and warmup duration",
            ],
        }

    def inspect_bfm(self, path: Path) -> dict:
        config = json.loads(path.read_text())
        model = config.get("agent", {}).get("model", {})
        actor = model.get("archi", {}).get("actor", {})
        filt = actor.get("input_filter", {})
        is_released = (
            "prism" in str(filt.get("name", "")).lower()
            and filt.get("history_key", "history_actor") == "history_actor"
        )
        checkpoint_paths = sorted(path.parent.glob("checkpoint/**/*"))
        files = [p for p in checkpoint_paths if p.is_file() and p.suffix in {".pt", ".pth", ".safetensors"}]
        return {
            "training_seed": seed_from_saved_config(config),
            "directory_seed_hint": directory_seed_hint(path),
            "training_config": self.fingerprint(path),
            "configured_environment_steps": config.get("num_env_steps"),
            "actor": {
                "variant_id": "bfm_history_gated_quadratic_v1"
                if is_released and filt.get("product_mode") == "gated_quadratic"
                else None,
                "status": "saved BFM actor configuration; paper correspondence unverified",
                "configuration": actor,
                "observation_normalizer": model.get("obs_normalizer"),
            },
            "checkpoint_candidates": [self.fingerprint(p) for p in files],
            "unknown": [
                "exact aligned 9.6M checkpoint",
                "scenario metadata",
                "mapping to released seed-1000 result",
            ],
        }

    def inspect_evaluation(self, path: Path, kind: str) -> dict:
        data = json.loads(path.read_text())
        record = {
            "file": self.fingerprint(path),
            "kind": kind,
            "checkpoint_reference": None,
            "evaluation_seed": None,
        }
        if kind == "g1":
            result = data.get("result", {})
            record["evaluation_seed"] = data.get("metadata", {}).get("seed")
            record["protocol"] = data.get("metadata")
            if result.get("checkpoint_path"):
                record["checkpoint_reference"] = self.reference(Path(result["checkpoint_path"]))
            record["metrics"] = {
                key: value for key, value in result.items() if isinstance(value, int | float)
            }
        else:
            record["overall"] = data.get("overall", {})
            record["per_task_counts"] = [
                {
                    "task_name": row.get("task"),
                    "n": len(row.get("metrics", {}).get("successes", [])),
                    "n_success": sum(bool(value) for value in row.get("metrics", {}).get("successes", [])),
                }
                for row in data.get("per_task", [])
            ]
            # The historical files usually contain metrics alone, not run metadata.
            record["metadata_present"] = "metadata" in data
            if isinstance(data.get("metadata"), dict):
                record["evaluation_seed"] = data["metadata"].get("seed")
            record["association_status"] = "candidate selected by path; no checkpoint identity assumed"
        return record

    def build(self, spec: dict) -> dict:
        records = []
        coverage = []
        for source in spec["sources"]:
            root = self.roots[source["root"]]
            matched_paths = list(root.glob(source["glob"]))
            paths = sorted({path.resolve() for path in matched_paths})
            coverage.append(
                {
                    "id": source["id"],
                    "root": source["root"],
                    "glob": source["glob"],
                    "matches": len(paths),
                    "matched_paths_including_aliases": len(matched_paths),
                }
            )
            for path in paths:
                inspector = {"g1": self.inspect_g1, "lerobot": self.inspect_lerobot, "bfm": self.inspect_bfm}[
                    source["kind"]
                ]
                record = inspector(path)
                record.update(
                    {
                        "source_id": source["id"],
                        "kind": source["kind"],
                        "paper_candidates": source.get("paper_candidates", []),
                    }
                )
                records.append(record)
        evaluations = []
        for source in spec.get("evaluation_sources", []):
            paths = sorted(self.roots[source["root"]].glob(source["glob"]))
            coverage.append(
                {"id": source["id"], "root": source["root"], "glob": source["glob"], "matches": len(paths)}
            )
            for path in paths:
                evaluation = self.inspect_evaluation(path, source["kind"])
                evaluation["source_id"] = source["id"]
                evaluations.append(evaluation)
        summaries = defaultdict(
            lambda: {"records": 0, "saved_training_seeds": set(), "unknown_training_seed_records": 0}
        )
        for record in records:
            row = summaries[record["source_id"]]
            row["records"] += 1
            seed = record["training_seed"]["value"]
            if seed is None:
                row["unknown_training_seed_records"] += 1
            else:
                row["saved_training_seeds"].add(seed)
        for row in summaries.values():
            row["saved_training_seeds"] = sorted(row["saved_training_seeds"])
        return normalize_nonfinite_values(
            {
                "schema_version": 1,
                "source_root_names": sorted(self.roots),
                "whole_file_hash_limit_bytes": self.max_hash_bytes,
                "scope": "Targeted local simulation evidence only; repeated checkpoint steps/tasks are not independent seeds",
                "coverage": coverage,
                "summary": dict(summaries),
                "checkpoints": records,
                "evaluations": evaluations,
                "g1_evaluation_aggregation": aggregate_g1_evaluations(evaluations),
                "paper_metric_checks": compare_paper_targets(spec.get("paper_targets", []), evaluations),
            }
        )


def aggregate_g1_evaluations(evaluations: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in evaluations:
        if row["kind"] == "g1":
            grouped[row["protocol"]["variant_name"]].append(row)
    summary = {}
    for method, rows in grouped.items():
        metric_names = [
            "avg_return",
            "avg_episode_length",
            "avg_lin_vel_error",
            "avg_yaw_vel_error",
            "success_rate",
        ]
        summary[method] = {
            "n_evaluation_files": len(rows),
            "evaluation_seeds": [row["evaluation_seed"] for row in rows],
            "training_seed_status": "not independently stored in checkpoint; do not relabel evaluation seeds",
            "mean_of_evaluation_file_means": {
                key: sum(row["metrics"][key] for row in rows) / len(rows) for key in metric_names
            },
            "source_files": [row["file"] for row in rows],
        }
    return summary


def compare_paper_targets(targets: list[dict], evaluations: list[dict]) -> list[dict]:
    """Check printed-number agreement; this is not a proof of seed/source identity."""
    checks = []
    g1 = aggregate_g1_evaluations(evaluations)
    for target in targets:
        checks_for_row = {}
        if target["kind"] == "g1":
            aggregate = g1.get(target["variant"])
            for metric, (reported, digits, scale) in target["expected"].items():
                observed = aggregate["mean_of_evaluation_file_means"][metric] * scale if aggregate else None
                checks_for_row[metric] = {
                    "reported": reported,
                    "observed": observed,
                    "matches_printed_precision": observed is not None and round(observed, digits) == reported,
                }
            n_files = aggregate["n_evaluation_files"] if aggregate else 0
        else:
            groups = defaultdict(list)
            for row in evaluations:
                if row["source_id"] not in target["evaluation_sources"]:
                    continue
                path = Path(row["file"]["path"])
                if path.parent.name != target["method_directory"]:
                    continue
                groups[path.parents[2].name].append(row["overall"]["pc_success"])
            for suite, reported in target["expected_suite_success_pct"].items():
                values = groups[suite]
                observed = sum(values) / len(values) if values else None
                checks_for_row[suite] = {
                    "reported": reported,
                    "observed": observed,
                    "n_task_files": len(values),
                    "matches_printed_precision": observed is not None and round(observed, 1) == reported,
                }
            n_files = sum(len(values) for values in groups.values())
        checks.append(
            {
                "table": target["table"],
                "method": target.get("variant", target.get("method_directory")),
                "n_evaluation_files": n_files,
                "meaning": "Agreement of archived metric values only; training seeds/checkpoint association require independent evidence",
                "checks": checks_for_row,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", action="append", required=True, help="Name=/absolute/source/root; repeat per root"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hash-mib", type=int, default=32)
    args = parser.parse_args()
    roots = {}
    for item in args.root:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or name in roots:
            parser.error("Each --root must have a unique Name=/path")
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            parser.error(f"Source root does not exist: {path}")
        roots[name] = path
    if args.max_hash_mib < 0:
        parser.error("--max-hash-mib must be nonnegative")
    spec = json.loads(args.spec.read_text())
    required = {source["root"] for source in spec["sources"] + spec.get("evaluation_sources", [])}
    if required - roots.keys():
        parser.error(f"Missing source roots: {sorted(required - roots.keys())}")
    result = Inventory(roots, args.max_hash_mib * 1024 * 1024).build(spec)
    result["selection_spec_sha256"] = sha256(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
