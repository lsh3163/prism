#!/usr/bin/env python3
"""Aggregate one benchmark metric per independently trained policy checkpoint.

Manifest records: method, training_seed, checkpoint, metrics_path, metric_path
(a JSON key/index list), and optional evaluation_seed. Task/episode aggregation
must already be complete inside each supplied metrics file.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate(manifest: Path, expected_seeds: int) -> dict:
    records = json.loads(manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty JSON list")
    grouped = defaultdict(list)
    for record in records:
        for key in ("method", "training_seed", "checkpoint", "metrics_path", "metric_path"):
            if key not in record:
                raise ValueError(f"Missing {key!r} in seed record")
        path = Path(record["metrics_path"]).expanduser()
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        value = json.loads(path.read_text())
        if not isinstance(record["metric_path"], list) or not record["metric_path"]:
            raise ValueError("metric_path must be a nonempty list of JSON keys or indices")
        for key in record["metric_path"]:
            value = value[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Expected finite scalar metric in {path}")
        grouped[record["method"]].append({**record, "metrics_path": str(path), "value": float(value)})
    summaries = {}
    for method, rows in grouped.items():
        seeds = [str(row["training_seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"{method}: repeated training seed; evaluation rollouts are not training replicates"
            )
        if len(seeds) != expected_seeds:
            raise ValueError(f"{method}: expected {expected_seeds} training seeds, received {len(seeds)}")
        checkpoints = [str(row["checkpoint"]) for row in rows]
        if len(checkpoints) != len(set(checkpoints)):
            raise ValueError(f"{method}: one checkpoint is assigned to multiple training seeds")
        values = [row["value"] for row in rows]
        summaries[method] = {
            "n_training_seeds": len(values),
            "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
            "runs": rows,
        }
    return {"aggregation": "Unweighted mean across training seeds; sample SD (ddof=1)", "methods": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-training-seeds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_training_seeds < 1:
        parser.error("--expected-training-seeds must be positive")
    result = aggregate(args.manifest, args.expected_training_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
