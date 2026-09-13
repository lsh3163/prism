#!/usr/bin/env python
"""Aggregate contact-response diagnostics from LIBERO eval_info.json files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _metrics_from_eval(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows: list[dict] = []
    for task in data.get("per_task", []):
        metrics = task.get("metrics", {})
        n = len(metrics.get("successes", []))
        for i in range(n):
            rows.append(
                {
                    "success": bool(metrics.get("successes", [False] * n)[i]),
                    "force": metrics.get("contact_force_traces", [[]])[i],
                    "state": metrics.get("state_traces", [[]])[i],
                    "smoothness": metrics.get("action_smoothnesses", [math.nan] * n)[i],
                    "peak": metrics.get("peak_contact_force_proxies", [math.nan] * n)[i],
                }
            )
    return rows


def _episode_contact_stats(
    force_trace: list[float],
    state_trace: list[list[float]],
    *,
    dt: float,
    contact_threshold: float,
    pre_window: int,
    post_window: int,
) -> dict[str, float] | None:
    force = np.asarray(force_trace, dtype=np.float64)
    states = np.asarray(state_trace, dtype=np.float64)
    if force.size < 3 or states.ndim != 2 or states.shape[0] < 3 or states.shape[1] < 3:
        return None
    n = min(force.shape[0], states.shape[0])
    force = np.nan_to_num(force[:n], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    eef_pos = states[:n, :3]
    speed = np.zeros(n, dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1) / max(dt, 1e-9)

    contact = np.flatnonzero(force > contact_threshold)
    if contact.size == 0:
        return None
    tc = int(contact[0])
    pre = speed[max(0, tc - pre_window) : tc]
    post = speed[tc : min(n, tc + post_window)]
    if pre.size == 0 or post.size == 0:
        return None

    after_force = force[tc:]
    after_speed = speed[tc:]
    return {
        "approach_speed": float(np.mean(pre)),
        "post_speed": float(np.mean(post)),
        "speed_drop": float(np.mean(pre) - np.mean(post)),
        "peak_force": float(np.max(after_force)),
        "contact_impulse": float(np.sum(after_force) * dt),
        "contact_power": float(np.mean(after_force * after_speed[: after_force.size])),
    }


def _summarize(paths: list[Path], args: argparse.Namespace) -> dict[str, float]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(_metrics_from_eval(path))
    successes = [r for r in rows if r["success"]]
    stats = [
        s
        for r in successes
        if (
            s := _episode_contact_stats(
                r["force"],
                r["state"],
                dt=args.dt,
                contact_threshold=args.contact_threshold,
                pre_window=args.pre_window,
                post_window=args.post_window,
            )
        )
        is not None
    ]

    def mean(key: str) -> float:
        vals = np.asarray([s[key] for s in stats], dtype=np.float64)
        return float(np.mean(vals)) if vals.size else float("nan")

    return {
        "n": len(rows),
        "n_success": len(successes),
        "n_contact_success": len(stats),
        "success_pct": 100.0 * len(successes) / len(rows) if rows else float("nan"),
        "approach_speed": mean("approach_speed"),
        "speed_drop": mean("speed_drop"),
        "peak_force": mean("peak_force"),
        "contact_impulse": mean("contact_impulse"),
        "contact_power": mean("contact_power"),
    }


def _fmt(x: float, digits: int = 2) -> str:
    return "--" if not np.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="Label=path/to/eval_info.json")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--contact-threshold", type=float, default=5.0)
    parser.add_argument("--pre-window", type=int, default=8)
    parser.add_argument("--post-window", type=int, default=8)
    args = parser.parse_args()

    summaries: dict[str, dict[str, float]] = {}
    run_paths: dict[str, list[Path]] = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be Label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        run_paths.setdefault(label, []).append(path)
    for label, paths in run_paths.items():
        summaries[label] = _summarize(paths, args)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summaries, indent=2))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Aggregate contact-response mechanism over successful episodes.} Metrics are aligned to first contact and averaged over successful contact episodes.}",
        r"\label{tab:aggregate_compliance_mechanism}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Success (\%) $\uparrow$ & Appr. Spd. $\uparrow$ & Spd. Drop $\uparrow$ & Peak $F$ $\downarrow$ & Impulse $\downarrow$ \\",
        r"\midrule",
    ]
    for label, s in summaries.items():
        lines.append(
            f"{label} & {_fmt(s['success_pct'], 1)} & {_fmt(s['approach_speed'])} & "
            f"{_fmt(s['speed_drop'])} & {_fmt(s['peak_force'], 1)} & {_fmt(s['contact_impulse'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text("\n".join(lines))
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.tex_output}")


if __name__ == "__main__":
    main()
