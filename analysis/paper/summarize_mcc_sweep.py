#!/usr/bin/env python
"""Summarize MCC-Sensorless validation sweeps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _episode_rows(info: dict) -> list[dict]:
    rows: list[dict] = []
    for task in info.get("per_task", []):
        metrics = task.get("metrics", {})
        n = len(metrics.get("successes", []))
        for i in range(n):
            force_trace = metrics.get("contact_force_traces", [[]])[i]
            force = np.asarray(force_trace, dtype=np.float64)
            force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
            rows.append(
                {
                    "success": bool(metrics.get("successes", [False] * n)[i]),
                    "peak_force": float(np.nanmax(force)) if force.size else float("nan"),
                    "contact_impulse": float(np.nansum(force) * 0.05) if force.size else float("nan"),
                    "smoothness": float(metrics.get("action_smoothnesses", [math.nan] * n)[i]),
                    "excess_contact": float(metrics.get("excess_contact_rates", [math.nan] * n)[i]),
                }
            )
    return rows


def _summarize(paths: list[Path]) -> dict:
    rows: list[dict] = []
    for path in paths:
        info = _load_eval(path)
        if info is not None:
            rows.extend(_episode_rows(info))

    def mean(key: str, *, success_only: bool = False) -> float:
        source = [r for r in rows if (r["success"] or not success_only)]
        vals = np.asarray([r[key] for r in source], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    n = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    return {
        "n": n,
        "n_success": n_success,
        "success_pct": 100.0 * n_success / n if n else float("nan"),
        "peak_force": mean("peak_force"),
        "contact_impulse": mean("contact_impulse"),
        "success_peak_force": mean("peak_force", success_only=True),
        "success_contact_impulse": mean("contact_impulse", success_only=True),
        "smoothness": mean("smoothness"),
        "excess_contact_pct": 100.0 * mean("excess_contact"),
    }


def _read_configs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cfg_id, gain, delay, ema, noise, clip = line.split()
        rows.append(
            {
                "cfg_id": cfg_id,
                "gain": float(gain),
                "delay": int(delay),
                "ema": float(ema),
                "noise": float(noise),
                "clip": float(clip),
            }
        )
    return rows


def _fmt(x: float, digits: int = 1) -> str:
    return "--" if not np.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for cfg in _read_configs(args.configs):
        paths = sorted((args.root / cfg["cfg_id"]).glob("*/diffusion_mcc_sensorless/eval_info.json"))
        summary = _summarize(paths)
        summaries.append({**cfg, **summary})

    # Rank by success first, then successful contact impulse, then peak force.
    summaries.sort(
        key=lambda row: (
            -row["success_pct"],
            row["success_contact_impulse"] if np.isfinite(row["success_contact_impulse"]) else float("inf"),
            row["peak_force"] if np.isfinite(row["peak_force"]) else float("inf"),
        )
    )
    if summaries:
        summaries[0]["selected"] = True

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summaries, indent=2))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Validation sweep for MCC-Sensorless.} We tune MCC evaluation-time parameters on a held-out validation subset and select by success rate, then successful-episode contact impulse.}",
        r"\label{tab:mcc_sensorless_sweep}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Cfg. & $k_f$ & Delay & EMA & Noise & Success (\%) $\uparrow$ & Impulse$^\dagger$ $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summaries[:8]:
        cfg = rf"\textbf{{{row['cfg_id']}}}" if row.get("selected") else row["cfg_id"]
        lines.append(
            f"{cfg} & {_fmt(row['gain'], 3)} & {row['delay']} & {_fmt(row['ema'], 2)} & "
            f"{_fmt(row['noise'], 2)} & {_fmt(row['success_pct'], 1)} & "
            f"{_fmt(row['success_contact_impulse'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text("\n".join(lines))
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.tex_output}")
    if summaries:
        best = summaries[0]
        print(
            "Selected "
            f"{best['cfg_id']}: gain={best['gain']}, delay={best['delay']}, "
            f"ema={best['ema']}, noise={best['noise']}, clip={best['clip']}, "
            f"success={best['success_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
