#!/usr/bin/env python3
"""Summarize LIBERO robustness perturbation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ["Diffusion", "MCC-Sens.", r"\sname{}"]
VARIANTS = [
    ("clean", "Clean"),
    ("action_delay", "Act. Delay"),
    ("proprio_noise", "Prop. Noise"),
    ("image_corrupt", "Image Corr."),
    ("combined", "Combined"),
]


def success_rate(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text())
    succ = 0
    total = 0
    for row in data.get("per_task", []):
        successes = row.get("metrics", {}).get("successes", [])
        succ += sum(1 for value in successes if bool(value))
        total += len(successes)
    return succ, total


def fmt(rate: float | None) -> str:
    return "--" if rate is None else f"{rate:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    method_dirs = {
        "Diffusion": "diffusion",
        "MCC-Sens.": "diffusion_mcc_sensorless",
        r"\sname{}": "prism",
    }
    results: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for variant_key, _ in VARIANTS:
        results[variant_key] = {}
        for method, method_dir in method_dirs.items():
            succ = total = 0
            for eval_path in sorted((args.root / variant_key).glob(f"*_task_*/{method_dir}/eval_info.json")):
                s, n = success_rate(eval_path)
                succ += s
                total += n
            rate = None if total == 0 else 100.0 * succ / total
            results[variant_key][method] = {"success": rate, "n_success": succ, "n": total}

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{",
        r"\textbf{Robustness under deployment-time perturbations.}",
        r"Entries report success rate on the contact-heavy diagnostic subset under evaluation-only perturbations.",
        r"All methods use the same trained checkpoints as the clean evaluation.",
        r"}",
        r"\label{tab:appendix_robustness}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Clean & Act. Delay & Prop. Noise & Image Corr. & Combined \\",
        r"\midrule",
    ]
    column_rates: dict[str, list[float]] = {}
    for variant_key, _ in VARIANTS:
        column_rates[variant_key] = [
            float(results[variant_key][method]["success"])
            for method in METHODS
            if isinstance(results[variant_key][method]["success"], float)
        ]
    for method in METHODS:
        cells = []
        for variant_key, _ in VARIANTS:
            v = results[variant_key][method]["success"]
            text = fmt(v if isinstance(v, float) else None)
            if isinstance(v, float):
                ordered = sorted(column_rates[variant_key], reverse=True)
                if v == ordered[0]:
                    text = rf"\textbf{{{text}}}"
                elif len(ordered) > 1 and v == ordered[1]:
                    text = rf"\underline{{{text}}}"
            cells.append(text)
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text("\n".join(lines), encoding="utf-8")
    print(args.tex_output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
