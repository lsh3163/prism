#!/usr/bin/env python3
"""Plot episode contact force and EEF speed, aligned to first contact.

Pass the simulation control timestep, not the frame rate of a rendered video.
This is a configurable diagnostic renderer; it does not select favorable trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Label=/path/to/eval_info.json")
    parser.add_argument("--episode", type=int, required=True, help="Same episode index in every supplied run")
    parser.add_argument("--dt", type=float, required=True, help="Seconds per recorded simulator step")
    parser.add_argument("--contact-threshold", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dt <= 0 or args.episode < 0:
        parser.error("--dt must be positive and --episode nonnegative")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
    metadata = {
        "dt": args.dt,
        "episode": args.episode,
        "contact_threshold_n": args.contact_threshold,
        "runs": [],
    }
    for item in args.run:
        label, separator, raw_path = item.partition("=")
        if not separator or not label:
            parser.error("--run must be Label=/path/to/eval_info.json")
        path = Path(raw_path)
        data = json.loads(path.read_text())
        if len(data["per_task"]) != 1:
            raise ValueError(f"Expected one task per file: {path}")
        metrics = data["per_task"][0]["metrics"]
        force = np.asarray(metrics["contact_force_traces"][args.episode], dtype=float)
        state = np.asarray(metrics["state_traces"][args.episode], dtype=float)
        if state.ndim != 2 or state.shape[1] < 3 or len(state) != len(force) or len(force) < 2:
            raise ValueError(f"Expected aligned force and state traces with at least 2 steps: {path}")
        if not np.isfinite(force).all() or not np.isfinite(state[:, :3]).all():
            raise ValueError(f"Non-finite traces: {path}")
        contact = np.flatnonzero(force > args.contact_threshold)
        if not contact.size:
            raise ValueError(f"Selected episode has no contact above the threshold: {path}")
        contact_step = int(contact[0])
        time = (np.arange(len(force)) - contact_step) * args.dt
        speed = np.zeros_like(force)
        speed[1:] = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1) / args.dt
        axes[0].plot(time, force, label=label)
        axes[1].plot(time, speed, label=label)
        metadata["runs"].append({"label": label, "path": str(path.resolve()), "contact_step": contact_step})
    for ax in axes:
        ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Contact force (N)")
    axes[1].set_ylabel("EEF speed (m/s)")
    axes[1].set_xlabel("Time relative to first contact (s)")
    axes[0].legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
