#!/usr/bin/env python3
"""Plot per-method t-SNE embeddings from BFM-Zero representation exports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SCENARIO_STYLE = {
    "Nominal": ("#4f7a45", "o"),
    "Low friction": ("#3478c7", "^"),
    "Payload mass": ("#d27a32", "s"),
}


def _parse_input(specification: str) -> tuple[str, str, Path]:
    label, separator, raw_path = specification.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected METHOD:SCENARIO=/path/to/file.npz")
    method, separator, scenario = label.partition(":")
    if not separator or not method or not scenario:
        raise argparse.ArgumentTypeError("Expected METHOD:SCENARIO=/path/to/file.npz")
    return method.strip(), scenario.strip(), Path(raw_path).expanduser()


def _load_features(path: Path, feature_key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if feature_key not in archive:
            available = ", ".join(sorted(archive.files))
            raise KeyError(f"{path} has no {feature_key!r}; available: {available}")
        features = np.asarray(archive[feature_key], dtype=np.float32)
    if features.ndim < 2:
        raise ValueError(f"Expected at least 2D features in {path}, got {features.shape}")
    return features.reshape(features.shape[0], -1)


def _embed(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if len(features) < 3:
        raise ValueError("Each method needs at least three representation samples")
    scaled = StandardScaler().fit_transform(features)
    effective_perplexity = min(perplexity, float(len(features) - 1))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(scaled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_parse_input,
        metavar="METHOD:SCENARIO=NPZ",
    )
    parser.add_argument("--feature-key", default="history_conditioned")
    parser.add_argument("--output", type=Path, default=Path("bfm_scenario_tsne.pdf"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for method, scenario, path in args.input:
        grouped[method].append((scenario, _load_features(path, args.feature_key)))

    figure, axes = plt.subplots(
        1,
        len(grouped),
        figsize=(4.2 * len(grouped), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (method, scenario_groups) in zip(axes[0], grouped.items()):
        feature_blocks = [features for _, features in scenario_groups]
        labels = np.concatenate(
            [np.full(len(features), scenario, dtype=object) for scenario, features in scenario_groups]
        )
        coordinates = _embed(np.concatenate(feature_blocks), args.seed, args.perplexity)

        for scenario in dict.fromkeys(labels):
            mask = labels == scenario
            color, marker = SCENARIO_STYLE.get(scenario, ("#777777", "o"))
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                label=scenario,
                color=color,
                marker=marker,
                s=54,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.45,
            )

        axis.set_title(method, fontweight="semibold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(frameon=False, fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
