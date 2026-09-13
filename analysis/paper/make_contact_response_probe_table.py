#!/usr/bin/env python3
"""Probe contact-conditioned future physical response from frozen LIBERO latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

# Paths are explicit to prevent silently selecting a different experiment/checkpoint.
PROBE_SPECS: list[dict] = []
OUT_TEX = Path("contact_response_probe_table.tex")
OUT_JSON = Path("contact_response_probe_table.json")
RIDGE = 10.0
MAX_SAMPLES = 12000
HORIZON = 5
CONTACT_THRESHOLD_N = 1.0
SKIPPED_NO_CONTACT: list[tuple[str, int]] = []
BASELINE_FEATURE_MODE: str | None = None
BASELINE_NAMES = {
    "state-first-contribution": "Diffusion State Contribution",
    "historical-surrogate-audit": "Historical Surrogate (Audit Only)",
}


def load_specs(path: Path) -> list[dict]:
    """Resolve each experiment's exact checkpoint and trace relative to a JSON spec."""
    specs = json.loads(path.read_text())
    if not isinstance(specs, list) or not specs:
        raise ValueError("--specs must contain a nonempty JSON list")
    for spec in specs:
        for key in ("name", "suite", "task_id", "base_policy", "prism_policy", "trace_path"):
            if key not in spec:
                raise ValueError(f"Missing {key!r} in probe spec")
        for key in ("base_policy", "prism_policy", "trace_path"):
            candidate = Path(spec[key]).expanduser()
            spec[key] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
    return specs


def base_policy_path(suite: dict, task_id: int) -> Path:
    return suite["base_policy"]


def prism_policy_path(suite: dict, task_id: int) -> Path:
    return suite["prism_policy"]


def probe_trace_path(suite: dict, task_id: int) -> Path:
    return suite["trace_path"]


def load_stats(policy_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    stats = load_file(str(policy_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    mean = stats["observation.state.mean"].float()
    std = stats["observation.state.std"].float().clamp_min(1e-6)
    return mean, std


def normalize_state_pair(prev: torch.Tensor, curr: torch.Tensor, policy_dir: Path) -> torch.Tensor:
    if BASELINE_FEATURE_MODE == "state-first-contribution":
        config = json.loads((policy_dir / "config.json").read_text())
        state_shape = config["input_features"]["observation.state"]["shape"]
        if config["n_obs_steps"] != 2 or list(prev.shape[1:]) != state_shape or curr.shape != prev.shape:
            raise ValueError(
                f"Recorded state history disagrees with the checkpoint configuration: {policy_dir}"
            )
        processor = json.loads((policy_dir / "policy_preprocessor.json").read_text())
        normalizers = [step for step in processor["steps"] if step["registry_name"] == "normalizer_processor"]
        if len(normalizers) != 1:
            raise ValueError(f"Expected one saved normalizer in {policy_dir}")
        step = normalizers[0]
        stats = load_file(str(policy_dir / step["state_file"]))
        mode = step["config"]["norm_map"]["STATE"]
        eps = float(step["config"]["eps"])
        if mode != config["normalization_mapping"]["STATE"]:
            raise ValueError(f"Saved policy and preprocessor STATE normalization disagree: {policy_dir}")
        return torch.cat(
            [
                normalize_saved_state(prev.float(), stats, mode, eps),
                normalize_saved_state(curr.float(), stats, mode, eps),
            ],
            dim=-1,
        )
    if BASELINE_FEATURE_MODE != "historical-surrogate-audit":
        raise ValueError("Select an explicit baseline feature mode before extracting representations")
    # Original analysis normalization, not necessarily the trained policy's normalization.
    mean, std = load_stats(policy_dir)
    return torch.cat([(prev.float() - mean) / std, (curr.float() - mean) / std], dim=-1)


def normalize_saved_state(x: torch.Tensor, stats: dict, mode: str, eps: float) -> torch.Tensor:
    """Match the pinned LeRobot normalizer for supported saved STATE modes."""
    if mode == "IDENTITY":
        return x
    if mode == "MEAN_STD":
        return (x - stats["observation.state.mean"]) / (stats["observation.state.std"] + eps)
    if mode == "MIN_MAX":
        minimum = stats["observation.state.min"]
        denominator = stats["observation.state.max"] - minimum
        denominator = torch.where(denominator == 0, torch.full_like(denominator, eps), denominator)
        return 2 * (x - minimum) / denominator - 1
    raise ValueError(f"Unsupported saved STATE normalization {mode!r}; use the full saved preprocessor")


def load_prism_weights(policy_dir: Path) -> dict[str, torch.Tensor]:
    if BASELINE_FEATURE_MODE == "state-first-contribution":
        config = json.loads((policy_dir / "config.json").read_text())
        if (
            not config.get("use_poly_kernel_conditioning")
            or config.get("poly_kernel_source") != "state"
            or config.get("poly_kernel_lift_mode") != "latent_quadratic"
        ):
            raise ValueError("PRISM features require the main state-source latent-quadratic conditioner")
    sd = load_file(str(policy_dir / "model.safetensors"))
    prefix = "diffusion.poly_kernel_conditioner."
    return {key.removeprefix(prefix): value.float() for key, value in sd.items() if key.startswith(prefix)}


def prism_latents(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weights["input_norm.weight"],
        bias=weights["input_norm.bias"],
    )
    left = torch.nn.functional.linear(x, weights["left_proj.weight"], weights["left_proj.bias"])
    right = torch.nn.functional.linear(x, weights["right_proj.weight"], weights["right_proj.bias"])
    linear = torch.cat([left, right], dim=-1)
    poly = left * right
    return linear, poly


def state_first_contribution(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, config: dict
) -> torch.Tensor:
    """State-dependent FiLM projection plus bias; excludes image/timestep contributions.

    Supported layout: [timestep embedding, flattened state history, image history,
    environment history], as in the released main Diffusion integration.
    """
    if config["type"] != "diffusion" or config.get("use_poly_kernel_conditioning", False):
        raise ValueError("State-first baseline requires a nominal Diffusion checkpoint")
    features = config["input_features"]
    state_shape = features["observation.state"]["shape"]
    if len(state_shape) != 1 or config["n_obs_steps"] != 2:
        raise ValueError("This contact-window extractor supports two observations of a flat robot state")
    state_dim = int(state_shape[0]) * int(config["n_obs_steps"])
    offset = int(config["diffusion_step_embed_dim"])
    if x.ndim != 2 or x.shape[1] != state_dim or state_dim < 1 or offset < 1:
        raise ValueError("Probe state shape disagrees with the saved checkpoint configuration")
    visual_count = sum(feature["type"] == "VISUAL" for feature in features.values())
    keypoints = config["spatial_softmax_num_keypoints"]
    if visual_count and (not isinstance(keypoints, int) or keypoints < 1):
        raise ValueError("Cannot infer visual conditioning width from saved keypoint configuration")
    environment_shapes = [feature["shape"] for feature in features.values() if feature["type"] == "ENV"]
    if any(len(shape) != 1 for shape in environment_shapes):
        raise ValueError("Only flat environment features are supported")
    environment_dim = sum(shape[0] for shape in environment_shapes)
    visual_dim = 2 * keypoints * visual_count if visual_count else 0
    global_dim = state_dim + config["n_obs_steps"] * (visual_dim + environment_dim)
    expected_rows = config["down_dims"][0] * (2 if config["use_film_scale_modulation"] else 1)
    if tuple(weight.shape) != (expected_rows, offset + global_dim) or tuple(bias.shape) != (expected_rows,):
        raise ValueError(
            "Conditioning projection dimensions disagree with the saved checkpoint configuration"
        )
    return torch.nn.functional.linear(
        torch.nn.functional.mish(x.float()), weight[:, offset : offset + state_dim], bias
    )


def historical_surrogate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Preserve historical tail-column + SiLU numerics solely to audit old outputs."""
    return torch.nn.functional.silu(torch.nn.functional.linear(x.float(), weight[:, -x.shape[-1] :], bias))


def diffusion_state_embedding(x: torch.Tensor, policy_dir: Path) -> torch.Tensor:
    sd = load_file(str(policy_dir / "model.safetensors"))
    weight = sd["diffusion.unet.down_modules.0.0.cond_encoder.1.weight"].float()
    bias = sd["diffusion.unet.down_modules.0.0.cond_encoder.1.bias"].float()
    if BASELINE_FEATURE_MODE == "historical-surrogate-audit":
        return historical_surrogate(x, weight, bias)
    if BASELINE_FEATURE_MODE == "state-first-contribution":
        config = json.loads((policy_dir / "config.json").read_text())
        return state_first_contribution(x, weight, bias, config)
    raise ValueError("Select an explicit baseline feature mode")


def make_representations(
    prev: torch.Tensor, curr: torch.Tensor, base_policy_dir: Path, prism_policy_dir: Path
) -> dict[str, torch.Tensor]:
    raw = normalize_state_pair(prev, curr, prism_policy_dir)
    baseline = normalize_state_pair(prev, curr, base_policy_dir)
    linear, poly = prism_latents(raw, load_prism_weights(prism_policy_dir))
    return {
        "Raw Proprioception": raw,
        BASELINE_NAMES[BASELINE_FEATURE_MODE]: diffusion_state_embedding(baseline, base_policy_dir),
        "PRISM Linear Latent": linear,
        "PRISM Poly. Latent": poly,
    }


def corrcoef(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.norm() * target.norm()).clamp_min(1e-12)
    return float((pred @ target / denom).item())


def linear_probe(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    valid = torch.isfinite(y)
    x = x[valid].float()
    y = y[valid].float()
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    generator = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=generator)
    if n > MAX_SAMPLES:
        idx = idx[:MAX_SAMPLES]
        n = MAX_SAMPLES
    x = x[idx]
    y = y[idx]
    split = int(0.8 * n)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    x_train = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=-1)
    x_test = torch.cat([x_test, torch.ones(x_test.shape[0], 1)], dim=-1)
    eye = torch.eye(x_train.shape[1])
    eye[-1, -1] = 0.0
    beta = torch.linalg.solve(x_train.T @ x_train + RIDGE * eye, x_train.T @ y_train)
    pred = x_test @ beta
    return float(torch.mean((pred - y_test) ** 2).item()), corrcoef(pred, y_test)


def trace_windows(trace_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    data = json.loads(trace_path.read_text())
    metrics = data["per_task"][0]["metrics"]
    prevs: list[torch.Tensor] = []
    currs: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {
        "Future Peak Force": [],
        "Contact Impulse": [],
        "Force-Weighted Translation": [],
        "Future Translation Energy": [],
        "Future Rotation Energy": [],
    }
    for state_trace, force_trace in zip(
        metrics["state_traces"], metrics["contact_force_traces"], strict=False
    ):
        states = torch.tensor(state_trace, dtype=torch.float32)
        force = torch.tensor(force_trace, dtype=torch.float32).clamp_min(0.0)
        n = min(states.shape[0], force.shape[0])
        if n <= HORIZON + 1:
            continue
        states = states[:n]
        force = force[:n]
        for t in range(1, n - HORIZON):
            f_future = force[t : t + HORIZON]
            if float(f_future.max().item()) <= CONTACT_THRESHOLD_N:
                continue
            prev = states[t - 1]
            curr = states[t]
            future = states[t + HORIZON]
            pos_path = states[t : t + HORIZON + 1, :3]
            step_disp = torch.linalg.norm(pos_path[1:] - pos_path[:-1], dim=-1)
            dx_future = future[:3] - curr[:3]
            dtheta_future = future[3:6] - curr[3:6]
            prevs.append(prev)
            currs.append(curr)
            targets["Future Peak Force"].append(torch.log1p(f_future.max()))
            targets["Contact Impulse"].append(torch.log1p(f_future.sum()))
            targets["Force-Weighted Translation"].append(torch.sum(f_future * step_disp))
            targets["Future Translation Energy"].append(torch.sum(dx_future.square()))
            targets["Future Rotation Energy"].append(torch.sum(dtheta_future.square()))
    if not prevs:
        raise RuntimeError(f"No contact-conditioned windows found in {trace_path}")
    return (
        torch.stack(prevs),
        torch.stack(currs),
        {name: torch.stack(values) for name, values in targets.items()},
    )


def collect_probe_inputs() -> dict[str, dict[str, list[torch.Tensor]]]:
    store: dict[str, dict[str, list[torch.Tensor]]] = {}
    for suite in PROBE_SPECS:
        task_id = suite["task_id"]
        trace_path = probe_trace_path(suite, task_id)
        base_path = base_policy_path(suite, task_id)
        prism_path = prism_policy_path(suite, task_id)
        if not trace_path.exists() or not base_path.exists() or not prism_path.exists():
            raise FileNotFoundError(
                f"Missing probe artifact for {suite['name']} task {task_id}: "
                f"trace={trace_path.exists()} base={base_path.exists()} prism={prism_path.exists()}"
            )
        try:
            prev, curr, targets = trace_windows(trace_path)
        except RuntimeError as exc:
            if "No contact-conditioned windows" not in str(exc):
                raise
            SKIPPED_NO_CONTACT.append((suite["suite"], task_id))
            continue
        reps = make_representations(prev, curr, base_path, prism_path)
        for target_name, target in targets.items():
            if target_name not in store:
                store[target_name] = {rep_name: [] for rep_name in reps}
                store[target_name]["target"] = []
            for rep_name, rep in reps.items():
                store[target_name][rep_name].append(rep.cpu())
            store[target_name]["target"].append(target.cpu())
    return store


def fmt(value: float) -> str:
    return f"{value:.3f}"


def rank_format(value: float, values: list[float], higher_is_better: bool) -> str:
    order = sorted(values, reverse=higher_is_better)
    text = fmt(value)
    if value == order[0]:
        return rf"\textbf{{{text}}}"
    if len(order) > 1 and value == order[1]:
        return rf"\underline{{{text}}}"
    return text


def gain_text(mse: float, baseline: float) -> str:
    gain = 100.0 * (baseline - mse) / baseline
    text = f"{gain:+.1f}\\%"
    if gain > 0:
        return rf"\textbf{{{text}}}"
    return text


def write_table(results: dict[str, dict[str, tuple[float, float]]]) -> None:
    quantities = [
        "Future Peak Force",
        "Contact Impulse",
        "Force-Weighted Translation",
        "Future Translation Energy",
        "Future Rotation Energy",
    ]
    proxy = {
        "Future Peak Force": r"$\max_{h \le H}\log(1+F_{t+h})$",
        "Contact Impulse": r"$\log(1+\sum_{h \le H}F_{t+h})$",
        "Force-Weighted Translation": r"$\sum_{h \le H}F_{t+h}\|\Delta x_{t+h}\|$",
        "Future Translation Energy": r"$\|x_{t+H}-x_t\|^2$",
        "Future Rotation Energy": r"$\|\theta_{t+H}-\theta_t\|^2$",
    }
    target_label = {
        "Future Peak Force": r"\shortstack{Future\\Peak Force}",
        "Contact Impulse": r"\shortstack{Contact\\Impulse}",
        "Force-Weighted Translation": r"\shortstack{Force-Weighted\\Translation}",
        "Future Translation Energy": r"\shortstack{Future\\Translation Energy}",
        "Future Rotation Energy": r"\shortstack{Future\\Rotation Energy}",
    }
    reps = [
        "Raw Proprioception",
        BASELINE_NAMES[BASELINE_FEATURE_MODE],
        "PRISM Linear Latent",
        "PRISM Poly. Latent",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"\textbf{Contact-response feature probing.}",
        (
            r"Baseline features are the saved Mish-before-linear state contribution plus bias, not a full U-Net embedding."
            if BASELINE_FEATURE_MODE == "state-first-contribution"
            else r"Audit only: the historical baseline surrogate uses tail columns and SiLU-after-linear; it is not a certified baseline embedding."
        ),
        rf"We use contact-conditioned windows from a cross-suite LIBERO diagnostic subset (Spatial, Object, Goal, and Long) with horizon $H={HORIZON}$ and train lightweight probes on frozen representations.",
        r"Targets measure future force response, contact impulse, and mechanics-inspired motion response.",
        r"MSE is computed after standardizing each target; Gain reports the relative MSE reduction of PRISM Poly. Latent over the PRISM Linear Latent, isolating the contribution of the polynomial interaction.",
        r"}",
        r"\label{tab:contact_response_probe}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cclccc}",
        r"\toprule",
        r"Target & Proxy & Representation & MSE $\downarrow$ & $r$ $\uparrow$ & Gain $\uparrow$ \\",
        r"\midrule",
    ]
    for qi, quantity in enumerate(quantities):
        mse_values = [results[quantity][rep][0] for rep in reps]
        r_values = [results[quantity][rep][1] for rep in reps]
        linear_baseline = results[quantity]["PRISM Linear Latent"][0]
        for ri, rep in enumerate(reps):
            mse, r = results[quantity][rep]
            gain = gain_text(mse, linear_baseline) if rep == "PRISM Poly. Latent" else "--"
            target_cell = f"\\multirow{{4}}{{*}}{{{target_label[quantity]}}}" if ri == 0 else ""
            proxy_cell = f"\\multirow{{4}}{{*}}{{{proxy[quantity]}}}" if ri == 0 else ""
            lines.append(
                f"{target_cell} "
                f"& {proxy_cell} "
                f"& {rep} & {rank_format(mse, mse_values, False)} & {rank_format(r, r_values, True)} & {gain} \\\\"
            )
        if qi != len(quantities) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global RIDGE, OUT_TEX, OUT_JSON, PROBE_SPECS, BASELINE_FEATURE_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True, help="JSON list of exact checkpoints and traces")
    parser.add_argument(
        "--baseline-features",
        choices=list(BASELINE_NAMES),
        required=True,
        help="Explicit checkpoint layout/feature interpretation; historical mode is audit only",
    )
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--output-tex", type=Path, default=OUT_TEX)
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    BASELINE_FEATURE_MODE = args.baseline_features
    RIDGE = float(args.ridge)
    if RIDGE < 0:
        parser.error("--ridge must be nonnegative")
    PROBE_SPECS = load_specs(args.specs)
    OUT_TEX = args.output_tex
    OUT_JSON = args.output_json
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    probe_inputs = collect_probe_inputs()
    results = {}
    counts = {}
    for quantity, rows in probe_inputs.items():
        target = torch.cat(rows["target"], dim=0)
        counts[quantity] = int(target.numel())
        results[quantity] = {
            name: linear_probe(torch.cat(rep_rows, dim=0), target)
            for name, rep_rows in rows.items()
            if name != "target"
        }
    OUT_JSON.write_text(
        json.dumps(
            {
                "metadata": {
                    "probe_specs": [
                        {key: str(value) if isinstance(value, Path) else value for key, value in spec.items()}
                        for spec in PROBE_SPECS
                    ],
                    "ridge": RIDGE,
                    "baseline_feature_mode": BASELINE_FEATURE_MODE,
                    "baseline_interpretation": BASELINE_NAMES[BASELINE_FEATURE_MODE],
                    "normalization": "saved preprocessor"
                    if BASELINE_FEATURE_MODE == "state-first-contribution"
                    else "historical mean/std surrogate",
                    "split": "seed-0 random windows, 80% train / 20% test; not held-out episodes",
                    "max_samples": MAX_SAMPLES,
                    "force_window": "[t, t+H), includes current force",
                    "horizon": HORIZON,
                    "contact_threshold_n": CONTACT_THRESHOLD_N,
                    "counts": counts,
                    "skipped_no_contact": SKIPPED_NO_CONTACT,
                    "mse": "target-standardized",
                },
                "results": {q: {r: list(v) for r, v in rows.items()} for q, rows in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_table(results)
    print(OUT_TEX.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
