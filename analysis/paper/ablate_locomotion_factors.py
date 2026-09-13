#!/usr/bin/env python3
"""Measure degree-2 factor ablations using the released Humanoid-Gym actor.

The historical Table 5 implementation uses the mean action-vector L2 distance,
scaled by 0.25 radians. Also report coordinate-wise MAE, which differs from L2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_actor(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = payload["model_state_dict"]
    gated = "actor_variant_version" in weights
    directory = Path(__file__).resolve().parents[2] / "integrations/humanoid-gym"
    actor_path = directory / ("gated_actor.py" if gated else "actor.py")
    spec = importlib.util.spec_from_file_location("prism_humanoid_actor", actor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {actor_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    common = dict(
        num_actor_obs=705,
        num_critic_obs=219,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        poly_hidden_dim=256,
        poly_degree=2,
        activation="elu",
    )
    if gated:
        actor = module.GatedPolyActorCritic(**common)
        actor.load_state_dict(weights, strict=True)
        return actor.eval()
    actor = module.PolyActorCritic(
        **common,
        actor_use_poly=True,
        critic_use_poly=False,
        actor_poly_mode="residual",
        actor_use_input_layer_norm=False,
        critic_use_input_layer_norm=False,
        actor_use_hidden_layer_norm=False,
        critic_use_hidden_layer_norm=False,
        actor_output_tanh=False,
        actor_poly_warmup_updates=500,
    )
    actor.load_state_dict(weights, strict=True)
    return actor.eval()


def input_name(index: int) -> str:
    joints = [
        f"{side}_{joint}"
        for side in ("L", "R")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    fields = ["sin_phase", "cos_phase", "command_vx", "command_vy", "command_yaw"]
    fields += [f"q_{joint}" for joint in joints]
    fields += [f"qd_{joint}" for joint in joints]
    fields += [f"previous_action_{joint}" for joint in joints]
    fields += ["base_wx", "base_wy", "base_wz", "roll", "pitch", "yaw"]
    frame, coordinate = divmod(index, 47)
    lag = 14 - frame
    return f"{fields[coordinate]}(t{'-' + str(lag) if lag else ''})"


@torch.no_grad()
def ablate(
    actor: torch.nn.Module, observations: torch.Tensor, factors: list[int], scale: float
) -> list[dict]:
    encoder = actor.actor_encoder
    if observations.ndim != 2 or observations.shape[1] != 705 or observations.shape[0] == 0:
        raise ValueError("observations must have nonempty shape (N, 705)")
    if not torch.isfinite(observations).all():
        raise ValueError("observations must be finite")
    full = actor.act_inference(observations)
    # Effective affine weights on the original 15 x 47 observation history.
    gated = hasattr(encoder, "interaction_scales")
    if encoder.degree != 2:
        raise ValueError("Factor ablation currently supports degree-2 actors only")
    if gated:
        left_weights, right_weights = encoder.factors[0].weight, encoder.factors[1].weight
    else:
        left_weights = encoder.A[1].weight @ encoder.poly_in_proj.weight
        right_weights = encoder.S[0].weight @ encoder.A[0].weight @ encoder.poly_in_proj.weight
    rows = []
    for factor in factors:
        if not 0 <= factor < left_weights.shape[0]:
            raise ValueError(f"Factor {factor} is outside the 256-channel encoder")

        def zero_factor(module, inputs, output, dimension=factor):
            modified = output.clone()
            modified[..., dimension] = 0
            return modified

        gate = None
        if gated:
            gate = encoder.interaction_scales[0, factor].clone()
            try:
                encoder.interaction_scales[0, factor] = 0
                difference = (full - actor.act_inference(observations)) * scale
            finally:
                encoder.interaction_scales[0, factor] = gate
        else:
            # Zero A1's channel: remove only A1(h)*S0(A0(h)), retaining A0(h).
            hook = encoder.A[1].register_forward_hook(zero_factor)
            try:
                difference = (full - actor.act_inference(observations)) * scale
            finally:
                hook.remove()
        left_index = int(left_weights[factor].abs().argmax())
        right_index = int(right_weights[factor].abs().argmax())
        # The historical naming helper excludes the left coordinate from the right
        # branch. Preserve that interpretation separately from the true argmax.
        distinct_right = right_weights[factor].abs().clone()
        distinct_right[left_index] = -1
        historical_right_index = int(distinct_right.argmax())
        rows.append(
            {
                "factor": factor,
                "learned_alpha": float(gate) if gated else None,
                "left_input": input_name(left_index),
                "right_input": input_name(right_index),
                "historical_distinct_right_input": input_name(historical_right_index),
                "mean_l2_joint_target_shift_rad": float(difference.norm(dim=-1).mean()),
                "mean_absolute_joint_target_shift_rad": float(difference.abs().mean()),
            }
        )
    return sorted(rows, key=lambda row: -row["mean_l2_joint_target_shift_rad"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--observations", type=Path, required=True, help="PT dict with observations shaped (N,705)"
    )
    parser.add_argument(
        "--factors",
        type=int,
        nargs="+",
        required=True,
        help="Explicit checkpoint-specific factor IDs; historical IDs do not generalize to new actors",
    )
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action_scale <= 0:
        parser.error("--action-scale must be positive")
    actor = load_actor(args.checkpoint)
    payload = torch.load(args.observations, map_location="cpu", weights_only=True)
    observations = payload["observations"].float()
    rows = ablate(actor, observations, args.factors, args.action_scale)
    result = {
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "observations": str(args.observations.resolve()),
            "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
            "n_observations": int(observations.shape[0]),
            "action_scale": args.action_scale,
            "actor_variant": getattr(actor, "actor_variant", "g1_residual_poly_v1"),
            "poly_scale": actor.get_actor_poly_scale() if hasattr(actor, "get_actor_poly_scale") else None,
            "term_labels": "Independent largest absolute effective affine weights; interpretation only",
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
