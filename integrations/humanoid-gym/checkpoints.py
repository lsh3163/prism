"""Explicit actor-schema checks before constructing a G1 simulator."""

from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import Tensor


def actor_variant(policy_class: str) -> Optional[str]:
    identities = {
        "ActorCritic": None,
        "PolyActorCritic": "g1_residual_poly_v1",
        "GatedPolyActorCritic": "g1_gated_poly_v2",
    }
    if policy_class not in identities:
        raise ValueError("Unknown G1 policy class: " + policy_class)
    return identities[policy_class]


def validate_state_schema(state: Mapping[str, Tensor], train_config: dict) -> Optional[str]:
    """Reject cross-variant loads early; the runner still performs a strict load."""
    expected = actor_variant(train_config["runner"]["policy_class_name"])
    marker = state.get("actor_variant_version")
    has_marker = isinstance(marker, Tensor) and marker.numel() == 1 and marker.item() == 2
    policy = train_config["policy"]
    if expected == "g1_gated_poly_v2":
        if not has_marker or "actor_encoder.raw_proj.weight" in state:
            raise ValueError("Expected g1_gated_poly_v2; historical checkpoints require legacy-* recipes")
        degree = policy["poly_degree"]
        width = policy["poly_hidden_dim"]
        factors = {
            key for key in state if key.startswith("actor_encoder.factors.") and key.endswith(".weight")
        }
        wanted = {"actor_encoder.factors.{}.weight".format(index) for index in range(degree)}
        if factors != wanted:
            raise ValueError("Gated checkpoint polynomial degree differs from the selected recipe")
        scales = state.get("actor_encoder.interaction_scales")
        if degree > 1 and (scales is None or tuple(scales.shape) != (degree - 1, width)):
            raise ValueError("Gated checkpoint alpha shape differs from the selected recipe")
        if degree == 1 and scales is not None:
            raise ValueError("Degree-one checkpoints must not contain interaction gates")
    elif expected == "g1_residual_poly_v1":
        required = {"actor_encoder.raw_proj.weight", "actor_encoder.poly_scale", "actor_poly_warmup_step"}
        if marker is not None or not required.issubset(state):
            raise ValueError("Expected historical g1_residual_poly_v1; select its explicit legacy-* recipe")
        warmup = policy.get("actor_poly_warmup_updates", 0)
        step = int(state["actor_poly_warmup_step"].item())
        scale = float(state["actor_encoder.poly_scale"].item())
        wanted_scale = min(step / float(warmup), 1.0) if warmup else 1.0
        if (not warmup and step != 0) or abs(scale - wanted_scale) > 1e-6:
            raise ValueError("Legacy checkpoint warmup state differs from the selected legacy recipe")
    elif marker is not None or any(key.startswith("actor_encoder.") for key in state):
        raise ValueError("Expected an MLP control checkpoint, not a PRISM actor")
    return expected


def validate_checkpoint(path: Path, train_config: dict) -> Optional[str]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Expected an RSL-RL checkpoint containing model_state_dict")
    return validate_state_schema(checkpoint["model_state_dict"], train_config)
