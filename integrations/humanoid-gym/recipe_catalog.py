"""Canonical, simulator-free catalog for the released G1 recipes.

The upstream-compatible classes in :mod:`config` stay deliberately explicit:
RSL-RL resolves them by name and historical checkpoints rely on those names.
This catalog is the readable index over those classes.  It owns public recipe
names, aliases, actor schema, and the policy fields that distinguish a recipe;
callers should select a recipe here instead of duplicating string literals.

It has no Isaac Gym or RSL-RL imports so ``train.py --dry-run`` and release
validation can inspect every setting on a CPU-only machine.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


# Values shared by every G1 PPO policy.  Recipe overrides below are relative to
# these values; environment, optimizer, and PPO schedule remain in config.py.
BASE_POLICY_SETTINGS = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": (512, 256, 128),
    "critic_hidden_dims": (768, 256, 128),
    "activation": "elu",
}


LEGACY_POLY_OVERRIDES = (
    ("poly_hidden_dim", 256),
    ("poly_degree", 2),
    ("actor_use_poly", True),
    ("critic_use_poly", False),
    ("actor_poly_mode", "residual"),
    ("actor_use_input_layer_norm", False),
    ("critic_use_input_layer_norm", False),
    ("actor_use_hidden_layer_norm", False),
    ("critic_use_hidden_layer_norm", False),
    ("actor_output_tanh", False),
)

GATED_POLY_OVERRIDES = (
    ("actor_variant", "g1_gated_poly_v2"),
    ("poly_hidden_dim", 256),
    ("poly_degree", 2),
    ("gate_init", 0.01),
)


def _replace_override(
    overrides: Tuple[Tuple[str, Any], ...], name: str, value: Any
) -> Tuple[Tuple[str, Any], ...]:
    """Return an immutable override list with one named value replaced."""

    replaced = False
    updated = []
    for key, old_value in overrides:
        if key == name:
            updated.append((key, value))
            replaced = True
        else:
            updated.append((key, old_value))
    if not replaced:
        updated.append((name, value))
    return tuple(updated)


GATED_DEGREE1_OVERRIDES = _replace_override(GATED_POLY_OVERRIDES, "poly_degree", 1)
GATED_DEGREE3_OVERRIDES = _replace_override(GATED_POLY_OVERRIDES, "poly_degree", 3)
LEGACY_DEGREE1_OVERRIDES = _replace_override(LEGACY_POLY_OVERRIDES, "poly_degree", 1)
LEGACY_DEGREE3_OVERRIDES = _replace_override(LEGACY_POLY_OVERRIDES, "poly_degree", 3)


@dataclass(frozen=True)
class RecipeSpec:
    """One named, externally selectable G1 training recipe."""

    name: str
    task_name: str
    config_class: str
    policy_class: str
    actor_variant: Optional[str]
    policy_type: str
    family: str
    label: str
    description: str
    policy_overrides: Tuple[Tuple[str, Any], ...] = ()
    aliases: Tuple[str, ...] = ()
    default_suite: bool = False
    status: str = "current"

    def resolved_policy_settings(self) -> Dict[str, Any]:
        """Return base policy fields plus the exact recipe-specific overrides."""

        settings = dict(BASE_POLICY_SETTINGS)
        settings.update(self.policy_overrides)
        return settings

    def as_dict(self, requested_name: Optional[str] = None) -> Dict[str, Any]:
        """Return JSON-friendly metadata used by dry-run and catalog commands."""

        return {
            "requested_variant": requested_name or self.name,
            "canonical_variant": self.name,
            "aliases": list(self.aliases),
            "label": self.label,
            "status": self.status,
            "family": self.family,
            "description": self.description,
            "task": self.task_name,
            "config_class": self.config_class,
            "policy_class": self.policy_class,
            "actor_variant": self.actor_variant,
            "policy_type": self.policy_type,
            "policy_overrides": dict(self.policy_overrides),
            "resolved_policy_settings": self.resolved_policy_settings(),
        }


@dataclass(frozen=True)
class MainTableSpec:
    """The immutable G1 main-table matrix and its shared evaluation protocol."""

    variants: Tuple[str, ...]
    training_seeds: Tuple[int, ...]
    evaluation_seeds: Tuple[int, ...]
    max_iterations: int
    training_envs: int
    evaluation_protocol_id: str
    evaluation_runner_protocol: str
    evaluation_episodes: int
    evaluation_envs: int
    condition_name: str
    terrain_mode: str
    push_mode: str

    def evaluation_settings(self) -> Dict[str, Any]:
        return {
            "protocol_id": self.evaluation_protocol_id,
            "episodes": self.evaluation_episodes,
            "num_envs": self.evaluation_envs,
            "condition_name": self.condition_name,
            "terrain_mode": self.terrain_mode,
            "push_mode": self.push_mode,
        }


# Keep this order stable: it is reflected in command-line choices and existing
# manifests.  Aliases are declared by the canonical recipe rather than copied.
RECIPE_SPECS = (
    RecipeSpec(
        name="baseline",
        task_name="g1_humanoidgym_ppo",
        config_class="G1HumanoidGymCfgPPO",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="main",
        label="Baseline MLP",
        description="Released 512-256-128 MLP policy and common G1 PPO schedule.",
        default_suite=True,
    ),
    RecipeSpec(
        name="larger",
        task_name="g1_humanoidgym_ppo_larger_v2",
        config_class="G1HumanoidGymCfgPPOLarger",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="main",
        label="Larger MLP control",
        description="MLP capacity control matched to the default gated actor.",
        policy_overrides=(("actor_hidden_dims", (648, 328, 160)),),
        default_suite=True,
    ),
    RecipeSpec(
        name="degree1",
        task_name="g1_humanoidgym_ppo_gated_d1",
        config_class="G1HumanoidGymCfgPPOGatedD1",
        policy_class="GatedPolyActorCritic",
        actor_variant="g1_gated_poly_v2",
        policy_type="gated_polynomial",
        family="degree_ablation",
        label="Gated degree 1",
        description="One affine factor; no learned interaction gate is instantiated.",
        policy_overrides=GATED_DEGREE1_OVERRIDES,
        default_suite=True,
    ),
    RecipeSpec(
        name="prism",
        task_name="g1_humanoidgym_ppo_gated_d2",
        config_class="G1HumanoidGymCfgPPOGated",
        policy_class="GatedPolyActorCritic",
        actor_variant="g1_gated_poly_v2",
        policy_type="gated_polynomial",
        family="main",
        label="PRISM (gated degree 2)",
        description="Default new G1 PRISM actor: per-feature learned alpha, initialized to 0.01.",
        policy_overrides=GATED_POLY_OVERRIDES,
        aliases=("degree2",),
        default_suite=True,
    ),
    RecipeSpec(
        name="degree3",
        task_name="g1_humanoidgym_ppo_gated_d3",
        config_class="G1HumanoidGymCfgPPOGatedD3",
        policy_class="GatedPolyActorCritic",
        actor_variant="g1_gated_poly_v2",
        policy_type="gated_polynomial",
        family="degree_ablation",
        label="Gated degree 3",
        description="Three affine factors with two learned per-feature interaction gates.",
        policy_overrides=GATED_DEGREE3_OVERRIDES,
        default_suite=True,
    ),
    RecipeSpec(
        name="legacy-larger",
        task_name="g1_humanoidgym_ppo_parammatch",
        config_class="G1HumanoidGymCfgPPOParamMatched",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="historical",
        label="Historical parameter-matched MLP",
        description="Historical MLP capacity control for residual-PRISM checkpoints.",
        policy_overrides=(("actor_hidden_dims", (816, 352, 160)),),
        status="legacy",
    ),
    RecipeSpec(
        name="legacy-degree1",
        task_name="g1_humanoidgym_ppo_poly_d1",
        config_class="G1HumanoidGymCfgPPOPolyD1",
        policy_class="PolyActorCritic",
        actor_variant="g1_residual_poly_v1",
        policy_type="residual_polynomial",
        family="historical",
        label="Historical residual degree 1",
        description="Historical residual actor schema with a single polynomial factor.",
        policy_overrides=LEGACY_DEGREE1_OVERRIDES,
        status="legacy",
    ),
    RecipeSpec(
        name="legacy-degree2",
        task_name="g1_humanoidgym_ppo_poly",
        config_class="G1HumanoidGymCfgPPOPoly",
        policy_class="PolyActorCritic",
        actor_variant="g1_residual_poly_v1",
        policy_type="residual_polynomial",
        family="historical",
        label="Historical residual degree 2",
        description="Historical residual actor without the 500-update warmup recipe.",
        policy_overrides=LEGACY_POLY_OVERRIDES,
        status="legacy",
    ),
    RecipeSpec(
        name="legacy-prism",
        task_name="g1_humanoidgym_ppo_poly_warmup",
        config_class="G1HumanoidGymCfgPPOPolyWarmup",
        policy_class="PolyActorCritic",
        actor_variant="g1_residual_poly_v1",
        policy_type="residual_polynomial",
        family="historical",
        label="Historical residual PRISM",
        description="Historical residual degree-two actor with a 500-update scalar warmup.",
        policy_overrides=LEGACY_POLY_OVERRIDES + (("actor_poly_warmup_updates", 500),),
        status="legacy",
    ),
    RecipeSpec(
        name="legacy-degree3",
        task_name="g1_humanoidgym_ppo_poly_d3",
        config_class="G1HumanoidGymCfgPPOPolyD3",
        policy_class="PolyActorCritic",
        actor_variant="g1_residual_poly_v1",
        policy_type="residual_polynomial",
        family="historical",
        label="Historical residual degree 3",
        description="Historical residual actor with three polynomial factors.",
        policy_overrides=LEGACY_DEGREE3_OVERRIDES,
        status="legacy",
    ),
    RecipeSpec(
        name="residual-learned-gate",
        task_name="g1_humanoidgym_ppo_residual_learned_gate",
        config_class="G1HumanoidGymCfgPPOResidualLearnedGate",
        policy_class="ResidualLearnedGateActorCritic",
        actor_variant="g1_residual_learned_gate_v1",
        policy_type="residual_learned_gate_diagnostic",
        family="diagnostic",
        label="Residual learned-gate diagnostic",
        description="Historical residual path with learned gates, gate_init=1.0, and a 500-update warmup.",
        policy_overrides=LEGACY_POLY_OVERRIDES
        + (
            ("actor_poly_warmup_updates", 500),
            ("actor_variant", "g1_residual_learned_gate_v1"),
            ("gate_init", 1.0),
        ),
        status="diagnostic",
    ),
    RecipeSpec(
        name="prism-1321k",
        task_name="g1_humanoidgym_ppo_gated_1321k",
        config_class="G1HumanoidGymCfgPPOGated1321",
        policy_class="GatedPolyActorCritic",
        actor_variant="g1_gated_poly_v2",
        policy_type="gated_polynomial",
        family="capacity",
        label="Gated PRISM 1.321M",
        description="Wider gated actor used for the matched 1.321M capacity comparison.",
        policy_overrides=(
            ("actor_variant", "g1_gated_poly_v2"),
            ("poly_hidden_dim", 334),
            ("poly_degree", 2),
            ("gate_init", 0.01),
            ("actor_hidden_dims", (513, 256, 128)),
        ),
    ),
    RecipeSpec(
        name="mlp-1321k",
        task_name="g1_humanoidgym_ppo_mlp_1321k",
        config_class="G1HumanoidGymCfgPPOMatched1321",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="capacity",
        label="MLP 1.321M",
        description="MLP parameter-matched to the 1.321M gated PRISM actor.",
        policy_overrides=(("actor_hidden_dims", (816, 352, 160)),),
    ),
    RecipeSpec(
        name="mlp-wide-1500k",
        task_name="g1_humanoidgym_ppo_mlp_wide_1500k",
        config_class="G1HumanoidGymCfgPPOWide1500",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="capacity",
        label="Wide MLP 1.500M",
        description="Three-hidden-layer MLP wider than the 1.321M gated actor.",
        policy_overrides=(("actor_hidden_dims", (928, 384, 224)),),
    ),
    RecipeSpec(
        name="mlp-deep-1500k",
        task_name="g1_humanoidgym_ppo_mlp_deep_1500k",
        config_class="G1HumanoidGymCfgPPODeep1500",
        policy_class="ActorCritic",
        actor_variant=None,
        policy_type="mlp",
        family="capacity",
        label="Deep MLP 1.500M",
        description="Four-hidden-layer MLP matched to the wide 1.500M control.",
        policy_overrides=(("actor_hidden_dims", (768, 512, 256, 128)),),
    ),
)


def _build_recipe_index() -> Dict[str, RecipeSpec]:
    index = {}
    for recipe in RECIPE_SPECS:
        for name in (recipe.name,) + recipe.aliases:
            if name in index:
                raise RuntimeError("Duplicate G1 recipe or alias: " + name)
            index[name] = recipe
    return index


RECIPES_BY_NAME = _build_recipe_index()
PUBLIC_VARIANT_NAMES = (
    "baseline",
    "larger",
    "degree1",
    "prism",
    "degree2",
    "degree3",
    "legacy-larger",
    "legacy-degree1",
    "legacy-degree2",
    "legacy-prism",
    "legacy-degree3",
    "residual-learned-gate",
    "prism-1321k",
    "mlp-1321k",
    "mlp-wide-1500k",
    "mlp-deep-1500k",
)
DEFAULT_VARIANT_NAMES = ("baseline", "larger", "degree1", "prism", "degree3")
MAIN_TABLE_SPEC = MainTableSpec(
    variants=("baseline", "larger", "prism"),
    training_seeds=(1, 2, 3, 4, 5),
    evaluation_seeds=(101, 102, 103),
    max_iterations=3001,
    training_envs=4096,
    evaluation_protocol_id="g1_balanced_timeout_v2",
    evaluation_runner_protocol="balanced",
    evaluation_episodes=200,
    evaluation_envs=100,
    condition_name="match_nopush",
    terrain_mode="match",
    push_mode="off",
)

if set(PUBLIC_VARIANT_NAMES) != set(RECIPES_BY_NAME):
    raise RuntimeError("Public G1 recipe names do not match the catalog")


def recipe_for_variant(name: str) -> RecipeSpec:
    """Resolve a public recipe name or explicit alias."""

    try:
        return RECIPES_BY_NAME[name]
    except KeyError as error:
        raise ValueError("Unknown G1 recipe: " + name) from error


def unique_recipes() -> Iterable[RecipeSpec]:
    """Yield canonical recipes once, in stable release order."""

    return iter(RECIPE_SPECS)


def actor_variant_for_policy_class(policy_class: str) -> Optional[str]:
    """Resolve the checkpoint schema identity from the registered policy class."""

    matches = {
        (recipe.policy_class, recipe.actor_variant)
        for recipe in RECIPE_SPECS
        if recipe.policy_class == policy_class
    }
    if not matches:
        raise ValueError("Unknown G1 policy class: " + policy_class)
    variants = {actor_variant for _, actor_variant in matches}
    if len(variants) != 1:
        raise RuntimeError("Ambiguous actor schema for G1 policy class: " + policy_class)
    return variants.pop()


def describe_variant(name: str) -> Dict[str, Any]:
    """Produce a stable, JSON-friendly recipe description for users and tools."""

    return recipe_for_variant(name).as_dict(requested_name=name)


def describe_main_table() -> Dict[str, Any]:
    """Return the exact G1 main-table settings in a serializable form."""

    return {
        "variants": list(MAIN_TABLE_SPEC.variants),
        "training_seeds": list(MAIN_TABLE_SPEC.training_seeds),
        "evaluation_seeds": list(MAIN_TABLE_SPEC.evaluation_seeds),
        "max_iterations": MAIN_TABLE_SPEC.max_iterations,
        "num_envs": MAIN_TABLE_SPEC.training_envs,
        "checkpoint_iteration": MAIN_TABLE_SPEC.max_iterations,
        "evaluation": MAIN_TABLE_SPEC.evaluation_settings(),
    }
