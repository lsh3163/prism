"""The public G1 recipe names resolve to one explicit, inspectable catalog."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "humanoid-gym"
sys.path.insert(0, str(INTEGRATION))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CATALOG = load_module("g1_recipe_catalog_test", INTEGRATION / "recipe_catalog.py")
RUNTIME = load_module("g1_recipe_runtime_test", INTEGRATION / "runtime.py")


class G1RecipeCatalogTest(unittest.TestCase):
    def test_public_mapping_retains_every_existing_recipe_and_alias(self):
        self.assertEqual(tuple(RUNTIME.VARIANTS), CATALOG.PUBLIC_VARIANT_NAMES)
        self.assertEqual(RUNTIME.DEFAULT_VARIANTS, CATALOG.DEFAULT_VARIANT_NAMES)
        self.assertEqual(RUNTIME.VARIANTS["degree2"], RUNTIME.VARIANTS["prism"])
        for name, task_and_config in RUNTIME.VARIANTS.items():
            with self.subTest(recipe=name):
                recipe = CATALOG.recipe_for_variant(name)
                self.assertEqual(task_and_config, (recipe.task_name, recipe.config_class))

    def test_recipe_descriptions_are_json_serializable_and_expose_effective_settings(self):
        description = CATALOG.describe_variant("prism-1321k")
        self.assertEqual(description["canonical_variant"], "prism-1321k")
        self.assertEqual(description["actor_variant"], "g1_gated_poly_v2")
        self.assertEqual(description["resolved_policy_settings"]["poly_hidden_dim"], 334)
        self.assertEqual(description["resolved_policy_settings"]["actor_hidden_dims"], (513, 256, 128))
        json.dumps(description, allow_nan=False)

    def test_policy_classes_have_exactly_one_checkpoint_schema(self):
        self.assertIsNone(CATALOG.actor_variant_for_policy_class("ActorCritic"))
        self.assertEqual(CATALOG.actor_variant_for_policy_class("GatedPolyActorCritic"), "g1_gated_poly_v2")
        self.assertEqual(
            CATALOG.actor_variant_for_policy_class("ResidualLearnedGateActorCritic"),
            "g1_residual_learned_gate_v1",
        )
        with self.assertRaisesRegex(ValueError, "Unknown G1 policy class"):
            CATALOG.actor_variant_for_policy_class("UnknownActor")

    def test_main_table_protocol_is_explicit_and_matches_the_paper_runner(self):
        specification = CATALOG.describe_main_table()
        self.assertEqual(specification["variants"], ["baseline", "larger", "prism"])
        self.assertEqual(specification["training_seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(specification["evaluation_seeds"], [101, 102, 103])
        self.assertEqual(specification["max_iterations"], 3001)
        self.assertEqual(specification["num_envs"], 4096)
        self.assertEqual(specification["evaluation"]["protocol_id"], "g1_balanced_timeout_v2")
        self.assertEqual(CATALOG.MAIN_TABLE_SPEC.evaluation_runner_protocol, "balanced")


if __name__ == "__main__":
    unittest.main()
