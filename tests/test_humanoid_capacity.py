"""Capacity controls change declared actor dimensions while retaining G1 PPO interfaces."""

import ast
import contextlib
import importlib.util
import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/humanoid-gym"
RECIPES = {
    "prism-1321k": ("G1HumanoidGymCfgPPOGated1321", [513, 256, 128], 921715, 1320576),
    "mlp-1321k": ("G1HumanoidGymCfgPPOMatched1321", [816, 352, 160], 922092, 1320953),
    "mlp-wide-1500k": ("G1HumanoidGymCfgPPOWide1500", [928, 384, 224], 1100844, 1499705),
    "mlp-deep-1500k": ("G1HumanoidGymCfgPPODeep1500", [768, 512, 256, 128], 1101708, 1500569),
}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, INTEGRATION / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACTOR = load_module("capacity_test_actor", "actor.py")
with mock.patch.dict(sys.modules, {"actor": ACTOR}):
    GATED = load_module("capacity_test_gated", "gated_actor.py")
RUNTIME = load_module("capacity_test_runtime", "runtime.py")
CHECKPOINTS = load_module("capacity_test_checkpoints", "checkpoints.py")


def policy_config_classes():
    """Execute actual PPO declarations without importing the simulator package.

    These config classes only need an upstream algorithm and runner base. The
    empty base intentionally supplies no simulator or invented PPO defaults;
    comparisons below concern the release's actual declarations and inheritance.
    """
    path = INTEGRATION / "config.py"
    parsed = ast.parse(path.read_text(), filename=str(path))
    declarations = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("G1HumanoidGymCfgPPO")
    ]
    namespace = {"LeggedRobotCfgPPO": type("LeggedRobotCfgPPO", (), {"algorithm": object, "runner": object})}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fields(config):
    return {name: getattr(config, name) for name in dir(config) if not name.startswith("_")}


def build(config):
    actor_class = (
        GATED.GatedPolyActorCritic
        if config.runner.policy_class_name == "GatedPolyActorCritic"
        else ACTOR.ActorCritic
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return actor_class(705, 219, 12, **fields(config.policy))


class HumanoidCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = policy_config_classes()
        cls.models = {variant: build(cls.configs[recipe[0]]) for variant, recipe in RECIPES.items()}

    def test_resolved_configs_only_change_declared_actor_capacity(self):
        for variant, (config_name, hidden_dims, _, _) in RECIPES.items():
            with self.subTest(variant=variant):
                self.assertEqual(RUNTIME.VARIANTS[variant][1], config_name)
                config = self.configs[config_name]
                baseline_name = (
                    "G1HumanoidGymCfgPPOGated" if variant == "prism-1321k" else "G1HumanoidGymCfgPPO"
                )
                baseline = self.configs[baseline_name]
                expected_policy = dict(fields(baseline.policy), actor_hidden_dims=hidden_dims)
                if variant == "prism-1321k":
                    expected_policy["poly_hidden_dim"] = 334
                self.assertEqual(fields(config.policy), expected_policy)
                self.assertEqual(fields(config.algorithm), fields(baseline.algorithm))
                runner, expected_runner = fields(config.runner), fields(baseline.runner)
                self.assertNotEqual(runner.pop("experiment_name"), expected_runner.pop("experiment_name"))
                self.assertEqual(runner, expected_runner)
                self.assertEqual(config.seed, baseline.seed)
                self.assertEqual(config.policy.critic_hidden_dims, [768, 256, 128])

    def test_actual_parameter_counts_and_common_critic_match_registered_capacity(self):
        expected_critic_shapes = {
            "0.weight": (768, 219),
            "0.bias": (768,),
            "2.weight": (256, 768),
            "2.bias": (256,),
            "4.weight": (128, 256),
            "4.bias": (128,),
            "6.weight": (1, 128),
            "6.bias": (1,),
        }
        for variant, policy in self.models.items():
            with self.subTest(variant=variant):
                _, hidden_dims, expected_actor, expected_total = RECIPES[variant]
                actor_count = sum(
                    value.numel()
                    for name, value in policy.named_parameters()
                    if name.startswith(("actor.", "actor_encoder."))
                )
                self.assertEqual(actor_count, expected_actor)
                self.assertEqual(sum(value.numel() for value in policy.parameters()), expected_total)
                self.assertEqual(sum(value.numel() for value in policy.critic.parameters()), 398849)
                self.assertEqual(
                    {name: tuple(value.shape) for name, value in policy.critic.named_parameters()},
                    expected_critic_shapes,
                )
                self.assertEqual(tuple(policy.std.shape), (12,))
                self.assertTrue(all(value.device.type == "cpu" for value in policy.parameters()))
                linears = [module for module in policy.actor if isinstance(module, torch.nn.Linear)]
                self.assertEqual([module.out_features for module in linears], hidden_dims + [12])
                self.assertEqual(linears[0].in_features, 334 if variant == "prism-1321k" else 705)
                self.assertEqual(policy.act_inference(torch.zeros(2, 705)).shape, (2, 12))
                self.assertEqual(policy.evaluate(torch.zeros(2, 219)).shape, (2, 1))
        prism = RECIPES["prism-1321k"][3]
        matched = RECIPES["mlp-1321k"][3]
        wide, deep = RECIPES["mlp-wide-1500k"][3], RECIPES["mlp-deep-1500k"][3]
        self.assertLess(abs(prism - 1320857) / 1320857, 0.001)
        self.assertLess(abs(prism - matched) / prism, 0.001)
        self.assertLess(abs(wide - deep) / deep, 0.001)
        self.assertGreater(min(wide, deep) / prism, 1.13)

    def test_expanded_actor_retains_learned_gated_formula_without_warmup(self):
        policy = self.models["prism-1321k"]
        self.assertIs(type(policy), GATED.GatedPolyActorCritic)
        self.assertEqual(policy.actor_variant, "g1_gated_poly_v2")
        self.assertEqual(policy.actor_variant_version.item(), 2)
        self.assertFalse(hasattr(policy, "advance_poly_warmup"))
        self.assertFalse(any("warmup" in name or "poly_scale" in name for name in policy.state_dict()))
        encoder = policy.actor_encoder
        self.assertEqual(tuple(encoder.interaction_scales.shape), (1, 334))
        torch.testing.assert_close(
            encoder.interaction_scales, torch.full_like(encoder.interaction_scales, 0.01), rtol=0, atol=0
        )
        self.assertEqual([tuple(factor.weight.shape) for factor in encoder.factors], [(334, 705)] * 2)
        self.assertEqual(tuple(encoder.projection.weight.shape), (334, 334))
        self.assertTrue(torch.equal(encoder.factors[1].bias, torch.zeros(334)))
        observations = torch.randn(3, 705)
        interaction = encoder.factors[0](observations) * (
            1.0 + encoder.interaction_scales[0] * encoder.factors[1](observations)
        )
        expected = policy.actor(torch.nn.functional.elu(encoder.projection(interaction)))
        actual = policy.act_inference(observations)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.autograd.grad(actual.square().sum(), encoder.interaction_scales)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)

    def test_selected_width_rejects_original_gated_checkpoint_before_simulation(self):
        expanded = self.models["prism-1321k"]
        original = build(self.configs["G1HumanoidGymCfgPPOGated"])
        config = self.configs[RECIPES["prism-1321k"][0]]
        selected = {"runner": fields(config.runner), "policy": fields(config.policy)}
        self.assertEqual(
            CHECKPOINTS.validate_state_schema(expanded.state_dict(), selected), "g1_gated_poly_v2"
        )
        with self.assertRaisesRegex(ValueError, "alpha shape"):
            CHECKPOINTS.validate_state_schema(original.state_dict(), selected)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            original.load_state_dict(expanded.state_dict(), strict=True)
        mlp_config = self.configs[RECIPES["mlp-1321k"][0]]
        selected_mlp = {"runner": fields(mlp_config.runner), "policy": fields(mlp_config.policy)}
        with self.assertRaisesRegex(ValueError, "MLP control"):
            CHECKPOINTS.validate_state_schema(expanded.state_dict(), selected_mlp)
        self.assertIsNone(
            CHECKPOINTS.validate_state_schema(self.models["mlp-1321k"].state_dict(), selected_mlp)
        )

    def test_each_recipe_dry_run_resolves_without_importing_simulator(self):
        for variant in RECIPES:
            arguments = ["train.py", "--variant=" + variant, "--seed=23", "--dry-run"]
            with self.subTest(variant=variant), contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        {"runtime": RUNTIME, "isaacgym": None, "legged_gym": None, "rsl_rl": None},
                    )
                )
                stack.enter_context(mock.patch.object(sys, "argv", arguments))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runpy.run_path(str(INTEGRATION / "train.py"), run_name="__main__")
            result = json.loads(output.getvalue())
            self.assertEqual(result["variant"], variant)
            self.assertEqual(result["task"], RUNTIME.VARIANTS[variant][0])
            self.assertEqual(result["arguments"], ["--seed=23"])


if __name__ == "__main__":
    unittest.main()
