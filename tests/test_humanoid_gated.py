"""Learned G1 gate mathematics, optimizer state and incompatible-checkpoint guards."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/humanoid-gym"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module("gated_test_legacy", INTEGRATION / "actor.py")
with mock.patch.dict(sys.modules, {"actor": LEGACY}):
    GATED = load_module("gated_test_actor", INTEGRATION / "gated_actor.py")
CHECKPOINTS = load_module("gated_test_checkpoints", INTEGRATION / "checkpoints.py")
RUNTIME = load_module("gated_test_runtime", INTEGRATION / "runtime.py")
PROVENANCE = load_module("gated_test_provenance", INTEGRATION / "provenance.py")


def model(degree=2, full_size=False):
    options = (
        {}
        if full_size
        else {"poly_hidden_dim": 32, "actor_hidden_dims": (24, 16), "critic_hidden_dims": (16, 8)}
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return GATED.GatedPolyActorCritic(705, 219, 12, poly_degree=degree, **options)


class GatedG1ActorTest(unittest.TestCase):
    def test_degree_three_matches_expanded_polynomial_and_derivatives(self):
        torch.manual_seed(14)
        encoder = GATED.GatedPolyEncoder(5, 7, degree=3).double()
        observations = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
        left, middle, right = [factor(observations) for factor in encoder.factors]
        alpha, beta = encoder.interaction_scales
        expanded = left + alpha * left * middle + beta * left * right + alpha * beta * left * middle * right
        actual = encoder.polynomial_features(observations)
        torch.testing.assert_close(actual, expanded, rtol=1e-12, atol=1e-12)
        parameters = [observations, encoder.interaction_scales]
        parameters.extend(parameter for factor in encoder.factors for parameter in factor.parameters())
        actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        expanded_gradients = torch.autograd.grad(expanded.square().sum(), parameters)
        for index, actual_gradient in enumerate(actual_gradients):
            expected_gradient = expanded_gradients[index]
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11)

    def test_zero_gate_retains_first_order_path_and_degree_one_has_no_gate(self):
        observations = torch.randn(3, 5)
        encoder = GATED.GatedPolyEncoder(5, 7, degree=3)
        with torch.no_grad():
            encoder.interaction_scales.zero_()
        torch.testing.assert_close(
            encoder.polynomial_features(observations), encoder.factors[0](observations), rtol=0, atol=0
        )
        degree_one = GATED.GatedPolyEncoder(5, 7, degree=1)
        self.assertIsNone(degree_one.interaction_scales)
        self.assertNotIn("interaction_scales", degree_one.state_dict())
        self.assertEqual(degree_one(observations).shape, (3, 7))

    def test_ppo_policy_optimizer_updates_gate_and_roundtrips_its_state(self):
        torch.manual_seed(25)
        policy = model()
        alpha = policy.actor_encoder.interaction_scales
        self.assertIsInstance(alpha, torch.nn.Parameter)
        torch.testing.assert_close(alpha, torch.full_like(alpha, 0.01), rtol=0, atol=0)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.assertTrue(
            any(parameter is alpha for group in optimizer.param_groups for parameter in group["params"])
        )
        observations, privileged = torch.randn(8, 705), torch.randn(8, 219)
        before = alpha.detach().clone()
        policy.act(observations)
        actions = policy.action_mean.detach() + 0.2
        loss = -policy.get_actions_log_prob(actions).mean() + policy.evaluate(privileged).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertGreater(alpha.grad.abs().max().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, alpha))
        self.assertFalse(hasattr(policy, "advance_poly_warmup"))
        self.assertNotIn("actor_poly_warmup_step", policy.state_dict())

        saved = io.BytesIO()
        torch.save(
            {"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, saved
        )
        saved.seek(0)
        checkpoint = torch.load(saved, map_location="cpu", weights_only=True)
        restored = model()
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.testing.assert_close(
            restored.act_inference(observations), policy.act_inference(observations), rtol=0, atol=0
        )
        restored_alpha = restored.actor_encoder.interaction_scales
        torch.testing.assert_close(
            restored_optimizer.state[restored_alpha]["exp_avg"],
            optimizer.state[alpha]["exp_avg"],
            rtol=0,
            atol=0,
        )

    def test_actor_never_reads_privileged_critic_input(self):
        policy = model()
        policy.act_inference(torch.randn(4, 705)).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in policy.critic.parameters()))

    def test_new_and_legacy_checkpoints_cannot_be_interchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            legacy = LEGACY.PolyActorCritic(705, 219, 12, critic_use_poly=False, actor_poly_mode="residual")
        gated = model()
        with self.assertRaisesRegex(ValueError, "legacy"):
            gated.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "strict"):
            gated.load_state_dict(gated.state_dict(), strict=False)
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(gated.state_dict(), strict=True)
        malformed = dict(gated.state_dict())
        malformed["actor_variant_version"] = torch.tensor(1)
        with self.assertRaises(ValueError):
            gated.load_state_dict(malformed, strict=True)
        selected = {
            "runner": {"policy_class_name": "GatedPolyActorCritic"},
            "policy": {"poly_degree": 3, "poly_hidden_dim": 32},
        }
        with self.assertRaisesRegex(ValueError, "degree"):
            CHECKPOINTS.validate_state_schema(gated.state_dict(), selected)

    def test_default_recipe_and_parameter_matched_control(self):
        self.assertEqual(RUNTIME.VARIANTS["prism"][1], "G1HumanoidGymCfgPPOGated")
        self.assertEqual(RUNTIME.VARIANTS["degree2"], RUNTIME.VARIANTS["prism"])
        self.assertEqual(RUNTIME.VARIANTS["legacy-prism"][1], "G1HumanoidGymCfgPPOPolyWarmup")
        self.assertNotIn("degree2", RUNTIME.DEFAULT_VARIANTS)  # Avoid duplicate degree-two training.
        policy = model(full_size=True)
        actor_count = sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if name.startswith(("actor_encoder.", "actor."))
        )
        self.assertEqual(actor_count, 724876)
        self.assertEqual(sum(parameter.numel() for parameter in policy.parameters()), 1123737)
        with contextlib.redirect_stdout(io.StringIO()):
            larger = LEGACY.ActorCritic(
                705, 219, 12, actor_hidden_dims=[648, 328, 160], critic_hidden_dims=[768, 256, 128]
            )
        control_count = sum(parameter.numel() for parameter in larger.actor.parameters())
        self.assertEqual(control_count, 724932)
        self.assertLess(abs(control_count - actor_count) / actor_count, 0.001)

    def test_manifest_identifies_learned_gate_without_scheduled_warmup(self):
        policy = model()
        config = {
            "seed": 1,
            "runner": {"policy_class_name": "GatedPolyActorCritic"},
            "policy": {"actor_hidden_dims": [24, 16], "poly_degree": 2, "gate_init": 0.01},
        }
        with mock.patch.object(PROVENANCE, "source_snapshot", return_value={}):
            manifest = PROVENANCE.training_manifest(
                "prism",
                RUNTIME.VARIANTS["prism"][0],
                ["train.py"],
                types.SimpleNamespace(seed=1),
                {"seed": 1},
                config,
                types.SimpleNamespace(
                    current_learning_iteration=0, alg=types.SimpleNamespace(actor_critic=policy)
                ),
                {},
            )
        self.assertEqual(manifest["actor"]["actor_variant"], "g1_gated_poly_v2")
        self.assertEqual(manifest["actor"]["gate_semantics"], "learned_per_feature")
        self.assertEqual(manifest["actor"]["warmup_ppo_updates"], 0)


if __name__ == "__main__":
    unittest.main()
