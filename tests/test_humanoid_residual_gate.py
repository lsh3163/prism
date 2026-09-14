"""Alpha-only G1 residual control: exact legacy pairing and independent checkpoints."""

import ast
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


LEGACY = load_module("residual_gate_test_legacy", INTEGRATION / "actor.py")
with mock.patch.dict(sys.modules, {"actor": LEGACY}):
    CONTROL = load_module("residual_gate_test_actor", INTEGRATION / "residual_gated_actor.py")
    GATED = load_module("residual_gate_test_default", INTEGRATION / "gated_actor.py")
CHECKPOINTS = load_module("residual_gate_test_checkpoints", INTEGRATION / "checkpoints.py")
RUNTIME = load_module("residual_gate_test_runtime", INTEGRATION / "runtime.py")
PROVENANCE = load_module("residual_gate_test_provenance", INTEGRATION / "provenance.py")


def model(control=True, degree=2, full_size=False, **overrides):
    options = {
        "poly_degree": degree,
        "actor_use_poly": True,
        "critic_use_poly": False,
        "actor_poly_mode": "residual",
        "actor_use_input_layer_norm": False,
        "critic_use_input_layer_norm": False,
        "actor_use_hidden_layer_norm": False,
        "critic_use_hidden_layer_norm": False,
        "actor_poly_warmup_updates": 500,
        "actor_hidden_dims": (512, 256, 128) if full_size else (11, 7),
        "critic_hidden_dims": (768, 256, 128) if full_size else (13, 7),
        "poly_hidden_dim": 256 if full_size else 9,
    }
    options.update(overrides)
    dimensions = (705, 219, 12) if full_size else (5, 6, 3)
    actor_class = CONTROL.ResidualLearnedGateActorCritic if control else LEGACY.PolyActorCritic
    with contextlib.redirect_stdout(io.StringIO()):
        return actor_class(*dimensions, **options)


def paired_models(degree=2):
    torch.manual_seed(61)
    legacy = model(control=False, degree=degree)
    legacy_rng = torch.get_rng_state().clone()
    torch.manual_seed(61)
    control = model(degree=degree)
    return legacy, control, legacy_rng, torch.get_rng_state().clone()


class ResidualLearnedGateTest(unittest.TestCase):
    def test_complete_legacy_initialization_and_rng_state_are_preserved(self):
        for degree in (2, 3):
            with self.subTest(degree=degree):
                legacy, control, old_rng, new_rng = paired_models(degree)
                self.assertTrue(torch.equal(old_rng, new_rng))
                old_state, new_state = legacy.state_dict(), control.state_dict()
                self.assertEqual(
                    set(new_state) - set(old_state),
                    {"actor_encoder.interaction_scales", "actor_variant_version"},
                )
                for name, tensor in old_state.items():
                    self.assertTrue(torch.equal(tensor, new_state[name]), name)
                alpha = control.actor_encoder.interaction_scales
                self.assertEqual(tuple(alpha.shape), (degree - 1, 9))
                self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))
                self.assertIsInstance(control.actor_encoder, LEGACY.ResidualPolyEncoder)
                self.assertEqual(control.actor_variant, "g1_residual_learned_gate_v1")

    def test_unit_gates_preserve_outputs_and_all_legacy_gradients_exactly(self):
        for degree in (2, 3):
            for updates in (0, 137, 500):
                with self.subTest(degree=degree, updates=updates):
                    legacy, control, _, _ = paired_models(degree)
                    for _ in range(updates):
                        legacy.advance_poly_warmup()
                        control.advance_poly_warmup()
                    observations = torch.randn(8, 5, requires_grad=True)
                    control_observations = observations.detach().clone().requires_grad_()
                    privileged = torch.randn(8, 6, requires_grad=True)
                    control_privileged = privileged.detach().clone().requires_grad_()
                    actions = torch.randn(8, 3)
                    legacy.update_distribution(observations)
                    control.update_distribution(control_observations)
                    torch.testing.assert_close(control.action_mean, legacy.action_mean, rtol=0, atol=0)
                    legacy_value = legacy.evaluate(privileged)
                    control_value = control.evaluate(control_privileged)
                    torch.testing.assert_close(control_value, legacy_value, rtol=0, atol=0)
                    old_loss = -legacy.get_actions_log_prob(actions).mean() + legacy_value.square().mean()
                    new_loss = -control.get_actions_log_prob(actions).mean() + control_value.square().mean()
                    old_loss.backward()
                    new_loss.backward()
                    new_parameters = dict(control.named_parameters())
                    for name, parameter in legacy.named_parameters():
                        torch.testing.assert_close(new_parameters[name].grad, parameter.grad, rtol=0, atol=0)
                    torch.testing.assert_close(control_observations.grad, observations.grad, rtol=0, atol=0)
                    torch.testing.assert_close(control_privileged.grad, privileged.grad, rtol=0, atol=0)
                    gate_gradient = control.actor_encoder.interaction_scales.grad
                    self.assertTrue(torch.isfinite(gate_gradient).all())
                    if updates == 0:
                        self.assertEqual(torch.count_nonzero(gate_gradient).item(), 0)
                    else:
                        self.assertGreater(torch.count_nonzero(gate_gradient).item(), 0)

    def test_zero_gates_remove_only_interactions_preserving_raw_skip_and_projection(self):
        policy = model(degree=3)
        for _ in range(250):
            policy.advance_poly_warmup()
        encoder = policy.actor_encoder
        with torch.no_grad():
            encoder.interaction_scales.zero_()
        observations = torch.randn(2, 4, 5)
        first_order = encoder.A[0](encoder.poly_in_proj(observations))
        expected = encoder.activation(
            encoder.raw_proj(observations) + encoder.poly_scale * encoder.poly_out_proj(first_order)
        )
        torch.testing.assert_close(encoder(observations), expected, rtol=0, atol=0)

    def test_optimizer_learns_gate_and_roundtrips_model_optimizer_and_warmup(self):
        torch.manual_seed(19)
        policy = model()
        for _ in range(137):
            policy.advance_poly_warmup()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        alpha = policy.actor_encoder.interaction_scales
        self.assertTrue(
            any(parameter is alpha for group in optimizer.param_groups for parameter in group["params"])
        )
        observations, privileged = torch.randn(8, 5), torch.randn(8, 6)
        old_alpha = alpha.detach().clone()
        policy.update_distribution(observations)
        actions = policy.action_mean.detach() + 0.2
        loss = -policy.get_actions_log_prob(actions).mean() + policy.evaluate(privileged).square().mean()
        loss.backward()
        self.assertGreater(alpha.grad.abs().max().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(alpha, old_alpha))
        self.assertEqual(policy.actor_poly_warmup_step.item(), 137)

        stream = io.BytesIO()
        torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = model()
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(saved["optimizer"])
        restored_alpha = restored.actor_encoder.interaction_scales
        torch.testing.assert_close(
            restored.act_inference(observations), policy.act_inference(observations), rtol=0, atol=0
        )
        self.assertEqual(restored.actor_poly_warmup_step.item(), 137)
        self.assertEqual(restored.get_actor_poly_scale(), policy.get_actor_poly_scale())
        for name, value in optimizer.state[alpha].items():
            torch.testing.assert_close(restored_optimizer.state[restored_alpha][name], value, rtol=0, atol=0)
        for _ in range(501):
            restored.advance_poly_warmup()
        self.assertEqual(restored.actor_poly_warmup_step.item(), 500)
        self.assertEqual(restored.get_actor_poly_scale(), 1.0)

    def test_legacy_and_default_checkpoints_are_rejected_in_both_directions(self):
        legacy, control, _, _ = paired_models()
        with contextlib.redirect_stdout(io.StringIO()):
            default = GATED.GatedPolyActorCritic(
                5, 6, 3, actor_hidden_dims=(11, 7), critic_hidden_dims=(13, 7), poly_hidden_dim=9
            )
        for incompatible in (legacy, default):
            with self.subTest(actor=type(incompatible).__name__):
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    control.load_state_dict(incompatible.state_dict())
                with self.assertRaises((ValueError, RuntimeError)):
                    incompatible.load_state_dict(control.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "strict"):
            control.load_state_dict(control.state_dict(), strict=False)

    def test_malformed_gate_marker_and_warmup_schemas_are_rejected(self):
        policy = model()
        invalid = (
            ("actor_variant_version", torch.tensor(2)),
            ("actor_variant_version", torch.tensor(3.0)),
            ("actor_variant_version", torch.tensor([3])),
            ("actor_encoder.interaction_scales", torch.ones(9)),
            ("actor_encoder.interaction_scales", torch.ones(1, 9, dtype=torch.long)),
            ("actor_encoder.interaction_scales", torch.full((1, 9), float("nan"))),
            ("actor_poly_warmup_step", torch.tensor(501)),
            ("actor_poly_warmup_step", torch.tensor(-1)),
            ("actor_poly_warmup_step", torch.tensor(0.0)),
            ("actor_encoder.poly_scale", torch.tensor(1.0)),
            ("actor_encoder.poly_scale", torch.tensor(float("nan"))),
        )
        for name, value in invalid:
            with self.subTest(key=name, value=value):
                state = dict(policy.state_dict())
                state[name] = value
                with self.assertRaises(ValueError):
                    policy.load_state_dict(state)
        for name in ("actor_variant_version", "actor_encoder.interaction_scales", "actor_poly_warmup_step"):
            with self.subTest(missing=name):
                state = dict(policy.state_dict())
                del state[name]
                with self.assertRaises(ValueError):
                    policy.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "degree and width"):
            model(degree=3).load_state_dict(policy.state_dict())

    def test_control_contract_rejects_silent_architecture_changes(self):
        invalid = (
            {"actor_variant": "g1_gated_poly_v2"},
            {"gate_init": 0.01},
            {"gate_init": float("nan")},
            {"actor_poly_mode": "replace"},
            {"actor_use_poly": False},
            {"critic_use_poly": True},
            {"activation": "relu"},
            {"actor_output_tanh": True},
            {"actor_use_input_layer_norm": True},
            {"actor_use_hidden_layer_norm": True},
            {"critic_use_input_layer_norm": True},
            {"critic_use_hidden_layer_norm": True},
            {"actor_poly_warmup_updates": 0},
            {"poly_degree": 1},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                model(**options)

    def test_main_parameter_difference_is_exactly_one_gate_per_latent_feature(self):
        legacy, control = model(control=False, full_size=True), model(full_size=True)
        actor_count = sum(
            parameter.numel()
            for name, parameter in control.named_parameters()
            if name.startswith(("actor_encoder.", "actor."))
        )
        self.assertEqual(actor_count, 922252)
        self.assertEqual(sum(parameter.numel() for parameter in control.parameters()), 1321113)
        self.assertEqual(
            sum(parameter.numel() for parameter in control.parameters())
            - sum(parameter.numel() for parameter in legacy.parameters()),
            256,
        )

    def test_integration_schema_and_manifest_keep_the_diagnostic_distinct(self):
        legacy, policy, _, _ = paired_models()
        for _ in range(137):
            policy.advance_poly_warmup()
        config = {
            "seed": 1,
            "runner": {"policy_class_name": "ResidualLearnedGateActorCritic"},
            "policy": {
                "actor_variant": CONTROL.ACTOR_VARIANT,
                "actor_hidden_dims": [11, 7],
                "poly_degree": 2,
                "poly_hidden_dim": 9,
                "gate_init": 1.0,
                "actor_poly_warmup_updates": 500,
            },
        }
        state = policy.state_dict()
        self.assertEqual(CHECKPOINTS.validate_state_schema(state, config), CONTROL.ACTOR_VARIANT)
        for incompatible in ("PolyActorCritic", "GatedPolyActorCritic", "ActorCritic"):
            selected = dict(config, runner={"policy_class_name": incompatible})
            with self.subTest(policy_class=incompatible), self.assertRaises(ValueError):
                CHECKPOINTS.validate_state_schema(state, selected)
        wrong_warmup = dict(config, policy=dict(config["policy"], actor_poly_warmup_updates=0))
        with self.assertRaisesRegex(ValueError, "warmup|warm-up|500"):
            CHECKPOINTS.validate_state_schema(state, wrong_warmup)
        for _ in range(500):
            policy.advance_poly_warmup()
        wrong_warmup = dict(config, policy=dict(config["policy"], actor_poly_warmup_updates=250))
        with self.assertRaisesRegex(ValueError, "warmup|warm-up|500"):
            CHECKPOINTS.validate_state_schema(policy.state_dict(), wrong_warmup)
        variant = "residual-learned-gate"
        task, config_name = RUNTIME.VARIANTS[variant]
        self.assertEqual(config_name, "G1HumanoidGymCfgPPOResidualLearnedGate")
        self.assertNotIn(variant, RUNTIME.DEFAULT_VARIANTS)
        with mock.patch.object(PROVENANCE, "source_snapshot", return_value={}):
            manifest = PROVENANCE.training_manifest(
                variant,
                task,
                ["train.py", "--variant", variant],
                types.SimpleNamespace(seed=1),
                {"seed": 1},
                config,
                types.SimpleNamespace(
                    current_learning_iteration=0, alg=types.SimpleNamespace(actor_critic=policy)
                ),
                {},
            )
            legacy_manifest = PROVENANCE.training_manifest(
                "legacy-prism",
                RUNTIME.VARIANTS["legacy-prism"][0],
                ["train.py", "--variant", "legacy-prism"],
                types.SimpleNamespace(seed=1),
                {"seed": 1},
                dict(
                    config,
                    runner={"policy_class_name": "PolyActorCritic"},
                    policy={
                        name: value
                        for name, value in config["policy"].items()
                        if name not in {"actor_variant", "gate_init"}
                    },
                ),
                types.SimpleNamespace(
                    current_learning_iteration=0, alg=types.SimpleNamespace(actor_critic=legacy)
                ),
                {},
            )
        actor = manifest["actor"]
        self.assertEqual(actor["actor_variant"], CONTROL.ACTOR_VARIANT)
        self.assertEqual(actor["policy_type"], "residual_learned_gate_diagnostic")
        self.assertEqual(actor["class"], "ResidualLearnedGateActorCritic")
        self.assertEqual(actor["gate_semantics"], "learned_per_feature")
        self.assertEqual(actor["gate_init"], 1.0)
        self.assertEqual(actor["warmup_ppo_updates"], 500)
        self.assertEqual(len(actor["initial_shared_parameter_sha256"]), 64)
        self.assertEqual(
            actor["initial_shared_parameter_sha256"],
            legacy_manifest["actor"]["initial_shared_parameter_sha256"],
        )
        self.assertEqual(
            actor["total_actor_critic_parameters"],
            sum(parameter.numel() for parameter in policy.parameters()),
        )

    def test_actor_and_tests_parse_with_python_38_grammar(self):
        for path in (INTEGRATION / "residual_gated_actor.py", Path(__file__)):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
