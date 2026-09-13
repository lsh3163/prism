"""CPU tests for the archived Humanoid-Gym actor; Isaac Gym is not required."""

import contextlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "humanoid-gym"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACTOR = load_module("prism_humanoid_actor_test", INTEGRATION / "actor.py")
PROVENANCE = load_module("prism_humanoid_provenance_test", INTEGRATION / "provenance.py")
RUNTIME = load_module("prism_humanoid_runtime_test", INTEGRATION / "runtime.py")


def policy_options(degree=2, warmup=500):
    return {
        "num_actor_obs": 705,
        "num_critic_obs": 219,
        "num_actions": 12,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [768, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "poly_hidden_dim": 256,
        "poly_degree": degree,
        "actor_use_poly": True,
        "critic_use_poly": False,
        "actor_poly_mode": "residual",
        "actor_use_input_layer_norm": False,
        "critic_use_input_layer_norm": False,
        "actor_use_hidden_layer_norm": False,
        "critic_use_hidden_layer_norm": False,
        "actor_output_tanh": False,
        "actor_poly_warmup_updates": warmup,
    }


def build(cls, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(**kwargs)


class HumanoidActorTest(unittest.TestCase):
    def test_legacy_sources_parse_as_python38(self):
        import ast

        for path in INTEGRATION.glob("*.py"):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(), feature_version=(3, 8))

    def test_polynomial_warmup_and_checkpoint_resume(self):
        model = build(ACTOR.PolyActorCritic, **policy_options())
        x = torch.randn(2, 705)
        with torch.no_grad():
            # At update zero, the polynomial branch contributes exactly zero.
            expected = model.actor(model.actor_encoder.activation(model.actor_encoder.raw_proj(x)))
            torch.testing.assert_close(model.act_inference(x), expected, rtol=0, atol=0)
        for _ in range(137):
            model.advance_poly_warmup()
        resumed = build(ACTOR.PolyActorCritic, **policy_options())
        resumed.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(resumed.actor_poly_warmup_step.item(), 137)
        self.assertAlmostEqual(resumed.get_actor_poly_scale(), 137 / 500, places=6)
        resumed.advance_poly_warmup()
        self.assertAlmostEqual(resumed.get_actor_poly_scale(), 138 / 500, places=6)
        for _ in range(600):
            resumed.advance_poly_warmup()
        self.assertEqual(resumed.get_actor_poly_scale(), 1.0)

    def test_actor_cannot_read_privileged_critic_observations(self):
        model = build(ACTOR.PolyActorCritic, **policy_options(warmup=0))
        observations = torch.randn(2, 705, requires_grad=True)
        actions = model.act_inference(observations)
        self.assertEqual(actions.shape, (2, 12))
        actions.square().mean().backward()
        self.assertTrue(torch.isfinite(observations.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in model.critic.parameters()))


class HumanoidTrainingProvenanceTest(unittest.TestCase):
    def test_run_manifest_never_reuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            path = PROVENANCE.write_training_manifest(output, {"seed": 23})
            self.assertEqual(json.loads(path.read_text()), {"seed": 23})
            with self.assertRaises(FileExistsError):
                PROVENANCE.write_training_manifest(output, {"seed": 99})
            self.assertEqual(json.loads(path.read_text()), {"seed": 23})
            invalid_output = Path(directory) / "invalid"
            with self.assertRaises(ValueError):
                PROVENANCE.write_training_manifest(invalid_output, {"seed": float("nan")})
            self.assertFalse(invalid_output.exists())

    def test_file_identity_detects_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_3.pt"
            path.write_bytes(b"original checkpoint")
            before = PROVENANCE.file_identity(path)
            path.write_bytes(b"different checkpoint")
            after = PROVENANCE.file_identity(path)
            self.assertEqual(before["path"], after["path"])
            self.assertNotEqual(before["sha256"], after["sha256"])

    def test_manifest_records_resolved_recipe_without_labeling_controls_prism(self):
        policy = {
            "actor_hidden_dims": [512, 256, 128],
            "actor_use_poly": True,
            "poly_degree": 2,
            "actor_poly_warmup_updates": 500,
        }
        config = {"seed": 23, "policy": policy, "runner": {"policy_class_name": "PolyActorCritic"}}
        simulator_enum = types.SimpleNamespace(name="SIM_PHYSX")
        with mock.patch.object(PROVENANCE, "source_snapshot", return_value={"files": []}):
            result = PROVENANCE.training_manifest(
                "prism",
                "g1_humanoidgym_ppo_poly_warmup",
                ["python", "train.py", "--seed=23"],
                types.SimpleNamespace(seed=23, physics_engine=simulator_enum),
                {"seed": 23},
                config,
                types.SimpleNamespace(current_learning_iteration=42),
                {},
            )
            self.assertEqual(result["seed"], 23)
            self.assertEqual(result["training_config"]["policy"], policy)
            self.assertEqual(result["initial_learning_iteration"], 42)
            self.assertEqual(result["actor"]["actor_variant"], "g1_residual_poly_v1")
            self.assertEqual(result["actor"]["warmup_ppo_updates"], 500)
            self.assertIsInstance(result["arguments"]["physics_engine"], str)
            json.dumps(result, allow_nan=False)
            config["policy"] = {"actor_hidden_dims": [816, 352, 160]}
            config["runner"]["policy_class_name"] = "ActorCritic"
            control = PROVENANCE.training_manifest(
                "larger",
                "g1_humanoidgym_ppo_parammatch",
                ["python", "train.py"],
                types.SimpleNamespace(seed=23),
                {"seed": 23},
                config,
                types.SimpleNamespace(current_learning_iteration=0),
                {},
            )
            self.assertIsNone(control["actor"]["actor_variant"])
            self.assertEqual(control["actor"]["policy_type"], "mlp")
            self.assertIsNone(control["actor"]["polynomial_degree"])

    def test_suite_previews_by_default_and_executes_only_when_requested(self):
        for execute in (False, True):
            arguments = ["run_suite.py", "--seeds", "1", "2", "--variants", "baseline", "prism"]
            if execute:
                arguments.append("--execute")
            with contextlib.ExitStack() as stack:
                stack.enter_context(self.subTest(execute=execute))
                stack.enter_context(mock.patch.dict(sys.modules, {"runtime": RUNTIME}))
                stack.enter_context(mock.patch.object(sys, "argv", arguments))
                launch = stack.enter_context(mock.patch("subprocess.run"))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runpy.run_path(str(INTEGRATION / "run_suite.py"), run_name="__main__")
                self.assertEqual(launch.call_count, 4 if execute else 0)
                self.assertEqual(len(output.getvalue().splitlines()), 4)

    def test_train_dry_run_does_not_import_simulator(self):
        arguments = ["train.py", "--variant=prism", "--seed=23", "--dry-run"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"runtime": RUNTIME, "isaacgym": None}))
            stack.enter_context(mock.patch.object(sys, "argv", arguments))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runpy.run_path(str(INTEGRATION / "train.py"), run_name="__main__")
        result = json.loads(output.getvalue())
        self.assertEqual(result["task"], "g1_humanoidgym_ppo_poly_warmup")
        self.assertEqual(result["arguments"], ["--seed=23"])


@unittest.skipUnless(
    os.environ.get("RSL_RL_REFERENCE_ROOT"),
    "Set RSL_RL_REFERENCE_ROOT to compare the original research source",
)
class HistoricalActorParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ["RSL_RL_REFERENCE_ROOT"]) / "rsl_rl" / "modules"
        package_name = "prism_original_rsl_reference"
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
        cls.baseline = load_module(package_name + ".actor_critic", root / "actor_critic.py")
        cls.reference = load_module(package_name + ".actor_critic_sota", root / "actor_critic_sota.py")

    def test_all_variants_match_outputs_gradients_and_checkpoint_keys(self):
        torch.manual_seed(52)
        observations = torch.randn(3, 705)
        privileged = torch.randn(3, 219)
        for degree, warmup in ((1, 0), (2, 0), (2, 500), (3, 0)):
            with self.subTest(degree=degree, warmup=warmup):
                options = policy_options(degree, warmup)
                reference = build(self.reference.PolyActorCritic, **options)
                released = build(ACTOR.PolyActorCritic, **options)
                for _ in range(173):
                    reference.advance_poly_warmup()
                released.load_state_dict(reference.state_dict(), strict=True)
                torch.testing.assert_close(
                    released.act_inference(observations),
                    reference.act_inference(observations),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    released.evaluate(privileged), reference.evaluate(privileged), rtol=0, atol=0
                )
                for model in (reference, released):
                    model.act_inference(observations).square().mean().backward()
                for name, parameter in released.named_parameters():
                    expected = dict(reference.named_parameters())[name]
                    if expected.grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)

        for hidden in ([512, 256, 128], [816, 352, 160]):
            options = {
                "num_actor_obs": 705,
                "num_critic_obs": 219,
                "num_actions": 12,
                "actor_hidden_dims": hidden,
                "critic_hidden_dims": [768, 256, 128],
                "activation": "elu",
                "init_noise_std": 1.0,
            }
            with self.subTest(baseline_hidden=hidden):
                reference = build(self.baseline.ActorCritic, **options)
                released = build(ACTOR.ActorCritic, **options)
                released.load_state_dict(reference.state_dict(), strict=True)
                torch.testing.assert_close(
                    released.act_inference(observations),
                    reference.act_inference(observations),
                    rtol=0,
                    atol=0,
                )

    @unittest.skipUnless(
        os.environ.get("HUMANOID_CHECKPOINT"), "Set HUMANOID_CHECKPOINT to verify a trained model"
    )
    def test_original_trained_checkpoint_matches_exactly(self):
        checkpoint = torch.load(os.environ["HUMANOID_CHECKPOINT"], map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        reference = build(self.reference.PolyActorCritic, **policy_options())
        released = build(ACTOR.PolyActorCritic, **policy_options())
        reference.load_state_dict(state, strict=True)
        released.load_state_dict(state, strict=True)
        observations = torch.randn(8, 705)
        torch.testing.assert_close(
            released.act_inference(observations), reference.act_inference(observations), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
