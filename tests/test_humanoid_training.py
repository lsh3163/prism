"""Training metadata must prove completion and retain the learned gate identity."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/humanoid-gym"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, INTEGRATION / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load_module("humanoid_training_wrapper_test", "train.py")
PROVENANCE = load_module("humanoid_training_provenance_test", "provenance.py")


class GateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_encoder = torch.nn.Module()
        self.actor_encoder.interaction_scales = torch.nn.Parameter(torch.full((1, 4), 0.01))
        self.weight = torch.nn.Parameter(torch.ones(4))


class Runner:
    """Retain v1.0.2's stale periodic iteration behavior for the regression test."""

    def __init__(self, log_dir, gated=True):
        model = GateModel() if gated else torch.nn.Linear(3, 2)
        self.alg = types.SimpleNamespace(actor_critic=model, optimizer=torch.optim.Adam(model.parameters()))
        self.log_dir = str(log_dir)
        self.current_learning_iteration = 0
        self.writer = mock.Mock()

    def log(self, locs):
        self.logged_step = locs["it"]

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )


class HumanoidTrainingObserverTest(unittest.TestCase):
    def install(self, runner, updates=2, gated=True):
        actor = {
            "actor_variant": "g1_gated_poly_v2" if gated else None,
            "recipe_variant": "prism" if gated else "baseline",
            "polynomial_degree": 2 if gated else None,
        }
        observer = TRAIN.TrainingObserver(runner, 3, actor, updates)
        manifest = Path(runner.log_dir) / "prism_run_manifest.json"
        manifest.write_text(json.dumps({"seed": 3, "actor": actor}))
        with mock.patch.dict(sys.modules, {"provenance": PROVENANCE}):
            observer.install(PROVENANCE.file_identity(manifest))
        return observer

    def test_optimizer_update_metadata_gate_scalars_and_complete_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner = Runner(output)
            observer = self.install(runner)
            initial = runner.alg.actor_critic.actor_encoder.interaction_scales.detach().clone()
            for step in range(2):
                alpha = runner.alg.actor_critic.actor_encoder.interaction_scales
                loss = (alpha * runner.alg.actor_critic.weight).sum()
                runner.alg.optimizer.zero_grad()
                loss.backward()
                runner.alg.optimizer.step()
                runner.log({"it": step})
                if step == 0:
                    runner.save(output / "model_0.pt")
                    self.assertEqual(runner.current_learning_iteration, 0)
            runner.current_learning_iteration = 2
            runner.save(output / "model_2.pt")
            complete = json.loads(observer.finish().read_text())
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(complete["seed"], 3)
            self.assertEqual(complete["completed_iteration"], 2)
            self.assertEqual(complete["gate"]["optimizer_steps"], 2)
            self.assertGreater(complete["gate"]["max_abs_change_from_start"], 0)
            self.assertEqual(complete["checkpoint"], PROVENANCE.file_identity(output / "model_2.pt"))
            self.assertEqual(
                complete["training_manifest"], PROVENANCE.file_identity(output / "prism_run_manifest.json")
            )
            self.assertFalse(torch.equal(initial, alpha))
            periodic = torch.load(output / "model_0.pt", map_location="cpu", weights_only=True)
            self.assertEqual(periodic["iter"], 1)
            self.assertEqual(periodic["infos"]["prism_training"]["completed_iteration"], 1)
            self.assertEqual(periodic["infos"]["prism_training"]["actor_variant"], "g1_gated_poly_v2")
            final = torch.load(output / "model_2.pt", map_location="cpu", weights_only=True)
            restored = GateModel()
            restored.load_state_dict(final["model_state_dict"], strict=True)
            torch.testing.assert_close(restored.actor_encoder.interaction_scales, alpha, rtol=0, atol=0)
            self.assertEqual(len((output / "prism_checkpoints.jsonl").read_text().splitlines()), 2)
            self.assertEqual(runner.writer.add_scalar.call_count, 14)
            runner.writer.close.assert_called_once()

    def test_missing_or_frozen_gate_optimizer_entry_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = Runner(temporary)
            runner.alg.optimizer = torch.optim.Adam([runner.alg.actor_critic.weight])
            with self.assertRaisesRegex(RuntimeError, "exactly once"):
                self.install(runner)
            runner.alg.optimizer = torch.optim.Adam(runner.alg.actor_critic.parameters())
            runner.alg.actor_critic.actor_encoder.interaction_scales.requires_grad_(False)
            with self.assertRaisesRegex(RuntimeError, "require gradients"):
                self.install(runner)

    def test_partial_training_never_creates_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner = Runner(output)
            observer = self.install(runner)
            runner.log({"it": 0})
            runner.save(output / "model_0.pt")
            with self.assertRaisesRegex(RuntimeError, "requested updates"):
                observer.finish()
            failure = json.loads(observer.finish(RuntimeError("interrupted")).read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["completed_iteration"], 1)
            self.assertFalse((output / "prism_training_complete.json").exists())

    def test_control_has_explicit_null_gate_and_checkpoint_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner = Runner(output, gated=False)
            observer = self.install(runner, updates=1, gated=False)
            self.assertIsNone(observer.initial_gate_statistics)
            runner.log({"it": 0})
            runner.current_learning_iteration = 1
            checkpoint = output / "model_1.pt"
            runner.save(checkpoint)
            identity = PROVENANCE.file_identity(checkpoint)
            with self.assertRaises(FileExistsError):
                runner.save(checkpoint)
            self.assertEqual(PROVENANCE.file_identity(checkpoint), identity)
            complete = json.loads(observer.finish().read_text())
            self.assertIsNone(complete["actor"]["actor_variant"])
            self.assertIsNone(complete["gate"])
            runner.writer.add_scalar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
