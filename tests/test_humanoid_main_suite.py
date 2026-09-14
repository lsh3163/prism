"""Protect the registered matrix and prevent adopting incomplete training runs."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations/humanoid-gym/run_main_table.py"
SPEC = importlib.util.spec_from_file_location("humanoid_main_suite", MODULE_PATH)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class MainSuiteTest(unittest.TestCase):
    def test_registered_matrix_has_fifteen_fresh_runs_and_shared_evaluation_seeds(self):
        plan = suite.planned_suite("/tmp/example_suite")
        pairs = {(run["variant"], run["training_seed"]) for run in plan["runs"]}
        self.assertEqual(
            pairs, {(variant, seed) for variant in ("baseline", "larger", "prism") for seed in range(1, 6)}
        )
        self.assertEqual(len(plan["runs"]), 15)
        self.assertEqual(plan["evaluation_seeds"], [101, 102, 103])
        self.assertEqual(plan["num_envs"], 4096)
        self.assertEqual(plan["max_iterations"], 3001)
        self.assertEqual(plan["evaluation"]["protocol_id"], "g1_balanced_timeout_v2")
        for run in plan["runs"]:
            self.assertIsNone(run["training_manifest"])
            command = suite.training_command("/python", "/code", "/output", run, "cuda:0")
            self.assertIn("--seed=" + str(run["training_seed"]), command)
            self.assertFalse(any("resume" in arg or "model_path" in arg for arg in command))
            self.assertEqual(len(run["evaluations"]), 3)

    def test_checkpoint_alone_is_not_training_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "training/run"
            directory.mkdir(parents=True)
            (directory / "model_3001.pt").write_bytes(b"partial checkpoint")
            self.assertIsNone(suite.completed_training(Path(temporary)))

    def test_completion_rejects_tampered_weights_or_wrong_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "training/run"
            directory.mkdir(parents=True)
            checkpoint = directory / "model_3001.pt"
            checkpoint.write_bytes(b"verified checkpoint")
            manifest = directory / "prism_run_manifest.json"
            manifest.write_text("{}")
            marker = {
                "status": "completed",
                "completed_iteration": 3001,
                "initial_iteration": 0,
                "requested_final_iteration": 3001,
                "training_manifest": {"path": str(manifest), "sha256": suite.sha256(manifest)},
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            }
            marker_path = directory / "prism_training_complete.json"
            marker_path.write_text(json.dumps(marker))
            self.assertEqual(suite.completed_training(temporary), (checkpoint, manifest))
            marker["completed_iteration"] = 2
            marker_path.write_text(json.dumps(marker))
            with self.assertRaisesRegex(RuntimeError, "budget"):
                suite.completed_training(temporary)
            marker["completed_iteration"] = 3001
            marker_path.write_text(json.dumps(marker))
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "identity"):
                suite.completed_training(temporary)

    def test_snapshot_rejects_code_or_inventory_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            code = source / "train.py"
            code.write_text("pass\n")
            plan = {"source_files": {"train.py": suite.sha256(code)}}
            suite.verify_snapshot(temporary, plan)
            code.write_text("print('changed')\n")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                suite.verify_snapshot(temporary, plan)
            code.write_text("pass\n")
            (source / "extra.py").write_text("pass\n")
            with self.assertRaisesRegex(RuntimeError, "inventory"):
                suite.verify_snapshot(temporary, plan)


if __name__ == "__main__":
    unittest.main()
