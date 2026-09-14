"""Closed capacity comparison and evidence validation reuse the residual suite engine."""

import copy
import dataclasses
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_residual_gate_controls import CONTROL, INTEGRATION, fixture, write

SPEC = CONTROL.CAPACITY_SPEC


def rebind_manifest(run, mutate):
    path = Path(run["training_manifest"])
    manifest = json.loads(path.read_text())
    mutate(manifest)
    write(path, manifest)
    marker_path = path.parent / "prism_training_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["actor"] = copy.deepcopy(manifest["actor"])
    marker["training_manifest"]["sha256"] = CONTROL.queue.sha256(path)
    write(marker_path, marker)


class CapacityControlsTest(unittest.TestCase):
    def test_closed_catalog_twenty_runs_in_registered_order_and_cpu_cli(self):
        self.assertEqual(SPEC.variants, ("prism-1321k", "mlp-1321k", "mlp-wide-1500k", "mlp-deep-1500k"))
        plan = CONTROL.planned_suite("/tmp/capacity-example", SPEC)
        self.assertEqual(len(plan["runs"]), 20)
        self.assertEqual([row["variant"] for row in plan["runs"][:4]], list(SPEC.variants))
        self.assertEqual(plan["evaluation_seeds"], [101, 102, 103])
        self.assertEqual(plan["num_envs"], 4096)
        self.assertEqual(plan["max_iterations"], 3001)
        self.assertEqual(plan["evaluation"]["episodes"], 200)
        for run in plan["runs"]:
            command = CONTROL.queue.training_command("/python", "/source", "/output", run, "cuda:0")
            self.assertIn("--variant=" + run["variant"], command)
            self.assertIn("--max_iterations=3001", command)
            self.assertNotIn("--resume", command)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SPEC.kind = "changed"
        with self.assertRaisesRegex(ValueError, "registered immutable"):
            CONTROL.planned_suite("/tmp/example", dataclasses.replace(SPEC, title="different"))
        changed = SPEC.methods
        changed["prism-1321k"]["actor_mean_parameters"] = 0
        self.assertEqual(SPEC.methods["prism-1321k"]["actor_mean_parameters"], 921715)
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(INTEGRATION / "run_capacity_controls.py"),
                "--output-dir=/tmp/not-created-capacity-fixture",
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["suite_kind"], SPEC.kind)

    def test_capacity_summary_requires_twenty_runs_and_sixty_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output, capacity=True)
            report = CONTROL.write_report(output, require_complete=True, spec=SPEC)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["verified_training_runs"], 20)
            self.assertEqual(report["verified_evaluations"], 60)
            self.assertFalse(report["paired_initialization_required"])
            self.assertTrue(all(row["initial_shared_parameter_sha256"] is None for row in report["records"]))
            value = report["methods"]["prism-1321k"]["metrics"]["avg_lin_vel_error"]
            self.assertAlmostEqual(value["mean"], 3.102)
            self.assertAlmostEqual(value["sample_sd"], math.sqrt(2.5))
            self.assertTrue((output / "report/capacity_summary.md").is_file())
            self.assertFalse((output / "report/diagnostic_summary.json").exists())
            for identity in report["reporter"].values():
                self.assertEqual(identity["sha256"], CONTROL.queue.sha256(identity["path"]))

    def test_missing_evaluation_keeps_all_summary_statistics_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output, capacity=True)
            Path(suite["runs"][-1]["evaluations"][-1]).unlink()
            report = CONTROL.build_report(output, SPEC)
            self.assertEqual(report["status"], "pending")
            self.assertEqual(report["verified_training_runs"], 20)
            self.assertEqual(report["verified_evaluations"], 59)
            self.assertTrue(all(row["metrics"] is None for row in report["methods"].values()))
            with self.assertRaisesRegex(ValueError, "pending"):
                CONTROL.write_report(output, require_complete=True, spec=SPEC)

    def test_default_gated_width_cannot_be_adopted_as_capacity_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output, capacity=True)
            rebind_manifest(
                suite["runs"][0],
                lambda manifest: manifest["training_config"]["policy"].update(poly_hidden_dim=256),
            )
            with self.assertRaisesRegex(ValueError, "exact registered architecture"):
                CONTROL.build_report(output, SPEC)

    def test_gate_shape_and_shared_ppo_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output, capacity=True)
            first = suite["runs"][0]
            marker_path = Path(first["training_manifest"]).parent / "prism_training_complete.json"
            marker = json.loads(marker_path.read_text())
            marker["gate"]["shape"] = [1, 256]
            write(marker_path, marker)
            with self.assertRaisesRegex(ValueError, "gate shape"):
                CONTROL.build_report(output, SPEC)
            marker["gate"]["shape"] = [1, 334]
            write(marker_path, marker)
            rebind_manifest(
                suite["runs"][1],
                lambda manifest: manifest["training_config"]["algorithm"].update(learning_rate=1e-3),
            )
            with self.assertRaisesRegex(ValueError, "Training PPO"):
                CONTROL.build_report(output, SPEC)

    def test_capacity_and_residual_suites_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output, capacity=True)
            with self.assertRaisesRegex(ValueError, "selected frozen"):
                CONTROL.build_report(output)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output, complete=False)
            with self.assertRaisesRegex(ValueError, "selected frozen"):
                CONTROL.build_report(output, SPEC)

    def test_report_calculator_must_match_frozen_source(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output, complete=False, capacity=True)
            calculator = output / "source/integration/run_residual_gate_controls.py"
            calculator.write_text("# a different report implementation\n")
            plan_path = output / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["source_files"]["integration/run_residual_gate_controls.py"] = CONTROL.queue.sha256(
                calculator
            )
            write(plan_path, plan)
            CONTROL.queue.verify_snapshot(output, plan)
            with self.assertRaisesRegex(ValueError, "Executing reporter differs"):
                CONTROL.build_report(output, SPEC)


if __name__ == "__main__":
    unittest.main()
