"""CPU checks for residual-only scope, recipe identity and reference parity."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_residual_gate_controls import INTEGRATION, fixture, write
from test_robustness_runner import robustness_fixture

with mock.patch.object(sys, "path", [str(INTEGRATION)] + sys.path):
    SPEC = importlib.util.spec_from_file_location(
        "residual_robustness_test", INTEGRATION / "run_residual_robustness.py"
    )
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)


def read(path):
    return json.loads(Path(path).read_text())


class ResidualRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.residual = self.root / "residual"
        fixture(self.residual)
        self.residual_plan = read(self.residual / "plan.json")
        self.bound = RUNNER.training(self.residual, self.residual_plan)
        capacity, reference, cap_plan, ref_plan = robustness_fixture(self.root)
        self.output = self.root / "new_residual_robustness"
        shutil.copytree(reference / "source", self.output / "source")
        source = [
            RUNNER.identity(path) for path in sorted((self.output / "source").rglob("*")) if path.is_file()
        ]
        self.plan = {
            "python": str(Path(sys.executable).resolve()),
            "device": "cuda:0",
            "environment": {},
            "source_files": {
                str(Path(row["path"]).relative_to(self.output / "source")): row["sha256"] for row in source
            },
            "source_bridge": {},
            "reference_summary": RUNNER.identity(reference / "plan.json"),
        }
        write(self.output / "plan.json", self.plan)
        write(self.output / "queue_state.json", {"status": "prepared", "jobs": {}})
        other = RUNNER.base.training(capacity, cap_plan, "prism-1321k")
        self.references = {}
        for condition in RUNNER.CONDITIONS:
            for seed in RUNNER.EVALUATION_SEEDS:
                ref_path = RUNNER.base.destination(reference, "prism-1321k", condition, seed)
                _, signature, realized, ref_payload = RUNNER.base.validate_evaluation(
                    reference, ref_plan, other, "prism-1321k", condition, seed, ref_path
                )
                self.references[(condition, seed)] = {"signature": signature, "realized": realized}
                payload = read(self.residual / "runs/legacy-prism/seed1" / f"eval_{seed}.json")
                meta = payload["metadata"]
                for name in (
                    "evaluation_config",
                    "evaluation_arguments",
                    "robustness",
                    "push_interval_s",
                    "max_push_vel_xy",
                    "condition_name",
                ):
                    meta[name] = copy.deepcopy(ref_payload["metadata"][name])
                meta["source"] = {"files": copy.deepcopy(source)}
                write(self.path(condition, seed), payload)
        with mock.patch.object(RUNNER.base, "load_plan", return_value=(ref_plan, capacity, cap_plan)):
            reference_report = RUNNER.base.build_report(reference)
        write(reference / "robustness_summary.json", reference_report)
        self.plan["reference_summary"] = RUNNER.identity(reference / "robustness_summary.json")
        write(self.output / "plan.json", self.plan)
        patch = mock.patch.object(
            RUNNER, "load_context", return_value=(self.plan, self.bound, self.references)
        )
        patch.start()
        self.addCleanup(patch.stop)

    def path(self, condition="nominal", seed=101):
        return RUNNER.base.destination(self.output, "legacy-prism", condition, seed)

    def change(self, path, operation):
        value = read(path)
        operation(value)
        write(path, value)

    def test_nine_evaluations_keep_one_training_seed_and_no_sd(self):
        report = RUNNER.build_report(self.output)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["verified_evaluations"], 9)
        self.assertEqual(report["total_episodes"], 1800)
        self.assertEqual(report["variants"], ["legacy-prism"])
        self.assertEqual(len(report["records"]), 3)
        for row in report["records"]:
            self.assertEqual(row["training_seed"], 1)
            self.assertEqual(row["metrics"]["success_rate"], 0.5)
            self.assertNotIn("sample_sd", row["metrics"])

    def test_partial_condition_has_no_average(self):
        self.path("payload_mass", 103).unlink()
        report = RUNNER.build_report(self.output)
        self.assertEqual(report["verified_evaluations"], 8)
        self.assertIsNone(report["records"][2]["metrics"])
        with self.assertRaisesRegex(ValueError, "pending"):
            RUNNER.write_report(self.output, require_complete=True)

    def test_writer_extends_reference_to_five_models_without_mutating_reference(self):
        source = self.plan["reference_summary"]
        original = Path(source["path"]).read_bytes()
        self.path("payload_mass", 103).unlink()
        RUNNER.write_report(self.output)
        comparison = read(self.output / "comparison_summary.json")
        self.assertEqual(comparison["verified_evaluations"], 44)
        self.assertEqual(len(comparison["records"]), 15)
        self.assertEqual(len(comparison["survival_percent"]), 5)
        self.assertIsNone(comparison["survival_percent"][-1]["payload_mass"])
        self.assertEqual(comparison["survival_percent"][-1]["nominal"], 50.0)
        self.assertEqual(Path(source["path"]).read_bytes(), original)
        self.assertEqual(comparison["reference_summary"], source)
        self.assertTrue((self.output / "comparison_summary.md").is_file())

    def test_canceled_variant_cannot_be_selected_or_adopted(self):
        with self.assertRaisesRegex(ValueError, "Only legacy-prism"):
            RUNNER.execute(self.output, ["residual-learned-gate"])
        with self.assertRaisesRegex(ValueError, "Only legacy-prism"):
            RUNNER.training(self.residual, self.residual_plan, "residual-learned-gate")
        (self.output / "evaluations/residual-learned-gate").mkdir()
        with self.assertRaisesRegex(ValueError, "Only legacy-prism"):
            RUNNER.build_report(self.output)

    def test_unknown_condition_and_seed_are_rejected(self):
        outside = self.output / "evaluations/legacy-prism/nominal/eval_999.json"
        write(outside, {})
        with self.assertRaisesRegex(ValueError, "scope"):
            RUNNER.build_report(self.output)

    def test_checkpoint_and_manifest_binding_cannot_be_changed(self):
        for field in ("checkpoint", "training_manifest"):
            with self.subTest(field=field):
                old = read(self.path())
                self.change(self.path(), lambda p, field=field: p["metadata"][field].update(sha256="0" * 64))
                with self.assertRaisesRegex(ValueError, "checkpoint identity|manifest binding"):
                    RUNNER.build_report(self.output)
                write(self.path(), old)

    def test_changed_actor_counts_are_rejected(self):
        self.change(self.path(), lambda p: p["metadata"]["actor"].update(actor_mean_parameters=1))
        with self.assertRaisesRegex(ValueError, "Wrong diagnostic actor"):
            RUNNER.build_report(self.output)

    def test_training_source_stays_strict_even_for_audited_recipe_modules(self):
        def mutate(payload):
            item = next(x for x in payload["metadata"]["source"]["files"] if x["path"].endswith("config.py"))
            item["sha256"] = "f" * 64
            self.plan["source_files"]["integration/config.py"] = item["sha256"]

        self.change(self.path(), mutate)
        with self.assertRaisesRegex(ValueError, "Training/evaluation implementation"):
            RUNNER.build_report(self.output)

    def test_reference_environment_and_runtime_mismatch_rejected(self):
        for field, key, value in (("environment_config", "new_setting", 1), ("runtime", "torch", "other")):
            with self.subTest(field=field):
                before = copy.deepcopy(self.references[("nominal", 101)])
                self.references[("nominal", 101)]["signature"][field][key] = value
                with self.assertRaisesRegex(ValueError, "differs from reference"):
                    RUNNER.build_report(self.output)
                self.references[("nominal", 101)] = before

    def test_reference_shared_source_mismatch_is_rejected(self):
        self.references[("nominal", 101)]["signature"]["source"]["integration/legacy_actor.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from reference"):
            RUNNER.build_report(self.output)

    def test_realized_physics_must_equal_reference(self):
        self.references[("nominal", 101)]["realized"]["first_actor"]["bodies"][0]["com"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "physics differ from reference"):
            RUNNER.build_report(self.output)

    def test_stress_without_nominal_is_rejected(self):
        self.path("nominal", 101).unlink()
        with self.assertRaisesRegex(ValueError, "needs its nominal"):
            RUNNER.build_report(self.output)

    def test_wrong_budget_and_seed_markers_are_rejected(self):
        marker = self.bound["manifest_path"].parent / "prism_training_complete.json"
        self.change(marker, lambda p: p.update(completed_iteration=3000))
        with self.assertRaisesRegex(RuntimeError, "3001-update"):
            RUNNER.training(self.residual, self.residual_plan)

    def test_only_validated_legacy_recipe_fields_are_removed_from_training_comparison(self):
        signature = copy.deepcopy(self.bound["training_signature"])
        signature["training_config"]["policy"]["actor_poly_warmup_updates"] = 0
        with self.assertRaisesRegex(ValueError, "Legacy architecture setting"):
            RUNNER.comparable_signature(signature, True, True)


class SourceBridgeTests(unittest.TestCase):
    def test_bridge_only_accepts_capacity_additions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / "old", root / "new"
            for directory in (old, new):
                (directory / "source/integration").mkdir(parents=True)
            (old / "source/integration/config.py").write_text("class Legacy:\n    value = 1\n")
            additions = "".join(
                "\nclass " + name + ":\n    pass\n" for name in sorted(RUNNER.CAPACITY_CLASSES)
            )
            (new / "source/integration/config.py").write_text("class Legacy:\n    value = 1\n" + additions)
            (old / "source/integration/runtime.py").write_text(
                "VARIANTS = {'legacy-prism': ('task', 'Legacy')}\n"
            )
            registry = {"legacy-prism": ("task", "Legacy")}
            registry.update(
                {variant: (variant, "Capacity") for variant in RUNNER.controls.CAPACITY_SPEC.variants}
            )
            (new / "source/integration/runtime.py").write_text("VARIANTS = " + repr(registry) + "\n")
            bridge = RUNNER.audit_source_bridge(old, new)
            self.assertEqual(set(bridge), {"integration/config.py", "integration/runtime.py"})
            path = new / "source/integration/config.py"
            path.write_text(path.read_text().replace("value = 1", "value = 2"))
            with self.assertRaisesRegex(ValueError, "Undeclared source difference"):
                RUNNER.audit_source_bridge(old, new)

    def test_cli_help_has_no_simulator_dependency(self):
        result = subprocess.run(
            [sys.executable, "-S", str(INTEGRATION / "run_residual_robustness.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--reference-robustness-suite", result.stdout)


if __name__ == "__main__":
    unittest.main()
