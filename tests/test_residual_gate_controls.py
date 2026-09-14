"""Separate residual-gate registration, provenance rejection and nested statistics."""

import copy
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/humanoid-gym"
with mock.patch.object(sys, "path", [str(INTEGRATION)] + sys.path):
    SPEC = importlib.util.spec_from_file_location(
        "residual_controls_test", INTEGRATION / "run_residual_gate_controls.py"
    )
    CONTROL = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CONTROL)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def gate(steps=0, width=256, initial=1.0):
    return {
        "shape": [1, width],
        "features": width,
        "finite": True,
        "requires_grad": True,
        "optimizer_occurrences": 1,
        "optimizer_steps": steps,
        "min": initial - 0.1 if steps else initial,
        "max": initial + 0.1 if steps else initial,
        "max_abs_change_from_start": 0.1 if steps else 0.0,
    }


def fixture(output, complete=True, capacity=False):
    spec = CONTROL.CAPACITY_SPEC if capacity else CONTROL.RESIDUAL_SPEC
    source = []
    names = CONTROL.metrics.TRAIN_CODE | {
        "residual_gated_actor.py",
        "evaluate.py",
        "evaluation_protocol.py",
        "run_residual_gate_controls.py",
        "run_capacity_controls.py",
        "build_main_table.py",
        "run_main_table.py",
    }
    for name in names:
        path = output / "source/integration" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in {
            "run_residual_gate_controls.py",
            "run_capacity_controls.py",
            "build_main_table.py",
            "run_main_table.py",
        }:
            path.write_bytes((INTEGRATION / name).read_bytes())
        else:
            path.write_text("# fixture only: " + name)
        source.append({"path": str(path), "sha256": CONTROL.queue.sha256(path)})
    plan = {
        "suite_kind": spec.kind,
        "source_files": {
            str(Path(row["path"]).relative_to(output / "source")): row["sha256"] for row in source
        },
    }
    write(output / "plan.json", plan)
    suite = CONTROL.planned_suite(output, spec)
    write(output / "suite.json", suite)
    if not complete:
        return suite
    runtime = {
        "python": "3.8.20",
        "torch": "2.3.1",
        "numpy": "1.20.0",
        "isaacgym": "preview4",
        "torch_cuda": "12.1",
    }
    for run in suite["runs"]:
        variant, seed = run["variant"], run["training_seed"]
        learned = variant == ("prism-1321k" if capacity else "residual-learned-gate")
        width, gate_init = (334, 0.01) if capacity else (256, 1.0)
        hidden_dims = {
            "prism-1321k": [513, 256, 128],
            "mlp-1321k": [816, 352, 160],
            "mlp-wide-1500k": [928, 384, 224],
            "mlp-deep-1500k": [768, 512, 256, 128],
        }.get(variant, [512, 256, 128])
        directory = output / "runs" / variant / ("seed" + str(seed)) / "training/fresh"
        directory.mkdir(parents=True)
        actor = {key: value for key, value in spec.methods[variant].items() if key not in ("label", "task")}
        actor.update(
            recipe_variant=variant,
            actor_hidden_dims=hidden_dims,
            polynomial_degree=2 if learned or not capacity else None,
            warmup_ppo_updates=(0 if learned else None) if capacity else 500,
            gate_semantics="learned_per_feature"
            if learned
            else (None if capacity else "legacy_scalar_scale"),
            gate_init=gate_init if learned else None,
        )
        if not capacity:
            actor["initial_shared_parameter_sha256"] = f"{seed:064x}"
        policy = {
            "actor_hidden_dims": hidden_dims,
            "critic_hidden_dims": [768, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "poly_hidden_dim": 256,
            "poly_degree": 2,
            "actor_use_poly": True,
            "critic_use_poly": False,
            "actor_poly_mode": "residual",
            "actor_use_input_layer_norm": False,
            "critic_use_input_layer_norm": False,
            "actor_use_hidden_layer_norm": False,
            "critic_use_hidden_layer_norm": False,
            "actor_output_tanh": False,
            "actor_poly_warmup_updates": 500,
        }
        if capacity:
            policy = {
                key: value
                for key, value in policy.items()
                if key in {"actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std"}
            }
        if learned:
            policy.update(actor_variant=actor["actor_variant"], gate_init=gate_init)
            if capacity:
                policy.update(poly_hidden_dim=334, poly_degree=2)
        manifest = {
            "schema_version": 1,
            "record_kind": "training_start",
            "seed": seed,
            "task": spec.methods[variant]["task"],
            "actor": actor,
            "initial_learning_iteration": 0,
            "resume_checkpoint": None,
            "gate_initial": gate(width=width, initial=gate_init) if learned else None,
            "training_config": {
                "seed": seed,
                "policy": policy,
                "algorithm": {"learning_rate": 1e-5},
                "runner": {
                    "policy_class_name": actor["class"],
                    "num_steps_per_env": 60,
                    "max_iterations": 3001,
                    "resume": False,
                },
            },
            "environment_config": {"seed": seed, "env": {"num_envs": 4096, "episode_length_s": 24}},
            "arguments": {"seed": seed, "task": spec.methods[variant]["task"], "physics_engine": "physx"},
            "source": {"files": copy.deepcopy(source)},
            "runtime": runtime,
        }
        manifest_path = directory / "prism_run_manifest.json"
        write(manifest_path, manifest)
        checkpoint = directory / "model_3001.pt"
        checkpoint.write_bytes((variant + "/" + str(seed) + ": synthetic metadata fixture").encode())
        checkpoint_identity = {"path": str(checkpoint), "sha256": CONTROL.queue.sha256(checkpoint)}
        manifest_identity = {"path": str(manifest_path), "sha256": CONTROL.queue.sha256(manifest_path)}
        marker = {
            "status": "completed",
            "completed_iteration": 3001,
            "initial_iteration": 0,
            "requested_final_iteration": 3001,
            "seed": seed,
            "actor": actor,
            "checkpoint": checkpoint_identity,
            "training_manifest": manifest_identity,
            "gate_initial": gate(width=width, initial=gate_init) if learned else None,
            "gate": gate(24008, width=width, initial=gate_init) if learned else None,
        }
        write(directory / "prism_training_complete.json", marker)
        run["training_manifest"] = str(manifest_path)
        for evaluation_index, eval_seed in enumerate(CONTROL.EVAL_SEEDS):
            destination = run["evaluations"][evaluation_index]
            episodes = [
                {
                    "env_id": env,
                    "episode_ordinal": ordinal,
                    "length": 100 + seed,
                    "return": seed * 10,
                    "avg_lin_vel_error": seed + eval_seed / 1000,
                    "avg_yaw_vel_error": seed / 10,
                    "timeout": env < 50,
                    "success": env < 50,
                }
                for env in range(100)
                for ordinal in (1, 2)
            ]
            result = {
                "task": manifest["task"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_identity["sha256"],
                "checkpoint_iteration": 3001,
                "actor_variant": actor["actor_variant"],
                "num_episodes": 200,
                "num_envs": 100,
                "protocol_id": CONTROL.metrics.PROTOCOL_ID,
                "protocol": {
                    "id": CONTROL.metrics.PROTOCOL_ID,
                    "version": 2,
                    "episode_allocation": "equal_per_environment_quota",
                    "episode_quotas": [2] * 100,
                    "success_definition": "environment_timeout",
                    "tracking_error_timing": "before_terminal_reset",
                },
                "episodes": episodes,
            }
            for name, episode_key in (
                ("avg_episode_length", "length"),
                ("avg_return", "return"),
                ("avg_lin_vel_error", "avg_lin_vel_error"),
                ("avg_yaw_vel_error", "avg_yaw_vel_error"),
                ("success_rate", "success"),
            ):
                result[name] = statistics.mean(row[episode_key] for row in episodes)
            payload = {
                "schema_version": 2,
                "result": result,
                "metadata": {
                    "training_seed": seed,
                    "evaluation_seed": eval_seed,
                    "seed": eval_seed,
                    "recipe_variant": variant,
                    "training_manifest": dict(manifest_identity, payload=manifest),
                    "checkpoint": checkpoint_identity,
                    "checkpoint_iteration": 3001,
                    "actor": actor,
                    "actor_variant": actor["actor_variant"],
                    "protocol_id": result["protocol_id"],
                    "evaluation_protocol": result["protocol"],
                    "episodes": 200,
                    "num_envs": 100,
                    "condition_name": "match_nopush",
                    "terrain_mode": "match",
                    "push_mode": "off",
                    "evaluation_config": {
                        "seed": eval_seed,
                        "env": {"num_envs": 100, "episode_length_s": 24},
                    },
                    "evaluation_arguments": {"seed": eval_seed, "physics_engine": "physx"},
                    "source": {"files": copy.deepcopy(source)},
                    "runtime": runtime,
                },
            }
            write(Path(destination), payload)
    write(output / "suite.json", suite)
    return suite


class ResidualGateControlsTest(unittest.TestCase):
    def test_prepare_reserves_a_separate_frozen_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            unitree, rsl = root / "unitree", root / "rsl"
            for directory, package in ((unitree, "legged_gym"), (rsl, "rsl_rl")):
                (directory / package).mkdir(parents=True)
                (directory / package / "__init__.py").write_text("# fixture\n")
            output = root / "diagnostic"
            plan = CONTROL.prepare(output, sys.executable, unitree, rsl, "cuda:0")
            self.assertEqual(plan["suite_kind"], CONTROL.KIND)
            suite = json.loads((output / "suite.json").read_text())
            CONTROL.validate_suite(output, suite)
            CONTROL.queue.verify_snapshot(output, plan)
            self.assertIn("integration/run_residual_gate_controls.py", plan["source_files"])
            with self.assertRaises(FileExistsError):
                CONTROL.prepare(output, sys.executable, unitree, rsl, "cuda:0")

    def test_registration_and_optional_dependency_free_cli(self):
        suite = CONTROL.planned_suite("/tmp/residual-example")
        self.assertEqual(len(suite["runs"]), 10)
        self.assertEqual(suite["suite_kind"], CONTROL.KIND)
        self.assertEqual({run["variant"] for run in suite["runs"]}, set(CONTROL.VARIANTS))
        self.assertEqual(suite["num_envs"], 4096)
        self.assertEqual(suite["max_iterations"], 3001)
        self.assertEqual(suite["evaluation"]["episodes"], 200)
        command = [
            sys.executable,
            "-S",
            str(INTEGRATION / "run_residual_gate_controls.py"),
            "--output-dir=/tmp/not-created-residual-fixture",
            "--dry-run",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["suite_kind"], CONTROL.KIND)

    def test_pending_has_no_summary_values_and_require_complete_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output, complete=False)
            report = CONTROL.write_report(output)
            self.assertEqual(report["status"], "pending")
            self.assertEqual(report["verified_training_runs"], 0)
            self.assertTrue(all(row["metrics"] is None for row in report["methods"].values()))
            with self.assertRaisesRegex(ValueError, "pending"):
                CONTROL.write_report(output, require_complete=True)
            self.assertFalse((output / "main_table.tex").exists())

    def test_complete_aggregates_eval_seeds_then_five_training_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fixture(output)
            report = CONTROL.write_report(output, require_complete=True)
            self.assertEqual(report["verified_training_runs"], 10)
            self.assertEqual(report["verified_evaluations"], 30)
            self.assertEqual(report["status"], "complete")
            metric = report["methods"]["residual-learned-gate"]["metrics"]["avg_lin_vel_error"]
            self.assertAlmostEqual(metric["mean"], 3.102)
            self.assertAlmostEqual(metric["sample_sd"], math.sqrt(2.5))

    def test_tampering_and_shortened_eval_list_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output)
            original_suite = copy.deepcopy(suite)
            suite["runs"][0]["evaluations"].pop()
            write(output / "suite.json", suite)
            with self.assertRaisesRegex(ValueError, "registration"):
                CONTROL.build_report(output)
            write(output / "suite.json", original_suite)
            path = Path(original_suite["runs"][0]["evaluations"][0])
            original = json.loads(path.read_text())
            for change, message in (
                (lambda value: value["metadata"].update(training_seed=99), "training seed"),
                (lambda value: value["metadata"].update(actor_variant="g1_gated_poly_v2"), "identity"),
                (lambda value: value["result"]["episodes"][0].update(env_id=99), "balanced"),
            ):
                value = copy.deepcopy(original)
                change(value)
                write(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    CONTROL.build_report(output)
            write(path, original)
            checkpoint = Path(
                json.loads(Path(original_suite["runs"][0]["training_manifest"]).read_text())["source"][
                    "files"
                ][0]["path"]
            )
            checkpoint.write_text("changed frozen code")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                CONTROL.build_report(output)

    def test_changed_checkpoint_and_unchanged_gate_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output)
            run = next(row for row in suite["runs"] if row["variant"] == "residual-learned-gate")
            directory = Path(run["training_manifest"]).parent
            path = directory / "prism_training_complete.json"
            marker = json.loads(path.read_text())
            marker["gate"]["max_abs_change_from_start"] = 0.0
            write(path, marker)
            with self.assertRaisesRegex(ValueError, "optimizer updates"):
                CONTROL.build_report(output)
            marker["gate"]["max_abs_change_from_start"] = 0.1
            write(path, marker)
            (directory / "model_3001.pt").write_bytes(b"replaced")
            with self.assertRaisesRegex(RuntimeError, "identity"):
                CONTROL.build_report(output)

    def test_paired_seed_requires_identical_initial_shared_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            suite = fixture(output)
            run = next(row for row in suite["runs"] if row["variant"] == "residual-learned-gate")
            path = Path(run["training_manifest"])
            manifest = json.loads(path.read_text())
            manifest["actor"]["initial_shared_parameter_sha256"] = "f" * 64
            write(path, manifest)
            marker_path = path.parent / "prism_training_complete.json"
            marker = json.loads(marker_path.read_text())
            marker["actor"] = manifest["actor"]
            marker["training_manifest"]["sha256"] = CONTROL.queue.sha256(path)
            write(marker_path, marker)
            with self.assertRaisesRegex(ValueError, "identical shared parameters"):
                CONTROL.build_report(output)


if __name__ == "__main__":
    unittest.main()
