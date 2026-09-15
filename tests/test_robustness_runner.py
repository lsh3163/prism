"""CPU-only robustness provenance, physical controls, and report validation."""

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_residual_gate_controls import INTEGRATION, fixture, write

with mock.patch.object(sys, "path", [str(INTEGRATION)] + sys.path):
    SPEC = importlib.util.spec_from_file_location("robustness_runner_test", INTEGRATION / "run_robustness.py")
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)


def read(path):
    return json.loads(path.read_text())


def physics(condition):
    """Two synthetic links with reciprocal inertia and a declared material."""
    scale = 1.15 if condition == "payload_mass" else 1.0
    friction = 0.2 if condition == "low_friction" else 1.0
    ground = {
        "static_friction": friction,
        "dynamic_friction": friction,
        "source": "configured_terrain_parameters",
    }
    bodies = []
    for name, mass in (("base", 4.0), ("leg", 1.0)):
        bodies.append(
            {
                "name": name,
                "mass": mass * scale,
                "nominal_mass": mass,
                "inv_mass": 1.0 / (mass * scale),
                "com": [0.0, 0.0, 0.1],
                "inertia": [[scale if i == j else 0.0 for j in range(3)] for i in range(3)],
                "inv_inertia": [[1.0 / scale if i == j else 0.0 for j in range(3)] for i in range(3)],
            }
        )
    return {
        "spec": {
            "schema_version": 1,
            "condition": condition,
            "ground": ground,
            "robot_shape_friction": friction if condition == "low_friction" else None,
            "rigid_body_mass_scale": scale,
            "mass_scope": "all_robot_rigid_bodies",
            "inertia_update": "upstream_recomputeInertia_True",
            "payload_interpretation": "uniform_link_mass_scaling_at_fixed_geometry_and_COM",
        },
        "realized": {
            "schema_version": 1,
            "condition": condition,
            "num_environments": 100,
            "all_envs_consistent": True,
            "configured_ground": copy.deepcopy(ground),
            "first_actor": {
                "body_names": [body["name"] for body in bodies],
                "bodies": bodies,
                "shape_friction": [friction, friction],
                "total_mass": 5.0 * scale,
                "nominal_total_mass": 5.0,
            },
            "validation": {
                "all_actors_checked": 100,
                "mass_scale": scale,
                "mass_scale_verified": True,
                "mass_reference": "nominal_readback" if condition == "nominal" else "pre_scaling_hook",
                "robot_shape_friction_verified": condition == "low_friction",
                "observation_caches_consistent": True,
                "inertia_scaling": "requires_nominal_comparison"
                if condition == "payload_mass"
                else "not_scaled",
                "ground_properties": "configured_only_no_simulator_readback",
            },
        },
    }


def nominal_config(seed):
    return {
        "seed": seed,
        "env": {"num_envs": 100, "episode_length_s": 24},
        "terrain": {"mesh_type": "plane", "static_friction": 1.0, "dynamic_friction": 1.0},
        "noise": {"add_noise": False},
        "domain_rand": {
            "randomize_friction": False,
            "randomize_base_mass": False,
            "randomize_pd_gains": False,
            "push_robots": False,
            "action_delay": False,
            "action_noise": 0.0,
        },
    }


def robustness_fixture(root):
    capacity, output = root / "capacity", root / "robustness"
    fixture(capacity, capacity=True)
    cap_plan = read(capacity / "plan.json")
    shutil.copytree(capacity / "source", output / "source")
    for name in ("run_robustness.py", "robustness.py"):
        shutil.copy2(INTEGRATION / name, output / "source/integration" / name)
    source = [RUNNER.identity(path) for path in sorted((output / "source").rglob("*")) if path.is_file()]
    plan = {
        "python": str(Path(sys.executable).resolve()),
        "device": "cuda:0",
        "environment": {},
        "source_files": {
            str(Path(row["path"]).relative_to(output / "source")): row["sha256"] for row in source
        },
    }
    write(output / "plan.json", plan)
    for variant in RUNNER.VARIANTS:
        for seed in RUNNER.EVALUATION_SEEDS:
            old_path = capacity / "runs" / variant / "seed1" / f"eval_{seed}.json"
            original = read(old_path)
            old_meta = original["metadata"]
            old_meta["evaluation_config"] = nominal_config(seed)
            old_meta["evaluation_arguments"].update(
                condition_name="match_nopush", sim_device="cuda:0", rl_device="cuda:0"
            )
            write(old_path, original)
            for condition in RUNNER.CONDITIONS:
                payload = copy.deepcopy(original)
                meta = payload["metadata"]
                meta.update(
                    source={"files": copy.deepcopy(source)},
                    condition_name=condition,
                    robustness=physics(condition),
                    push_interval_s=None,
                    max_push_vel_xy=None,
                )
                meta["evaluation_arguments"].update(condition_name=condition, robustness_condition=condition)
                if condition == "low_friction":
                    meta["evaluation_config"]["terrain"].update(static_friction=0.2, dynamic_friction=0.2)
                write(RUNNER.destination(output, variant, condition, seed), payload)
    return capacity, output, cap_plan, plan


class RobustnessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.capacity, self.output, self.cap_plan, self.plan = robustness_fixture(Path(self.temporary.name))
        self.plan_patch = mock.patch.object(
            RUNNER, "load_plan", return_value=(self.plan, self.capacity, self.cap_plan)
        )
        self.plan_patch.start()
        self.addCleanup(self.plan_patch.stop)

    def result_path(self, condition="nominal", variant="prism-1321k", seed=101):
        return RUNNER.destination(self.output, variant, condition, seed)

    def change(self, path, mutate):
        payload = read(path)
        mutate(payload)
        write(path, payload)

    def test_all_four_models_have_36_evaluations_without_training_seed_sd(self):
        report = RUNNER.build_report(self.output)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["verified_evaluations"], 36)
        self.assertEqual(report["total_episodes"], 7200)
        self.assertEqual(len(report["records"]), 12)
        for record in report["records"]:
            self.assertEqual(record["training_seed"], 1)
            self.assertEqual(record["metrics"]["success_rate"], 0.5)
            self.assertNotIn("sample_sd", record["metrics"])
            if record["condition"] == "nominal":
                self.assertEqual(record["nominal_parity_verified_eval_seeds"], [101, 102, 103])

    def test_wrong_training_seed_is_rejected(self):
        self.change(self.result_path(), lambda p: p["metadata"].update(training_seed=2))
        with self.assertRaisesRegex(ValueError, "training seed"):
            RUNNER.build_report(self.output)

    def test_pooled_protocol_is_rejected_even_when_aliases_agree(self):
        def mutate(payload):
            payload["result"]["protocol"]["episode_allocation"] = "first_completed_episodes"
            payload["metadata"]["evaluation_protocol"] = copy.deepcopy(payload["result"]["protocol"])

        self.change(self.result_path(), mutate)
        with self.assertRaisesRegex(ValueError, "metric semantics"):
            RUNNER.build_report(self.output)

    def test_wrong_checkpoint_hash_is_rejected(self):
        self.change(self.result_path(), lambda p: p["metadata"]["checkpoint"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            RUNNER.build_report(self.output)

    def test_low_friction_must_change_robot_shapes_and_ground(self):
        path = self.result_path("low_friction")
        original = read(path)
        for target in ("ground", "shape"):
            with self.subTest(target=target):
                changed = copy.deepcopy(original)
                if target == "ground":
                    changed["metadata"]["evaluation_config"]["terrain"]["dynamic_friction"] = 0.3
                else:
                    changed["metadata"]["robustness"]["realized"]["first_actor"]["shape_friction"][1] = 1.0
                write(path, changed)
                with self.assertRaisesRegex(ValueError, "friction differs"):
                    RUNNER.build_report(self.output)

    def test_physics_capture_must_verify_all_environments(self):
        for condition in ("nominal", "low_friction"):
            with self.subTest(condition=condition):
                payload = read(self.result_path(condition))
                payload["metadata"]["robustness"]["realized"]["validation"]["all_actors_checked"] = 99
                with self.assertRaises(ValueError):
                    RUNNER.validate_physics(payload["metadata"], condition)

    def test_robot_physics_cannot_be_empty(self):
        for field in ("shape_friction", "bodies"):
            with self.subTest(field=field):
                payload = read(self.result_path("low_friction"))
                payload["metadata"]["robustness"]["realized"]["first_actor"][field] = []
                with self.assertRaises(ValueError):
                    RUNNER.validate_physics(payload["metadata"], "low_friction")

    def test_same_condition_must_have_same_realized_bodies_across_models(self):
        for condition in RUNNER.CONDITIONS:
            for seed in RUNNER.EVALUATION_SEEDS:
                path = self.result_path(condition, "mlp-1321k", seed)
                payload = read(path)
                actor = payload["metadata"]["robustness"]["realized"]["first_actor"]
                for body in actor["bodies"]:
                    body["mass"] *= 2.0
                    body["nominal_mass"] *= 2.0
                    body["inv_mass"] /= 2.0
                    body["inertia"] = [[value * 2.0 for value in row] for row in body["inertia"]]
                    body["inv_inertia"] = [[value / 2.0 for value in row] for row in body["inv_inertia"]]
                actor["total_mass"] *= 2.0
                actor["nominal_total_mass"] *= 2.0
                write(path, payload)
        with self.assertRaises(ValueError):
            RUNNER.build_report(self.output)

    def test_payload_must_scale_mass_and_inertia_without_moving_com(self):
        path = self.result_path("payload_mass")
        original = read(path)
        for field in ("mass", "inertia", "com"):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                body = changed["metadata"]["robustness"]["realized"]["first_actor"]["bodies"][0]
                if field == "mass":
                    body[field] = 4.0
                elif field == "inertia":
                    body[field][0][0] = 1.0
                else:
                    body[field][0] = 0.2
                write(path, changed)
                with self.assertRaisesRegex(ValueError, "payload"):
                    RUNNER.build_report(self.output)

    def test_undeclared_stress_config_change_is_rejected(self):
        self.change(
            self.result_path("low_friction"),
            lambda p: p["metadata"]["evaluation_config"].update(controller_gain=999),
        )
        with self.assertRaisesRegex(ValueError, "Undeclared condition difference"):
            RUNNER.build_report(self.output)

    def test_nominal_episode_parity_is_checked_after_aggregate_validation(self):
        def mutate(payload):
            episodes = payload["result"]["episodes"]
            episodes[0]["avg_lin_vel_error"] += 0.1
            episodes[1]["avg_lin_vel_error"] -= 0.1

        self.change(self.result_path(), mutate)
        with self.assertRaisesRegex(ValueError, "Nominal trajectory changed"):
            RUNNER.build_report(self.output)

    def test_nominal_config_must_match_original_capacity_evaluation(self):
        self.change(
            self.result_path(), lambda p: p["metadata"]["evaluation_config"].update(controller_gain=999)
        )
        with self.assertRaisesRegex(ValueError, "Nominal reference differs"):
            RUNNER.build_report(self.output)

    def test_missing_evaluation_remains_pending_and_final_assertion_fails(self):
        self.result_path("payload_mass", "mlp-deep-1500k", 103).unlink()
        report = RUNNER.build_report(self.output)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["verified_evaluations"], 35)
        self.assertIsNone(report["records"][-1]["metrics"])
        with self.assertRaisesRegex(ValueError, "pending"):
            RUNNER.write_report(self.output, require_complete=True)

    def test_successful_subprocess_without_result_cannot_complete_selected_variants(self):
        self.result_path("payload_mass", "prism-1321k", 103).unlink()
        write(self.output / "queue_state.json", {"schema_version": 1, "status": "prepared", "jobs": {}})
        with mock.patch.object(RUNNER.queue, "execute_command", return_value=None) as execute:
            with self.assertRaisesRegex(ValueError, "result"):
                RUNNER.execute(self.output, ["prism-1321k"])
            execute.assert_called_once()
        self.assertEqual(read(self.output / "queue_state.json")["status"], "failed")


class RobustnessPreparationTests(unittest.TestCase):
    def test_prepare_copies_source_without_subprocess_and_allows_training_progress(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            capacity, output = root / "capacity", root / "robustness"
            fixture(capacity, complete=False, capacity=True)
            cap_plan = read(capacity / "plan.json")
            cap_plan.update(python=str(Path(sys.executable).resolve()), device="cuda:0")
            for root_name, package in (("unitree_rl_gym", "legged_gym"), ("rsl_rl", "rsl_rl")):
                path = capacity / "source" / root_name / package / "__init__.py"
                path.parent.mkdir(parents=True)
                path.write_text("# Synthetic upstream source for CPU preparation test.\n")
                cap_plan["source_files"][str(path.relative_to(capacity / "source"))] = RUNNER.queue.sha256(
                    path
                )
            write(capacity / "plan.json", cap_plan)
            with (
                mock.patch("subprocess.Popen", side_effect=AssertionError("Preparation launched a process")),
                mock.patch("subprocess.run", side_effect=AssertionError("Preparation launched a process")),
            ):
                plan = RUNNER.prepare(output, capacity)
                self.assertEqual(plan["scope"]["evaluations"], 36)
                self.assertEqual(read(output / "queue_state.json")["status"], "prepared")
                self.assertIn("integration/robustness.py", plan["source_files"])
                suite = read(capacity / "suite.json")
                suite["runs"][0]["training_manifest"] = str(capacity / "runs/completed/manifest.json")
                write(capacity / "suite.json", suite)
                RUNNER.load_plan(output)
                suite["runs"][0]["evaluations"][0] = str(capacity / "unregistered.json")
                write(capacity / "suite.json", suite)
                with self.assertRaisesRegex(ValueError, "registration changed"):
                    RUNNER.load_plan(output)


if __name__ == "__main__":
    unittest.main()
