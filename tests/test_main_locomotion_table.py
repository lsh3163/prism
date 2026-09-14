"""Five independent training runs, evidence binding, and nested aggregation."""

import contextlib
import copy
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/humanoid-gym/build_main_table.py"
SPEC = importlib.util.spec_from_file_location("main_locomotion_table", SCRIPT)
TABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLE)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False))


def source():
    return {
        "files": [
            {"path": "/fixture/integration/" + name, "sha256": "1" * 64}
            for name in sorted(TABLE.TRAIN_CODE | {"evaluate.py", "evaluation_protocol.py"})
        ]
    }


def runtime():
    return {
        "python": "3.8.20",
        "torch": "2.3.1",
        "numpy": "1.20.0",
        "isaacgym": "preview4",
        "torch_cuda": "12.1",
    }


def make_manifest(variant, seed):
    gated = variant == "prism"
    actor = {
        "class": "GatedPolyActorCritic" if gated else "ActorCritic",
        "actor_variant": "g1_gated_poly_v2" if gated else None,
        "recipe_variant": variant,
        "actor_hidden_dims": TABLE.HIDDEN_DIMS[variant],
        "polynomial_degree": 2 if gated else None,
        "gate_semantics": "learned_per_feature" if gated else None,
        "gate_init": 0.01 if gated else None,
        "warmup_ppo_updates": 0 if gated else None,
        **TABLE.expected_counts()[variant],
    }
    policy = {
        "actor_hidden_dims": TABLE.HIDDEN_DIMS[variant],
        "critic_hidden_dims": [768, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    if gated:
        policy.update(actor_variant="g1_gated_poly_v2", poly_degree=2, poly_hidden_dim=256, gate_init=0.01)
    return {
        "schema_version": 1,
        "record_kind": "training_start",
        "seed": seed,
        "task": TABLE.TASKS[variant],
        "actor": actor,
        "initial_learning_iteration": 0,
        "resume_checkpoint": None,
        "training_config": {
            "seed": seed,
            "policy": policy,
            "algorithm": {"learning_rate": 1e-5, "gamma": 0.994},
            "runner": {
                "policy_class_name": actor["class"],
                "max_iterations": 3001,
                "num_steps_per_env": 60,
                "resume": False,
                "experiment_name": variant,
            },
        },
        "environment_config": {
            "seed": seed,
            "env": {"num_envs": 4096, "episode_length_s": 24},
            "commands": {"resampling_time": 8},
        },
        "arguments": {"seed": seed, "task": TABLE.TASKS[variant], "physics_engine": "physx"},
        "source": source(),
        "runtime": runtime(),
    }


def make_eval(manifest, manifest_path, checkpoint, eval_seed):
    variant, seed = manifest["actor"]["recipe_variant"], manifest["seed"]
    identity = {"path": str(checkpoint), "sha256": TABLE.sha256(checkpoint)}
    episodes = []
    for env_id in (0, 1):
        for ordinal in (1, 2):
            episodes.append(
                {
                    "env_id": env_id,
                    "episode_ordinal": ordinal,
                    "completion_step": ordinal * 100,
                    "length": 100 + seed,
                    "return": seed * 10 + env_id,
                    "avg_lin_vel_error": seed + eval_seed / 1000 + env_id / 100,
                    "avg_yaw_vel_error": seed / 10 + ordinal / 100,
                    "timeout": env_id == 0,
                    "success": env_id == 0,
                }
            )
    result = {
        "task": TABLE.TASKS[variant],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": identity["sha256"],
        "checkpoint_iteration": 3001,
        "actor_variant": manifest["actor"]["actor_variant"],
        "protocol_id": TABLE.PROTOCOL_ID,
        "protocol": {
            "id": TABLE.PROTOCOL_ID,
            "version": 2,
            "episode_allocation": "equal_per_environment_quota",
            "episode_quotas": [2, 2],
            "success_definition": "environment_timeout",
            "tracking_error_timing": "before_terminal_reset",
        },
        "episodes": episodes,
        "num_episodes": 4,
    }
    for metric, key in (
        ("avg_episode_length", "length"),
        ("avg_return", "return"),
        ("avg_lin_vel_error", "avg_lin_vel_error"),
        ("avg_yaw_vel_error", "avg_yaw_vel_error"),
        ("success_rate", "success"),
    ):
        result[metric] = statistics.mean(row[key] for row in episodes)
    return {
        "schema_version": 2,
        "metadata": {
            "protocol_id": result["protocol_id"],
            "evaluation_protocol": copy.deepcopy(result["protocol"]),
            "recipe_variant": variant,
            "actor_variant": manifest["actor"]["actor_variant"],
            "training_seed": seed,
            "evaluation_seed": eval_seed,
            "seed": eval_seed,
            "training_manifest": {
                "path": str(manifest_path),
                "sha256": TABLE.sha256(manifest_path),
                "payload": copy.deepcopy(manifest),
            },
            "checkpoint": identity,
            "checkpoint_iteration": 3001,
            "actor": copy.deepcopy(manifest["actor"]),
            "episodes": 4,
            "num_envs": 2,
            "condition_name": "match_nopush",
            "terrain_mode": "match",
            "push_mode": "off",
            "evaluation_config": {
                "seed": eval_seed,
                "env": {"num_envs": 2, "episode_length_s": 24},
                "commands": {"resampling_time": 8},
            },
            "evaluation_arguments": {
                "seed": eval_seed,
                "task": TABLE.TASKS[variant],
                "model_path": str(checkpoint),
                "physics_engine": "physx",
            },
            "source": source(),
            "runtime": runtime(),
        },
        "result": result,
    }


def fixture(directory):
    suite = {
        "schema_version": 1,
        "training_seeds": [1, 2, 3, 4, 5],
        "evaluation_seeds": [101, 102, 103],
        "max_iterations": 3001,
        "checkpoint_iteration": 3001,
        "num_envs": 4096,
        "evaluation": {
            "protocol_id": TABLE.PROTOCOL_ID,
            "episodes": 4,
            "num_envs": 2,
            "condition_name": "match_nopush",
            "terrain_mode": "match",
            "push_mode": "off",
        },
        "runs": [],
    }
    for variant in TABLE.VARIANTS:
        for seed in TABLE.TRAINING_SEEDS:
            run = directory / variant / str(seed)
            manifest_path = run / "prism_run_manifest.json"
            manifest = make_manifest(variant, seed)
            write_json(manifest_path, manifest)
            checkpoint = run / "model_3001.pt"
            checkpoint.write_bytes(f"synthetic evidence fixture only: {variant}/{seed}".encode())
            evaluations = []
            for eval_seed in suite["evaluation_seeds"]:
                path = run / f"eval_{eval_seed}.json"
                write_json(path, make_eval(manifest, manifest_path, checkpoint, eval_seed))
                evaluations.append(str(path.relative_to(directory)))
            suite["runs"].append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "training_manifest": str(manifest_path.relative_to(directory)),
                    "evaluations": evaluations,
                }
            )
    path = directory / "suite.json"
    write_json(path, suite)
    return path, suite


class MainLocomotionTableTest(unittest.TestCase):
    def test_parameter_counts_match_instantiated_released_actors(self):
        integration = SCRIPT.parent
        sys.path.insert(0, str(integration))
        try:
            from actor import ActorCritic
            from gated_actor import GatedPolyActorCritic
        finally:
            sys.path.pop(0)
        for variant in TABLE.VARIANTS:
            constructor = GatedPolyActorCritic if variant == "prism" else ActorCritic
            kwargs = (
                {"poly_hidden_dim": 256, "poly_degree": 2, "gate_init": 0.01} if variant == "prism" else {}
            )
            with contextlib.redirect_stdout(io.StringIO()):
                model = constructor(
                    705,
                    219,
                    12,
                    actor_hidden_dims=TABLE.HIDDEN_DIMS[variant],
                    critic_hidden_dims=[768, 256, 128],
                    activation="elu",
                    **kwargs,
                )
            expected = TABLE.expected_counts()[variant]
            with self.subTest(variant=variant):
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()), expected["total_actor_critic_parameters"]
                )
                self.assertEqual(
                    sum(
                        p.numel()
                        for name, p in model.named_parameters()
                        if name.startswith(("actor.", "actor_encoder."))
                    ),
                    expected["actor_mean_parameters"],
                )

    def test_means_then_sample_sd_across_five_training_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(Path(folder))
            report = TABLE.build_table(path)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["verified_training_runs"], 15)
            self.assertEqual(report["verified_evaluations"], 45)
            row = report["methods"]["prism"]["metrics"]["avg_lin_vel_error"]
            self.assertAlmostEqual(row["mean"], 3.107)
            self.assertAlmostEqual(row["sample_sd"], math.sqrt(2.5))
            self.assertEqual(report["methods"]["larger"]["total_actor_critic_parameters"], 1123793)
            self.assertEqual(report["methods"]["prism"]["total_actor_critic_parameters"], 1123737)
            latex = TABLE.render_latex(report)
            self.assertIn(r"\label{tab:locomotion_main}", latex)
            self.assertIn(r"\sname{} & \cmark", latex)
            self.assertIn("Bold indicates the best mean.", latex)
            self.assertNotIn(r"\textbf", latex)
            self.assertIn(r"\resizebox{\columnwidth}{!}{%", latex)
            self.assertEqual(latex.count(r"\midrule"), 2)
            self.assertIn(r"\vspace{-1em}", latex)
            TABLE.write_report(report, Path(folder) / "table")
            self.assertEqual(TABLE.load_json(Path(folder) / "table/main_table.json")["status"], "complete")

    def test_complete_table_bolds_actual_best_method_and_pending_bolds_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(Path(folder))
            report = TABLE.build_table(path)
            report["methods"]["baseline"]["metrics"]["success_rate"]["mean"] = 0.9
            report["methods"]["larger"]["metrics"]["avg_lin_vel_error"]["mean"] = 0.1
            latex = TABLE.render_latex(report)
            baseline = next(line for line in latex.splitlines() if line.startswith("MLP Baseline &"))
            larger = next(line for line in latex.splitlines() if line.startswith("Larger MLP &"))
            prism = next(line for line in latex.splitlines() if line.startswith(r"\sname{} &"))
            self.assertIn(r"\mathbf{90.00", baseline)
            self.assertIn(r"\mathbf{0.1000", larger)
            self.assertNotIn(r"\mathbf{50.00", prism)
            report["status"] = "pending"
            pending = TABLE.render_latex(report)
            self.assertNotIn(r"\mathbf", pending)
            self.assertNotIn("best mean", pending)

    def test_missing_training_run_or_evaluation_never_prints_partial_statistics(self):
        for missing in ("training_run", "evaluation"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                path, suite = fixture(Path(folder))
                if missing == "training_run":
                    suite["runs"].pop()
                    write_json(path, suite)
                else:
                    (Path(folder) / suite["runs"][-1]["evaluations"][-1]).unlink()
                report = TABLE.build_table(path)
                self.assertEqual(report["status"], "pending")
                self.assertTrue(all(row["metrics"] is None for row in report["methods"].values()))
                latex = TABLE.render_latex(report)
                self.assertIn("-- & -- & -- & --", latex)
                self.assertNotIn(r"\pm", latex)
                self.assertTrue(report["missing"])

    def test_wrong_seed_set_duplicate_eval_legacy_actor_and_budget_rejected(self):
        mutations = (
            ("seed_set", "exact training seeds"),
            ("duplicate_eval", "Duplicate evaluation seed"),
            ("legacy_actor", "Wrong actor variant"),
            ("checkpoint_budget", "Wrong checkpoint budget"),
            ("checkpoint_hash", "Checkpoint hash mismatch"),
            ("protocol", "Protocol metadata disagrees"),
            ("episode_mean", "Aggregate disagrees"),
            ("unbalanced", "not balanced"),
            ("evaluation_shift", "Evaluation protocol/environment/source/runtime differ"),
        )
        for mutation, message in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path, suite = fixture(Path(folder))
                target = Path(folder) / suite["runs"][-1]["evaluations"][-1]
                payload = TABLE.load_json(target)
                if mutation == "seed_set":
                    suite["training_seeds"] = [1, 2]
                elif mutation == "duplicate_eval":
                    suite["runs"][-1]["evaluations"].append(suite["runs"][-1]["evaluations"][-1])
                elif mutation == "legacy_actor":
                    payload["metadata"]["actor"]["actor_variant"] = "g1_residual_poly_v1"
                elif mutation == "checkpoint_budget":
                    payload["metadata"]["checkpoint_iteration"] = 100
                elif mutation == "checkpoint_hash":
                    payload["result"]["checkpoint_sha256"] = "0" * 64
                elif mutation == "protocol":
                    payload["result"]["protocol_id"] = "legacy-pooled"
                elif mutation == "episode_mean":
                    payload["result"]["success_rate"] = 0.95
                elif mutation == "unbalanced":
                    payload["result"]["episodes"][0]["env_id"] = 1
                elif mutation == "evaluation_shift":
                    payload["metadata"]["evaluation_config"]["commands"]["resampling_time"] = 1
                write_json(target, payload)
                write_json(path, suite)
                with self.assertRaisesRegex(ValueError, message):
                    TABLE.build_table(path)

    def test_training_hyperparameter_and_manifest_binding_changes_rejected(self):
        for mutation, message in (
            ("hyperparameter", "Training hyperparameters"),
            ("binding", "not bound"),
            ("resume", "independent fresh training"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path, suite = fixture(Path(folder))
                manifest_path = Path(folder) / suite["runs"][-1]["training_manifest"]
                manifest = TABLE.load_json(manifest_path)
                if mutation == "hyperparameter":
                    manifest["training_config"]["algorithm"]["learning_rate"] = 1e-3
                elif mutation == "binding":
                    manifest["new_audit_note"] = "changed after evaluation"
                else:
                    manifest["initial_learning_iteration"] = 100
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, message):
                    TABLE.build_table(path)

    def test_help_uses_standard_library_only_and_empty_suite_is_pending(self):
        help_result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with tempfile.TemporaryDirectory() as folder:
            path, suite = fixture(Path(folder))
            suite["runs"] = []
            write_json(path, suite)
            report = TABLE.build_table(path)
            self.assertEqual(report["status"], "pending")
            self.assertEqual(len(report["missing"]), 15)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    "--suite",
                    str(path),
                    "--output-dir",
                    str(Path(folder) / "table"),
                    "--require-complete",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(TABLE.load_json(Path(folder) / "table/main_table.json")["status"], "pending")


if __name__ == "__main__":
    unittest.main()
