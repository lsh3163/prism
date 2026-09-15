"""Episode selection and actual evaluation-loop checks without Isaac Gym."""

import contextlib
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
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


PROTOCOL = load_module("humanoid_eval_protocol_test", INTEGRATION / "evaluation_protocol.py")
PROVENANCE = load_module("humanoid_eval_provenance_test", INTEGRATION / "provenance.py")
ROBUSTNESS = load_module("humanoid_eval_robustness_test", INTEGRATION / "robustness.py")
legged_gym = types.ModuleType("legged_gym")
legged_gym.LEGGED_GYM_ROOT_DIR = "unused"
isaacgym = types.ModuleType("isaacgym")
isaacgym.gymutil = types.SimpleNamespace()
with mock.patch.dict(
    sys.modules,
    {
        "isaacgym": isaacgym,
        "legged_gym": legged_gym,
        "legged_gym.utils": types.SimpleNamespace(class_to_dict=lambda value: value, get_load_path=None),
        "evaluation_protocol": PROTOCOL,
        "provenance": PROVENANCE,
        "robustness": ROBUSTNESS,
        "checkpoints": types.SimpleNamespace(actor_variant=lambda name: None, validate_checkpoint=None),
        "runtime": types.SimpleNamespace(VARIANTS={}, register_tasks=lambda: None),
    },
):
    EVALUATE = load_module("humanoid_evaluator_test", INTEGRATION / "evaluate.py")


def record(collector, env_id, step, length, timeout=False):
    return collector.add_episode(env_id, step, length, float(length), 0.2, 0.1, timeout)


class EpisodeQuotaTest(unittest.TestCase):
    def test_fast_repeated_falls_cannot_take_other_environments_quota(self):
        collector = PROTOCOL.EpisodeCollector(2, 4, 10)
        for step in range(1, 7):
            record(collector, 0, step, 1)
        self.assertFalse(collector.complete)
        self.assertEqual(collector.counted_counts, [2, 0])
        record(collector, 1, 10, 10, timeout=True)
        record(collector, 1, 20, 10, timeout=True)
        result = collector.summary()
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["episode_counts_by_environment"], [2, 2])
        self.assertEqual(len(result["episode_events"]), 8)
        self.assertEqual(result["protocol_id"], "g1_balanced_timeout_v2")

    def test_simultaneous_completions_keep_all_environment_quotas(self):
        collector = PROTOCOL.EpisodeCollector(3, 6, 10)
        record(collector, 0, 1, 1)
        record(collector, 0, 2, 1)
        record(collector, 0, 3, 1)
        for env_id in [1, 2]:
            record(collector, env_id, 10, 10, timeout=True)
        for env_id in [0, 1, 2]:
            record(collector, env_id, 20, 10, timeout=True)
        result = collector.summary()
        self.assertEqual(result["episode_counts_by_environment"], [2, 2, 2])
        self.assertEqual(result["success_rate"], 4 / 6)
        self.assertEqual(result["episode_events"][-1]["env_id"], 2)
        self.assertTrue(result["episode_events"][-1]["counted"])

    def test_near_horizon_fall_is_not_timeout_success(self):
        balanced = PROTOCOL.EpisodeCollector(2, 2, 10)
        legacy = PROTOCOL.EpisodeCollector(2, 2, 10, "legacy-pooled")
        for collector in [balanced, legacy]:
            record(collector, 0, 9, 9, timeout=False)
            record(collector, 1, 10, 10, timeout=True)
        self.assertEqual(balanced.summary()["success_rate"], 0.5)
        self.assertEqual(legacy.summary()["success_rate"], 1.0)
        self.assertFalse(balanced.episodes[0]["timeout"])
        self.assertNotEqual(balanced.summary()["protocol_id"], legacy.summary()["protocol_id"])

    def test_legacy_pool_only_counts_earliest_completions(self):
        collector = PROTOCOL.EpisodeCollector(2, 2, 10, "legacy-pooled")
        record(collector, 0, 1, 1)
        record(collector, 0, 2, 1)
        record(collector, 1, 2, 2, timeout=True)
        result = collector.summary()
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["episode_counts_by_environment"], [2, 0])
        self.assertFalse(result["episode_events"][-1]["counted"])

    def test_invalid_or_incomplete_quota_fails(self):
        for episodes, envs in [(0, 2), (2, 0), (1, 2), (3, 2), (True, 1)]:
            with self.subTest(episodes=episodes, envs=envs), self.assertRaises(ValueError):
                PROTOCOL.episode_quotas(episodes, envs)
        self.assertEqual(PROTOCOL.episode_quotas(200, 100), [2] * 100)
        with self.assertRaises(RuntimeError):
            PROTOCOL.EpisodeCollector(2, 4, 10).summary()
        with self.assertRaises(ValueError):
            PROTOCOL.EpisodeCollector(1, 1, 10).add_episode(0, 1, 1, float("nan"), 0.0, 0.0, False)


class TensorEnvironment:
    """Two independent timelines; resets intentionally corrupt exposed commands."""

    def __init__(self, fail=False, omit_callback=False):
        self.num_envs = 2
        self.max_episode_length = 3
        self.dt = 0.01
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.commands = torch.zeros(2, 4)
        self.base_lin_vel = torch.zeros(2, 3)
        self.base_ang_vel = torch.zeros(2, 3)
        self.time_out_buf = torch.zeros(2, dtype=torch.bool)
        self.fail = fail
        self.omit_callback = omit_callback

    def get_observations(self):
        return torch.zeros(2, 1)

    def reset_idx(self, ids):
        self.episode_length_buf[ids] = 0
        self.commands[ids] = 100.0
        self.time_out_buf[ids] = False

    def step(self, actions):
        if self.fail:
            raise RuntimeError("simulated step failure")
        self.episode_length_buf += 1
        self.commands[:] = 0
        self.commands[:, 0] = 2
        self.commands[:, 2] = 3
        self.base_lin_vel[:] = 0
        self.base_lin_vel[:, 0] = 1
        self.base_ang_vel[:] = 0
        self.base_ang_vel[:, 2] = 1
        self.time_out_buf = self.episode_length_buf >= 3
        # Env0 falls immediately every episode; env1 reaches a real timeout.
        dones = self.time_out_buf.clone()
        dones[0] = True
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if not self.omit_callback:
            self.reset_idx(ids)
        return self.get_observations(), None, torch.ones(2), dones, {}


def bundle(env):
    return {"env": env, "policy": lambda observations: observations, "task": "test", "checkpoint_path": "x"}


class EvaluationLoopTest(unittest.TestCase):
    def test_terminal_errors_and_timeouts_are_captured_before_reset_without_rng(self):
        env = TensorEnvironment()
        torch_state = torch.get_rng_state().clone()
        random_state = random.getstate()
        result = EVALUATE.evaluate_bundle(bundle(env), 4)
        self.assertEqual(result["evaluation_steps"], 6)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["avg_lin_vel_error"], 1.0)
        self.assertEqual(result["avg_yaw_vel_error"], 2.0)
        self.assertEqual(result["episode_counts_by_environment"], [2, 2])
        self.assertEqual(len(result["episode_events"]), 8)
        self.assertNotIn("reset_idx", vars(env))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(random_state, random.getstate())

    def test_legacy_errors_remain_explicitly_historical(self):
        result = EVALUATE.evaluate_bundle(bundle(TensorEnvironment()), 2, "legacy-pooled")
        self.assertGreater(result["avg_lin_vel_error"], 100)
        self.assertGreater(result["avg_yaw_vel_error"], 90)
        self.assertEqual(result["success_rate"], 0)
        self.assertEqual(result["protocol_id"], "g1_legacy_pooled_length_v1")

    def test_callback_is_restored_when_step_raises_or_contract_is_missing(self):
        for env in [TensorEnvironment(fail=True), TensorEnvironment(omit_callback=True)]:
            with self.subTest(env=env), self.assertRaises(RuntimeError):
                EVALUATE.evaluate_bundle(bundle(env), 4)
            self.assertNotIn("reset_idx", vars(env))

    def test_empty_reset_ids_still_capture_and_existing_override_is_restored(self):
        env = TensorEnvironment()
        override = mock.Mock()
        env.reset_idx = override
        with PROTOCOL.BeforeResetCapture(env, lambda: "before reset") as capture:
            capture.begin_step()
            env.reset_idx([])
            self.assertEqual(capture.snapshot(), "before reset")
            override.assert_called_once_with([])
        self.assertIs(env.reset_idx, override)

    def test_saved_schema_retains_raw_episodes_and_distinct_training_seed(self):
        result = EVALUATE.evaluate_bundle(bundle(TensorEnvironment()), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            args = types.SimpleNamespace(
                save_path=str(path),
                variant_name="prism",
                variant="prism",
                condition_name="match_nopush",
                terrain_mode="match",
                push_mode="off",
                push_interval_s=None,
                max_push_vel_xy=None,
                seed=101,
                episodes=4,
                num_envs=2,
            )
            provenance = {"training_seed": 3, "checkpoint_iteration": 3001}
            EVALUATE.maybe_save_results(args, result, {"provenance": provenance})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["metadata"]["training_seed"], 3)
            self.assertEqual(saved["metadata"]["evaluation_seed"], 101)
            self.assertEqual(saved["metadata"]["protocol_id"], "g1_balanced_timeout_v2")
            self.assertEqual(saved["result"]["episodes"], result["episodes"])
            with self.assertRaises(FileExistsError):
                EVALUATE.maybe_save_results(args, result)


def as_dict(value):
    if isinstance(value, types.SimpleNamespace):
        return {name: as_dict(item) for name, item in vars(value).items()}
    return value


class EvaluationProvenanceTest(unittest.TestCase):
    def setup_bundle(self, directory, manifest=True):
        path = Path(directory) / "model_999.pt"
        path.write_bytes(b"test checkpoint identity; model loading is mocked")
        payload = {
            "schema_version": 1,
            "record_kind": "training_start",
            "seed": 3,
            "task": "g1",
            "actor": {"class": "ActorCritic", "actor_variant": None},
        }
        if manifest:
            (path.parent / "prism_run_manifest.json").write_text(json.dumps(payload))
        args = types.SimpleNamespace(
            save_path=None,
            task="g1",
            num_envs=2,
            terrain_mode="plane",
            push_mode="off",
            push_interval_s=None,
            max_push_vel_xy=None,
            seed=101,
            model_path=str(path),
            experiment_name=None,
            load_run=None,
            checkpoint=None,
        )
        cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(num_envs=2),
            terrain=types.SimpleNamespace(mesh_type="plane"),
            noise=types.SimpleNamespace(),
            domain_rand=types.SimpleNamespace(),
        )
        train_cfg = types.SimpleNamespace(runner=types.SimpleNamespace(policy_class_name="ActorCritic"))
        env = TensorEnvironment()

        class ActorCritic(torch.nn.Module):
            pass

        model = ActorCritic()
        model.actor = torch.nn.Linear(1, 1)
        runner = types.SimpleNamespace(
            alg=types.SimpleNamespace(actor_critic=model),
            current_learning_iteration=3001,
            load=mock.Mock(),
            get_inference_policy=lambda device: lambda observations: observations,
        )

        def make_env(**kwargs):
            env.cfg = kwargs["env_cfg"]
            return env, env.cfg

        registry = types.SimpleNamespace(
            get_cfgs=lambda name: (cfg, train_cfg),
            make_env=make_env,
            make_alg_runner=lambda **kwargs: (runner, train_cfg),
        )
        return path, payload, args, registry

    def load(self, args, registry):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(EVALUATE, "task_registry", registry))
            stack.enter_context(mock.patch.object(EVALUATE, "validate_checkpoint"))
            stack.enter_context(mock.patch.object(EVALUATE, "class_to_dict", side_effect=as_dict))
            stack.enter_context(mock.patch.object(EVALUATE, "source_snapshot", return_value={"files": []}))
            return EVALUATE.load_eval_bundle(args)

    def test_manifest_payload_hash_checkpoint_and_iteration_are_actual(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, args, registry = self.setup_bundle(directory)
            provenance = self.load(args, registry)["provenance"]
            self.assertEqual(provenance["training_seed"], 3)
            self.assertEqual(provenance["evaluation_seed"], 101)
            self.assertEqual(provenance["checkpoint_iteration"], 3001)
            self.assertEqual(
                provenance["checkpoint"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            manifest = provenance["training_manifest"]
            self.assertEqual(manifest["payload"], payload)
            self.assertEqual(
                manifest["sha256"], hashlib.sha256(Path(manifest["path"]).read_bytes()).hexdigest()
            )
            self.assertEqual(provenance["evaluation_config"]["seed"], 101)

    def test_missing_training_manifest_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, args, registry = self.setup_bundle(directory, manifest=False)
            provenance = self.load(args, registry)["provenance"]
            self.assertIsNone(provenance["training_seed"])
            self.assertIsNone(provenance["training_manifest"])

    def test_checkpoint_symlink_uses_manifest_beside_actual_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, args, registry = self.setup_bundle(directory)
            alias_dir = path.parent / "alias"
            alias_dir.mkdir()
            alias = alias_dir / "model_3001.pt"
            alias.symlink_to(path)
            payload["seed"] = 999
            (alias_dir / "prism_run_manifest.json").write_text(json.dumps(payload))
            args.model_path = str(alias)
            provenance = self.load(args, registry)["provenance"]
            self.assertEqual(provenance["training_seed"], 3)
            self.assertEqual(provenance["checkpoint"]["path"], str(path.resolve()))

    def test_mismatched_manifest_rejected_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, args, registry = self.setup_bundle(directory)
            payload["actor"]["actor_variant"] = "g1_gated_poly_v2"
            (path.parent / "prism_run_manifest.json").write_text(json.dumps(payload))
            registry.make_env = mock.Mock()
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.load(args, registry)
            registry.make_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
