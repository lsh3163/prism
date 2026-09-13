"""CPU contracts for the historical Diffusion actor and simulation recipes."""

import argparse
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/lerobot"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Conditioner = load_module(
    "prism_diffusion_release", INTEGRATION / "diffusion_conditioner.py"
).PolynomialKernelConditioner


class DiffusionActorTest(unittest.TestCase):
    def test_checkpoint_layout_and_roundtrip(self):
        model = Conditioner(16, 16, "latent_quadratic", 256, 256)
        expected = {
            "input_norm.weight",
            "input_norm.bias",
            "left_proj.weight",
            "left_proj.bias",
            "right_proj.weight",
            "right_proj.bias",
            "net.0.weight",
            "net.0.bias",
            "net.1.weight",
            "net.1.bias",
            "net.3.weight",
            "net.3.bias",
        }
        self.assertEqual(set(model.state_dict()), expected)
        self.assertEqual(model.left_proj.weight.shape, (256, 16))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = Conditioner(16, 16, "latent_quadratic", 256, 256)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        inputs = torch.randn(3, 16)
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)

    def test_factorized_polynomial_contains_bias_linear_and_cross_terms(self):
        model = Conditioner(2, 2, "latent_quadratic", 2, 4)
        # Isolate the polynomial stage from the two historical normalizations/MLP.
        model.input_norm = nn.Identity()
        model.net = nn.Identity()
        with torch.no_grad():
            model.left_proj.weight.copy_(torch.eye(2))
            model.left_proj.bias.copy_(torch.tensor([1.0, -1.0]))
            model.right_proj.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            model.right_proj.bias.copy_(torch.tensor([2.0, 3.0]))
        x = torch.tensor([[2.0, 4.0], [-3.0, 1.0]], requires_grad=True)
        expected = torch.stack([(x[:, 0] + 1) * (x[:, 1] + 2), (x[:, 1] - 1) * (x[:, 0] + 3)], dim=-1)
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.left_proj.weight.grad)
        self.assertIsNotNone(model.right_proj.weight.grad)

    def test_history_shape_and_gradients(self):
        for mode in ("raw", "latent_quadratic"):
            model = Conditioner(16, 16, mode, 32, 64)
            inputs = torch.randn(4, 16, requires_grad=True)
            outputs = model(inputs)
            self.assertEqual(outputs.shape, inputs.shape)
            outputs.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
            self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_rejects_different_actor_variants(self):
        for mode in ("prism_gated", "prism_factorized", "explicit_monomials"):
            with self.assertRaises(ValueError):
                Conditioner(16, 16, mode, 32, 64)


class LiberoEpisodeTerminationTest(unittest.TestCase):
    def test_success_valued_backend_done_does_not_end_continuation_episode(self):
        helper = load_module("prism_libero_episode", INTEGRATION / "libero_episode.py")
        self.assertFalse(helper.should_terminate_episode(True, True, False))
        self.assertTrue(helper.should_terminate_episode(True, True, True))

    def test_non_success_termination_and_normal_steps_are_preserved(self):
        helper = load_module("prism_libero_episode", INTEGRATION / "libero_episode.py")
        for stop_on_success in (False, True):
            self.assertTrue(helper.should_terminate_episode(True, False, stop_on_success))
            self.assertFalse(helper.should_terminate_episode(False, False, stop_on_success))


class SimulationRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The scripts are also directly executable; resolve their local common module.
        cls.previous_common = sys.modules.get("common")
        sys.modules["common"] = load_module("common", INTEGRATION / "scripts/common.py")
        cls.train = load_module("prism_train_diffusion_recipe", INTEGRATION / "scripts/train_diffusion.py")
        cls.evaluate = load_module("prism_eval_diffusion_recipe", INTEGRATION / "scripts/eval_diffusion.py")
        cls.common = sys.modules["common"]

    @classmethod
    def tearDownClass(cls):
        if cls.previous_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = cls.previous_common

    def test_complete_recorded_task_splits(self):
        data = json.loads((INTEGRATION / "recipes/diffusion_tasks.json").read_text())
        self.assertEqual(len(data["tasks"]), 40)
        self.assertEqual(len(data["source_configs"]), 80)
        for task in data["tasks"]:
            self.assertGreater(len(task["episodes"]), 0)
            self.assertEqual(len(task["episodes"]), len(set(task["episodes"])))
            self.assertEqual(task["profiles"]["historical-baseline"]["batch_size"], 8)
            self.assertEqual(task["profiles"]["prism"]["batch_size"], 64)

    def test_matched_control_changes_only_actor_and_output(self):
        task = json.loads((INTEGRATION / "recipes/diffusion_tasks.json").read_text())["tasks"][0]
        args = argparse.Namespace(
            profile="prism", output_root=Path("runs"), steps=20000, seed=0, dataset_revision=None
        )
        prism, _ = self.train.build_command(args, task)
        args.profile = "matched-baseline"
        baseline, _ = self.train.build_command(args, task)
        differences = [(left, right) for left, right in zip(prism, baseline, strict=False) if left != right]
        self.assertEqual(len(differences), 2)
        self.assertTrue(differences[0][0].startswith("--policy.use_poly_kernel_conditioning="))
        self.assertTrue(differences[1][0].startswith("--output_dir="))

    def test_oracle_uses_contact_without_sensorless_noise_or_delay(self):
        args = self.evaluate.make_parser().parse_args(
            [
                "--lerobot-root",
                "/tmp/lerobot",
                "--baseline",
                "/tmp/baseline",
                "--prism",
                "/tmp/prism",
                "--suite",
                "libero_spatial",
                "--task-id",
                "1",
                "--output-root",
                "/tmp/eval",
            ]
        )
        sensorless, _, _ = self.evaluate.build_command(args, "mcc")
        oracle, _, _ = self.evaluate.build_command(args, "oracle")
        prism, _, _ = self.evaluate.build_command(args, "prism")
        self.assertIn("--env.mcc_force_source=sensorless", sensorless)
        self.assertIn("--env.mcc_delay=2", sensorless)
        self.assertIn("--env.mcc_force_source=contact", oracle)
        self.assertIn("--env.mcc_noise_std=0.0", oracle)
        self.assertIn("--env.mcc_delay=0", oracle)
        self.assertIn("--env.mcc_noisy_enabled=false", prism)
        for command in (sensorless, oracle, prism):
            self.assertIn("--env.controller_kp=50", command)
            self.assertIn("--seed=1000", command)

    def test_every_cli_help_works_without_optional_dependencies(self):
        scripts = [INTEGRATION / "install.py", *sorted((INTEGRATION / "scripts").glob("*.py"))]
        for script in scripts:
            if script.name == "common.py":
                continue
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_cli_rejects_invalid_numbers_and_duplicate_rows(self):
        prefix = [
            "--lerobot-root",
            "/tmp/lerobot",
            "--suite",
            "libero_spatial",
            "--task-id",
            "0",
            "--output-root",
            "/tmp/eval",
        ]
        for option in (
            ["--rows", "prism,prism"],
            ["--mcc-ema", "1.1"],
            ["--image-noise", "nan"],
            ["--episodes", "0"],
            ["--render-episodes", "-1"],
        ):
            with (
                self.subTest(option=option),
                patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.evaluate.make_parser().parse_args([*prefix, *option])

    def test_missing_later_checkpoint_prevents_every_run_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(lerobot_root=root, execute=True, python=None, egl_vendor_file=None)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"type":"diffusion"}')
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
            metadata = {"policy": "diffusion", "profile": "baseline", "actor_variant": None}
            runs = [
                self.common.Run(["unused"], Path("first"), metadata, checkpoint=checkpoint),
                self.common.Run(["unused"], Path("second"), metadata, checkpoint=root / "missing"),
            ]
            with (
                patch.object(self.common, "verify_installed", return_value={}),
                patch.object(self.common.subprocess, "run") as execute,
                self.assertRaises(FileNotFoundError),
            ):
                self.common.run_commands(runs, args, extra_files={Path("configs.txt"): "fixture"})
            execute.assert_not_called()
            self.assertFalse((root / "first.command.json").exists())
            self.assertFalse((root / "configs.txt").exists())

    def test_existing_eval_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "eval").mkdir()
            args = argparse.Namespace(lerobot_root=root, execute=True, python=None, egl_vendor_file=None)
            run = self.common.Run(["unused"], Path("eval"), {"policy": "diffusion"})
            with patch.object(self.common.subprocess, "run") as execute, self.assertRaises(FileExistsError):
                self.common.run_commands([run], args)
            execute.assert_not_called()

    def test_wrong_git_revision_is_rejected_before_file_identity(self):
        with (
            patch.object(self.common.subprocess, "check_output", return_value="0" * 40 + "\n"),
            patch.object(self.common, "sha256") as hash_file,
            self.assertRaisesRegex(ValueError, "Expected LeRobot"),
        ):
            self.common.verify_installed(Path("/tmp/lerobot"), with_smolvla=False)
        hash_file.assert_not_called()

    def test_execution_selects_source_and_records_status_hashes_and_control_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/prism_recipe_probe.py").write_text('MARKER = "selected checkout"\n')
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"type":"diffusion"}')
            (checkpoint / "model.safetensors").write_bytes(b"weights fixture")
            (checkpoint / "policy_preprocessor.json").write_text("{}")
            args = argparse.Namespace(
                lerobot_root=root, execute=True, python=Path(sys.executable), egl_vendor_file=None
            )
            code = "from pathlib import Path; import prism_recipe_probe; Path('selected.txt').write_text(prism_recipe_probe.MARKER)"
            run = self.common.Run(
                [sys.executable, "-c", code],
                Path("eval"),
                {"policy": "diffusion", "profile": "baseline", "actor_variant": None},
                checkpoint=checkpoint,
            )
            with (
                patch.object(self.common, "verify_installed", return_value={"fixture": True}),
                redirect_stdout(io.StringIO()),
            ):
                self.common.run_commands([run], args)
            record = json.loads((root / "eval.command.json").read_text())
            self.assertEqual((root / "selected.txt").read_text(), "selected checkout")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["exit_code"], 0)
            self.assertIsNone(record["run"]["actor_variant"])
            self.assertEqual(
                record["checkpoint"]["weights_sha256"], self.common.sha256(checkpoint / "model.safetensors")
            )
            self.assertIn("policy_preprocessor.json", record["checkpoint"]["auxiliary_files"])

    def test_failed_child_records_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(lerobot_root=root, execute=True, python=None, egl_vendor_file=None)
            run = self.common.Run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                Path("eval"),
                {"policy": "diffusion", "actor_variant": None},
            )
            with (
                patch.object(self.common, "verify_installed", return_value={}),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.common.run_commands([run], args)
            self.assertEqual(error.exception.code, 7)
            record = json.loads((root / "eval.command.json").read_text())
            self.assertEqual((record["status"], record["exit_code"]), ("failed", 7))

    def test_smolvla_checkpoint_identity_requires_complete_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "type": "smolvla",
                "state_conditioner_type": "prism",
                "state_conditioner_product_mode": "gated_quadratic",
                "state_conditioner_num_layers": 2,
                "state_conditioner_use_rmsnorm": True,
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            (root / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(
                self.common.checkpoint_metadata(root, "smolvla", None)["actor_variant"],
                "smolvla_gated_quadratic_v1",
            )
            for change in (
                {"state_conditioner_num_layers": 3},
                {"state_conditioner_use_rmsnorm": False},
                {"state_conditioner_product_mode": "vanilla"},
            ):
                path.write_text(json.dumps({**config, **change}))
                with self.assertRaises(ValueError):
                    self.common.checkpoint_metadata(root, "smolvla", None)
            path.write_text('{"type":"smolvla","use_poly_interaction":true}')
            with self.assertRaises(ValueError):
                self.common.checkpoint_metadata(root, "smolvla", None)


class InstallerTest(unittest.TestCase):
    def test_verified_install_is_idempotent_and_rejects_modified_target(self):
        installer = load_module("prism_release_installer", INTEGRATION / "install.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, target = root / "bundle", root / "target"
            bundle.mkdir()
            target.mkdir()
            (bundle / "change.patch").write_text("fixture patch")
            (bundle / "adapter.py").write_text("adapter")
            (target / "source.py").write_text("upstream")
            release = root / "release.py"
            release.write_text("release")
            manifest = {
                "upstream_commit": "fixture_revision",
                "patch": "change.patch",
                "patch_sha256": installer.file_hash(bundle / "change.patch"),
                "files": [
                    {
                        "upstream_path": "source.py",
                        "upstream_sha256": installer.file_hash(target / "source.py"),
                        "release_sha256": installer.file_hash(release),
                    }
                ],
                "copied_modules": [
                    {
                        "source": "adapter.py",
                        "target": "adapter.py",
                        "sha256": installer.file_hash(bundle / "adapter.py"),
                    }
                ],
            }
            (bundle / "source_manifest.json").write_text(json.dumps(manifest))

            def apply_patch(command, **kwargs):
                if "--check" not in command:
                    (target / "source.py").write_text("release")
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(installer, "ROOT", bundle),
                patch.object(installer.subprocess, "check_output", return_value="fixture_revision\n"),
                patch.object(installer.subprocess, "run", side_effect=apply_patch) as apply,
                redirect_stdout(io.StringIO()),
            ):
                installer.install(target, apply=True)
                self.assertEqual((target / "adapter.py").read_text(), "adapter")
                calls = apply.call_count
                installer.install(target, apply=True)
                self.assertEqual(apply.call_count, calls)
                (target / "adapter.py").write_text("local edit")
                with self.assertRaisesRegex(ValueError, "overwrite modified adapter"):
                    installer.install(target, apply=True)
                (bundle / "change.patch").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "Bundle patch hash mismatch"):
                    installer.install(target, apply=True)


if __name__ == "__main__":
    unittest.main()
