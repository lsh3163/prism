"""Golden contracts for the declarative LeRobot recipe and evaluation specs."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "lerobot"
SCRIPTS = INTEGRATION / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluation_specs  # noqa: E402
import profile_catalog  # noqa: E402
import smolvla  # noqa: E402
import sweep_diffusion  # noqa: E402
import train_diffusion  # noqa: E402


def argument_with_prefix(command, prefix):
    return next(argument for argument in command if argument.startswith(prefix))


class LeRobotProfileCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = json.loads((INTEGRATION / "recipes/diffusion_tasks.json").read_text())["tasks"][0]

    def test_diffusion_public_profiles_preserve_routes_and_effective_settings(self):
        self.assertEqual(
            profile_catalog.DIFFUSION_PROFILE_NAMES,
            (
                "prism",
                "baseline",
                "legacy-prism",
                "legacy-baseline",
                "historical-baseline",
                "matched-baseline",
            ),
        )
        expected = {
            "prism": ("prism", "legacy-prism", 64, 8, True, "gated_quadratic", "gated_state"),
            "baseline": ("baseline", "historical-baseline", 64, 8, False, "gated_quadratic", "nominal"),
            "legacy-prism": (
                "legacy-prism",
                "legacy-prism",
                64,
                8,
                True,
                "latent_quadratic",
                "factorized_state",
            ),
            "legacy-baseline": (
                "legacy-baseline",
                "historical-baseline",
                8,
                4,
                False,
                "latent_quadratic",
                "nominal",
            ),
            "historical-baseline": (
                "legacy-baseline",
                "historical-baseline",
                8,
                4,
                False,
                "latent_quadratic",
                "nominal",
            ),
            "matched-baseline": (
                "baseline",
                "historical-baseline",
                64,
                8,
                False,
                "gated_quadratic",
                "nominal",
            ),
        }
        for name, values in expected.items():
            with self.subTest(profile=name):
                resolved = profile_catalog.resolve_diffusion_profile(name, self.task["profiles"])
                canonical, source, batch, workers, enabled, lift_mode, architecture = values
                self.assertEqual(resolved.specification.canonical_name, canonical)
                self.assertEqual(resolved.specification.task_settings_profile, source)
                self.assertEqual((resolved.batch_size, resolved.num_workers), (batch, workers))
                self.assertEqual(resolved.specification.conditioning_enabled, enabled)
                self.assertEqual(resolved.specification.lift_mode, lift_mode)
                self.assertEqual(resolved.specification.architecture, architecture)
                arguments = argparse.Namespace(
                    profile=name,
                    output_root=Path("runs"),
                    steps=20000,
                    seed=0,
                    dataset_revision=None,
                )
                command, output = train_diffusion.build_command(arguments, self.task)
                self.assertEqual(output, Path("runs") / name / self.task["suite"] / "task_0")
                self.assertEqual(
                    argument_with_prefix(command, "--policy.use_poly_kernel_conditioning="),
                    "--policy.use_poly_kernel_conditioning=" + str(enabled).lower(),
                )
                self.assertEqual(
                    argument_with_prefix(command, "--policy.poly_kernel_lift_mode="),
                    "--policy.poly_kernel_lift_mode=" + lift_mode,
                )
                self.assertIn(f"--batch_size={batch}", command)
                self.assertIn(f"--num_workers={workers}", command)
                json.dumps(resolved.as_dict(), allow_nan=False)

    def test_smolvla_profiles_record_active_and_inactive_conditioner_settings(self):
        expected = {
            "baseline": ("linear", 2, None, None),
            "larger": ("mlp", 3, 2048, None),
            "prism": ("prism", 2, None, "smolvla_gated_quadratic_v1"),
        }
        self.assertEqual(profile_catalog.SMOLVLA_PROFILE_NAMES, tuple(expected))
        for name, (conditioner, layers, hidden, actor_variant) in expected.items():
            with self.subTest(profile=name):
                specification = profile_catalog.smolvla_profile(name)
                self.assertEqual(specification.conditioner_type, conditioner)
                self.assertEqual(specification.num_layers, layers)
                self.assertEqual(specification.hidden_dim, hidden)
                self.assertEqual(specification.actor_variant, actor_variant)
                arguments = argparse.Namespace(
                    action="train",
                    profile=name,
                    steps=100000,
                    seed=1000,
                    output=Path("runs"),
                    checkpoint=None,
                    device="cuda",
                    python=None,
                )
                run = smolvla.build_run(arguments)
                self.assertIn(f"--policy.state_conditioner_type={conditioner}", run.command)
                self.assertIn(f"--policy.state_conditioner_num_layers={layers}", run.command)
                self.assertIn("--policy.state_conditioner_product_mode=gated_quadratic", run.command)
                self.assertIn("--policy.state_conditioner_gate_scale_init=0.01", run.command)
                self.assertIn("--policy.state_conditioner_use_rmsnorm=true", run.command)
                self.assertEqual(run.metadata["actor_variant"], actor_variant)
                if hidden is None:
                    self.assertFalse(any("state_conditioner_hidden_dim" in item for item in run.command))
                else:
                    hidden_argument = f"--policy.state_conditioner_hidden_dim={hidden}"
                    self.assertEqual(
                        run.command.index(hidden_argument), run.command.index("--wandb.enable=false") + 1
                    )


class LeRobotEvaluationSpecTest(unittest.TestCase):
    def test_rows_conditions_and_mcc_file_are_stable(self):
        self.assertEqual(evaluation_specs.EVALUATION_ROW_NAMES, ("baseline", "mcc", "oracle", "prism"))
        self.assertEqual(
            [row.output_name for row in evaluation_specs.EVALUATION_ROW_SPECS],
            ["diffusion", "diffusion_mcc_sensorless", "diffusion_mcc_oracle", "prism"],
        )
        self.assertEqual(
            [condition.name for condition in evaluation_specs.ROBUSTNESS_PERTURBATIONS],
            ["clean", "action_delay", "proprio_noise", "image_corrupt", "combined"],
        )
        self.assertEqual(
            evaluation_specs.mcc_configs_text(),
            "c00 0.005 0 0.7 0.05 0.05\n"
            "c01 0.005 1 0.9 0.05 0.05\n"
            "c02 0.01 0 0.7 0.05 0.05\n"
            "c03 0.01 1 0.9 0.05 0.05\n"
            "c04 0.015 1 0.9 0.15 0.05\n"
            "c05 0.015 2 0.9 0.15 0.05\n"
            "c06 0.02 1 0.9 0.15 0.05\n"
            "c07 0.01 1 0.7 0.15 0.1\n"
            "c08 0.015 1 0.7 0.05 0.1\n",
        )

    def test_sweep_plans_preserve_counts_order_and_output_layout(self):
        checkpoint_map = {
            "libero_spatial:1": {"baseline": "baseline/spatial", "prism": "prism/spatial"},
            "libero_object:3": {"baseline": "baseline/object", "prism": "prism/object"},
            "libero_goal:7": {"baseline": "baseline/goal", "prism": "prism/goal"},
            "libero_10:1": {"baseline": "baseline/long", "prism": "prism/long"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoints.json"
            checkpoint_path.write_text(json.dumps(checkpoint_map))
            common = {
                "checkpoint_map": checkpoint_path,
                "output_root": root / "output",
                "seed": 1000,
                "controller": "shared",
                "python": None,
                "device": "cuda",
                "episodes": None,
            }
            robustness, robustness_files = sweep_diffusion.build_plan(
                argparse.Namespace(kind="robustness", **common)
            )
            self.assertEqual(len(robustness), 60)
            self.assertEqual(robustness_files, {})
            self.assertEqual(
                robustness[0].output,
                root / "output/clean/libero_spatial_task_1/diffusion",
            )
            self.assertEqual(
                robustness[-1].output,
                root / "output/combined/libero_10_task_1/prism",
            )
            self.assertEqual(robustness[1].metadata["protocol"]["mcc"]["correction_clip"], 0.05)
            mcc, files = sweep_diffusion.build_plan(argparse.Namespace(kind="mcc", **common))
            self.assertEqual(len(mcc), 36)
            self.assertEqual(
                files[root / "output/configs.txt"],
                evaluation_specs.mcc_configs_text(),
            )
            self.assertEqual(
                files[root / "output/tasks.txt"],
                "libero_spatial:1 libero_object:3 libero_goal:7 libero_10:1\n",
            )
            self.assertEqual(
                mcc[0].output, root / "output/c00/libero_spatial_task_1/diffusion_mcc_sensorless"
            )
            self.assertEqual(mcc[-1].output, root / "output/c08/libero_10_task_1/diffusion_mcc_sensorless")


if __name__ == "__main__":
    unittest.main()
