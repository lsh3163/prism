"""Check that nominal and shifted evaluations request different physical settings."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/bfm-zero/evaluate_scenarios.py"
SPEC = importlib.util.spec_from_file_location("bfm_scenario_recipes", SCRIPT)
SCENARIOS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIOS)


class BFMScenarioTest(unittest.TestCase):
    def test_existing_later_scenario_prevents_the_entire_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "humanoidverse").mkdir()
            (root / "humanoidverse/tracking_eval.py").touch()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "checkpoint").mkdir()
            (model / "tracking_eval_release_payload_mass").mkdir()
            data = root / "motions.pkl"
            data.touch()
            arguments = [
                str(SCRIPT),
                "--bfm-root",
                str(root),
                "--model-folder",
                str(model),
                "--data-path",
                str(data),
                "--execute",
            ]
            with patch.object(sys, "argv", arguments), patch.object(SCENARIOS.subprocess, "run") as run:
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    SCENARIOS.main()
                self.assertIn("Evaluation output already exists", stderr.getvalue())
                run.assert_not_called()

    def test_fixed_dynamics_shifts_enable_only_the_required_event(self) -> None:
        configs = {
            scenario: dict(item.split("=", 1) for item in SCENARIOS.scenario_overrides(scenario))
            for scenario in SCENARIOS.SCENARIOS
        }
        self.assertTrue(all(value == "false" for value in configs["nominal"].values()))
        friction = configs["low_friction"]
        self.assertEqual(friction["domain_rand.randomize_friction"], "true")
        self.assertEqual(friction["domain_rand.friction_range"], "[0.2,0.2]")
        self.assertEqual(friction["domain_rand.randomize_link_mass"], "false")
        payload = configs["payload_mass"]
        self.assertEqual(payload["domain_rand.randomize_link_mass"], "true")
        self.assertEqual(payload["domain_rand.link_mass_range"], "[1.15,1.15]")
        self.assertEqual(payload["domain_rand.randomize_friction"], "false")
        self.assertTrue(all(config["domain_rand.push_robots"] == "false" for config in configs.values()))


if __name__ == "__main__":
    unittest.main()
