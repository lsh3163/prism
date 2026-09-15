"""Validate portable release metadata and CPU-only command discovery."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


class ReleaseBundleTest(unittest.TestCase):
    def test_public_metadata_is_strict_json(self) -> None:
        for directory in ("configs", "integrations", "analysis", "validation"):
            for path in sorted((ROOT / directory).rglob("*.json")):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    json.loads(path.read_text(), parse_constant=reject_nonfinite)

    def test_entrypoint_help_does_not_require_simulators(self) -> None:
        scripts = (
            "integrations/humanoid-gym/train.py",
            "integrations/humanoid-gym/run_suite.py",
            "integrations/humanoid-gym/run_main_table.py",
            "integrations/humanoid-gym/run_residual_gate_controls.py",
            "integrations/humanoid-gym/run_capacity_controls.py",
            "integrations/humanoid-gym/run_robustness.py",
            "integrations/humanoid-gym/run_residual_robustness.py",
            "integrations/humanoid-gym/build_main_table.py",
            "integrations/humanoid-gym/verify_environment.py",
            "integrations/bfm-zero/evaluate_scenarios.py",
            "validation/checkpoint_inventory.py",
            "validation/check_distribution.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, "-S", str(ROOT / script), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
