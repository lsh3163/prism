"""Keep offline features and ablations faithful to versioned gated checkpoints."""

import ast
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from prism_robot import PRISMConditioner

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GatedAnalysisTest(unittest.TestCase):
    def test_g1_checkpoint_inventory_and_ablation_preserve_learned_gates(self) -> None:
        directory = ROOT / "integrations/humanoid-gym"
        with patch.object(sys, "path", [str(directory), *sys.path]):
            actor_module = load_module("gated_analysis_actor", directory / "gated_actor.py")
        with redirect_stdout(io.StringIO()):
            actor = actor_module.GatedPolyActorCritic(705, 219, 12).eval()
        with torch.no_grad():
            actor.actor_encoder.interaction_scales[0, 0] = -0.5
            actor.actor_encoder.interaction_scales[0, 1] = 0
        ablation = load_module("gated_ablation", ROOT / "analysis/paper/ablate_locomotion_factors.py")
        inventory = load_module("gated_inventory", ROOT / "validation/checkpoint_inventory.py")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model_state_dict": actor.state_dict(), "infos": {"seed": 7}}, checkpoint)
            with redirect_stdout(io.StringIO()):
                restored = ablation.load_actor(checkpoint)
            observations = torch.randn(4, 705)
            torch.testing.assert_close(
                restored.act_inference(observations), actor.act_inference(observations)
            )
            before = {key: value.clone() for key, value in restored.state_dict().items()}
            rows = {row["factor"]: row for row in ablation.ablate(restored, observations, [0, 1], 0.25)}
            self.assertEqual(rows[0]["learned_alpha"], -0.5)
            self.assertGreater(rows[0]["mean_l2_joint_target_shift_rad"], 0)
            self.assertEqual(rows[1]["mean_l2_joint_target_shift_rad"], 0)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            record = inventory.Inventory({"run": Path(directory)}, 0).inspect_g1(checkpoint)
            self.assertEqual(record["actor"]["variant_id"], "g1_gated_poly_v2")
            self.assertEqual(record["actor"]["degree_from_gated_factors"], 2)
            self.assertEqual(record["actor"]["gate_shape"], [1, 256])
            self.assertEqual(record["training_seed"]["value"], 7)

    def test_contact_probe_uses_saved_gate_and_rejects_schema_disagreement(self) -> None:
        # Execute real extractor functions without importing optional plotting/data packages.
        source = ROOT / "analysis/paper/make_contact_response_probe_table.py"
        functions = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in {"load_prism_weights", "prism_latents"}
        ]
        scope = {
            "torch": torch,
            "json": json,
            "Path": Path,
            "BASELINE_FEATURE_MODE": "state-first-contribution",
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
        model = PRISMConditioner(4, 5, hidden_dim=5)
        with torch.no_grad():
            model.interaction_scales.copy_(torch.tensor([[-2.0, 0, 0.01, 0.7, 3.0]]))
        weights = {"input_norm.weight": torch.ones(4), "input_norm.bias": torch.zeros(4)}
        for index, prefix in enumerate(("left_proj", "right_proj")):
            weights.update(
                {f"{prefix}.{key}": value for key, value in model.factors[index].state_dict().items()}
            )
        weights["quadratic_scale"] = model.interaction_scales[0].detach().clone()
        inputs = torch.randn(6, 4)
        normalized = torch.nn.functional.layer_norm(inputs, (4,))
        _, features = scope["prism_latents"](inputs, weights)
        torch.testing.assert_close(features, model.polynomial_features(normalized))
        scope["load_file"] = lambda _: {
            f"diffusion.poly_kernel_conditioner.{key}": value for key, value in weights.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = {
                "use_poly_kernel_conditioning": True,
                "poly_kernel_source": "state",
                "poly_kernel_lift_mode": "gated_quadratic",
            }
            (folder / "config.json").write_text(json.dumps(config))
            self.assertIn("quadratic_scale", scope["load_prism_weights"](folder))
            config["poly_kernel_lift_mode"] = "latent_quadratic"
            (folder / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "gate tensor disagrees"):
                scope["load_prism_weights"](folder)
            weights.pop("quadratic_scale")
            scope["load_prism_weights"](folder)
            config["poly_kernel_lift_mode"] = "gated_quadratic"
            (folder / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "gate tensor disagrees"):
                scope["load_prism_weights"](folder)


if __name__ == "__main__":
    unittest.main()
