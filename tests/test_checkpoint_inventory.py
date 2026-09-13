"""Keep training-seed provenance separate from path labels and evaluation seeds."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "validation/checkpoint_inventory.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_inventory", SCRIPT)
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


class CheckpointInventoryTest(unittest.TestCase):
    def test_saved_training_seed_wins_over_folder_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "seed1000/checkpoints/020000/pretrained_model"
            folder.mkdir(parents=True)
            config = {"seed": 7, "policy": {"type": "diffusion"}, "dataset": {}}
            source = folder / "train_config.json"
            source.write_text(json.dumps(config))
            record = INVENTORY.Inventory({"source": root}, 1024).inspect_lerobot(source)
            self.assertEqual(record["training_seed"]["value"], 7)
            self.assertEqual(record["directory_seed_hint"], 1000)
            self.assertFalse(record["checkpoint"]["exists"])
            self.assertIsNone(INVENTORY.seed_from_saved_config({"evaluation_seed": 4})["value"])

    def test_large_model_fingerprint_is_explicitly_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.safetensors"
            header = json.dumps(
                {"state_proj.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            ).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 2.0))
            inspector = INVENTORY.Inventory({"source": root}, 0)
            self.assertIsNone(inspector.fingerprint(path)["sha256"])
            schema = inspector.safetensors_schema(path)
            self.assertEqual(schema["n_stored_elements"], 1)
            self.assertEqual(schema["selected_tensors"]["state_proj.weight"]["shape"], [1])
            self.assertIn("header_sha256", schema)

    def test_evaluation_metadata_does_not_establish_training_seed(self) -> None:
        metrics = {
            key: 1.0
            for key in [
                "avg_return",
                "avg_episode_length",
                "avg_lin_vel_error",
                "avg_yaw_vel_error",
                "success_rate",
            ]
        }
        rows = [
            {
                "kind": "g1",
                "protocol": {"variant_name": "prism"},
                "evaluation_seed": seed,
                "metrics": metrics,
                "file": {"root": "unitree", "path": f"eval{seed}.json"},
            }
            for seed in (1, 2)
        ]
        result = INVENTORY.aggregate_g1_evaluations(rows)["prism"]
        self.assertEqual(result["n_evaluation_files"], 2)
        self.assertEqual(result["evaluation_seeds"], [1, 2])
        self.assertIn("not independently stored", result["training_seed_status"])

    def test_nonfinite_metrics_are_unavailable_without_imputing_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, success in enumerate((50.0, float("nan"))):
                path = root / f"suite/task{index}/diffusion/eval_info.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"overall": {"pc_success": success, "optional": [float("inf")]}}))
                paths.append(path)
            original_contents = [path.read_bytes() for path in paths]
            spec = {
                "sources": [],
                "evaluation_sources": [
                    {"id": "eval", "root": "source", "kind": "lerobot", "glob": "**/eval_info.json"}
                ],
                "paper_targets": [
                    {
                        "kind": "lerobot",
                        "table": "test",
                        "evaluation_sources": ["eval"],
                        "method_directory": "diffusion",
                        "expected_suite_success_pct": {"suite": 50.0},
                    }
                ],
            }
            result = INVENTORY.Inventory({"source": root}, 1024).build(spec)
            self.assertIsNone(result["evaluations"][1]["overall"]["pc_success"])
            check = result["paper_metric_checks"][0]["checks"]["suite"]
            self.assertIsNone(check["observed"])
            self.assertFalse(check["matches_printed_precision"])
            self.assertEqual(len(result["nonfinite_values"]["replacements"]), 4)
            json.dumps(result, allow_nan=False)
            self.assertEqual([path.read_bytes() for path in paths], original_contents)


if __name__ == "__main__":
    unittest.main()
