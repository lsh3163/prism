"""Check the public pretrained artifacts without installing the simulator."""

import hashlib
import json
import unittest
from pathlib import Path

import torch

DIRECTORY = Path(__file__).resolve().parents[1] / "checkpoints/g1"


class ReleasedG1Test(unittest.TestCase):
    def test_trained_actors_are_tensor_only_complete_and_checksum_bound(self):
        manifest = json.loads((DIRECTORY / "manifest.json").read_text())
        self.assertEqual(set(manifest["actors"]), {"prism", "mlp"})
        self.assertEqual((manifest["observation_dim"], manifest["action_dim"]), (286, 29))
        for name, entry in manifest["actors"].items():
            with self.subTest(actor=name):
                path = DIRECTORY / entry["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                self.assertEqual(path.stat().st_size, entry["bytes"])
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
                self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
                self.assertEqual(tuple(state["obs_normalizer._mean"].shape), (1, 286))
                self.assertTrue((state["obs_normalizer._std"] > 0).all())
                self.assertTrue((state["obs_normalizer._var"] >= 0).all())
                self.assertGreater(int(state["obs_normalizer.count"]), 0)
                policy_parameters = sum(
                    value.numel() for key, value in state.items() if not key.startswith("obs_normalizer.")
                )
                self.assertEqual(policy_parameters, entry["trainable_parameters"])
                self.assertFalse(any("critic" in key or "optimizer" in key for key in state))
                self.assertEqual(entry["completed_updates"], 30000)
                if name == "prism":
                    self.assertEqual(tuple(state["mlp.alpha"].shape), (256,))
                    self.assertEqual(tuple(state["mlp.factor_u.weight"].shape), (256, 99))


if __name__ == "__main__":
    unittest.main()
