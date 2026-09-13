"""Keep the manuscript's architecture metadata aligned with release configurations."""

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((ROOT / "configs/actor_contracts.json").read_text())["contracts"]


def class_constants(path: Path, *names: str) -> dict:
    node = ast.parse(path.read_text())
    for name in names:
        node = next(child for child in node.body if isinstance(child, ast.ClassDef) and child.name == name)
    result = {}
    for child in node.body:
        if isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name):
            try:
                result[child.targets[0].id] = ast.literal_eval(child.value)
            except (ValueError, TypeError):
                # Only literals are needed; inherited expressions are checked
                # through their independent source dimensions below.
                pass
    return result


class ActorContractTest(unittest.TestCase):
    def test_g1_contract_matches_task_dimensions_and_warmup_variants(self) -> None:
        contract = CONTRACTS["g1_residual_poly_v1"]
        source = ROOT / "integrations/humanoid-gym/config.py"
        env = class_constants(source, "G1HumanoidGymCfg", "env")
        policy = class_constants(source, "G1HumanoidGymCfgPPO", "policy")
        poly = class_constants(source, "G1HumanoidGymCfgPPOPoly", "policy")
        warmup = class_constants(source, "G1HumanoidGymCfgPPOPolyWarmup", "policy")
        self.assertEqual(contract["input_dim"], env["actor_history_length"] * env["num_single_obs"])
        self.assertEqual(
            contract["critic_input_dim"], env["critic_history_length"] * env["single_num_privileged_obs"]
        )
        self.assertEqual(contract["output_dim"], env["num_actions"])
        self.assertEqual(contract["actor_hidden_dims"], policy["actor_hidden_dims"])
        self.assertEqual(contract["critic_hidden_dims"], policy["critic_hidden_dims"])
        self.assertEqual(contract["representation_dim"], poly["poly_hidden_dim"])
        self.assertEqual(contract["main_variant"]["degree"], poly["poly_degree"])
        self.assertEqual(contract["main_variant"]["warmup_ppo_updates"], warmup["actor_poly_warmup_updates"])
        self.assertFalse(poly["critic_use_poly"])
        self.assertFalse(poly["actor_use_input_layer_norm"])
        self.assertFalse(poly["actor_use_hidden_layer_norm"])

    def test_diffusion_contract_matches_saved_training_configuration(self) -> None:
        contract = CONTRACTS["diffusion_factorized_state_v1"]
        config = json.loads(
            (ROOT / "integrations/lerobot/recipes/prism_task0_train_config.json").read_text()
        )["policy"]
        state_dim = config["input_features"]["observation.state"]["shape"][0]
        self.assertEqual(contract["input_dim"], config["n_obs_steps"] * state_dim)
        self.assertEqual(contract["state_dim_per_step"], state_dim)
        self.assertEqual(contract["action_dim"], config["output_features"]["action"]["shape"][0])
        self.assertEqual(contract["latent_dim"], config["poly_kernel_latent_dim"])
        self.assertEqual(contract["hidden_dim"], config["poly_kernel_hidden_dim"])
        self.assertEqual(contract["lift_mode"], config["poly_kernel_lift_mode"])
        self.assertEqual(config["poly_kernel_source"], "state")
        self.assertTrue(config["use_poly_kernel_conditioning"])
        self.assertFalse(config["use_poly_compliance"])

    def test_all_contract_sources_and_manifest_ids_resolve(self) -> None:
        for name, contract in CONTRACTS.items():
            with self.subTest(contract=name):
                self.assertTrue((ROOT / contract["source"]).is_file())
        manifest = json.loads((ROOT / "integrations/lerobot/source_manifest.json").read_text())
        self.assertIn(manifest["actor_variant"], CONTRACTS)
        self.assertIn(manifest["smolvla"]["actor_variant"], CONTRACTS)
        humanoid = json.loads((ROOT / "integrations/humanoid-gym/SOURCE_MANIFEST.json").read_text())
        self.assertIn(humanoid["actor_variant"], CONTRACTS)


if __name__ == "__main__":
    unittest.main()
