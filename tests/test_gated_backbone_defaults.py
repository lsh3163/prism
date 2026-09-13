"""Exercise new-run gates and legacy checkpoint boundaries in released patches."""

import ast
import copy
import io
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_backbone_consistency import copy_legacy_weights, load_patch_classes
from torch import nn

from prism_robot import PRISMConditioner

ROOT = Path(__file__).resolve().parents[1]


def added_definition(filename: str, kind: str, name: str) -> ast.AST:
    lines = (ROOT / "integrations" / filename).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"+{kind} {name}"))
    added = []
    for line in lines[start:]:
        if not line.startswith("+") or line.startswith("+++"):
            break
        added.append(line[1:])
    return next(node for node in ast.parse("\n".join(added)).body if getattr(node, "name", None) == name)


class ConfigNode(SimpleNamespace):
    def model_copy(self, *, deep: bool, update: dict):
        result = copy.deepcopy(self) if deep else copy.copy(self)
        result.__dict__.update(update)
        return result


class DictFilter(ConfigNode):
    pass


class HistoryFilter(ConfigNode):
    pass


class GatedBackboneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smol = load_patch_classes(
            "lerobot-smolvla.patch", ("RMSNorm", "MLPStateConditioner", "PRISMStateConditioner")
        )
        cls.bfm = load_patch_classes(
            "bfm-zero.patch", ("_HistoryActorPostMLP", "RMSNorm", "HistoryActorPRISMConcatFilter")
        )

    def make_branch(self, backbone: str, mode: str):
        if backbone == "smolvla":
            branch = self.smol["PRISMStateConditioner"](7, 11, 13, 2, mode, 0.01)
            norm = self.smol["RMSNorm"](11)
            # Nest the conditioner to exercise the prefix used by parent loaders.
            model = nn.Sequential(branch, norm)
            return branch, norm, model, lambda model, value: model(value)
        cfg = SimpleNamespace(
            history_key="history_actor",
            key=["history_actor"],
            output_dim=11,
            hidden_dim=13,
            product_mode=mode,
            gate_scale_init=0.01,
            num_layers=2,
            use_rmsnorm=True,
            rmsnorm_eps=1e-6,
        )
        input_space = SimpleNamespace(spaces={"history_actor": SimpleNamespace(shape=(7,))})
        filter_class = self.bfm["HistoryActorPRISMConcatFilter"]
        with patch.object(filter_class, "get_output_space", return_value=None):
            branch = filter_class(input_space, cfg)
        return (
            branch,
            branch.output_norm,
            nn.Sequential(branch),
            lambda model, value: model({"history_actor": value}),
        )

    def test_gate_trains_roundtrips_and_matches_core_after_update(self):
        for backbone in ("smolvla", "bfm"):
            with self.subTest(backbone=backbone):
                torch.manual_seed(307)
                branch, norm, model, forward = self.make_branch(backbone, "gated_quadratic")
                gate = branch.quadratic_scale
                self.assertEqual(tuple(gate.shape), (13,))
                self.assertTrue(gate.requires_grad)
                torch.testing.assert_close(gate, torch.full_like(gate, 0.01), rtol=0, atol=0)
                inputs, target = torch.randn(8, 7), torch.randn(8, 11)
                initial = gate.detach().clone()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                (forward(model, inputs) - target).square().mean().backward()
                self.assertTrue(torch.isfinite(gate.grad).all())
                self.assertGreater(gate.grad.abs().max().item(), 0)
                optimizer.step()
                self.assertGreater((gate - initial).abs().max().item(), 0)

                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                _, _, restored, _ = self.make_branch(backbone, "gated_quadratic")
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                torch.testing.assert_close(forward(restored, inputs), forward(model, inputs), rtol=0, atol=0)
                core = PRISMConditioner(
                    7,
                    11,
                    hidden_dim=13,
                    activation="silu" if backbone == "smolvla" else "mish",
                    use_rmsnorm=True,
                )
                copy_legacy_weights(branch, core)
                core.output_norm.load_state_dict(norm.state_dict(), strict=True)
                torch.testing.assert_close(core(inputs), forward(model, inputs))

    def test_gate_schema_mismatch_is_rejected_even_by_non_strict_parent_load(self):
        for backbone in ("smolvla", "bfm"):
            with self.subTest(backbone=backbone):
                _, _, gated, _ = self.make_branch(backbone, "gated_quadratic")
                _, _, legacy, _ = self.make_branch(backbone, "vanilla")
                for receiver, state in ((gated, legacy.state_dict()), (legacy, gated.state_dict())):
                    with self.assertRaisesRegex(RuntimeError, "original representation configuration"):
                        receiver.load_state_dict(state, strict=False)
                # Explicitly selected old representations still load their own keys.
                legacy.load_state_dict(legacy.state_dict(), strict=True)

    def test_raw_missing_mode_defaults_remain_legacy_compatible(self):
        legacy = self.smol["PRISMStateConditioner"](7, 11, 13, 2)
        self.assertEqual(legacy.product_mode, "vanilla")
        self.assertNotIn("quadratic_scale", legacy.state_dict())
        cfg = added_definition("bfm-zero.patch", "class", "HistoryActorPRISMFilterConfig")
        defaults = {
            node.target.id: ast.literal_eval(node.value)
            for node in cfg.body
            if isinstance(node, ast.AnnAssign) and node.value is not None
        }
        self.assertEqual(defaults["product_mode"], "vanilla")
        self.assertFalse(defaults["use_rmsnorm"])

    def test_new_smolvla_command_explicitly_selects_learnable_gate(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "integrations/lerobot/scripts/smolvla.py"),
                "train",
                "--lerobot-root=/example/lerobot",
                "--output=outputs/new_prism",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for setting in (
            "state_conditioner_type=prism",
            "state_conditioner_product_mode=gated_quadratic",
            "state_conditioner_gate_scale_init=0.01",
            "state_conditioner_use_rmsnorm=true",
            "train_state_proj=true",
        ):
            self.assertIn(f"--policy.{setting}", result.stdout)

    def test_bfm_new_training_defaults_and_explicit_legacy_override(self):
        names = (
            "_read_env_int",
            "_read_env_str",
            "_read_env_bool",
            "_read_env_optional_int",
            "apply_bfm_zero_env_overrides",
        )
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body
            + [added_definition("bfm-zero.patch", "def", name) for name in names],
            type_ignores=[],
        )
        namespace = {"os": os}
        exec(compile(module, "bfm-zero.patch", "exec"), namespace)
        numeric = (
            "seed online_parallel_envs num_env_steps update_agent_every num_seed_steps "
            "num_agent_updates buffer_size eval_every_steps log_every_updates checkpoint_every_steps"
        )
        archi = ConfigNode(
            **{
                key: ConfigNode(input_filter=DictFilter(key=["history_actor"]))
                for key in ("f", "actor", "critic", "aux_critic")
            }
        )
        config = ConfigNode(
            **dict.fromkeys(numeric.split(), 1),
            work_dir="outputs/run",
            buffer_device="cpu",
            use_wandb=False,
            env=ConfigNode(hydra_overrides=[]),
            agent=ConfigNode(compile=False, model=ConfigNode(archi=archi)),
        )
        filters = types.ModuleType("humanoidverse.agents.nn_filters")
        filters.DictInputFilterConfig = DictFilter
        filters.HistoryActorPRISMFilterConfig = HistoryFilter
        cases = (
            ({}, "gated_quadratic", True),
            (
                {
                    "BFM_HISTORY_CONDITIONER_PRODUCT_MODE": "vanilla",
                    "BFM_HISTORY_CONDITIONER_USE_RMSNORM": "false",
                },
                "vanilla",
                False,
            ),
        )
        for overrides, expected_mode, expected_norm in cases:
            with (
                self.subTest(mode=expected_mode),
                patch.dict(os.environ, {"BFM_HISTORY_CONDITIONER_TYPE": "prism", **overrides}, clear=True),
                patch.dict(sys.modules, {filters.__name__: filters}),
            ):
                result = namespace["apply_bfm_zero_env_overrides"](config)
                actual = result.agent.model.archi.actor.input_filter
                self.assertEqual(actual.product_mode, expected_mode)
                self.assertEqual(actual.use_rmsnorm, expected_norm)
                self.assertEqual(actual.gate_scale_init, 0.01)
                self.assertIsInstance(config.agent.model.archi.actor.input_filter, DictFilter)


if __name__ == "__main__":
    unittest.main()
