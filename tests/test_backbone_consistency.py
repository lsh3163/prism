"""Compare the shared conditioner with the actual code shipped in backbone patches.

These CPU tests cover the representation branches in float32. They do not load
full policies, certify historical checkpoints, or validate simulator interfaces.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import Tensor, nn

from prism_robot import PRISMConditioner

ROOT = Path(__file__).resolve().parents[1]


def load_patch_classes(filename: str, names: tuple[str, ...]) -> dict:
    """Load complete added class definitions directly from a released patch."""
    lines = (ROOT / "integrations" / filename).read_text().splitlines()
    nodes = []
    for name in names:
        start = next(i for i, line in enumerate(lines) if line.startswith(f"+class {name}("))
        added = []
        for line in lines[start:]:
            if not line.startswith("+") or line.startswith("+++"):
                break
            added.append(line[1:])
        parsed = ast.parse("\n".join(added))
        nodes.append(
            next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
        )
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = {"torch": torch, "Tensor": Tensor, "nn": nn}
    exec(compile(module, str(ROOT / "integrations" / filename), "exec"), namespace)
    return namespace


def copy_legacy_weights(reference: nn.Module, core: PRISMConditioner) -> None:
    core.factors[0].load_state_dict(reference.left_proj.state_dict(), strict=True)
    core.factors[1].load_state_dict(reference.right_proj.state_dict(), strict=True)
    core.post_mlp.load_state_dict(reference.post_mlp.state_dict(), strict=True)
    with torch.no_grad():
        core.interaction_scales.copy_(reference.quadratic_scale.unsqueeze(0))


class BackboneConsistencyTest(unittest.TestCase):
    def test_smolvla_patch_matches_core_with_explicit_weight_mapping(self) -> None:
        classes = load_patch_classes(
            "lerobot-smolvla.patch", ("RMSNorm", "MLPStateConditioner", "PRISMStateConditioner")
        )
        reference = classes["PRISMStateConditioner"](7, 11, 13, 2, "gated_quadratic", 0.01)
        reference_norm = classes["RMSNorm"](11)
        core = PRISMConditioner(7, 11, hidden_dim=13, use_rmsnorm=True)
        copy_legacy_weights(reference, core)
        core.output_norm.load_state_dict(reference_norm.state_dict(), strict=True)

        inputs = torch.randn(5, 3, 7, requires_grad=True)
        actual = core(inputs)
        expected = reference_norm(reference(inputs))
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_bfm_patch_matches_core_and_preserves_other_observation_branches(self) -> None:
        classes = load_patch_classes(
            "bfm-zero.patch", ("_HistoryActorPostMLP", "RMSNorm", "HistoryActorPRISMConcatFilter")
        )
        cfg = SimpleNamespace(
            history_key="history_actor",
            key=["command", "history_actor", "other"],
            output_dim=11,
            hidden_dim=13,
            product_mode="gated_quadratic",
            gate_scale_init=0.01,
            num_layers=2,
            use_rmsnorm=True,
            rmsnorm_eps=1e-6,
        )
        input_space = SimpleNamespace(spaces={"history_actor": SimpleNamespace(shape=(7,))})
        filter_class = classes["HistoryActorPRISMConcatFilter"]
        # Space construction belongs to the simulator integration test; this test
        # executes the released filter's real constructor and forward computation.
        with patch.object(filter_class, "get_output_space", return_value=None):
            reference = filter_class(input_space, cfg)
        core = PRISMConditioner(7, 11, hidden_dim=13, activation="mish", use_rmsnorm=True)
        copy_legacy_weights(reference, core)
        core.output_norm.load_state_dict(reference.output_norm.state_dict(), strict=True)

        history = torch.randn(5, 7, requires_grad=True)
        command, other = torch.randn(5, 3), torch.randn(5, 2)
        actual = reference({"command": command, "history_actor": history, "other": other})
        expected = torch.cat([command, core(history), other], dim=-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, :3], command, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, -2:], other, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual.square().sum(), history, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), history)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
