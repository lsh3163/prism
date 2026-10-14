"""Compare released adapter branches against their frozen camera-ready source.

Only PyTorch is required. This checks CPU feature outputs, input gradients,
initialization and strict state loading; it does not run either simulator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def classes(path: Path, names: tuple[str, ...]) -> dict:
    tree = ast.parse(path.read_text())
    constants = {"ROBOT_COMMAND_DIM", "FACTOR_DIM", "ALPHA_INITIAL", "RMS_EPS"}
    namespace = {"torch": torch, "nn": nn, "OrderedDict": OrderedDict}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if len(definitions) != len(names):
        raise ValueError(f"Missing adapter classes in {path.name}")
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + definitions, type_ignores=[]), path.name, "exec"), namespace)
    return namespace


def compare(reference: nn.Module, released: nn.Module, dim: int) -> None:
    released.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 3, dim, requires_grad=True)
    a, b = reference(x), released(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a.square().mean(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().mean(), x)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-source", required=True, type=Path)
    parser.add_argument("--mjlab-source", required=True, type=Path)
    args = parser.parse_args()
    records = json.loads((ROOT / "docs/camera-ready-sources.json").read_text())
    for path, key in [(args.act_source, "act_source_sha256"), (args.mjlab_source, "mjlab_source_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[key]:
            raise ValueError(f"Source hash does not match the frozen record: {path.name}")

    act_names = ("BaseConditioner", "PrismConditioner")
    act_ref = classes(args.act_source, act_names)
    act_new = classes(ROOT / "integrations/act/conditioner.py", act_names)
    projection = nn.Linear(18, 256)
    for name in act_names:
        torch.manual_seed(17)
        a = act_ref[name](projection)
        torch.manual_seed(17)
        b = act_new[name](projection)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        compare(a, b, 18)
    baseline = act_new["BaseConditioner"](projection)
    polynomial = act_new["PrismConditioner"](projection)
    with torch.no_grad():
        polynomial.alpha.zero_()
    x = torch.randn(8, 18)
    torch.testing.assert_close(baseline(x), polynomial(x), rtol=0, atol=0)

    mj_names = ("StableSharedRMSNorm", "SharedRmsHead", "NativeSharedRmsPrismMean")
    mj_ref = classes(args.mjlab_source, mj_names)
    mj_new = classes(ROOT / "integrations/mjlab/actors_mjlab_shared_rms.py", mj_names)

    def head() -> nn.Sequential:
        # Repeated activation objects verify that all ELU positions survive.
        elu = nn.ELU()
        return nn.Sequential(
            nn.Linear(286, 512),
            elu,
            nn.Linear(512, 256),
            elu,
            nn.Linear(256, 128),
            elu,
            nn.Linear(128, 29),
        )

    for name in ("SharedRmsHead", "NativeSharedRmsPrismMean"):
        torch.manual_seed(29)
        a = mj_ref[name](head())
        torch.manual_seed(29)
        b = mj_new[name](head())
        compare(a, b, 286)
    print(
        json.dumps(
            {
                "status": "PASS",
                "ACT": "exact initialization, outputs, gradients, strict loading and alpha-zero baseline",
                "mjlab": "exact head/mean outputs, gradients and strict loading",
                "scope": "CPU adapter branches; no full-policy or benchmark evaluation",
            }
        )
    )


if __name__ == "__main__":
    main()
