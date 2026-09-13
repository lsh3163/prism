"""Check the installed Diffusion integration on CPU without datasets or a simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import verify_installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument(
        "--conditioner-checkpoint", type=Path, help="Optional historical main PRISM model.safetensors"
    )
    args = parser.parse_args()
    try:
        verify_installed(args.lerobot_root.resolve(), with_smolvla=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.path.insert(0, str(args.lerobot_root.resolve() / "src"))
    import draccus
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
    from lerobot.policies.diffusion.prism_conditioner import PolynomialKernelConditioner

    torch.set_num_threads(2)
    recipe_root = Path(__file__).resolve().parents[1] / "recipes"
    for profile in ("historical-baseline", "prism"):
        config_data = json.loads((recipe_root / f"{profile}_task0_train_config.json").read_text())["policy"]
        config_data.pop("type")
        config_data["device"] = "cpu"
        draccus.decode(DiffusionConfig, config_data)
        config = DiffusionConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            down_dims=(32, 64),
            diffusion_step_embed_dim=32,
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            num_inference_steps=2,
            use_poly_kernel_conditioning=profile == "prism",
            poly_kernel_source="state",
        )
        model = DiffusionModel(config)
        batch = {
            "observation.state": torch.randn(2, 2, 8),
            "observation.environment_state": torch.randn(2, 2, 2),
            "action": torch.randn(2, 16, 7),
            "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        }
        state = batch["observation.state"].flatten(1)
        if model.poly_kernel_conditioner is not None:
            state = model.poly_kernel_conditioner(state)
        expected = torch.cat([state, batch["observation.environment_state"].flatten(1)], dim=-1)
        torch.testing.assert_close(model._prepare_global_conditioning(batch), expected)
        loss = model.compute_loss(batch)
        loss.backward()
        if model.poly_kernel_conditioner is not None:
            assert all(parameter.grad is not None for parameter in model.poly_kernel_conditioner.parameters())
        model.eval()
        with torch.no_grad():
            actions = model.generate_actions(batch, noise=torch.zeros(2, 16, 7))
        assert actions.shape == (2, 8, 7)
        assert torch.isfinite(actions).all()
        print(
            f"{profile}: config parsing, history ordering, backward pass, and action sampling passed",
            flush=True,
        )

    if args.conditioner_checkpoint is not None:
        from safetensors import safe_open

        prefix = "diffusion.poly_kernel_conditioner."
        with safe_open(args.conditioner_checkpoint, framework="pt", device="cpu") as checkpoint:
            weights = {
                key.removeprefix(prefix): checkpoint.get_tensor(key)
                for key in checkpoint.keys()
                if key.startswith(prefix)
            }
        model = PolynomialKernelConditioner(16, 16, "latent_quadratic", 256, 256)
        model.load_state_dict(weights, strict=True)
        assert torch.isfinite(model(torch.randn(2, 16))).all()
        print(f"Historical main conditioner: strict checkpoint loading passed ({len(weights)} tensors)")


if __name__ == "__main__":
    main()
