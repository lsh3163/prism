"""Check the installed Diffusion integration on CPU without datasets or a simulator."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from common import verify_installed


def check_rgb_policy() -> None:
    """Exercise actual policy loss and pretrained serialization with two cameras."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    for mode in ("latent_quadratic", "gated_quadratic"):
        torch.manual_seed(17)
        config = DiffusionConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            down_dims=(32, 64),
            diffusion_step_embed_dim=32,
            spatial_softmax_num_keypoints=4,
            pretrained_backbone_weights=None,
            crop_shape=None,
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            num_inference_steps=2,
            use_poly_kernel_conditioning=True,
            poly_kernel_source="state",
            poly_kernel_lift_mode=mode,
        )
        policy = DiffusionPolicy(config)
        batch = {
            "observation.state": torch.randn(2, 2, 8),
            "observation.images.front": torch.rand(2, 2, 3, 64, 64),
            "observation.images.wrist": torch.rand(2, 2, 3, 64, 64),
            "action": torch.randn(2, 16, 7),
            "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        }
        gate = policy.diffusion.poly_kernel_conditioner.quadratic_scale
        before = None if gate is None else gate.detach().clone()
        loss, _ = policy(batch)
        assert torch.isfinite(loss)
        loss.backward()
        if gate is not None:
            assert torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0
        torch.optim.Adam(policy.get_optim_params(), lr=1e-3).step()
        if gate is not None:
            assert not torch.equal(before, gate)
        with tempfile.TemporaryDirectory(prefix="prism-rgb-policy-") as temporary:
            policy.save_pretrained(temporary)
            restored = DiffusionPolicy.from_pretrained(temporary, local_files_only=True, strict=True)
            assert restored.config.poly_kernel_lift_mode == mode
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            saved_config_path = Path(temporary) / "config.json"
            saved_config = json.loads(saved_config_path.read_text())
            saved_config["poly_kernel_lift_mode"] = (
                "latent_quadratic" if mode == "gated_quadratic" else "gated_quadratic"
            )
            saved_config_path.write_text(json.dumps(saved_config))
            try:
                DiffusionPolicy.from_pretrained(temporary, local_files_only=True, strict=False)
            except RuntimeError as error:
                assert "gate schema" in str(error)
            else:
                raise AssertionError("LeRobot accepted a checkpoint with the wrong saved gate mode")
        print(f"RGB DiffusionPolicy {mode}: loss/backward/update and saved config/full-policy loading passed")


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
    profiles = {
        "historical-baseline": "historical-baseline_task0_train_config.json",
        "legacy-prism": "prism_task0_train_config.json",
        "prism": "gated_prism_task0_train_config.json",
    }
    for profile, recipe in profiles.items():
        config_data = json.loads((recipe_root / recipe).read_text())["policy"]
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
            use_poly_kernel_conditioning=profile != "historical-baseline",
            poly_kernel_source="state",
            poly_kernel_lift_mode=config_data.get("poly_kernel_lift_mode", "latent_quadratic"),
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
        if profile == "prism":
            gate = model.poly_kernel_conditioner.quadratic_scale
            assert torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0
            before = gate.detach().clone()
            torch.optim.Adam(model.parameters(), lr=1e-3).step()
            assert not torch.equal(before, gate)
            restored = DiffusionModel(config)
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(restored.poly_kernel_conditioner.quadratic_scale, gate, rtol=0, atol=0)
            print(
                "Gated PRISM: alpha gradient, optimizer update, and full-model checkpoint round trip passed"
            )
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

    check_rgb_policy()


if __name__ == "__main__":
    main()
