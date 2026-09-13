"""Compare extracted G1 observations and rewards to source without creating a simulator."""

import argparse
import ast
import copy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="Original modified unitree_rl_gym")
    args = parser.parse_args()

    import isaacgym  # noqa: F401  # Simulator libraries must precede torch.
    import numpy as np
    import torch
    from config import G1HumanoidGymCfg
    from g1_env import G1Robot
    from legacy_base import HumanoidLeggedRobot

    # Compile the reference class itself, without importing unrelated sensors
    # from its module. No simulator or robot object is constructed.
    source = (args.reference_root / "legged_gym/envs/g1/g1_env.py").read_text()
    node = next(
        item for item in ast.parse(source).body if isinstance(item, ast.ClassDef) and item.name == "G1Robot"
    )
    namespace = {"torch": torch, "np": np, "LeggedRobot": HumanoidLeggedRobot}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<original-g1-class>", "exec"), namespace)
    original = object.__new__(namespace["G1Robot"])
    cfg = G1HumanoidGymCfg()
    original.cfg = cfg
    original.num_envs, original.num_actions, original.feet_num = 4, 12, 2
    original.device, original.dt = "cpu", 0.01
    original.obs_scales = cfg.normalization.obs_scales
    torch.manual_seed(16)
    shapes = {
        "dof_pos": (4, 12),
        "dof_vel": (4, 12),
        "actions": (4, 12),
        "last_actions": (4, 12),
        "last_last_actions": (4, 12),
        "default_dof_pos": (1, 12),
        "ref_dof_pos": (4, 12),
        "commands": (4, 4),
        "base_lin_vel": (4, 3),
        "base_ang_vel": (4, 3),
        "rpy": (4, 3),
        "rand_push_force": (4, 3),
        "rand_push_torque": (4, 3),
        "friction_coeffs": (4, 1),
        "base_masses": (4, 1),
        "feet_pos": (4, 2, 3),
        "knee_pos": (4, 2, 3),
        "feet_vel": (4, 2, 3),
        "contact_forces": (4, 2, 3),
        "root_states": (4, 13),
        "last_root_vel": (4, 6),
        "projected_gravity": (4, 3),
        "feet_air_time": (4, 2),
        "last_feet_z": (4, 2),
        "feet_height": (4, 2),
    }
    for name, shape in shapes.items():
        setattr(original, name, torch.randn(*shape))
    original.phase = torch.tensor([0.0, 0.1, 0.5, 0.75])
    original.feet_indices = torch.tensor([0, 1])
    original.last_contacts = torch.zeros(4, 2, dtype=torch.bool)
    original.commands_scale = torch.tensor([2.0, 2.0, 1.0])
    original.actor_obs_history = torch.randn(4, 15, 47)
    original.critic_obs_history = torch.randn(4, 3, 73)
    original.noise_scale_vec = original._get_noise_scale_vec(cfg)
    original.sim_sound_sensor = None
    original._update_depth_observation = lambda: None
    released = object.__new__(G1Robot)
    released.__dict__.update(copy.deepcopy(original.__dict__))
    for _ in range(3):
        torch.manual_seed(17)
        original.compute_observations()
        torch.manual_seed(17)
        released.compute_observations()
        torch.testing.assert_close(released.obs_buf, original.obs_buf, rtol=0, atol=0)
        torch.testing.assert_close(released.privileged_obs_buf, original.privileged_obs_buf, rtol=0, atol=0)
    self_defined_rewards = [name for name in G1Robot.__dict__ if name.startswith("_reward_")]
    for name in self_defined_rewards:
        torch.testing.assert_close(getattr(released, name)(), getattr(original, name)(), rtol=0, atol=0)
    print(
        "Exact parity: noisy 705D actor history, 219D critic history, and {} G1 rewards.".format(
            len(self_defined_rewards)
        )
    )


if __name__ == "__main__":
    main()
