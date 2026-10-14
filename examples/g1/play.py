"""Play the released G1 actors in native mjlab, without training or experiment files."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/mjlab"))
# Keep software rendering and small inference workloads from spawning a worker
# per host core (including on shared servers and Colab CPU runtimes).
for variable in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def robot_spec(mass_ratio: float, payload: float):
    """Scale original links coherently and weld the declared load to the torso."""
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    spec = get_spec()
    for body in spec.bodies[1:]:
        body.mass *= mass_ratio
        body.inertia[:] *= mass_ratio
    if payload:
        size = (0.20, 0.12, 0.16)
        x, y, z = size
        inertia = (
            payload * (y * y + z * z) / 12,
            payload * (x * x + z * z) / 12,
            payload * (x * x + y * y) / 12,
        )
        load = spec.body("torso_link").add_body(
            name="prism_payload",
            pos=(-0.15, 0.0, 0.20),
            mass=payload,
            ipos=(0, 0, 0),
            iquat=(1, 0, 0, 0),
            inertia=inertia,
            explicitinertial=True,
        )
        # Visible, noncolliding backpack; inertial properties are specified above.
        load.add_geom(
            name="backpack",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=tuple(value / 2 for value in size),
            rgba=(0.12, 0.20, 0.32, 1),
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )
    return spec


def reset_robot(env, env_ids):
    """Repeatable home-pose reset, centered on the selected terrain tile."""
    import torch

    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    terrain = env.scene.terrain
    if terrain.terrain_origins is not None:
        center = terrain.terrain_origins.shape[0] // 2
        terrain.terrain_levels[env_ids] = center
        terrain.terrain_types[env_ids] = center
        terrain.env_origins[env_ids] = terrain.terrain_origins[center, center]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 2] += 0.08
    state[:, 7:] = 0
    home = robot.data.default_joint_pos[env_ids]
    robot.write_joint_state_to_sim(home, torch.zeros_like(home), env_ids=env_ids)
    robot.write_root_link_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(state[:, 7:], env_ids=env_ids)


def environment_config(args):
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

    cfg = unitree_g1_rough_env_cfg(play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # Host initialization also checks contacts before the first home-pose reset.
    cfg.sim.nconmax = 192
    cfg.auto_reset = not args.headless
    cfg.episode_length_s = args.seconds if args.headless else int(1e9)
    cfg.curriculum = {}
    cfg.viewer.width = 640
    cfg.viewer.height = 360
    cfg.viewer.distance = 2.8
    cfg.viewer.elevation = -15.0
    cfg.viewer.azimuth = 130.0
    cfg.viewer.enable_shadows = False
    cfg.viewer.enable_reflections = False
    for sensor in cfg.scene.sensors:
        sensor.debug_vis = False
    # Inference only: retain every actor input and the native physics/controller,
    # without computing unused critic observations, contact rewards or metrics.
    cfg.observations = {"actor": cfg.observations["actor"]}
    cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if sensor.name == "terrain_scan")
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.scene.entities["robot"].spec_fn = functools.partial(robot_spec, args.mass_ratio, args.payload)
    friction = cfg.events["foot_friction"]
    friction.params["ranges"] = (args.friction, args.friction)
    cfg.events = {"foot_friction": friction, "reset_robot": EventTermCfg(func=reset_robot, mode="reset")}
    cfg.terminations = {name: cfg.terminations[name] for name in ("time_out", "fell_over")}
    if args.terrain == "rough":
        generator = cfg.scene.terrain.terrain_generator
        rough = generator.sub_terrains["random_rough"]
        rough.proportion = 1.0
        generator.sub_terrains = {"random_rough": rough}
        generator.seed = args.seed
        generator.num_rows = generator.num_cols = 5
        generator.curriculum = False
        generator.difficulty_range = (0.5, 0.5)
        generator.border_width = 0.0
        cfg.scene.terrain.max_init_terrain_level = 0
    else:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        # Keep the native 187-height scan: these actors always consume 286 inputs.
    command = cfg.commands["twist"]
    command.resampling_time_range = (1e9, 1e9)
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (args.velocity[0],) * 2
    command.ranges.lin_vel_y = (args.velocity[1],) * 2
    command.ranges.ang_vel_z = (args.velocity[2],) * 2
    for name in (
        "rel_standing_envs",
        "rel_heading_envs",
        "rel_forward_envs",
        "rel_world_envs",
        "init_velocity_prob",
    ):
        setattr(command, name, 0.0)
    return cfg


def load_actor(observations, actor: str, device: str):
    import torch
    from actors_mjlab_shared_rms import NativeSharedRmsActor

    manifest = json.loads((ROOT / "checkpoints/g1/manifest.json").read_text())
    entry = manifest["actors"][actor]
    checkpoint = ROOT / "checkpoints/g1" / entry["file"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = NativeSharedRmsActor(
        observations,
        {"actor": ["actor"]},
        "actor",
        29,
        obs_normalization=True,
        variant=entry["variant"],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if model.obs_dim != 286 or int(model.obs_normalizer.count.item()) <= 0:
        raise ValueError("Invalid trained actor or observation normalizer")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=("prism", "mlp"), default="prism")
    parser.add_argument("--terrain", choices=("rough", "flat"), default="rough")
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument("--mass-ratio", type=float, default=1.0)
    parser.add_argument("--payload", type=float, default=0.0, help="Backpack mass in kg")
    parser.add_argument(
        "--velocity", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW")
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--viewer", choices=("viser", "native"), default="viser")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--headless", action="store_true", help="Run once without a viewer; stop on a fall")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--video", type=Path, help="Save an MP4 during a headless attempt")
    parser.add_argument("--video-fps", type=int, choices=(5, 10, 25), help="Default: 5 on CPU, 25 on CUDA")
    args = parser.parse_args()
    values = [args.friction, args.mass_ratio, args.payload, args.seconds, *args.velocity]
    if not all(math.isfinite(value) for value in values):
        parser.error("All physical values must be finite")
    if args.friction <= 0 or args.mass_ratio <= 0 or args.payload < 0 or args.seconds <= 0:
        parser.error("Friction, mass ratio and seconds must be positive; payload must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if args.video and not args.headless:
        parser.error("--video requires --headless")
    args.video_fps = args.video_fps or (5 if args.device == "cpu" else 25)

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    torch.set_num_threads(1)
    configure_torch_backends()
    started = time.monotonic()
    env = ManagerBasedRlEnv(
        cfg=environment_config(args),
        device=args.device,
        render_mode="rgb_array" if args.video else None,
    )
    try:
        wrapped = RslRlVecEnvWrapper(env)
        observations = wrapped.get_observations()
        model = load_actor(observations, args.actor, args.device)

        @torch.inference_mode()
        def policy(obs):
            actions = model(obs, stochastic_output=False)
            if not torch.isfinite(actions).all():
                raise ValueError("Policy produced nonfinite actions")
            return actions

        if args.headless:
            start = env.scene["robot"].data.root_link_pos_w[0].clone()
            initial = (
                env.sim.data.qpos.detach().cpu().numpy().tobytes()
                + env.sim.data.qvel.detach().cpu().numpy().tobytes()
            )
            initial_state_sha256 = hashlib.sha256(initial).hexdigest()
            geometry_model = env.sim.mj_model
            terrain_sha256 = hashlib.sha256(
                b"".join(
                    array.tobytes()
                    for array in (
                        geometry_model.hfield_data,
                        geometry_model.geom_type,
                        geometry_model.geom_size,
                        geometry_model.geom_pos,
                        geometry_model.geom_quat,
                        geometry_model.geom_dataid,
                    )
                )
            ).hexdigest()
            writer = contextlib.nullcontext()
            if args.video:
                import mediapy

                args.video.parent.mkdir(parents=True, exist_ok=True)
                writer = mediapy.VideoWriter(str(args.video), shape=(360, 640), fps=args.video_fps)
            with writer as video:
                if video:
                    video.add_image(env.render())
                for tick in range(max(1, math.ceil(args.seconds / env.step_dt))):
                    observations, _, done, _ = wrapped.step(policy(observations))
                    if video and ((tick + 1) % (50 // args.video_fps) == 0 or done.any()):
                        video.add_image(env.render())
                    if done.any():
                        break
            end = env.scene["robot"].data.root_link_pos_w[0]
            print(
                json.dumps(
                    {
                        "actor": args.actor,
                        "terrain": args.terrain,
                        "seed": args.seed,
                        "elapsed_seconds": (tick + 1) * env.step_dt,
                        "fell_over": bool(env.termination_manager.get_term("fell_over")[0]),
                        "forward_distance_m": float(end[0] - start[0]),
                        "friction": args.friction,
                        "mass_ratio": args.mass_ratio,
                        "payload_kg": args.payload,
                        "qualitative_demo": True,
                        "physics_backend": "MuJoCo-Warp",
                        "initial_state_sha256": initial_state_sha256,
                        "terrain_sha256": terrain_sha256,
                        "wall_seconds": time.monotonic() - started,
                        "video_fps": args.video_fps if args.video else None,
                        "video": str(args.video) if args.video else None,
                    }
                )
            )
        elif args.viewer == "viser":
            import viser
            from mjlab.viewer import ViserPlayViewer

            # Native joystick widgets require positive bounds even for a fixed
            # zero lateral/yaw command. Adjust widget bounds only, then restore
            # the simulation's fixed sampling ranges and set its initial command.
            term = env.command_manager.get_term("twist")
            native_gui = term.create_gui

            def command_gui(*gui_args, **gui_kwargs):
                ranges = term.cfg.ranges
                names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
                original = [getattr(ranges, name) for name in names]
                try:
                    for name, value in zip(names, args.velocity, strict=False):
                        limit = min(10.0, max(1.0, abs(value)))
                        setattr(ranges, name, (-limit, limit))
                    native_gui(*gui_args, **gui_kwargs)
                finally:
                    for name, value in zip(names, original, strict=False):
                        setattr(ranges, name, value)
                for slider, value in zip(term._joystick_sliders, args.velocity, strict=False):
                    slider.value = max(slider.min, min(slider.max, value))
                term._joystick_enabled.value = True

            term.create_gui = command_gui
            server = viser.ViserServer(
                host="127.0.0.1", port=args.port, label=f"PRISM · {args.actor.upper()}"
            )
            try:
                ViserPlayViewer(wrapped, policy, viser_server=server).run()
            finally:
                server.stop()
        else:
            from mjlab.viewer import NativeMujocoViewer

            NativeMujocoViewer(wrapped, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
