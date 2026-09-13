"""Train the released G1 simulation tasks using the legacy Isaac Gym environment."""

import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    # Parse our small wrapper before loading the simulator so --help is usable
    # without Isaac Gym. All remaining arguments retain upstream CLI spelling.
    from runtime import VARIANTS

    command = [sys.executable] + sys.argv[:]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--variant", choices=VARIANTS, default="prism")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the selected task without importing Isaac Gym"
    )
    wrapper, remaining = parser.parse_known_args()
    if wrapper.help:
        parser.print_help()
        print("Pass upstream flags such as --seed=1 --num_envs=16 --max_iterations=1 --headless.")
        return
    if any(arg == "--task" or arg.startswith("--task=") for arg in remaining):
        parser.error("select the task with --variant")
    if wrapper.dry_run:
        print(
            json.dumps(
                {"variant": wrapper.variant, "task": VARIANTS[wrapper.variant][0], "arguments": remaining},
                indent=2,
            )
        )
        return

    import isaacgym  # noqa: F401  # Required before torch imports.
    from runtime import register_tasks

    registry = register_tasks()
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import class_to_dict, get_args, get_load_path
    from legged_gym.utils.helpers import update_cfg_from_args
    from checkpoints import validate_checkpoint
    from provenance import file_identity, training_manifest, write_training_manifest

    task_name = VARIANTS[wrapper.variant][0]
    sys.argv = [sys.argv[0], "--task=" + task_name] + remaining
    args = get_args()
    env_cfg, train_cfg = registry.get_cfgs(task_name)
    env_cfg, train_cfg = copy.deepcopy(env_cfg), copy.deepcopy(train_cfg)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    if args.seed is not None:
        # Upstream v1.0.2's helper updates only the training seed. The local
        # experiment also updated the environment seed before construction.
        env_cfg.seed = train_cfg.seed = args.seed
    if train_cfg.seed < 0:
        parser.error("use an explicit nonnegative seed; upstream --seed=-1 does not retain its sampled seed")
    resume_checkpoint = None
    if train_cfg.runner.resume:
        if train_cfg.runner.load_run in (None, -1, "-1") or train_cfg.runner.checkpoint < 0:
            parser.error("resuming requires explicit --load_run=<run-directory> and --checkpoint=<number>")
        log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
        resume_checkpoint = file_identity(
            get_load_path(
                str(log_root),
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
        )
        validate_checkpoint(Path(resume_checkpoint["path"]), class_to_dict(train_cfg))
    env, env_cfg = registry.make_env(task_name, args=args, env_cfg=env_cfg)
    runner, train_cfg = registry.make_alg_runner(env, task_name, args=args, train_cfg=train_cfg)
    manifest = training_manifest(
        wrapper.variant,
        task_name,
        command,
        args,
        class_to_dict(env_cfg),
        class_to_dict(train_cfg),
        runner,
        sys.modules,
    )
    manifest["resume_checkpoint"] = resume_checkpoint
    if resume_checkpoint and file_identity(resume_checkpoint["path"]) != resume_checkpoint:
        raise RuntimeError(
            "Resume checkpoint changed during runner initialization; refusing ambiguous provenance"
        )
    path = write_training_manifest(runner.log_dir, manifest)
    print("Training provenance: " + str(path), flush=True)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
