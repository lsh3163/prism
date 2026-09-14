"""Train the released G1 simulation tasks using the legacy Isaac Gym environment."""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


class TrainingObserver:
    """Record training progress and learned gates without changing PPO updates."""

    def __init__(self, runner, seed, actor, num_learning_iterations):
        self.runner = runner
        self.seed = seed
        self.actor = actor
        self.initial_iteration = runner.current_learning_iteration
        self.completed_iteration = self.initial_iteration
        self.expected_final_iteration = self.initial_iteration + num_learning_iterations
        self.training_manifest = None
        self.last_checkpoint = None
        self.gate = getattr(
            getattr(runner.alg.actor_critic, "actor_encoder", None), "interaction_scales", None
        )
        self.initial_gate = None if self.gate is None else self.gate.detach().cpu().double().clone()
        self.initial_gate_statistics = self.gate_statistics()
        if (
            actor["actor_variant"] in {"g1_gated_poly_v2", "g1_residual_learned_gate_v1"}
            and actor["polynomial_degree"] > 1
        ):
            if self.gate is None:
                raise RuntimeError("The selected gated actor has no learned alpha parameter")
        if self.gate is not None:
            if not self.gate.requires_grad or self.initial_gate_statistics["optimizer_occurrences"] != 1:
                raise RuntimeError(
                    "Learned alpha must require gradients and occur exactly once in the optimizer"
                )
            if not self.initial_gate_statistics["finite"]:
                raise RuntimeError("Learned alpha contains non-finite values before training")

    def gate_statistics(self):
        if self.gate is None:
            return None
        import torch

        values = self.gate.detach().cpu().double()
        finite = bool(torch.isfinite(values).all())
        optimizer = self.runner.alg.optimizer
        step = optimizer.state.get(self.gate, {}).get("step", 0)
        result = {
            "shape": list(values.shape),
            "features": values.numel(),
            "finite": finite,
            "requires_grad": self.gate.requires_grad,
            "optimizer_occurrences": sum(
                parameter is self.gate for group in optimizer.param_groups for parameter in group["params"]
            ),
            "optimizer_steps": int(step.item() if hasattr(step, "item") else step),
        }
        statistics = {
            "mean": values.mean,
            "std": lambda: values.std(unbiased=False),
            "min": values.min,
            "max": values.max,
            "mean_abs": lambda: values.abs().mean(),
            "max_abs_change_from_start": lambda: (values - self.initial_gate).abs().max(),
        }
        result.update({name: float(function()) if finite else None for name, function in statistics.items()})
        return result

    def install(self, training_manifest):
        """Wrap the runner's existing logging/save calls; preserve its optimizer."""
        from provenance import file_identity

        self.training_manifest = training_manifest
        original_log, original_save = self.runner.log, self.runner.save

        def log(locs, *args, **kwargs):
            # The upstream runner calls log after the PPO update for this index.
            self.completed_iteration = int(locs["it"]) + 1
            result = original_log(locs, *args, **kwargs)
            statistics = self.gate_statistics()
            if statistics is not None and self.runner.writer is not None:
                for key in (
                    "mean",
                    "std",
                    "min",
                    "max",
                    "mean_abs",
                    "max_abs_change_from_start",
                    "optimizer_steps",
                ):
                    value = statistics[key]
                    if value is not None:
                        self.runner.writer.add_scalar("Policy/alpha_" + key, value, locs["it"])
            return result

        def save(path, infos=None):
            path = Path(path)
            if path.exists():
                raise FileExistsError("Refusing to overwrite a training checkpoint: " + str(path))
            metadata = {
                "seed": self.seed,
                "actor_variant": self.actor["actor_variant"],
                "recipe_variant": self.actor["recipe_variant"],
                "completed_iteration": self.completed_iteration,
                "training_manifest": self.training_manifest,
                "gate": self.gate_statistics(),
            }
            payload = {"prism_training": metadata, "upstream_infos": infos}
            # v1.0.2 increments current_learning_iteration only after learn().
            # Store the completed update count in periodic checkpoints as well,
            # so resuming them does not repeat already completed update indices.
            old_iteration = self.runner.current_learning_iteration
            try:
                self.runner.current_learning_iteration = self.completed_iteration
                result = original_save(str(path), infos=payload)
            finally:
                self.runner.current_learning_iteration = old_iteration
            self.last_checkpoint = file_identity(path)
            record = dict(metadata, checkpoint=self.last_checkpoint)
            index = Path(self.runner.log_dir) / "prism_checkpoints.jsonl"
            with index.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            return result

        self.runner.log, self.runner.save = log, save

    def finish(self, error=None):
        """Write completion only after the full budget and final save succeeded."""
        if error is None:
            expected_name = "model_{}.pt".format(self.expected_final_iteration)
            if (
                self.completed_iteration != self.expected_final_iteration
                or self.runner.current_learning_iteration != self.expected_final_iteration
                or self.last_checkpoint is None
                or Path(self.last_checkpoint["path"]).name != expected_name
            ):
                raise RuntimeError("Training returned without the requested updates and final checkpoint")
        if self.runner.writer is not None:
            self.runner.writer.flush()
            self.runner.writer.close()
        payload = {
            "schema_version": 1,
            "status": "completed" if error is None else "failed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "actor": self.actor,
            "initial_iteration": self.initial_iteration,
            "completed_iteration": self.completed_iteration,
            "requested_final_iteration": self.expected_final_iteration,
            "gate_initial": self.initial_gate_statistics,
            "gate": self.gate_statistics(),
            "checkpoint": self.last_checkpoint,
            "training_manifest": self.training_manifest,
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        }
        name = "prism_training_complete.json" if error is None else "prism_training_failed.json"
        path = Path(self.runner.log_dir) / name
        with path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path


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
    from checkpoints import validate_checkpoint
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import class_to_dict, get_args, get_load_path
    from legged_gym.utils.helpers import update_cfg_from_args
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
    if train_cfg.runner.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    if env_cfg.seed != train_cfg.seed:
        raise RuntimeError("Environment and training seeds must match before environment construction")
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
    manifest["seed_application"] = {
        "environment_seed": env_cfg.seed,
        "training_seed": train_cfg.seed,
        "mechanism": "task_registry.make_env calls set_seed before environment construction; network initialization continues that seeded RNG stream",
    }
    observer = TrainingObserver(runner, train_cfg.seed, manifest["actor"], train_cfg.runner.max_iterations)
    manifest["gate_initial"] = observer.initial_gate_statistics
    if resume_checkpoint and file_identity(resume_checkpoint["path"]) != resume_checkpoint:
        raise RuntimeError(
            "Resume checkpoint changed during runner initialization; refusing ambiguous provenance"
        )
    path = write_training_manifest(runner.log_dir, manifest)
    observer.install(file_identity(path))
    print("Training provenance: " + str(path), flush=True)
    try:
        runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
        completion = observer.finish()
    except BaseException as error:
        observer.finish(error)
        raise
    print("Training complete: " + str(completion), flush=True)


if __name__ == "__main__":
    main()
