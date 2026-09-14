"""Prepare and run the five-training-seed gated G1 main-table experiment."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TRAINING_SEEDS = (1, 2, 3, 4, 5)
EVALUATION_SEEDS = (101, 102, 103)
VARIANTS = ("baseline", "larger", "prism")
ITERATIONS = 3001
TRAINING_ENVS = 4096


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_suite(output):
    output = Path(output).resolve()
    runs = []
    for seed in TRAINING_SEEDS:
        for variant in VARIANTS:
            directory = output / "runs" / variant / ("seed" + str(seed))
            runs.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "training_manifest": None,
                    "evaluations": [
                        str(directory / ("eval_" + str(value) + ".json")) for value in EVALUATION_SEEDS
                    ],
                }
            )
    return {
        "schema_version": 1,
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "max_iterations": ITERATIONS,
        "num_envs": TRAINING_ENVS,
        "checkpoint_iteration": ITERATIONS,
        "evaluation": {
            "protocol_id": "g1_balanced_timeout_v2",
            "episodes": 200,
            "num_envs": 100,
            "condition_name": "match_nopush",
            "terrain_mode": "match",
            "push_mode": "off",
        },
        "runs": runs,
    }


def training_command(python, integration, output, run, device):
    directory = Path(output) / "runs" / run["variant"] / ("seed" + str(run["training_seed"]))
    return [
        str(python),
        str(Path(integration) / "train.py"),
        "--variant=" + run["variant"],
        "--seed=" + str(run["training_seed"]),
        "--max_iterations=" + str(ITERATIONS),
        "--num_envs=" + str(TRAINING_ENVS),
        "--experiment_name=" + str(directory / "training"),
        "--run_name=main_table",
        "--sim_device=" + device,
        "--rl_device=" + device,
        "--headless",
    ]


def evaluation_command(python, integration, run, checkpoint, seed, destination, device):
    return [
        str(python),
        str(Path(integration) / "evaluate.py"),
        "--variant=" + run["variant"],
        "--variant_name=" + run["variant"],
        "--model_path=" + str(checkpoint),
        "--seed=" + str(seed),
        "--episodes=200",
        "--num_envs=100",
        "--evaluation_protocol=balanced",
        "--condition_name=match_nopush",
        "--terrain_mode=match",
        "--push_mode=off",
        "--sim_device=" + device,
        "--rl_device=" + device,
        "--headless",
        "--save_path=" + str(destination),
    ]


def prepare(output, python, unitree_root, rsl_root, device):
    output = Path(output).resolve()
    python = Path(python).resolve()
    roots = {"unitree_rl_gym": Path(unitree_root).resolve(), "rsl_rl": Path(rsl_root).resolve()}
    for name, root in roots.items():
        package = "legged_gym" if name == "unitree_rl_gym" else "rsl_rl"
        if not (root / package / "__init__.py").is_file():
            raise ValueError("Invalid upstream checkout: " + str(root))
    if not python.is_file():
        raise ValueError("Python interpreter does not exist: " + str(python))
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    integration = source / "integration"
    integration.mkdir(parents=True)
    for original in Path(__file__).resolve().parent.iterdir():
        if original.is_file() and original.suffix in {".py", ".json", ".md", ".unitree", ".rsl-rl"}:
            shutil.copy2(original, integration / original.name)
    for name, root in roots.items():
        shutil.copytree(
            root,
            source / name,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "logs", ".pytest_cache"),
        )
    source_files = {
        str(path.relative_to(source)): sha256(path) for path in sorted(source.rglob("*")) if path.is_file()
    }
    environment = {
        "PATH": str(python.parent) + os.pathsep + os.environ.get("PATH", os.defpath),
        "PYTHONPATH": os.pathsep.join(
            str(source / name) for name in ["integration", "unitree_rl_gym", "rsl_rl"]
        ),
        "LD_LIBRARY_PATH": str(python.parent.parent / "lib"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    plan = {
        "schema_version": 1,
        "created_utc": now(),
        "python": str(python),
        "device": device,
        "source_origins": {name: str(root) for name, root in roots.items()},
        "integration_origin": str(Path(__file__).resolve().parent),
        "source_files": source_files,
        "environment": environment,
        "note": "Frozen inputs for 15 independent fresh training runs. No old checkpoint is reused.",
    }
    write_json(output / "plan.json", plan)
    write_json(output / "suite.json", planned_suite(output))
    write_json(output / "queue_state.json", {"schema_version": 1, "status": "prepared", "jobs": {}})
    return plan


def verify_snapshot(output, plan):
    source = Path(output) / "source"
    actual = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual != set(plan["source_files"]):
        raise RuntimeError("Frozen source file inventory changed; create a new suite.")
    for relative, expected in plan["source_files"].items():
        if sha256(source / relative) != expected:
            raise RuntimeError("Frozen source changed: " + relative)


def update_report(output, plan, environment):
    command = [
        plan["python"],
        str(output / "source/integration/build_main_table.py"),
        "--suite",
        str(output / "suite.json"),
        "--output-dir",
        str(output / "table"),
    ]
    result = subprocess.run(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    (output / "reporter.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("Table validation failed; see reporter.log")
    return read_json(output / "table/main_table.json")


def execute_command(command, logfile, job, state, state_path, environment, phase):
    job["phase"] = phase
    job["command"] = command
    job["phase_started_utc"] = now()
    print(now() + " " + phase + " " + shlex.join(command), flush=True)
    with Path(logfile).open("x") as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        job["pid"] = process.pid
        write_json(state_path, state)
        code = process.wait()
    job["pid"] = None
    job["last_exit_code"] = code
    job["phase_finished_utc"] = now()
    write_json(state_path, state)
    if code:
        raise RuntimeError("Subprocess failed with exit {}: {}".format(code, logfile))


def completed_training(directory, run=None):
    markers = sorted(Path(directory).glob("training/*/prism_training_complete.json"))
    if len(markers) > 1:
        raise RuntimeError("Ambiguous completed training runs in " + str(directory))
    if not markers:
        return None
    marker = read_json(markers[0])
    if marker["status"] != "completed":
        raise RuntimeError("Training completion marker does not declare completed status")
    checkpoint = Path(marker["checkpoint"]["path"])
    if marker["completed_iteration"] != ITERATIONS:
        raise RuntimeError("Training did not complete the registered 3001-update budget")
    if marker["initial_iteration"] != 0 or marker["requested_final_iteration"] != ITERATIONS:
        raise RuntimeError("The registered training budget requires a fresh run")
    if sha256(checkpoint) != marker["checkpoint"]["sha256"]:
        raise RuntimeError("Completed checkpoint identity changed")
    manifest = markers[0].parent / "prism_run_manifest.json"
    if manifest.resolve() != Path(marker["training_manifest"]["path"]).resolve():
        raise RuntimeError("Completed training manifest path changed")
    if sha256(manifest) != marker["training_manifest"]["sha256"]:
        raise RuntimeError("Completed training manifest identity changed")
    if run is not None:
        inputs = read_json(manifest)
        if inputs["seed"] != run["training_seed"] or marker["seed"] != run["training_seed"]:
            raise RuntimeError("Completed training seed does not match this run")
        if inputs["actor"]["recipe_variant"] != run["variant"]:
            raise RuntimeError("Completed training variant does not match this run")
        if (
            inputs["environment_config"]["env"]["num_envs"] != TRAINING_ENVS
            or inputs["training_config"]["runner"]["max_iterations"] != ITERATIONS
            or inputs["initial_learning_iteration"] != 0
            or inputs.get("resume_checkpoint") is not None
        ):
            raise RuntimeError("Completed training does not match the fresh-run recipe")
    return checkpoint, manifest


def execute(output):
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    state_path = output / "queue_state.json"
    environment = os.environ.copy()
    environment.update(plan["environment"])
    integration = output / "source/integration"
    with (output / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This suite already has a running queue") from None
        verify_snapshot(output, plan)
        suite = read_json(output / "suite.json")
        expected = planned_suite(output)
        for field in (
            "training_seeds",
            "evaluation_seeds",
            "max_iterations",
            "num_envs",
            "checkpoint_iteration",
            "evaluation",
        ):
            if suite[field] != expected[field]:
                raise RuntimeError("Registered experiment protocol changed: " + field)
        if [(run["variant"], run["training_seed"]) for run in suite["runs"]] != [
            (run["variant"], run["training_seed"]) for run in expected["runs"]
        ]:
            raise RuntimeError("Registered training matrix changed")
        for run, registered in zip(suite["runs"], expected["runs"]):
            if run["evaluations"] != registered["evaluations"]:
                raise RuntimeError("Registered evaluation destinations changed")
        state = read_json(state_path)
        state.update(status="running", queue_pid=os.getpid(), updated_utc=now())
        write_json(state_path, state)
        try:
            update_report(output, plan, environment)
            for run in suite["runs"]:
                verify_snapshot(output, plan)
                key = run["variant"] + "/seed" + str(run["training_seed"])
                directory = output / "runs" / key
                directory.mkdir(parents=True, exist_ok=True)
                job = state["jobs"].setdefault(key, {})
                previous_pid = job.get("pid")
                if previous_pid and Path("/proc", str(previous_pid), "cmdline").exists():
                    raise RuntimeError(
                        "A prior child process may still be active; inspect PID " + str(previous_pid)
                    )
                trained = completed_training(directory, run)
                if trained is None:
                    if (directory / "training.log").exists():
                        raise RuntimeError(
                            "Incomplete previous training; preserve it and inspect " + str(directory)
                        )
                    if shutil.disk_usage(output).free < 2 * 1024**3:
                        raise RuntimeError("Less than 2 GiB free; refusing to start another training run")
                    execute_command(
                        training_command(plan["python"], integration, output, run, plan["device"]),
                        directory / "training.log",
                        job,
                        state,
                        state_path,
                        environment,
                        "training",
                    )
                    trained = completed_training(directory, run)
                    if trained is None:
                        raise RuntimeError("Training exited without its verified completion marker")
                checkpoint, manifest = trained
                run["training_manifest"] = str(manifest)
                job["checkpoint"] = str(checkpoint)
                job["training_manifest"] = str(manifest)
                state["completed_training_runs"] = sum(
                    value.get("checkpoint") is not None for value in state["jobs"].values()
                )
                write_json(output / "suite.json", suite)
                for seed, destination in zip(EVALUATION_SEEDS, run["evaluations"]):
                    if not Path(destination).is_file():
                        logfile = directory / ("eval_" + str(seed) + ".log")
                        execute_command(
                            evaluation_command(
                                plan["python"],
                                integration,
                                run,
                                checkpoint,
                                seed,
                                destination,
                                plan["device"],
                            ),
                            logfile,
                            job,
                            state,
                            state_path,
                            environment,
                            "evaluation_" + str(seed),
                        )
                    if not Path(destination).is_file():
                        raise RuntimeError("Evaluation exited without writing its result: " + destination)
                    update_report(output, plan, environment)
                job.update(phase="complete", finished_utc=now())
                state["completed_evaluated_runs"] = sum(
                    value.get("phase") == "complete" for value in state["jobs"].values()
                )
                state["updated_utc"] = now()
                write_json(state_path, state)
            report = update_report(output, plan, environment)
            if report["status"] != "complete":
                raise RuntimeError("The five-seed table is still pending after the queue finished")
            if report["verified_training_runs"] != 15 or report["verified_evaluations"] != 45:
                raise RuntimeError("The final table does not contain all 15 training runs and 45 evaluations")
            state.update(status="complete", finished_utc=now(), queue_pid=None)
            write_json(state_path, state)
            print(
                "All 15 training runs and 45 evaluations completed: " + str(output / "table/main_table.tex"),
                flush=True,
            )
        except BaseException as error:
            state.update(status="stopped", error=str(error), updated_utc=now(), queue_pid=None)
            write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--unitree-root", type=Path)
    parser.add_argument("--rsl-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--prepare", action="store_true", help="Freeze source inputs and write the registered matrix"
    )
    action.add_argument(
        "--execute", action="store_true", help="Execute an already prepared matrix, one GPU job at a time"
    )
    action.add_argument(
        "--dry-run", action="store_true", help="Print the matrix without creating files (default)"
    )
    args = parser.parse_args()
    if args.prepare:
        if args.unitree_root is None or args.rsl_root is None:
            parser.error("--prepare requires --unitree-root and --rsl-root")
        prepare(args.output_dir, args.python, args.unitree_root, args.rsl_root, args.device)
        print("Prepared: " + str(args.output_dir.resolve() / "suite.json"))
    elif args.execute:
        execute(args.output_dir)
    else:
        print(json.dumps(planned_suite(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
