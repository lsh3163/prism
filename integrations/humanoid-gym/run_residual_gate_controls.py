"""Separate five-seed diagnostic of a learned gate on the historical G1 residual actor.

This suite never adopts archived checkpoints or writes a main-table result.
Python 3.8 and standard library only; the training subprocess owns simulator imports.
"""
# ruff: noqa: B905 -- retain Python 3.8 compatibility.

import argparse
import fcntl
import json
import math
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import build_main_table as metrics
import run_main_table as queue

KIND = "g1_residual_gate_diagnostic_v1"
VARIANTS = ("legacy-prism", "residual-learned-gate")
SEEDS = (1, 2, 3, 4, 5)
EVAL_SEEDS = (101, 102, 103)
METHODS = {
    "legacy-prism": {
        "label": "Historical residual PRISM, fresh training",
        "actor_variant": "g1_residual_poly_v1",
        "class": "PolyActorCritic",
        "task": "g1_humanoidgym_ppo_poly_warmup",
        "actor_mean_parameters": 921996,
        "total_actor_critic_parameters": 1320857,
    },
    "residual-learned-gate": {
        "label": "Historical residual actor with learned per-feature gate",
        "actor_variant": "g1_residual_learned_gate_v1",
        "class": "ResidualLearnedGateActorCritic",
        "task": "g1_humanoidgym_ppo_residual_learned_gate",
        "actor_mean_parameters": 922252,
        "total_actor_critic_parameters": 1321113,
    },
}
require = metrics.require


@dataclass(frozen=True)
class SuiteSpec:
    """Closed experiment catalog; nested descriptors contain immutable scalars."""

    kind: str
    title: str
    method_items: tuple
    report_stem: str
    paired_initialization: bool

    @property
    def methods(self):
        return {name: dict(items) for name, items in self.method_items}

    @property
    def variants(self):
        return tuple(name for name, _ in self.method_items)

    @property
    def training_runs(self):
        return len(self.variants) * len(SEEDS)

    @property
    def evaluations(self):
        return self.training_runs * len(EVAL_SEEDS)


RESIDUAL_SPEC = SuiteSpec(
    KIND,
    "Residual-gate diagnostic",
    tuple((name, tuple(value.items())) for name, value in METHODS.items()),
    "diagnostic_summary",
    True,
)
CAPACITY_METHODS = {
    "prism-1321k": {
        "label": "PRISM v2, 1.321M",
        "actor_variant": "g1_gated_poly_v2",
        "class": "GatedPolyActorCritic",
        "task": "g1_humanoidgym_ppo_gated_1321k",
        "actor_mean_parameters": 921715,
        "total_actor_critic_parameters": 1320576,
    },
    "mlp-1321k": {
        "label": "MLP, matched capacity",
        "actor_variant": None,
        "class": "ActorCritic",
        "task": "g1_humanoidgym_ppo_mlp_1321k",
        "actor_mean_parameters": 922092,
        "total_actor_critic_parameters": 1320953,
    },
    "mlp-wide-1500k": {
        "label": "Wider MLP, 1.500M",
        "actor_variant": None,
        "class": "ActorCritic",
        "task": "g1_humanoidgym_ppo_mlp_wide_1500k",
        "actor_mean_parameters": 1100844,
        "total_actor_critic_parameters": 1499705,
    },
    "mlp-deep-1500k": {
        "label": "Deeper MLP, 1.501M",
        "actor_variant": None,
        "class": "ActorCritic",
        "task": "g1_humanoidgym_ppo_mlp_deep_1500k",
        "actor_mean_parameters": 1101708,
        "total_actor_critic_parameters": 1500569,
    },
}
CAPACITY_SPEC = SuiteSpec(
    "g1_capacity_control_v1",
    "G1 capacity comparison",
    tuple((name, tuple(value.items())) for name, value in CAPACITY_METHODS.items()),
    "capacity_summary",
    False,
)


def selected_spec(spec):
    if spec is None:
        return RESIDUAL_SPEC
    require(
        spec is RESIDUAL_SPEC or spec is CAPACITY_SPEC, "Choose a registered immutable suite specification"
    )
    return spec


def planned_suite(output, spec=None):
    spec = selected_spec(spec)
    suite = queue.planned_suite(output)
    suite["suite_kind"] = spec.kind
    suite["methods"] = spec.methods
    suite["note"] = (
        str(spec.training_runs)
        + " independent fresh training runs; separate from the default gated main table and archived evaluations."
    )
    suite["runs"] = []
    for seed in SEEDS:
        for variant in spec.variants:
            directory = Path(output).resolve() / "runs" / variant / ("seed" + str(seed))
            suite["runs"].append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "training_manifest": None,
                    "evaluations": [
                        str(directory / ("eval_" + str(value) + ".json")) for value in EVAL_SEEDS
                    ],
                }
            )
    return suite


def validate_suite(output, suite, spec=None):
    spec = selected_spec(spec)
    expected = planned_suite(output, spec)
    for key in (
        "schema_version",
        "suite_kind",
        "methods",
        "training_seeds",
        "evaluation_seeds",
        "max_iterations",
        "num_envs",
        "checkpoint_iteration",
        "evaluation",
    ):
        require(suite.get(key) == expected[key], "Diagnostic registration changed: " + key)
    require(len(suite["runs"]) == spec.training_runs, "Wrong number of registered training runs")
    for run, registered in zip(suite["runs"], expected["runs"]):
        for key in ("variant", "training_seed", "evaluations"):
            require(run.get(key) == registered[key], "Diagnostic run registration changed: " + key)


def prepare(output, python, unitree_root, rsl_root, device, spec=None):
    spec = selected_spec(spec)
    plan = queue.prepare(output, python, unitree_root, rsl_root, device)
    # The shared helper reserves a fresh directory. Replace its default matrix
    # before this dedicated diagnostic plan can ever be executed.
    plan["suite_kind"] = spec.kind
    plan["note"] = (
        "Frozen "
        + spec.title
        + ": "
        + str(spec.training_runs)
        + " fresh runs and "
        + str(spec.evaluations)
        + " evaluations."
    )
    queue.write_json(Path(output) / "plan.json", plan)
    queue.write_json(Path(output) / "suite.json", planned_suite(output, spec))
    return plan


def source_signature(source, include_evaluator=False):
    signature = metrics.source_signature(source, include_evaluator)
    # Include the new actor as well as the original builder's maintained files.
    matches = [item for item in source["files"] if Path(item["path"]).name == "residual_gated_actor.py"]
    require(len(matches) == 1 and len(matches[0]["sha256"]) == 64, "Missing residual gated actor source")
    signature["integration/residual_gated_actor.py"] = matches[0]["sha256"]
    return signature


def check_actor(actor, variant, spec=None):
    spec = selected_spec(spec)
    for key in ("class", "actor_variant", "actor_mean_parameters", "total_actor_critic_parameters"):
        require(actor.get(key) == spec.methods[variant][key], "Wrong diagnostic actor: " + key)


def validate_source_binding(source, output, plan):
    root = (Path(output) / "source").resolve()
    for item in source["files"]:
        path = Path(item["path"]).resolve()
        require(root in path.parents, "Recorded source is outside the frozen diagnostic snapshot")
        relative = str(path.relative_to(root))
        require(
            plan["source_files"].get(relative) == item["sha256"],
            "Recorded source differs from frozen plan: " + relative,
        )


def _validate_residual_training(manifest, marker, run):
    variant, seed = run["variant"], run["training_seed"]
    actor, config = manifest["actor"], manifest["training_config"]
    require(
        manifest["schema_version"] == 1 and manifest["record_kind"] == "training_start",
        "Invalid training manifest",
    )
    require(
        manifest["seed"] == config["seed"] == manifest["environment_config"]["seed"] == seed,
        "Training seed mismatch",
    )
    require(manifest["task"] == METHODS[variant]["task"], "Wrong training task")
    check_actor(actor, variant)
    initial_hash = actor.get("initial_shared_parameter_sha256")
    require(
        isinstance(initial_hash, str)
        and len(initial_hash) == 64
        and all(character in "0123456789abcdef" for character in initial_hash),
        "Missing initial shared-parameter identity",
    )
    require(marker["actor"] == actor, "Completion actor differs from training manifest")
    require(actor["recipe_variant"] == variant and actor["polynomial_degree"] == 2, "Wrong residual recipe")
    require(actor["warmup_ppo_updates"] == 500, "Both diagnostic actors retain the 500-update warmup")
    require(config["runner"]["policy_class_name"] == actor["class"], "Runner policy class differs")
    require(config["runner"]["num_steps_per_env"] == 60, "Rollout budget differs")
    require(config["runner"].get("resume", False) is False, "Diagnostic training must be fresh")
    require(manifest["environment_config"]["env"]["episode_length_s"] == 24, "Training horizon differs")
    policy = config["policy"]
    expected_policy = {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [768, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "poly_hidden_dim": 256,
        "poly_degree": 2,
        "actor_use_poly": True,
        "critic_use_poly": False,
        "actor_poly_mode": "residual",
        "actor_use_input_layer_norm": False,
        "critic_use_input_layer_norm": False,
        "actor_use_hidden_layer_norm": False,
        "critic_use_hidden_layer_norm": False,
        "actor_output_tanh": False,
        "actor_poly_warmup_updates": 500,
    }
    for key, expected in expected_policy.items():
        require(policy.get(key) == expected, "Residual policy setting differs: " + key)
    if variant == "residual-learned-gate":
        require(
            policy.get("actor_variant") == METHODS[variant]["actor_variant"]
            and policy.get("gate_init") == 1.0,
            "Learned residual gate must start at one",
        )
        require(
            actor["gate_semantics"] == "learned_per_feature" and actor["gate_init"] == 1.0,
            "Wrong learned gate semantics",
        )
        initial, final = marker["gate_initial"], marker["gate"]
        require(initial == manifest["gate_initial"], "Initial gate statistics differ")
        for snapshot in (initial, final):
            require(
                snapshot["shape"] == [1, 256] and snapshot["features"] == 256 and snapshot["finite"] is True,
                "Invalid learned gate shape or values",
            )
            require(
                snapshot["requires_grad"] is True and snapshot["optimizer_occurrences"] == 1,
                "Learned gate is not trainable exactly once",
            )
        require(
            initial["min"] == initial["max"] == 1.0 and initial["optimizer_steps"] == 0,
            "Wrong gate initialization",
        )
        require(
            final["optimizer_steps"] == 3001 * 2 * 4 and final["max_abs_change_from_start"] > 0,
            "Learned gate did not complete the registered optimizer updates",
        )
    else:
        require(
            actor["gate_semantics"] == "legacy_scalar_scale" and actor["gate_init"] is None,
            "Wrong historical scale semantics",
        )
        require(
            marker["gate"] is None and manifest["gate_initial"] is None,
            "Legacy control contains a learned gate",
        )
        require(
            "gate_init" not in policy and "actor_variant" not in policy,
            "Legacy policy contains new gate settings",
        )
    # Only the explicit gate identity/initialization and class are allowed to
    # differ. All remaining PPO, environment, runtime and source inputs match.
    signature = metrics.training_signature(manifest)
    signature["source"] = source_signature(manifest["source"])
    return signature


def _validate_capacity_training(manifest, marker, run):
    variant, seed = run["variant"], run["training_seed"]
    descriptor = CAPACITY_SPEC.methods[variant]
    actor, config = manifest["actor"], manifest["training_config"]
    require(
        manifest["schema_version"] == 1 and manifest["record_kind"] == "training_start",
        "Invalid training manifest",
    )
    require(
        manifest["seed"] == config["seed"] == manifest["environment_config"]["seed"] == seed,
        "Training seed mismatch",
    )
    require(manifest["task"] == descriptor["task"], "Wrong capacity training task")
    check_actor(actor, variant, CAPACITY_SPEC)
    require(
        marker["actor"] == actor and actor["recipe_variant"] == variant, "Capacity recipe identity differs"
    )
    require(config["runner"]["policy_class_name"] == descriptor["class"], "Runner policy class differs")
    require(
        config["runner"]["num_steps_per_env"] == 60 and config["runner"].get("resume", False) is False,
        "Capacity training requires fresh 60-step rollouts",
    )
    require(manifest["environment_config"]["env"]["episode_length_s"] == 24, "Training horizon differs")
    hidden_dims = {
        "prism-1321k": [513, 256, 128],
        "mlp-1321k": [816, 352, 160],
        "mlp-wide-1500k": [928, 384, 224],
        "mlp-deep-1500k": [768, 512, 256, 128],
    }[variant]
    expected_policy = {
        "actor_hidden_dims": hidden_dims,
        "critic_hidden_dims": [768, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    gated = variant == "prism-1321k"
    if gated:
        expected_policy.update(
            actor_variant="g1_gated_poly_v2", poly_hidden_dim=334, poly_degree=2, gate_init=0.01
        )
    require(
        config["policy"] == expected_policy, "Capacity policy differs from the exact registered architecture"
    )
    require(actor["actor_hidden_dims"] == hidden_dims, "Actor dimensions metadata differs")
    require(actor["polynomial_degree"] == (2 if gated else None), "Wrong capacity polynomial degree")
    require(
        actor["warmup_ppo_updates"] == (0 if gated else None), "Capacity recipes have no polynomial warmup"
    )
    require(
        actor["gate_semantics"] == ("learned_per_feature" if gated else None), "Wrong capacity gate semantics"
    )
    require(actor["gate_init"] == (0.01 if gated else None), "Wrong capacity gate initialization")
    require(marker["gate_initial"] == manifest["gate_initial"], "Initial gate statistics differ")
    if gated:
        initial, final = marker["gate_initial"], marker["gate"]
        for snapshot in (initial, final):
            require(
                snapshot["shape"] == [1, 334] and snapshot["features"] == 334 and snapshot["finite"] is True,
                "Wrong capacity gate shape or values",
            )
            require(
                snapshot["requires_grad"] is True and snapshot["optimizer_occurrences"] == 1,
                "Capacity gate is not trainable exactly once",
            )
        require(
            initial["optimizer_steps"] == 0
            and all(math.isclose(initial[key], 0.01, abs_tol=1e-8) for key in ("min", "max")),
            "Wrong capacity gate initialization values",
        )
        require(
            final["optimizer_steps"] == 3001 * 2 * 4 and final["max_abs_change_from_start"] > 0,
            "Capacity gate did not complete its optimizer updates",
        )
    else:
        require(
            marker["gate"] is None and marker["gate_initial"] is None, "MLP control contains a learned gate"
        )
    # Architecture has been validated independently; only those declared
    # differences are removed from the shared PPO/environment comparison.
    signature = metrics.training_signature(manifest)
    signature["source"] = source_signature(manifest["source"])
    return signature


def validate_training(manifest, marker, run, spec=None):
    spec = selected_spec(spec)
    validator = _validate_residual_training if spec is RESIDUAL_SPEC else _validate_capacity_training
    return validator(manifest, marker, run)


def validate_evaluation(
    payload, path, manifest, manifest_path, checkpoint, run, expected_seed, suite, spec=None
):
    spec = selected_spec(spec)
    require(payload["schema_version"] == 2, "Invalid evaluation schema")
    meta, result = payload["metadata"], payload["result"]
    variant = run["variant"]
    require(meta["training_seed"] == run["training_seed"], "Evaluation training seed differs")
    require(
        meta["evaluation_seed"] == meta["seed"] == expected_seed,
        "Evaluation seed differs from registered destination",
    )
    require(
        meta["recipe_variant"] == variant and result["task"] == spec.methods[variant]["task"],
        "Wrong evaluation recipe",
    )
    binding = meta["training_manifest"]
    require(
        binding is not None
        and binding["payload"] == manifest
        and binding["sha256"] == queue.sha256(manifest_path),
        "Evaluation training manifest binding differs",
    )
    require(
        metrics.resolve(binding["path"], path.parent) == manifest_path,
        "Evaluation training manifest path differs",
    )
    require(meta["checkpoint"] == checkpoint, "Evaluation checkpoint identity differs")
    require(
        metrics.resolve(result["checkpoint_path"], path.parent) == Path(checkpoint["path"]),
        "Result checkpoint path differs",
    )
    require(result["checkpoint_sha256"] == checkpoint["sha256"], "Result checkpoint hash differs")
    require(
        meta["checkpoint_iteration"] == result["checkpoint_iteration"] == 3001, "Wrong checkpoint iteration"
    )
    check_actor(meta["actor"], variant, spec)
    require(
        meta["actor_variant"] == result["actor_variant"] == spec.methods[variant]["actor_variant"],
        "Actor identity aliases disagree",
    )
    require(
        meta["protocol_id"] == result["protocol_id"] and meta["evaluation_protocol"] == result["protocol"],
        "Protocol aliases disagree",
    )
    for key in ("episodes", "num_envs", "condition_name", "terrain_mode", "push_mode"):
        require(meta[key] == suite["evaluation"][key], "Evaluation setting differs: " + key)
    config = meta["evaluation_config"]
    require(
        config["seed"] == expected_seed and config["env"]["num_envs"] == result["num_envs"] == 100,
        "Resolved evaluation seed/environment count differs",
    )
    require(config["env"]["episode_length_s"] == 24, "Evaluation horizon differs")
    source = source_signature(meta["source"], True)
    train_source = source_signature(manifest["source"])
    require(
        all(
            source.get(key) == value for key, value in train_source.items() if key.startswith("integration/")
        ),
        "Training/evaluation implementation differs",
    )
    for key in ("python", "torch", "numpy", "isaacgym", "torch_cuda"):
        require(
            meta["runtime"][key] == manifest["runtime"][key], "Training/evaluation runtime differs: " + key
        )
    values = metrics.verify_episode_metrics(result, suite["evaluation"])
    require(
        {(row["env_id"], row["episode_ordinal"]) for row in result["episodes"]}
        == {(env, ordinal) for env in range(100) for ordinal in (1, 2)},
        "Wrong episode ordinals",
    )
    signature = {
        "environment_config": metrics.without_seed(config),
        "arguments": metrics.arguments_signature(meta["evaluation_arguments"]),
        "source": source,
        "runtime": metrics.runtime_signature(meta["runtime"]),
        "protocol": result["protocol"],
    }
    return values, signature


def reporter_identity(plan, spec):
    """Use the recorded implementation when regenerating an immutable suite."""
    entrypoint = (
        Path(__file__).with_name("run_capacity_controls.py") if spec is CAPACITY_SPEC else Path(__file__)
    )
    paths = {
        "implementation": __file__,
        "metrics": metrics.__file__,
        "queue": queue.__file__,
        "entrypoint_source": entrypoint,
    }
    identities = {}
    for name, path in paths.items():
        path = Path(path).resolve()
        digest = queue.sha256(path)
        require(
            plan["source_files"].get("integration/" + path.name) == digest,
            "Executing reporter differs from the frozen plan; use source/integration/" + entrypoint.name,
        )
        identities[name] = {"path": str(path), "sha256": digest}
    return identities


def build_report(output, spec=None):
    spec = selected_spec(spec)
    output = Path(output).resolve()
    plan = metrics.load_json(output / "plan.json")
    require(plan.get("suite_kind") == spec.kind, "Not the selected frozen experiment plan")
    queue.verify_snapshot(output, plan)
    reporter = reporter_identity(plan, spec)
    suite = metrics.load_json(output / "suite.json")
    validate_suite(output, suite, spec)
    records, missing, owners, initial_identities = [], [], {}, {}
    shared_training, shared_evaluation = None, None
    for run in suite["runs"]:
        label = run["variant"] + "/seed" + str(run["training_seed"])
        directory = output / "runs" / label
        completed = queue.completed_training(directory, run)
        if completed is None:
            require(
                not any(Path(path).exists() for path in run["evaluations"]),
                "Evaluation exists without completed training",
            )
            missing.append(label + ": training pending")
            continue
        checkpoint_path, manifest_path = completed
        manifest_path, checkpoint_path = manifest_path.resolve(), checkpoint_path.resolve()
        require(checkpoint_path.parent == manifest_path.parent, "Checkpoint does not belong to training run")
        require(checkpoint_path.name == "model_3001.pt", "Require the final checkpoint")
        if run["training_manifest"] is not None:
            require(
                metrics.resolve(run["training_manifest"], output) == manifest_path,
                "Suite manifest binding differs",
            )
        manifest = metrics.load_json(manifest_path)
        validate_source_binding(manifest["source"], output, plan)
        marker = metrics.load_json(manifest_path.parent / "prism_training_complete.json")
        signature = validate_training(manifest, marker, run, spec)
        initial_hash = manifest["actor"].get("initial_shared_parameter_sha256")
        if spec.paired_initialization:
            previous = initial_identities.setdefault(run["training_seed"], initial_hash)
            require(previous == initial_hash, "Paired methods did not start with identical shared parameters")
        if shared_training is None:
            shared_training = signature
        require(signature == shared_training, "Training PPO/environment/source/runtime differ")
        checkpoint = {"path": str(checkpoint_path), "sha256": queue.sha256(checkpoint_path)}
        require(checkpoint["sha256"] not in owners, "Checkpoint reused across independent training runs")
        owners[checkpoint["sha256"]] = label
        evaluation_values = {}
        for seed, value in zip(EVAL_SEEDS, run["evaluations"]):
            path = Path(value)
            if not path.is_file():
                missing.append(label + ": evaluation seed " + str(seed) + " pending")
                continue
            payload = metrics.load_json(path)
            validate_source_binding(payload["metadata"]["source"], output, plan)
            values, signature = validate_evaluation(
                payload, path, manifest, manifest_path, checkpoint, run, seed, suite, spec
            )
            if shared_evaluation is None:
                shared_evaluation = signature
            require(signature == shared_evaluation, "Evaluation protocol/environment/source/runtime differ")
            evaluation_values[str(seed)] = values
        records.append(
            {
                "variant": run["variant"],
                "training_seed": run["training_seed"],
                "initial_shared_parameter_sha256": initial_hash,
                "training_manifest": {"path": str(manifest_path), "sha256": queue.sha256(manifest_path)},
                "checkpoint": checkpoint,
                "gate": marker["gate"],
                "evaluation_metrics": evaluation_values,
                "training_seed_metrics": None
                if len(evaluation_values) != 3
                else {
                    key: statistics.mean(row[key] for row in evaluation_values.values())
                    for key in metrics.METRICS
                },
            }
        )
    complete = not missing and len(records) == spec.training_runs
    summaries = {}
    for variant in spec.variants:
        rows = [
            record["training_seed_metrics"]
            for record in records
            if record["variant"] == variant and record["training_seed_metrics"] is not None
        ]
        summaries[variant] = dict(
            spec.methods[variant],
            verified_training_seeds=len(rows),
            metrics=None
            if not complete
            else {
                key: {
                    "mean": statistics.mean(row[key] for row in rows),
                    "sample_sd": statistics.stdev(row[key] for row in rows),
                }
                for key in metrics.METRICS
            },
        )
    return {
        "schema_version": 1,
        "suite_kind": spec.kind,
        "status": "complete" if complete else "pending",
        "suite": {"path": str(output / "suite.json"), "sha256": queue.sha256(output / "suite.json")},
        "verified_training_runs": len(records),
        "verified_evaluations": sum(len(record["evaluation_metrics"]) for record in records),
        "aggregation": "Equal mean across three evaluation seeds within each training run, then mean and sample SD (ddof=1) across five fresh training seeds. "
        + spec.title
        + "; separate from the original default PRISM main table.",
        "paired_initialization_required": spec.paired_initialization,
        "reporter": reporter,
        "methods": summaries,
        "records": records,
        "missing": missing,
        "verified_training_signature": shared_training,
        "verified_evaluation_signature": shared_evaluation,
    }


def write_report(output, require_complete=False, spec=None):
    spec = selected_spec(spec)
    report = build_report(output, spec)
    directory = Path(output) / "report"
    directory.mkdir(parents=True, exist_ok=True)
    queue.write_json(directory / (spec.report_stem + ".json"), report)
    lines = [
        spec.title,
        "",
        "Status: " + report["status"],
        "",
        report["aggregation"],
        "",
        "| Method | Training seeds evaluated | Length | Linear error | Yaw error | Timeout success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in spec.variants:
        method = report["methods"][variant]
        values = (
            ["--"] * 4
            if method["metrics"] is None
            else [
                "{:.6g} ± {:.6g}".format(method["metrics"][key]["mean"], method["metrics"][key]["sample_sd"])
                for key in metrics.METRICS
            ]
        )
        lines.append(
            "| " + " | ".join([method["label"], str(method["verified_training_seeds"])] + values) + " |"
        )
    temporary = directory / (spec.report_stem + ".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(directory / (spec.report_stem + ".md"))
    require(not require_complete or report["status"] == "complete", "Diagnostic remains pending")
    return report


def execute(output, spec=None):
    spec = selected_spec(spec)
    output = Path(output).resolve()
    plan = metrics.load_json(output / "plan.json")
    require(plan.get("suite_kind") == spec.kind, "Use this experiment's --prepare first")
    environment = os.environ.copy()
    environment.update(plan["environment"])
    state_path = output / "queue_state.json"
    with (output / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The diagnostic queue is already running") from None
        queue.verify_snapshot(output, plan)
        suite = metrics.load_json(output / "suite.json")
        validate_suite(output, suite, spec)
        state = metrics.load_json(state_path)
        state.update(status="running", queue_pid=os.getpid(), updated_utc=queue.now())
        queue.write_json(state_path, state)
        try:
            write_report(output, spec=spec)
            for run in suite["runs"]:
                queue.verify_snapshot(output, plan)
                key = run["variant"] + "/seed" + str(run["training_seed"])
                directory = output / "runs" / key
                directory.mkdir(parents=True, exist_ok=True)
                job = state["jobs"].setdefault(key, {})
                pid = job.get("pid")
                if pid and Path("/proc", str(pid), "cmdline").exists():
                    raise RuntimeError("Prior diagnostic child may still be active: " + str(pid))
                completed = queue.completed_training(directory, run)
                if completed is None:
                    require(
                        not (directory / "training.log").exists(),
                        "Preserve and inspect incomplete training: " + key,
                    )
                    require(shutil.disk_usage(output).free >= 2 * 1024**3, "Less than 2 GiB free")
                    queue.execute_command(
                        queue.training_command(
                            plan["python"], output / "source/integration", output, run, plan["device"]
                        ),
                        directory / "training.log",
                        job,
                        state,
                        state_path,
                        environment,
                        "training",
                    )
                    completed = queue.completed_training(directory, run)
                    require(completed is not None, "Training exited without completion evidence")
                checkpoint, manifest = completed
                run["training_manifest"] = str(manifest)
                job.update(checkpoint=str(checkpoint), training_manifest=str(manifest))
                queue.write_json(output / "suite.json", suite)
                write_report(output, spec=spec)  # Reject incompatible training before GPU evaluation.
                for seed, destination in zip(EVAL_SEEDS, run["evaluations"]):
                    if not Path(destination).is_file():
                        queue.execute_command(
                            queue.evaluation_command(
                                plan["python"],
                                output / "source/integration",
                                run,
                                checkpoint,
                                seed,
                                destination,
                                plan["device"],
                            ),
                            directory / ("eval_" + str(seed) + ".log"),
                            job,
                            state,
                            state_path,
                            environment,
                            "evaluation_" + str(seed),
                        )
                    require(Path(destination).is_file(), "Evaluation exited without its result")
                    report = write_report(output, spec=spec)
                job.update(phase="complete", finished_utc=queue.now())
                state.update(
                    completed_training_runs=report["verified_training_runs"],
                    verified_evaluations=report["verified_evaluations"],
                    updated_utc=queue.now(),
                )
                queue.write_json(state_path, state)
            report = write_report(output, require_complete=True, spec=spec)
            require(
                report["verified_training_runs"] == spec.training_runs
                and report["verified_evaluations"] == spec.evaluations,
                "Incomplete diagnostic matrix",
            )
            state.update(status="complete", queue_pid=None, finished_utc=queue.now())
            queue.write_json(state_path, state)
        except BaseException as error:
            state.update(status="stopped", queue_pid=None, error=str(error), updated_utc=queue.now())
            queue.write_json(state_path, state)
            raise


def main(spec=None):
    spec = selected_spec(spec)
    parser = argparse.ArgumentParser(
        description=spec.title + ". Prepare frozen inputs, execute sequentially, or verify reports."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--unitree-root", type=Path)
    parser.add_argument("--rsl-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--require-complete", action="store_true", help="With --report, fail if any run is pending"
    )
    action = parser.add_mutually_exclusive_group()
    for name in ("prepare", "execute", "dry-run", "report"):
        action.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.require_complete and not args.report:
        parser.error("--require-complete requires --report")
    if args.prepare:
        if args.unitree_root is None or args.rsl_root is None:
            parser.error("--prepare requires --unitree-root and --rsl-root")
        prepare(args.output_dir, args.python, args.unitree_root, args.rsl_root, args.device, spec)
        print("Prepared " + spec.title + ": " + str(args.output_dir.resolve()))
    elif args.execute:
        execute(args.output_dir, spec)
    elif args.report:
        write_report(args.output_dir, args.require_complete, spec)
    else:
        print(json.dumps(planned_suite(args.output_dir, spec), indent=2))


if __name__ == "__main__":
    main()
