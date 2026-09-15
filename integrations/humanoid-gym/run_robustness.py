"""Frozen, evaluation-only seed-1 robustness of the four G1 capacity controls.

No simulator imports in this controller. Each result keeps its training binding,
raw episodes, evaluator source, and realized physics. No retraining or adaptation.
"""
# ruff: noqa: B905 -- the simulator environment uses Python 3.8.

import argparse
import copy
import fcntl
import json
import math
import os
import statistics
from pathlib import Path

import build_main_table as metrics
import run_main_table as queue
import run_residual_gate_controls as controls
from robustness import validate_mass_scaling

KIND = "g1_capacity_seed1_robustness_v1"
VARIANTS = controls.CAPACITY_SPEC.variants
CONDITIONS = ("nominal", "low_friction", "payload_mass")
EVALUATION_SEEDS = (101, 102, 103)
require = metrics.require


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": queue.sha256(path)}


def scope():
    return {
        "suite_kind": KIND,
        "variants": list(VARIANTS),
        "training_seed": 1,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "conditions": list(CONDITIONS),
        "episodes": 200,
        "num_envs": 100,
        "protocol_id": metrics.PROTOCOL_ID,
        "evaluations": 36,
        "total_episodes": 7200,
        "perturbations": {
            "nominal": "Unmodified flat-plane evaluation with domain randomization disabled",
            "low_friction": "Ground static/dynamic and every robot collision-shape friction fixed at 0.20",
            "payload_mass": "Every robot rigid-body mass multiplied by 1.15 before simulation; upstream recomputeInertia=True",
        },
        "interpretation": "One training seed, not five-seed evidence. Uniform link mass scaling is not an attached payload. Gates are frozen during evaluation.",
    }


def prepare(output, capacity):
    capacity = Path(capacity).resolve()
    cap_plan = metrics.load_json(capacity / "plan.json")
    require(cap_plan["suite_kind"] == controls.CAPACITY_SPEC.kind, "Expected a capacity training suite")
    queue.verify_snapshot(capacity, cap_plan)
    plan = queue.prepare(
        output,
        cap_plan["python"],
        capacity / "source/unitree_rl_gym",
        capacity / "source/rsl_rl",
        cap_plan["device"],
    )
    plan.update(
        suite_kind=KIND,
        capacity_suite=str(capacity),
        capacity_plan=identity(capacity / "plan.json"),
        capacity_definition_at_preparation=identity(capacity / "suite.json"),
        capacity_registration=controls.planned_suite(capacity, controls.CAPACITY_SPEC),
        scope=scope(),
        note="Predeclared seed-1 robustness. Final 3001-update capacity checkpoints only; no training is launched.",
    )
    queue.write_json(output / "plan.json", plan)
    queue.write_json(output / "suite.json", scope())
    return plan


def load_plan(output):
    plan = metrics.load_json(output / "plan.json")
    require(plan.get("suite_kind") == KIND and plan.get("scope") == scope(), "Robustness scope changed")
    require(metrics.load_json(output / "suite.json") == scope(), "Robustness suite definition changed")
    queue.verify_snapshot(output, plan)
    for module in (__file__, queue.__file__, controls.__file__, metrics.__file__):
        require(
            plan["source_files"].get("integration/" + Path(module).name) == queue.sha256(module),
            "Use the frozen source/integration/run_robustness.py",
        )
    for key in ("capacity_plan",):
        require(identity(plan[key]["path"]) == plan[key], "Pinned capacity inputs changed: " + key)
    capacity = Path(plan["capacity_suite"])
    cap_plan = metrics.load_json(capacity / "plan.json")
    queue.verify_snapshot(capacity, cap_plan)
    require(cap_plan["suite_kind"] == controls.CAPACITY_SPEC.kind, "Wrong capacity suite")
    controls.validate_suite(capacity, metrics.load_json(capacity / "suite.json"), controls.CAPACITY_SPEC)
    require(
        plan["capacity_registration"] == controls.planned_suite(capacity, controls.CAPACITY_SPEC),
        "Pinned capacity registration changed",
    )
    require(plan["python"] == cap_plan["python"] and plan["device"] == cap_plan["device"], "Runtime changed")
    return plan, capacity, cap_plan


def training(capacity, cap_plan, variant):
    run = {"variant": variant, "training_seed": 1}
    directory = capacity / "runs" / variant / "seed1"
    completed = queue.completed_training(directory, run)
    if completed is None:
        return None
    checkpoint, manifest_path = (path.resolve() for path in completed)
    require(
        checkpoint.name == "model_3001.pt" and checkpoint.parent == manifest_path.parent,
        "Wrong final checkpoint",
    )
    manifest = metrics.load_json(manifest_path)
    registration = next(
        row
        for row in metrics.load_json(capacity / "suite.json")["runs"]
        if row["variant"] == variant and row["training_seed"] == 1
    )
    if registration["training_manifest"] is not None:
        require(
            Path(registration["training_manifest"]).resolve() == manifest_path,
            "Capacity training manifest binding changed",
        )
    marker = metrics.load_json(manifest_path.parent / "prism_training_complete.json")
    controls.validate_source_binding(manifest["source"], capacity, cap_plan)
    signature = controls.validate_training(manifest, marker, run, controls.CAPACITY_SPEC)
    return {
        "run": run,
        "checkpoint": identity(checkpoint),
        "manifest_path": manifest_path,
        "manifest": manifest,
        "training_signature": signature,
    }


def destination(output, variant, condition, seed):
    return output / "evaluations" / variant / condition / ("eval_" + str(seed) + ".json")


def evaluation_spec(condition):
    return {
        "protocol_id": metrics.PROTOCOL_ID,
        "episodes": 200,
        "num_envs": 100,
        "condition_name": condition,
        "terrain_mode": "match",
        "push_mode": "off",
    }


def validate_physics(meta, condition):
    physics = meta["robustness"]
    spec, realized = physics["spec"], physics["realized"]
    require(spec["condition"] == realized["condition"] == condition, "Physical condition label differs")
    require(
        realized["num_environments"] == 100 and realized["all_envs_consistent"] is True,
        "Physics not verified in all environments",
    )
    validation = realized["validation"]
    require(validation["all_actors_checked"] == 100, "Physics did not check all actors")
    require(
        validation["mass_scale_verified"] is True and validation["observation_caches_consistent"] is True,
        "Mass or observation caches were not verified",
    )
    actor = realized["first_actor"]
    require(
        bool(actor["shape_friction"])
        and bool(actor["body_names"])
        and len(actor["body_names"]) == len(actor["bodies"]),
        "Missing body or shape records",
    )
    require([body["name"] for body in actor["bodies"]] == actor["body_names"], "Body names differ")
    require(realized["configured_ground"] == spec["ground"], "Configured ground records differ")
    require(
        spec["rigid_body_mass_scale"] == (1.15 if condition == "payload_mass" else 1.0),
        "Mass specification differs",
    )
    expected_friction = 0.2 if condition == "low_friction" else None
    require(spec["robot_shape_friction"] == expected_friction, "Friction specification differs")
    require(spec["mass_scope"] == "all_robot_rigid_bodies", "Mass scope differs")
    config = meta["evaluation_config"]
    require(config["terrain"]["mesh_type"] == "plane", "Robustness terrain must be a flat plane")
    require(config["noise"]["add_noise"] is False, "Observation noise is enabled")
    rand = config["domain_rand"]
    for key in ("randomize_friction", "randomize_base_mass", "push_robots", "action_delay"):
        require(rand[key] is False, "Unexpected randomization: " + key)
    require(rand["action_noise"] == 0, "Action noise is enabled")
    if condition == "low_friction":
        for key in ("static_friction", "dynamic_friction"):
            require(config["terrain"][key] == spec["ground"][key] == 0.2, "Ground friction differs")
        require(
            all(
                math.isclose(value, 0.2, abs_tol=1e-6) for value in realized["first_actor"]["shape_friction"]
            ),
            "Robot shape friction differs",
        )
    return realized


def validate_evaluation(output, plan, bound, variant, condition, seed, path):
    payload = metrics.load_json(path)
    meta = payload["metadata"]
    controls.validate_source_binding(meta["source"], output, plan)
    recorded = {Path(item["path"]).name: item["sha256"] for item in meta["source"]["files"]}
    require(
        recorded.get("robustness.py") == plan["source_files"]["integration/robustness.py"],
        "Missing robustness source binding",
    )
    values, signature = controls.validate_evaluation(
        payload,
        path,
        bound["manifest"],
        bound["manifest_path"],
        bound["checkpoint"],
        bound["run"],
        seed,
        {"evaluation": evaluation_spec(condition)},
        controls.CAPACITY_SPEC,
    )
    args = meta["evaluation_arguments"]
    require(args["robustness_condition"] == condition, "Wrong robustness CLI condition")
    require(meta["push_interval_s"] is None and meta["max_push_vel_xy"] is None, "Unexpected push override")
    require(args["sim_device"] == args["rl_device"] == plan["device"], "Device differs")
    realized = validate_physics(meta, condition)
    return values, signature, realized, payload


def check_nominal_parity(original, repeated):
    """The read-only physics instrumentation must preserve nominal trajectories."""
    first, second = original["result"]["episodes"], repeated["result"]["episodes"]
    require(len(first) == len(second), "Nominal episode counts changed")

    def by_key(rows):
        return {(row["env_id"], row["episode_ordinal"]): row for row in rows}

    first, second = by_key(first), by_key(second)
    require(first.keys() == second.keys(), "Nominal episode identities changed")
    for key, before in first.items():
        after = second[key]
        require(before.keys() == after.keys(), "Nominal episode fields changed")
        for field, value in before.items():
            if isinstance(value, float):
                require(
                    math.isclose(value, after[field], rel_tol=1e-6, abs_tol=1e-7),
                    "Nominal trajectory changed: " + field,
                )
            else:
                require(value == after[field], "Nominal trajectory changed: " + field)


def build_report(output):
    plan, capacity, cap_plan = load_plan(output)
    records, missing, shared, shared_training = [], [], {}, None
    shared_physics = {}
    for variant in VARIANTS:
        bound = training(capacity, cap_plan, variant)
        if bound is None:
            require(
                not (output / "evaluations" / variant).exists(),
                "Evaluations exist without completed training",
            )
            missing.append(variant + ": training pending")
            continue
        if shared_training is None:
            shared_training = bound["training_signature"]
        require(bound["training_signature"] == shared_training, "Training protocol differs across models")
        nominal_physics, nominal_signature, nominal_payloads = None, None, {}
        for condition in CONDITIONS:
            rows, physics = {}, None
            condition_comparison = None
            parity = []
            for seed in EVALUATION_SEEDS:
                path = destination(output, variant, condition, seed)
                if not path.exists():
                    missing.append(variant + "/" + condition + ": evaluation " + str(seed) + " pending")
                    continue
                values, signature, realized, payload = validate_evaluation(
                    output, plan, bound, variant, condition, seed, path
                )
                require(
                    signature == shared.setdefault(condition, signature),
                    "Evaluation differs across models/seeds",
                )
                require(
                    realized == shared_physics.setdefault(condition, realized),
                    "Realized physics differ across models",
                )
                if physics is None:
                    physics = realized
                require(physics == realized, "Realized physics differ between evaluation seeds")
                if condition == "nominal":
                    nominal_signature, nominal_physics = signature, realized
                    nominal_payloads[seed] = payload
                    old = capacity / "runs" / variant / "seed1" / ("eval_" + str(seed) + ".json")
                    if old.exists():
                        old_payload = metrics.load_json(old)
                        controls.validate_source_binding(
                            old_payload["metadata"]["source"], capacity, cap_plan
                        )
                        _, old_signature = controls.validate_evaluation(
                            old_payload,
                            old,
                            bound["manifest"],
                            bound["manifest_path"],
                            bound["checkpoint"],
                            bound["run"],
                            seed,
                            metrics.load_json(capacity / "suite.json"),
                            controls.CAPACITY_SPEC,
                        )
                        for field in ("environment_config", "runtime", "protocol"):
                            require(
                                signature[field] == old_signature[field],
                                "Nominal reference differs: " + field,
                            )
                        old_arguments = dict(old_signature["arguments"])
                        old_arguments.update(condition_name="nominal", robustness_condition="nominal")
                        require(
                            signature["arguments"] == old_arguments, "Nominal simulator arguments changed"
                        )
                        check_nominal_parity(old_payload, payload)
                        parity.append(seed)
                else:
                    require(seed in nominal_payloads, "Stress result needs its nominal reference")
                    expected = copy.deepcopy(nominal_signature)
                    expected["arguments"]["condition_name"] = condition
                    expected["arguments"]["robustness_condition"] = condition
                    if condition == "low_friction":
                        expected["environment_config"]["terrain"].update(
                            static_friction=0.2, dynamic_friction=0.2
                        )
                        require(
                            realized["first_actor"]["bodies"] == nominal_physics["first_actor"]["bodies"],
                            "Low friction changed body physics",
                        )
                        condition_comparison = {"body_properties_unchanged": True, "friction_verified": True}
                    else:
                        condition_comparison = validate_mass_scaling(nominal_physics, realized)
                        require(
                            realized["first_actor"]["shape_friction"]
                            == nominal_physics["first_actor"]["shape_friction"],
                            "Mass stress changed friction",
                        )
                    require(signature == expected, "Undeclared condition difference")
                rows[str(seed)] = {"metrics": values, "result": identity(path)}
            average = (
                None
                if len(rows) != 3
                else {
                    key: statistics.mean(row["metrics"][key] for row in rows.values())
                    for key in metrics.METRICS
                }
            )
            nominal_row = next(
                (row for row in records if row["variant"] == variant and row["condition"] == "nominal"), None
            )
            delta = None
            if average is not None and nominal_row is not None and nominal_row["metrics"] is not None:
                delta = 100 * (average["success_rate"] - nominal_row["metrics"]["success_rate"])
            records.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "training_seed": 1,
                    "parameters": controls.CAPACITY_SPEC.methods[variant],
                    "checkpoint": bound["checkpoint"],
                    "training_manifest": identity(bound["manifest_path"]),
                    "evaluations": rows,
                    "metrics": average,
                    "survival_delta_percentage_points": delta,
                    "realized_dynamics": physics,
                    "condition_comparison": condition_comparison,
                    "nominal_parity_verified_eval_seeds": parity,
                }
            )
    return {
        **scope(),
        "status": "pending" if missing else "complete",
        "created_utc": queue.now(),
        "verified_evaluations": sum(len(row["evaluations"]) for row in records),
        "aggregation": "Mean over three evaluation seeds, each 200 balanced episodes. One fixed training seed; no training-seed SD or causal claim.",
        "records": records,
        "missing": missing,
        "plan": identity(output / "plan.json"),
    }


def write_report(output, require_complete=False):
    report = build_report(output)
    queue.write_json(output / "robustness_summary.json", report)
    lines = [
        "# G1 seed-1 robustness",
        "",
        report["aggregation"],
        "",
        "Status: " + report["status"],
        "",
        "| Model | Condition | Evaluations / 3 | Episode length | Linear error | Yaw error | Survival (%) | Change (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["records"]:
        values = row["metrics"]
        cells = [row["variant"], row["condition"], str(len(row["evaluations"]))]
        cells += (
            ["pending"] * 4
            if values is None
            else [
                "{:.2f}".format(values["avg_episode_length"]),
                "{:.4f}".format(values["avg_lin_vel_error"]),
                "{:.4f}".format(values["avg_yaw_vel_error"]),
                "{:.2f}".format(100 * values["success_rate"]),
            ]
        )
        delta = row["survival_delta_percentage_points"]
        cells.append("—" if delta is None else "{:+.2f}".format(delta))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Payload mass denotes uniform link-mass scaling by 1.15, not a physically attached object.",
        "",
    ]
    (output / "robustness_summary.md").write_text("\n".join(lines))
    require(not require_complete or report["status"] == "complete", "Robustness report is pending")
    return report


def execute(output, variants):
    require(len(set(variants)) == len(variants) and set(variants).issubset(VARIANTS), "Invalid variants")
    plan, capacity, cap_plan = load_plan(output)
    environment = dict(os.environ, **plan["environment"])
    with (output / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "queue_state.json"
        state = metrics.load_json(state_path)
        state.update(status="running", queue_pid=os.getpid())
        queue.write_json(state_path, state)
        try:
            write_report(output)
            for variant in variants:
                bound = training(capacity, cap_plan, variant)
                require(bound is not None, "Training is pending: " + variant)
                for condition in CONDITIONS:
                    for seed in EVALUATION_SEEDS:
                        path = destination(output, variant, condition, seed)
                        if path.exists():
                            continue  # The full report above validated existing records.
                        path.parent.mkdir(parents=True, exist_ok=True)
                        command = queue.evaluation_command(
                            plan["python"],
                            output / "source/integration",
                            bound["run"],
                            bound["checkpoint"]["path"],
                            seed,
                            path,
                            plan["device"],
                        )
                        command = [
                            "--condition_name=" + condition if item.startswith("--condition_name=") else item
                            for item in command
                        ]
                        command.append("--robustness_condition=" + condition)
                        label = variant + "/" + condition + "/" + str(seed)
                        job = state.setdefault("jobs", {}).setdefault(label, {})
                        queue.execute_command(
                            command,
                            path.with_suffix(".log"),
                            job,
                            state,
                            state_path,
                            environment,
                            "evaluation",
                        )
                        require(path.is_file(), "Evaluation exited without its result: " + str(path))
                        write_report(output)
                        job["phase"] = "complete"
                        queue.write_json(state_path, state)
            report = write_report(output)
            selected = [row for row in report["records"] if row["variant"] in variants]
            require(
                len(selected) == 3 * len(variants) and all(row["metrics"] is not None for row in selected),
                "Selected robustness evaluations are incomplete",
            )
            state["status"] = "selected_variants_complete"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            state.update(queue_pid=None, updated_utc=queue.now())
            queue.write_json(state_path, state)
        return write_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity-suite", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "dry-run", "execute", "report"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.prepare:
        require(args.capacity_suite is not None, "--prepare needs --capacity-suite")
        prepare(output, args.capacity_suite)
        result = scope()
    elif args.dry_run:
        plan, _, _ = load_plan(output)
        result = dict(
            scope(), plan=identity(output / "plan.json"), frozen_source_files=len(plan["source_files"])
        )
    elif args.execute:
        result = execute(output, args.variants)
    else:
        result = write_report(output, args.require_complete)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
