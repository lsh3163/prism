"""Evaluate one verified fresh residual PRISM checkpoint under frozen robustness conditions.

This controller never trains. The completed four-model robustness suite is the
reference for physics, evaluation and training protocol, not a source of samples
to pool. The separate residual-learned-gate diagnostic cannot be selected.
"""

import argparse
import ast
import copy
import fcntl
import json
import os
import shutil
import statistics
from pathlib import Path

import run_robustness as base

queue, controls, metrics = base.queue, base.controls, base.metrics
require, identity = base.require, base.identity
KIND = "g1_residual_seed1_robustness_v1"
VARIANTS = ("legacy-prism",)
CONDITIONS, EVALUATION_SEEDS = base.CONDITIONS, base.EVALUATION_SEEDS
BRIDGE_FILES = ("config.py", "runtime.py")
CAPACITY_CLASSES = {
    "G1HumanoidGymCfgPPOGated1321",
    "G1HumanoidGymCfgPPOMatched1321",
    "G1HumanoidGymCfgPPOWide1500",
    "G1HumanoidGymCfgPPODeep1500",
}
LEGACY_POLICY_FIELDS = {
    "actor_output_tanh": False,
    "actor_poly_mode": "residual",
    "actor_poly_warmup_updates": 500,
    "actor_use_hidden_layer_norm": False,
    "actor_use_input_layer_norm": False,
    "actor_use_poly": True,
    "critic_use_hidden_layer_norm": False,
    "critic_use_input_layer_norm": False,
    "critic_use_poly": False,
}


def scope():
    result = base.scope()
    result.update(
        suite_kind=KIND,
        variants=list(VARIANTS),
        evaluations=9,
        total_episodes=1800,
        interpretation="One fresh training seed of historical residual PRISM, with its original 500-update scalar warmup. Evaluation only; no learned-alpha diagnostic or retraining.",
    )
    return result


def audit_source_bridge(residual, reference):
    """Allow only the already declared capacity registration additions."""
    bridge = {}
    for name in BRIDGE_FILES:
        old = Path(residual) / "source/integration" / name
        new = Path(reference) / "source/integration" / name
        before, after = ast.parse(old.read_text()), ast.parse(new.read_text())
        if name == "config.py":
            additions = after.body[len(before.body) :]
            require(
                len(additions) == 4
                and all(isinstance(node, ast.ClassDef) for node in additions)
                and {node.name for node in additions} == CAPACITY_CLASSES,
                "Config bridge must only append the four capacity classes",
            )
            after.body = after.body[: len(before.body)]
        else:
            assignments = [
                node
                for node in after.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets)
            ]
            require(
                len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
                "Runtime registry differs",
            )
            registry = assignments[0].value
            removed, pairs = set(), []
            for key, value in zip(registry.keys, registry.values):
                label = ast.literal_eval(key)
                if label in controls.CAPACITY_SPEC.variants:
                    removed.add(label)
                else:
                    pairs.append((key, value))
            require(
                removed == set(controls.CAPACITY_SPEC.variants),
                "Runtime bridge must remove four capacity registrations",
            )
            registry.keys = [key for key, _ in pairs]
            registry.values = [value for _, value in pairs]
        require(ast.dump(before) == ast.dump(after), "Undeclared source difference in bridge: " + name)
        bridge["integration/" + name] = {"legacy": identity(old), "reference": identity(new)}
    return bridge


def training(residual, plan, variant="legacy-prism"):
    require(variant in VARIANTS, "Only legacy-prism is authorized")
    run = {"variant": variant, "training_seed": 1}
    completed = queue.completed_training(Path(residual) / "runs" / variant / "seed1", run)
    require(completed is not None, "Fresh residual seed 1 training is incomplete")
    checkpoint, manifest_path = (path.resolve() for path in completed)
    require(
        checkpoint.name == "model_3001.pt" and checkpoint.parent == manifest_path.parent,
        "Wrong final checkpoint",
    )
    manifest = metrics.load_json(manifest_path)
    registration = next(
        row
        for row in metrics.load_json(Path(residual) / "suite.json")["runs"]
        if row["variant"] == variant and row["training_seed"] == 1
    )
    if registration["training_manifest"] is not None:
        require(
            Path(registration["training_manifest"]).resolve() == manifest_path,
            "Residual manifest registration differs",
        )
    marker_path = manifest_path.parent / "prism_training_complete.json"
    controls.validate_source_binding(manifest["source"], residual, plan)
    signature = controls.validate_training(
        manifest, metrics.load_json(marker_path), run, controls.RESIDUAL_SPEC
    )
    return {
        "run": run,
        "checkpoint": identity(checkpoint),
        "manifest_path": manifest_path,
        "manifest": manifest,
        "marker": identity(marker_path),
        "training_signature": signature,
    }


def comparable_signature(signature, training_protocol=False, legacy=False):
    result = copy.deepcopy(signature)
    for name in BRIDGE_FILES:
        require("integration/" + name in result["source"], "Missing audited source bridge")
        del result["source"]["integration/" + name]
    if training_protocol and legacy:
        policy = result["training_config"]["policy"]
        for name, expected in LEGACY_POLICY_FIELDS.items():
            require(policy.get(name) == expected, "Legacy architecture setting differs: " + name)
            del policy[name]
    return result


def reference_evidence(reference):
    plan, capacity, cap_plan = base.load_plan(reference)
    report = base.build_report(reference)
    require(
        report["status"] == "complete" and report["verified_evaluations"] == 36,
        "Reference robustness must be complete",
    )
    saved = metrics.load_json(Path(reference) / "robustness_summary.json")
    require(
        saved["status"] == "complete" and saved["verified_evaluations"] == 36,
        "Final reference summary is missing",
    )

    # Later capacity nominal repeats can add parity annotations. Performance,
    # checkpoint bindings and physics in the pinned summary must stay identical.
    def stable_rows(rows):
        return [
            {key: value for key, value in row.items() if key != "nominal_parity_verified_eval_seeds"}
            for row in rows
        ]

    require(
        stable_rows(saved["records"]) == stable_rows(report["records"]),
        "Reference summary differs from raw verified results",
    )
    evidence, signatures, bindings = [], {}, {}
    for variant in base.VARIANTS:
        bound = base.training(capacity, cap_plan, variant)
        require(bound is not None, "Reference training is incomplete")
        bindings[variant] = bound
        for condition in CONDITIONS:
            for seed in EVALUATION_SEEDS:
                path = base.destination(reference, variant, condition, seed)
                _, signature, realized, _ = base.validate_evaluation(
                    reference, plan, bound, variant, condition, seed, path
                )
                key = (condition, seed)
                record = {"signature": signature, "realized": realized}
                require(record == signatures.setdefault(key, record), "Reference controls disagree")
                evidence.append(identity(path))
    return plan, bindings, signatures, evidence


def prepare(output, residual, reference):
    output, residual, reference = (Path(path).resolve() for path in (output, residual, reference))
    residual_plan = metrics.load_json(residual / "plan.json")
    require(
        residual_plan["suite_kind"] == controls.RESIDUAL_SPEC.kind, "Expected the residual training suite"
    )
    queue.verify_snapshot(residual, residual_plan)
    controls.validate_suite(residual, metrics.load_json(residual / "suite.json"), controls.RESIDUAL_SPEC)
    bound = training(residual, residual_plan)
    reference_plan, bindings, _, evidence = reference_evidence(reference)
    bridge = audit_source_bridge(residual, reference)
    for other in bindings.values():
        require(
            comparable_signature(bound["training_signature"], True, True)
            == comparable_signature(other["training_signature"], True),
            "Residual/reference training PPO, environment, runtime or shared source differs",
        )
    require(
        residual_plan["python"] == reference_plan["python"]
        and residual_plan["device"] == reference_plan["device"],
        "Training runtimes differ",
    )
    plan = queue.prepare(
        output,
        reference_plan["python"],
        reference / "source/unitree_rl_gym",
        reference / "source/rsl_rl",
        reference_plan["device"],
    )
    # Preserve the exact trained residual recipe in this NEW snapshot. Existing
    # snapshots are read-only, and the generic actor/source validator stays strict.
    for relative, pair in bridge.items():
        shutil.copy2(pair["legacy"]["path"], output / "source" / relative)
        plan["source_files"][relative] = pair["legacy"]["sha256"]
    plan.update(
        suite_kind=KIND,
        scope=scope(),
        residual_suite=str(residual),
        residual_plan=identity(residual / "plan.json"),
        residual_registration=controls.planned_suite(residual, controls.RESIDUAL_SPEC),
        reference_suite=str(reference),
        reference_plan=identity(reference / "plan.json"),
        reference_summary=identity(reference / "robustness_summary.json"),
        reference_results=evidence,
        checkpoint=bound["checkpoint"],
        training_manifest=identity(bound["manifest_path"]),
        training_completion=bound["marker"],
        source_bridge=bridge,
        declared_training_recipe_fields=LEGACY_POLICY_FIELDS,
        note="Only config.py/runtime.py capacity-registration additions differ between source generations. These two modules are copied from the trained legacy snapshot. All remaining shared training/evaluation source, PPO, environment, runtime and realized physics must match the completed reference.",
    )
    queue.write_json(output / "plan.json", plan)
    queue.write_json(output / "suite.json", scope())
    return plan


def load_context(output):
    output = Path(output).resolve()
    plan = metrics.load_json(output / "plan.json")
    require(
        plan.get("suite_kind") == KIND and plan.get("scope") == scope(), "Residual robustness scope changed"
    )
    require(metrics.load_json(output / "suite.json") == scope(), "Residual robustness registration changed")
    require(
        plan["declared_training_recipe_fields"] == LEGACY_POLICY_FIELDS, "Declared recipe comparison changed"
    )
    queue.verify_snapshot(output, plan)
    for module in (__file__, base.__file__, queue.__file__, controls.__file__, metrics.__file__):
        require(
            plan["source_files"].get("integration/" + Path(module).name) == queue.sha256(module),
            "Use the frozen source/integration/run_residual_robustness.py",
        )
    for key in (
        "residual_plan",
        "reference_plan",
        "reference_summary",
        "checkpoint",
        "training_manifest",
        "training_completion",
    ):
        require(identity(plan[key]["path"]) == plan[key], "Pinned input changed: " + key)
    residual, reference = Path(plan["residual_suite"]), Path(plan["reference_suite"])
    residual_plan = metrics.load_json(residual / "plan.json")
    require(plan["residual_plan"] == identity(residual / "plan.json"), "Residual plan path differs")
    require(plan["reference_plan"] == identity(reference / "plan.json"), "Reference plan path differs")
    require(
        plan["reference_summary"] == identity(reference / "robustness_summary.json"),
        "Reference summary path differs",
    )
    queue.verify_snapshot(residual, residual_plan)
    controls.validate_suite(residual, metrics.load_json(residual / "suite.json"), controls.RESIDUAL_SPEC)
    require(
        plan["residual_registration"] == controls.planned_suite(residual, controls.RESIDUAL_SPEC),
        "Residual registration differs",
    )
    bound = training(residual, residual_plan)
    require(
        plan["checkpoint"] == bound["checkpoint"]
        and plan["training_manifest"] == identity(bound["manifest_path"])
        and plan["training_completion"] == bound["marker"],
        "Training binding changed",
    )
    bridge = audit_source_bridge(residual, reference)
    require(plan["source_bridge"] == bridge, "Audited source bridge changed")
    for relative, pair in bridge.items():
        require(plan["source_files"][relative] == pair["legacy"]["sha256"], "Legacy source overlay changed")
    reference_plan, bindings, signatures, evidence = reference_evidence(reference)
    require(plan["reference_results"] == evidence, "Pinned reference results changed")
    require(
        plan["python"] == reference_plan["python"] == residual_plan["python"]
        and plan["device"] == reference_plan["device"] == residual_plan["device"],
        "Runtime selection changed",
    )
    for other in bindings.values():
        require(
            comparable_signature(bound["training_signature"], True, True)
            == comparable_signature(other["training_signature"], True),
            "Residual/reference training protocol differs",
        )
    return plan, bound, signatures


def validate_output_scope(output):
    for path in (Path(output) / "evaluations").glob("*/*/*.json"):
        relative = path.relative_to(Path(output) / "evaluations")
        require(
            relative.parts[0] in VARIANTS
            and relative.parts[1] in CONDITIONS
            and relative.name in {"eval_" + str(seed) + ".json" for seed in EVALUATION_SEEDS},
            "Result outside the authorized residual robustness scope",
        )
    for path in (Path(output) / "evaluations").glob("*"):
        require(path.name in VARIANTS, "Only legacy-prism evaluation is authorized")


def validate_evaluation(output, plan, bound, condition, seed, path):
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
        {"evaluation": base.evaluation_spec(condition)},
        controls.RESIDUAL_SPEC,
    )
    args = meta["evaluation_arguments"]
    require(args["robustness_condition"] == condition, "Wrong robustness CLI condition")
    require(meta["push_interval_s"] is None and meta["max_push_vel_xy"] is None, "Unexpected push override")
    require(args["sim_device"] == args["rl_device"] == plan["device"], "Evaluation device differs")
    return values, signature, base.validate_physics(meta, condition)


def build_report(output):
    plan, bound, reference = load_context(output)
    validate_output_scope(output)
    records, missing, nominal = [], [], {}
    for condition in CONDITIONS:
        rows, physics, comparison = {}, None, None
        for seed in EVALUATION_SEEDS:
            path = base.destination(output, "legacy-prism", condition, seed)
            if not path.exists():
                missing.append(condition + ": evaluation " + str(seed) + " pending")
                continue
            values, signature, realized = validate_evaluation(output, plan, bound, condition, seed, path)
            expected = reference[(condition, seed)]
            require(
                comparable_signature(signature) == comparable_signature(expected["signature"]),
                "Evaluation protocol differs from reference",
            )
            require(realized == expected["realized"], "Realized physics differ from reference")
            if condition == "nominal":
                nominal[seed] = realized
            else:
                require(seed in nominal, "Stress result needs its nominal reference")
                if condition == "payload_mass":
                    comparison = base.validate_mass_scaling(nominal[seed], realized)
                    require(
                        realized["first_actor"]["shape_friction"]
                        == nominal[seed]["first_actor"]["shape_friction"],
                        "Mass stress changed friction",
                    )
                else:
                    require(
                        realized["first_actor"]["bodies"] == nominal[seed]["first_actor"]["bodies"],
                        "Low friction changed body physics",
                    )
                    comparison = {"body_properties_unchanged": True, "friction_verified": True}
            require(physics is None or physics == realized, "Realized physics differ across evaluation seeds")
            physics = realized
            rows[str(seed)] = {"metrics": values, "result": identity(path)}
        average = (
            {key: statistics.mean(row["metrics"][key] for row in rows.values()) for key in metrics.METRICS}
            if len(rows) == 3
            else None
        )
        records.append(
            {
                "variant": "legacy-prism",
                "condition": condition,
                "training_seed": 1,
                "parameters": controls.RESIDUAL_SPEC.methods["legacy-prism"],
                "checkpoint": bound["checkpoint"],
                "training_manifest": identity(bound["manifest_path"]),
                "evaluations": rows,
                "metrics": average,
                "realized_dynamics": physics,
                "condition_comparison": comparison,
            }
        )
    return {
        **scope(),
        "status": "pending" if missing else "complete",
        "created_utc": queue.now(),
        "verified_evaluations": sum(len(row["evaluations"]) for row in records),
        "aggregation": "Mean over three evaluation seeds, each 200 balanced episodes. One fixed training seed; no training-seed SD.",
        "records": records,
        "missing": missing,
        "plan": identity(Path(output) / "plan.json"),
        "source_bridge": plan["source_bridge"],
        "declared_training_recipe_fields": LEGACY_POLICY_FIELDS,
        "reference_summary": plan["reference_summary"],
    }


def write_report(output, require_complete=False):
    report = build_report(output)
    queue.write_json(Path(output) / "robustness_summary.json", report)
    lines = [
        "# Fresh residual PRISM seed 1 robustness",
        "",
        report["aggregation"],
        "",
        "Status: " + report["status"],
        "",
        "| Condition | Evaluations / 3 | Episode length | Linear error | Yaw error | Survival (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["records"]:
        values = row["metrics"]
        cells = [row["condition"], str(len(row["evaluations"]))]
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
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Mass stress scales every link by 1.15; it is not an attached payload. The original residual actor uses a scalar warmup, not the separate learned-alpha diagnostic.",
        "",
    ]
    (Path(output) / "robustness_summary.md").write_text("\n".join(lines))
    write_comparison(output, report)
    require(not require_complete or report["status"] == "complete", "Residual robustness report is pending")
    return report


def write_comparison(output, report):
    """Extend the four-model table without editing or pooling its reference."""
    reference_identity = report["reference_summary"]
    require(identity(reference_identity["path"]) == reference_identity, "Pinned reference summary changed")
    reference = metrics.load_json(reference_identity["path"])
    require(
        reference["status"] == "complete" and reference["verified_evaluations"] == 36,
        "Reference table is incomplete",
    )
    extension_identity = identity(Path(output) / "robustness_summary.json")
    rows = []
    for origin, summary, source in (
        ("reference", reference, reference_identity),
        ("residual_extension", report, extension_identity),
    ):
        for record in summary["records"]:
            rows.append(
                {
                    "variant": record["variant"],
                    "condition": record["condition"],
                    "training_seed": 1,
                    "parameters": record["parameters"],
                    "checkpoint": record["checkpoint"],
                    "training_manifest": record["training_manifest"],
                    "metrics": record["metrics"],
                    "evaluations": record["evaluations"],
                    "origin": origin,
                    "summary_source": source,
                }
            )
    require(
        len(rows) == 15 and len({(row["variant"], row["condition"]) for row in rows}) == 15,
        "Comparison must contain five models and three conditions",
    )
    matrix = []
    for variant in tuple(base.VARIANTS) + VARIANTS:
        row = {"variant": variant, "training_seed": 1}
        for condition in CONDITIONS:
            result = next(
                item for item in rows if item["variant"] == variant and item["condition"] == condition
            )
            row[condition] = None if result["metrics"] is None else 100 * result["metrics"]["success_rate"]
        matrix.append(row)
    combined = {
        "suite_kind": "g1_five_model_seed1_robustness_comparison_v1",
        "created_utc": queue.now(),
        "status": report["status"],
        "training_seed": 1,
        "verified_evaluations": 36 + report["verified_evaluations"],
        "planned_evaluations": 45,
        "evaluation_episodes_per_model_condition": 600,
        "aggregation": report["aggregation"],
        "interpretation": "Five models, one training seed each. No pooled training-seed inference. Every link mass is multiplied by 1.15 at fixed geometry; this is not an attached payload.",
        "reference_summary": reference_identity,
        "residual_summary": extension_identity,
        "records": rows,
        "survival_percent": matrix,
    }
    queue.write_json(Path(output) / "comparison_summary.json", combined)
    lines = [
        "# Five-model seed 1 robustness comparison",
        "",
        report["aggregation"],
        "",
        "Verified evaluations: " + str(combined["verified_evaluations"]) + "/45",
        "",
        "| Model | Nominal survival (%) | Friction 0.20 (%) | Uniform link mass +15% (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in matrix:
        cells = [row["variant"]] + [
            "pending" if row[key] is None else "{:.2f}".format(row[key]) for key in CONDITIONS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Model | Condition | Evaluations / 3 | Episode length | Linear error | Yaw error | Survival (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
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
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", combined["interpretation"], ""]
    (Path(output) / "comparison_summary.md").write_text("\n".join(lines))
    return combined


def execute(output, variants=VARIANTS):
    require(tuple(variants) == VARIANTS, "Only legacy-prism is authorized")
    plan, bound, _ = load_context(output)
    environment = dict(os.environ, **plan["environment"])
    with (Path(output) / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = Path(output) / "queue_state.json"
        state = metrics.load_json(state_path)
        state.update(status="running", queue_pid=os.getpid())
        queue.write_json(state_path, state)
        try:
            write_report(output)
            for condition in CONDITIONS:
                for seed in EVALUATION_SEEDS:
                    path = base.destination(output, "legacy-prism", condition, seed)
                    if path.exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    command = queue.evaluation_command(
                        plan["python"],
                        Path(output) / "source/integration",
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
                    job = state.setdefault("jobs", {}).setdefault(condition + "/" + str(seed), {})
                    logfile = path.with_suffix(".log")
                    # Preserve failed-attempt logs when safely resuming a job
                    # that produced no result. Never overwrite an earlier log.
                    attempt = 1
                    while logfile.exists():
                        logfile = path.with_name(path.stem + ".attempt" + str(attempt) + ".log")
                        attempt += 1
                    queue.execute_command(command, logfile, job, state, state_path, environment, "evaluation")
                    require(path.is_file(), "Evaluation exited without a result")
                    write_report(output)
                    job["phase"] = "complete"
                    queue.write_json(state_path, state)
            report = write_report(output, require_complete=True)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            state.update(queue_pid=None, updated_utc=queue.now())
            queue.write_json(state_path, state)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residual-suite", type=Path)
    parser.add_argument("--reference-robustness-suite", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "dry-run", "execute", "report"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    require(tuple(args.variants) == VARIANTS, "Only legacy-prism is authorized")
    output = args.output_dir.resolve()
    if args.prepare:
        require(
            args.residual_suite is not None and args.reference_robustness_suite is not None,
            "--prepare needs --residual-suite and --reference-robustness-suite",
        )
        prepare(output, args.residual_suite, args.reference_robustness_suite)
        result = scope()
    elif args.execute:
        result = execute(output, args.variants)
    elif args.dry_run:
        result = build_report(output)
    else:
        result = write_report(output, args.require_complete)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
