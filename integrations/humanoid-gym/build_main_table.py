"""Build the five-training-seed G1 table from bound training/evaluation records.

Standard library only. Missing runs produce a pending table; contradictory
records fail validation. No simulator is imported and no results are invented.
"""
# ruff: noqa: B905 -- also runnable in the simulator's Python 3.8 environment.

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

VARIANTS = ("baseline", "larger", "prism")
TRAINING_SEEDS = (1, 2, 3, 4, 5)
PROTOCOL_ID = "g1_balanced_timeout_v2"
METRICS = ("avg_episode_length", "avg_lin_vel_error", "avg_yaw_vel_error", "success_rate")
HIDDEN_DIMS = {"baseline": [512, 256, 128], "larger": [648, 328, 160], "prism": [512, 256, 128]}
TASKS = {
    "baseline": "g1_humanoidgym_ppo",
    "larger": "g1_humanoidgym_ppo_larger_v2",
    "prism": "g1_humanoidgym_ppo_gated_d2",
}
TRAIN_CODE = {
    "train.py",
    "runtime.py",
    "config.py",
    "actor.py",
    "gated_actor.py",
    "g1_env.py",
    "legacy_base.py",
    "checkpoints.py",
    "provenance.py",
}
IDENTIFIER_ARGS = {
    "seed",
    "task",
    "variant",
    "variant_name",
    "experiment_name",
    "run_name",
    "save_path",
    "model_path",
    "load_run",
    "checkpoint",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def mlp_parameters(dimensions):
    return sum(a * b + b for a, b in zip(dimensions, dimensions[1:]))


def expected_counts():
    """Derive counts from the released architecture, including critic and std."""
    critic_and_std = mlp_parameters([219, 768, 256, 128, 1]) + 12
    actor = {
        "baseline": mlp_parameters([705, 512, 256, 128, 12]),
        "larger": mlp_parameters([705, 648, 328, 160, 12]),
        # Two affine factors, one trainable alpha vector, projection and head.
        "prism": 2 * (705 * 256 + 256) + 256 + 256 * 256 + 256 + mlp_parameters([256, 512, 256, 128, 12]),
    }
    return {
        name: {"actor_mean_parameters": count, "total_actor_critic_parameters": count + critic_and_std}
        for name, count in actor.items()
    }


def source_signature(source, include_evaluator=False):
    signatures = {}
    for item in source["files"]:
        path = Path(item["path"])
        if "legged_gym" in path.parts or "rsl_rl" in path.parts:
            index = next(i for i, part in enumerate(path.parts) if part in {"legged_gym", "rsl_rl"})
            key = "/".join(path.parts[index:])
        elif path.name in TRAIN_CODE or (
            include_evaluator and path.name in {"evaluate.py", "evaluation_protocol.py"}
        ):
            key = "integration/" + path.name
        else:
            continue
        require(key not in signatures, "Ambiguous duplicate source identity: " + key)
        require(len(item["sha256"]) == 64, "Invalid source SHA256: " + key)
        signatures[key] = item["sha256"]
    needed = TRAIN_CODE | ({"evaluate.py", "evaluation_protocol.py"} if include_evaluator else set())
    require(all("integration/" + name in signatures for name in needed), "Incomplete source snapshot")
    return signatures


def runtime_signature(runtime):
    require(
        all(key in runtime for key in ("python", "torch", "numpy", "isaacgym", "torch_cuda")),
        "Incomplete runtime provenance",
    )
    return {key: value for key, value in runtime.items() if key != "executable"}


def without_seed(config):
    result = copy.deepcopy(config)
    result.pop("seed", None)
    return result


def arguments_signature(arguments):
    require("physics_engine" in arguments, "Missing simulator arguments")
    return {key: value for key, value in arguments.items() if key not in IDENTIFIER_ARGS}


def training_signature(manifest):
    config = copy.deepcopy(manifest["training_config"])
    config.pop("seed")
    for key in ("actor_hidden_dims", "actor_variant", "poly_degree", "poly_hidden_dim", "gate_init"):
        config["policy"].pop(key, None)
    for key in ("policy_class_name", "experiment_name", "run_name", "load_run", "checkpoint", "resume_path"):
        config["runner"].pop(key, None)
    return {
        "training_config": config,
        "environment_config": without_seed(manifest["environment_config"]),
        "arguments": arguments_signature(manifest["arguments"]),
        "source": source_signature(manifest["source"]),
        "runtime": runtime_signature(manifest["runtime"]),
    }


def check_actor(actor, variant):
    gated = variant == "prism"
    require(actor["class"] == ("GatedPolyActorCritic" if gated else "ActorCritic"), "Wrong policy class")
    require(actor["actor_variant"] == ("g1_gated_poly_v2" if gated else None), "Wrong actor variant")
    for key, expected in expected_counts()[variant].items():
        require(actor[key] == expected, "Wrong parameter count: " + key)


def validate_training(manifest, variant, seed, suite):
    require(
        manifest["schema_version"] == 1 and manifest["record_kind"] == "training_start",
        "Unsupported training manifest",
    )
    require(
        manifest["seed"] == seed and manifest["training_config"]["seed"] == seed,
        "Training seed must come from the saved manifest, not a folder name",
    )
    require(manifest["environment_config"]["seed"] == seed, "Training environment seed differs")
    require(
        manifest["initial_learning_iteration"] == 0 and manifest.get("resume_checkpoint") is None,
        "Main table requires independent fresh training, not resumed checkpoints",
    )
    require(manifest["task"] == TASKS[variant], "Wrong training task")
    actor, config = manifest["actor"], manifest["training_config"]
    check_actor(actor, variant)
    require(actor["recipe_variant"] == variant, "Wrong training recipe")
    require(actor["actor_hidden_dims"] == HIDDEN_DIMS[variant], "Wrong actor dimensions")
    require(config["policy"]["actor_hidden_dims"] == HIDDEN_DIMS[variant], "Wrong saved actor dimensions")
    require(config["policy"]["critic_hidden_dims"] == [768, 256, 128], "Wrong critic dimensions")
    require(config["policy"]["activation"] == "elu", "Wrong policy activation")
    require(config["runner"]["policy_class_name"] == actor["class"], "Class/config mismatch")
    require(config["runner"]["max_iterations"] == suite["max_iterations"], "Training budget differs")
    require(config["runner"].get("resume", False) is False, "Resumed run is not an independent fresh seed")
    require(
        manifest["environment_config"]["env"]["num_envs"] == suite["num_envs"],
        "Training environment count differs",
    )
    require(
        manifest["environment_config"]["env"]["episode_length_s"] == 24,
        "Main locomotion table requires the 24-second horizon",
    )
    if variant == "prism":
        require(
            actor["polynomial_degree"] == 2
            and actor["gate_semantics"] == "learned_per_feature"
            and actor["gate_init"] == 0.01
            and actor["warmup_ppo_updates"] == 0,
            "Main PRISM must use degree-two learnable alpha initialized at 0.01",
        )
        require(
            config["policy"]["actor_variant"] == "g1_gated_poly_v2"
            and config["policy"]["poly_degree"] == 2
            and config["policy"]["poly_hidden_dim"] == 256
            and config["policy"]["gate_init"] == 0.01,
            "Wrong gated policy configuration",
        )
    return training_signature(manifest)


def verify_episode_metrics(result, expected):
    protocol = result["protocol"]
    require(result["protocol_id"] == protocol["id"] == PROTOCOL_ID, "Wrong evaluation protocol")
    require(protocol["version"] == 2, "Wrong evaluation protocol version")
    require(
        protocol["episode_allocation"] == "equal_per_environment_quota"
        and protocol["success_definition"] == "environment_timeout"
        and protocol["tracking_error_timing"] == "before_terminal_reset",
        "Wrong metric semantics",
    )
    episodes, count, num_envs = result["episodes"], expected["episodes"], expected["num_envs"]
    require(len(episodes) == result["num_episodes"] == count, "Incomplete episode records")
    require(count % num_envs == 0, "Balanced evaluation requires divisible episode count")
    quota = count // num_envs
    require(protocol["episode_quotas"] == [quota] * num_envs, "Wrong environment quotas")
    require(
        Counter(row["env_id"] for row in episodes) == {i: quota for i in range(num_envs)},
        "Episode allocation is not balanced",
    )
    require(
        len({(row["env_id"], row["episode_ordinal"]) for row in episodes}) == count,
        "Duplicate episode identity",
    )
    for row in episodes:
        require(
            row["success"] in (0, 1, False, True)
            and row["timeout"] in (0, 1, False, True)
            and row["success"] == row["timeout"],
            "Survival must equal environment timeout",
        )
        require(row["length"] > 0, "Invalid episode length")
    aliases = {
        "avg_episode_length": "length",
        "avg_lin_vel_error": "avg_lin_vel_error",
        "avg_yaw_vel_error": "avg_yaw_vel_error",
        "success_rate": "success",
        "avg_return": "return",
    }
    for metric, episode_key in aliases.items():
        mean = statistics.mean(row[episode_key] for row in episodes)
        require(
            math.isfinite(mean)
            and math.isfinite(result[metric])
            and math.isclose(result[metric], mean, rel_tol=1e-6, abs_tol=1e-7),
            "Aggregate disagrees with episode records: " + metric,
        )
    return {metric: result[metric] for metric in METRICS}


def validate_evaluation(payload, path, manifest, manifest_path, variant, seed, suite, hashes):
    require(payload["schema_version"] == 2, "Unsupported evaluation schema")
    meta, result = payload["metadata"], payload["result"]
    require(
        meta["protocol_id"] == result["protocol_id"] and meta["evaluation_protocol"] == result["protocol"],
        "Protocol metadata disagrees",
    )
    require(meta["training_seed"] == seed, "Evaluation training-seed provenance differs")
    require(meta["evaluation_seed"] == meta["seed"], "Evaluation seed aliases disagree")
    eval_seed = meta["evaluation_seed"]
    require(eval_seed in suite["evaluation_seeds"], "Unexpected evaluation seed")
    require(meta["recipe_variant"] == variant and result["task"] == TASKS[variant], "Wrong evaluation recipe")
    binding = meta["training_manifest"]
    require(
        binding is not None and binding["sha256"] == sha256(manifest_path) and binding["payload"] == manifest,
        "Evaluation is not bound to the selected training manifest",
    )
    require(resolve(binding["path"], path.parent) == manifest_path, "Training manifest path differs")
    checkpoint = meta["checkpoint"]
    checkpoint_path = resolve(checkpoint["path"], path.parent)
    require(checkpoint_path.parent == manifest_path.parent, "Checkpoint does not belong to this training run")
    require(checkpoint_path.is_file(), "Bound checkpoint file is missing")
    if checkpoint_path not in hashes:
        hashes[checkpoint_path] = sha256(checkpoint_path)
    actual_hash = hashes[checkpoint_path]
    require(actual_hash == checkpoint["sha256"] == result["checkpoint_sha256"], "Checkpoint hash mismatch")
    require(resolve(result["checkpoint_path"], path.parent) == checkpoint_path, "Checkpoint paths disagree")
    require(meta["checkpoint_iteration"] == suite["checkpoint_iteration"], "Wrong checkpoint budget")
    require(
        result["checkpoint_iteration"] == meta["checkpoint_iteration"],
        "Checkpoint iteration metadata disagrees",
    )
    check_actor(meta["actor"], variant)
    require(
        result["actor_variant"] == meta["actor"]["actor_variant"] == meta["actor_variant"],
        "Evaluation actor identities disagree",
    )
    expected = suite["evaluation"]
    for key in ("episodes", "num_envs", "condition_name", "terrain_mode", "push_mode"):
        require(meta[key] == expected[key], "Evaluation setting differs: " + key)
    config = meta["evaluation_config"]
    require(
        config["seed"] == eval_seed and config["env"]["num_envs"] == expected["num_envs"],
        "Resolved evaluation seed/environment count differs",
    )
    require(config["env"]["episode_length_s"] == 24, "Evaluation horizon differs")
    source = source_signature(meta["source"], include_evaluator=True)
    training_sources = source_signature(manifest["source"])
    require(
        all(
            source.get("integration/" + name) == training_sources["integration/" + name]
            for name in TRAIN_CODE
        ),
        "Training/evaluation implementation differs",
    )
    signature = {
        "environment_config": without_seed(config),
        "arguments": arguments_signature(meta["evaluation_arguments"]),
        "source": source,
        "runtime": runtime_signature(meta["runtime"]),
        "protocol": result["protocol"],
    }
    metrics = verify_episode_metrics(result, expected)
    return eval_seed, metrics, signature, {"path": str(checkpoint_path), "sha256": actual_hash}


def build_table(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = load_json(suite_path)
    require(suite["schema_version"] == 1, "Unsupported suite schema")
    require(suite["training_seeds"] == list(TRAINING_SEEDS), "Require exact training seeds [1,2,3,4,5]")
    eval_seeds = suite["evaluation_seeds"]
    require(
        bool(eval_seeds)
        and all(type(seed) is int and seed >= 0 for seed in eval_seeds)
        and len(set(eval_seeds)) == len(eval_seeds),
        "Evaluation seeds must be explicit and unique",
    )
    require(
        suite["max_iterations"] == suite["checkpoint_iteration"] and suite["max_iterations"] > 0,
        "Main table uses the final matched training budget",
    )
    require(
        suite["num_envs"] > 0 and suite["evaluation"]["protocol_id"] == PROTOCOL_ID, "Invalid suite protocol"
    )
    entries = {}
    for entry in suite["runs"]:
        key = (entry["variant"], entry["training_seed"])
        require(
            key[0] in VARIANTS and type(key[1]) is int and key[1] in TRAINING_SEEDS,
            "Unexpected method or training seed",
        )
        require(key not in entries, "Duplicate method/training-seed entry")
        entries[key] = entry
    records, missing, hashes, manifests, checkpoint_owners = [], [], {}, set(), {}
    shared_training, shared_evaluation = None, None
    for variant in VARIANTS:
        for seed in TRAINING_SEEDS:
            entry = entries.get((variant, seed))
            label = f"{variant}/seed{seed}"
            if not entry or not entry.get("training_manifest"):
                missing.append({"run": label, "reason": "training manifest missing"})
                continue
            manifest_path = resolve(entry["training_manifest"], suite_path.parent)
            if not manifest_path.is_file():
                missing.append({"run": label, "reason": "training manifest file pending"})
                continue
            require(manifest_path not in manifests, "Training manifest reused for independent runs")
            manifests.add(manifest_path)
            manifest = load_json(manifest_path)
            try:
                signature = validate_training(manifest, variant, seed, suite)
                if shared_training is None:
                    shared_training = signature
                require(
                    signature == shared_training, "Training hyperparameters/environment/source/runtime differ"
                )
                eval_metrics, checkpoint = {}, None
                for value in entry.get("evaluations", []):
                    path = resolve(value, suite_path.parent)
                    if not path.is_file():
                        continue
                    payload = load_json(path)
                    eval_seed, metrics, signature, identity = validate_evaluation(
                        payload, path, manifest, manifest_path, variant, seed, suite, hashes
                    )
                    require(eval_seed not in eval_metrics, "Duplicate evaluation seed within a training run")
                    require(
                        checkpoint is None or checkpoint == identity,
                        "Multiple checkpoints for one training seed",
                    )
                    checkpoint = identity
                    if shared_evaluation is None:
                        shared_evaluation = signature
                    require(
                        signature == shared_evaluation,
                        "Evaluation protocol/environment/source/runtime differ",
                    )
                    eval_metrics[eval_seed] = metrics
                if checkpoint:
                    owner = checkpoint_owners.setdefault(checkpoint["sha256"], label)
                    require(owner == label, "Checkpoint reused across independent training runs")
                absent = sorted(set(eval_seeds) - set(eval_metrics))
                if absent:
                    missing.append(
                        {"run": label, "reason": "evaluation seeds pending", "evaluation_seeds": absent}
                    )
                records.append(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "training_manifest": str(manifest_path),
                        "training_manifest_sha256": sha256(manifest_path),
                        "checkpoint": checkpoint,
                        "evaluation_metrics": eval_metrics,
                        "training_seed_metrics": None
                        if absent
                        else {
                            metric: statistics.mean(values[metric] for values in eval_metrics.values())
                            for metric in METRICS
                        },
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{label}: {error}") from error
    complete = not missing
    summaries = {}
    for variant in VARIANTS:
        rows = [
            record["training_seed_metrics"]
            for record in records
            if record["variant"] == variant and record["training_seed_metrics"] is not None
        ]
        summaries[variant] = {
            "verified_training_seeds": len(rows),
            **expected_counts()[variant],
            "metrics": None
            if not complete
            else {
                metric: {
                    "mean": statistics.mean(row[metric] for row in rows),
                    "sample_sd": statistics.stdev(row[metric] for row in rows),
                }
                for metric in METRICS
            },
        }
    return {
        "schema_version": 1,
        "status": "complete" if complete else "pending",
        "suite": {"path": str(suite_path), "sha256": sha256(suite_path)},
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": eval_seeds,
        "verified_training_runs": sum(record["training_seed_metrics"] is not None for record in records),
        "verified_evaluations": sum(len(record["evaluation_metrics"]) for record in records),
        "aggregation": "Equal mean across evaluation seeds within each training run, then mean and sample SD (ddof=1) across five independent training-run records. No pooled-episode SD.",
        "parameter_count_basis": "All trainable actor, critic and Gaussian standard-deviation parameters; buffers excluded. Pending counts describe planned released architectures.",
        "verified_training_signature": shared_training,
        "verified_evaluation_signature": shared_evaluation,
        "records": records,
        "missing": missing,
        "methods": summaries,
    }


def render_latex(report):
    complete = report["status"] == "complete"
    caption = (
        r"Humanoid-Gym~\cite{gu2024humanoid} locomotion with an MLP baseline, "
        r"a parameter-matched larger MLP, and "
        r"\sname{} with learned per-feature polynomial interaction scales. "
        r"Parameters include actor, critic, and learned action standard deviations. "
        r"Episode length is in control steps; survival means reaching the 24\,s environment timeout. "
        r"Each evaluation environment supplies the same number of episodes. "
    )
    caption += (
        r"Values are mean $\pm$ sample standard deviation across five independent training seeds; "
        r"evaluation seeds are averaged within each training seed. Bold indicates the best mean."
        if complete
        else r"Evaluation is pending; dashes mark unavailable five-seed results and parameter counts describe "
        r"the planned architectures."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:locomotion_main}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Poly Interact. & Params & Episode Length $\uparrow$ & Lin. Error $\downarrow$ "
        r"& Yaw Error $\downarrow$ & Survival Rate (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    for variant, label in zip(VARIANTS, ("MLP Baseline", "Larger MLP", r"\sname{}")):
        method = report["methods"][variant]
        if variant == "prism":
            lines.append(r"\midrule")
        cells = []
        for metric, decimals in zip(METRICS, (1, 4, 4, 2)):
            if not complete:
                cells.append("--")
            else:
                values = method["metrics"][metric]
                scale = 100 if metric == "success_rate" else 1
                text = (
                    f"{values['mean'] * scale:.{decimals}f} "
                    + r"\pm "
                    + f"{values['sample_sd'] * scale:.{decimals}f}"
                )
                chooser = max if metric in ("avg_episode_length", "success_rate") else min
                best = chooser(row["metrics"][metric]["mean"] for row in report["methods"].values())
                cells.append("$" + (r"\mathbf{" + text + "}" if values["mean"] == best else text) + "$")
        mark = r"\cmark" if variant == "prism" else r"\xmark"
        params = f"{method['total_actor_critic_parameters'] / 1e6:.3f}M"
        lines.append(" & ".join([label, mark, params, *cells]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}}", r"\vspace{-1em}", r"\end{table}", ""])


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main_table.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output_dir / "main_table.tex").write_text(render_latex(report))
    lines = [
        "# Main locomotion table",
        "",
        "Status: **" + report["status"] + "**.",
        "",
        report["aggregation"],
        "",
        "| Method | Verified training seeds | Parameters |",
        "|---|---:|---:|",
    ]
    for name, row in report["methods"].items():
        lines.append(
            f"| {name} | {row['verified_training_seeds']}/5 | {row['total_actor_critic_parameters']:,} |"
        )
    if report["missing"]:
        lines.extend(["", "No five-seed statistics are printed until all three methods are complete.", ""])
        lines.extend(f"- {row['run']}: {row['reason']}" for row in report["missing"])
    lines.extend(
        [
            "",
            "See main_table.json for exact provenance, missing evaluations, and per-training-seed means.",
            "Complete tables mark the best observed mean, without a causal claim.",
        ]
    )
    (output_dir / "inventory.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write the inventory, then exit 1 if any training/evaluation run is pending",
    )
    args = parser.parse_args()
    try:
        report = build_table(args.suite)
        write_report(report, args.output_dir)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"Invalid table evidence: {error}\n")
    print(f"{report['status']}: {args.output_dir / 'main_table.tex'}")
    if args.require_complete and report["status"] != "complete":
        parser.exit(1, "The five-training-seed matrix is pending.\n")


if __name__ == "__main__":
    main()
