"""Print explicit BFM-Zero evaluation scenarios; use --execute to run them.

These settings implement the dynamics described in REPRODUCIBILITY.md. They do
not establish which overrides produced an archived result without its metadata.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

SCENARIOS = ("nominal", "low_friction", "payload_mass")


def validate_execution(bfm_root: Path, model_folder: Path, data_path: Path, output_names: list[str]) -> None:
    """Validate the entire requested sweep before the first simulator process."""
    if not (bfm_root / "humanoidverse/tracking_eval.py").is_file():
        raise ValueError("Expected a patched BFM-Zero checkout containing humanoidverse/tracking_eval.py")
    if not model_folder.is_dir():
        raise ValueError(f"Model directory does not exist: {model_folder}")
    if not (model_folder / "config.json").is_file() or not (model_folder / "checkpoint").is_dir():
        raise ValueError(f"Model directory must contain config.json and checkpoint/: {model_folder}")
    if not data_path.is_file():
        raise ValueError(f"Motion data does not exist: {data_path}")
    for name in output_names:
        output = model_folder / name
        if output.exists():
            raise ValueError(f"Evaluation output already exists: {output}")


def scenario_overrides(scenario: str) -> list[str]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    settings = {
        "domain_rand.push_robots": "false",
        "domain_rand.randomize_base_com": "false",
        "domain_rand.randomize_link_mass": "false",
        "domain_rand.randomize_pd_gain": "false",
        "domain_rand.randomize_friction": "false",
        "domain_rand.randomize_base_mass": "false",
        "domain_rand.randomize_torque_rfi": "false",
        "domain_rand.randomize_rfi_lim": "false",
        "domain_rand.randomize_ctrl_delay": "false",
        "domain_rand.randomize_default_dof_pos": "false",
    }
    if scenario == "low_friction":
        settings.update(
            {
                "domain_rand.randomize_friction": "true",
                "domain_rand.friction_range": "[0.2,0.2]",
                "terrain.static_friction": "0.2",
                "terrain.dynamic_friction": "0.2",
            }
        )
    elif scenario == "payload_mass":
        settings.update(
            {
                "domain_rand.randomize_link_mass": "true",
                "domain_rand.link_mass_range": "[1.15,1.15]",
            }
        )
    return [f"{key}={value}" for key, value in settings.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-root", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--output-prefix", default="tracking_eval_release")
    parser.add_argument("--disable-obs-noise", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a nonempty directory name")
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    if args.execute:
        try:
            validate_execution(
                args.bfm_root,
                args.model_folder,
                args.data_path,
                [f"{args.output_prefix}_{scenario}" for scenario in scenarios],
            )
        except ValueError as error:
            parser.error(str(error))
    for scenario in scenarios:
        output_name = f"{args.output_prefix}_{scenario}"
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "humanoidverse.tracking_eval",
            f"--model-folder={args.model_folder.resolve()}",
            f"--data-path={args.data_path.resolve()}",
            f"--num-envs={args.num_envs}",
            f"--output-subdir={output_name}",
            f"--eval-log-name=humanoidverse_{scenario}",
            # The upstream blanket switch is applied AFTER Hydra overrides and
            # would disable the fixed friction/mass events. Control each flag.
            "--no-disable-dr",
            "--disable-obs-noise" if args.disable_obs_noise else "--no-disable-obs-noise",
            f"--hydra-overrides-json={json.dumps(scenario_overrides(scenario))}",
        ]
        print(f"cd {shlex.quote(str(args.bfm_root.resolve()))} && {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=args.bfm_root, check=True)


if __name__ == "__main__":
    main()
