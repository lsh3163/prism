"""Run an explicit simulation-only degree/capacity/seed matrix."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from runtime import DEFAULT_VARIANTS, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--max-iterations", type=int, default=3001)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--prefix", default="prism_g1")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="Execute the printed training matrix")
    execution.add_argument("--dry-run", action="store_true", help="Preview commands (the default)")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("max-iterations and num-envs must be positive")
    for seed in args.seeds:
        for variant in args.variants:
            command = [
                sys.executable,
                str(Path(__file__).with_name("train.py")),
                "--variant=" + variant,
                "--seed=" + str(seed),
                "--max_iterations=" + str(args.max_iterations),
                "--num_envs=" + str(args.num_envs),
                "--experiment_name={}_seed{}_{}".format(args.prefix, seed, variant),
                "--headless",
            ]
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
