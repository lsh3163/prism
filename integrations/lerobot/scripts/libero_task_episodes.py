#!/usr/bin/env python3

"""Resolve LIBERO suite task ids to HuggingFaceVLA/libero episode indices."""

from __future__ import annotations

import argparse
import json

from common import SUITES


def resolve_task(repo_id: str, suite_name: str, task_id: int) -> dict:
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.envs.libero import _get_suite

    meta = LeRobotDatasetMetadata(repo_id)
    suite = _get_suite(suite_name)
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} is out of range for {suite_name} ({len(suite.tasks)} tasks)")
    task_description = suite.get_task(task_id).language
    if task_description not in meta.tasks.index:
        raise ValueError(f"Task description not found in dataset metadata: {task_description!r}")
    dataset_task_index = int(meta.tasks.loc[task_description].task_index)
    episodes: list[int] = []
    for episode in meta.episodes:
        if task_description in episode["tasks"]:
            episodes.append(int(episode["episode_index"]))
    if not episodes:
        raise ValueError(f"No episodes found for task {suite_name}/{task_id}: {task_description!r}")
    return {
        "suite": suite_name,
        "task_id": task_id,
        "task": task_description,
        "dataset_task_index": dataset_task_index,
        "episodes": episodes,
        "num_episodes": len(episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args()

    result = resolve_task(args.repo_id, args.suite, args.task_id)
    if args.format == "csv":
        print(",".join(str(ep) for ep in result["episodes"]))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
