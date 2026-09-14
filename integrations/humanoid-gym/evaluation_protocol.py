"""Portable episode selection and terminal-state capture for G1 evaluation.

This module imports neither PyTorch nor the simulator and consumes no randomness.
"""

import math
import statistics
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_IDS = {
    "balanced": "g1_balanced_timeout_v2",
    "legacy-pooled": "g1_legacy_pooled_length_v1",
}


def episode_quotas(num_episodes: int, num_envs: int, protocol: str = "balanced") -> Optional[List[int]]:
    """Require equal, positive quotas instead of assigning extra episodes by env ID."""
    for name, value in (("num_episodes", num_episodes), ("num_envs", num_envs)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if protocol not in PROTOCOL_IDS:
        raise ValueError("Unknown evaluation protocol: " + protocol)
    if protocol == "legacy-pooled":
        return None
    if num_episodes < num_envs or num_episodes % num_envs:
        raise ValueError("Balanced evaluation requires episodes >= num_envs and divisible by num_envs")
    return [num_episodes // num_envs] * num_envs


class EpisodeCollector:
    """Select completed episodes without allowing fast failures to fill other quotas."""

    def __init__(
        self,
        num_envs: int,
        num_episodes: int,
        max_episode_length: int,
        protocol: str = "balanced",
        control_dt: float = 0.01,
    ) -> None:
        self.quotas = episode_quotas(num_episodes, num_envs, protocol)
        if max_episode_length < 1 or not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("Episode horizon and control timestep must be positive")
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.protocol = protocol
        self.control_dt = control_dt
        self.completed_counts = [0] * num_envs
        self.counted_counts = [0] * num_envs
        self.episodes = []  # type: List[Dict[str, Any]]
        self.events = []  # type: List[Dict[str, Any]]

    @property
    def complete(self) -> bool:
        return len(self.episodes) == self.num_episodes

    def add_episode(
        self,
        env_id: int,
        completion_step: int,
        length: int,
        episode_return: float,
        avg_lin_vel_error: float,
        avg_yaw_vel_error: float,
        timeout: bool,
    ) -> bool:
        """Record every terminal event; return whether it contributes to the result."""
        if not 0 <= env_id < self.num_envs or length < 1 or completion_step < 1:
            raise ValueError("Invalid environment ID or episode length/step")
        if not all(math.isfinite(value) for value in (episode_return, avg_lin_vel_error, avg_yaw_vel_error)):
            raise ValueError("Non-finite episode metrics")
        self.completed_counts[env_id] += 1
        counted = not self.complete and (
            self.quotas is None or self.counted_counts[env_id] < self.quotas[env_id]
        )
        success = bool(timeout) if self.protocol == "balanced" else length >= self.max_episode_length - 1
        event = {
            "env_id": env_id,
            "episode_ordinal": self.completed_counts[env_id],
            "completion_step": completion_step,
            "length": length,
            "length_seconds": length * self.control_dt,
            "return": episode_return,
            "avg_lin_vel_error": avg_lin_vel_error,
            "avg_yaw_vel_error": avg_yaw_vel_error,
            "timeout": bool(timeout),
            "success": success,
            "counted": counted,
        }
        self.events.append(event)
        if counted:
            self.counted_counts[env_id] += 1
            self.episodes.append(event)
        return counted

    def summary(self) -> Dict[str, Any]:
        """Return episode means and population standard deviations, with raw records."""
        if not self.complete:
            raise RuntimeError("Evaluation is incomplete: episode quotas have not been met")
        protocol = {
            "id": PROTOCOL_IDS[self.protocol],
            "version": 2 if self.protocol == "balanced" else 1,
            "episode_allocation": "equal_per_environment_quota"
            if self.protocol == "balanced"
            else "first_completed_global_pool",
            "episode_quotas": self.quotas,
            "success_definition": "environment_timeout"
            if self.protocol == "balanced"
            else "length_at_least_max_episode_length_minus_one",
            "tracking_error_timing": "before_terminal_reset"
            if self.protocol == "balanced"
            else "legacy_after_environment_step_and_reset",
            "max_episode_length": self.max_episode_length,
            "control_dt": self.control_dt,
        }
        result = {
            "protocol_id": protocol["id"],
            "protocol": protocol,
            "num_envs": self.num_envs,
            "num_episodes": len(self.episodes),
            "success_rate": statistics.mean(float(episode["success"]) for episode in self.episodes),
            "episode_counts_by_environment": list(self.counted_counts),
            "completed_episode_counts_by_environment": list(self.completed_counts),
            "episodes": list(self.episodes),
            "episode_events": list(self.events),
        }
        for source, target in (
            ("return", "return"),
            ("length", "episode_length"),
            ("avg_lin_vel_error", "lin_vel_error"),
            ("avg_yaw_vel_error", "yaw_vel_error"),
        ):
            values = [episode[source] for episode in self.episodes]
            result["avg_" + target] = statistics.mean(values)
            result["std_" + target] = statistics.pstdev(values)
        return result


class BeforeResetCapture:
    """Temporarily observe each step immediately before the environment resets.

    The released environment calls ``reset_idx`` once per step, including when
    its ID tensor is empty. A missing or duplicate callback fails explicitly.
    The callback must copy values that the reset mutates; it must not draw RNGs.
    """

    def __init__(self, env: Any, capture: Callable[[], Any]) -> None:
        self.env = env
        self.capture = capture
        self.calls = 0
        self.value = None

    def __enter__(self) -> "BeforeResetCapture":
        self.had_override = "reset_idx" in vars(self.env)
        self.original = self.env.reset_idx
        self.env.reset_idx = self._before_reset
        return self

    def _before_reset(self, env_ids: Any) -> Any:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected exactly one reset callback per evaluation step")
        self.value = self.capture()
        return self.original(env_ids)

    def begin_step(self) -> None:
        self.calls = 0
        self.value = None

    def snapshot(self) -> Any:
        if self.calls != 1:
            raise RuntimeError("Environment did not expose its pre-reset evaluation state")
        return self.value

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.had_override:
            self.env.reset_idx = self.original
        else:
            delattr(self.env, "reset_idx")
