# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI component profiles and memory bounds for RoLink placement."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _numeric_table(raw: dict[str, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in raw.items()}


def interpolate(table: dict[int, float], size: float) -> float:
    """Linearly interpolate and proportionally extrapolate a profile table."""
    if size <= 0:
        return 0.0
    if not table:
        raise ValueError("profile table is empty")
    points = sorted(table)
    if size <= points[0]:
        return table[points[0]]
    if size >= points[-1]:
        return table[points[-1]] * size / points[-1]
    upper_index = next(index for index, point in enumerate(points) if point >= size)
    lower, upper = points[upper_index - 1], points[upper_index]
    ratio = (size - lower) / (upper - lower)
    return table[lower] + ratio * (table[upper] - table[lower])


def inverse_batch_interpolate(table: dict[int, float], size: float) -> float:
    """Interpolate Actor time linearly in samples-per-rank (1 / GPU count)."""
    if size <= 0 or not table:
        raise ValueError("Actor profile size and table must be positive")
    points = sorted(table)
    if len(points) == 1:
        reference = points[0]
        return table[reference] * reference / size
    if size <= points[0]:
        lower, upper = points[0], points[1]
    elif size >= points[-1]:
        lower, upper = points[-2], points[-1]
    else:
        upper_index = next(index for index, point in enumerate(points) if point >= size)
        lower, upper = points[upper_index - 1], points[upper_index]
    x = 1.0 / size
    x_lower = 1.0 / lower
    x_upper = 1.0 / upper
    ratio = (x - x_lower) / (x_upper - x_lower)
    return max(0.0, table[lower] + ratio * (table[upper] - table[lower]))


@dataclass(frozen=True)
class MemoryPoint:
    static_mib: float
    peak_mib: float


def memory_at(raw: dict[str, Any], size: float) -> MemoryPoint:
    static = _numeric_table({key: value["static_mib"] for key, value in raw.items()})
    peak = _numeric_table({key: value["peak_mib"] for key, value in raw.items()})
    return MemoryPoint(interpolate(static, size), interpolate(peak, size))


@dataclass(frozen=True)
class OpenPIProfile:
    source: str
    env_time_s: dict[int, float]
    env_memory: dict[str, Any]
    infer_time_s: dict[str, dict[int, float]]
    infer_memory: dict[str, dict[str, Any]]
    actor_single_gpu_compute_s: float
    actor_scaling: str
    actor_reference_gpus: int
    actor_total_time_s: dict[int, float]
    actor_communication_s: dict[int, float]
    actor_peak_mib: dict[int, float]
    weight_sync_s: dict[int, float]
    gpu_limit_mib: dict[str, float]
    observation_mib: float
    trajectory_mib: float

    @classmethod
    def load(cls, path: Path) -> "OpenPIProfile":
        raw = json.loads(path.read_text())
        return cls(
            source=str(raw.get("source", "unknown")),
            env_time_s=_numeric_table(raw["env"]["time_s"]),
            env_memory=raw["env"]["memory_mib"],
            infer_time_s={
                tier: _numeric_table(values["time_s"])
                for tier, values in raw["inference"].items()
            },
            infer_memory={
                tier: values["memory_mib"] for tier, values in raw["inference"].items()
            },
            actor_single_gpu_compute_s=float(raw["actor"]["single_gpu_compute_s"]),
            actor_scaling=str(raw["actor"].get("scaling", "profile_table")),
            actor_reference_gpus=int(raw["actor"].get("reference_gpus", 1)),
            actor_total_time_s=_numeric_table(raw["actor"].get("total_time_s", {})),
            actor_communication_s=_numeric_table(raw["actor"]["communication_s"]),
            actor_peak_mib=_numeric_table(raw["actor"]["peak_mib"]),
            weight_sync_s=_numeric_table(raw["weight_sync"]["time_s"]),
            gpu_limit_mib={
                key: float(value) for key, value in raw["gpu_memory_limit_mib"].items()
            },
            observation_mib=float(raw["network"]["observation_mib"]),
            trajectory_mib=float(
                raw["network"].get("trajectory_mib", raw["network"]["observation_mib"])
            ),
        )

    def env_time(self, local_envs: int) -> float:
        return interpolate(self.env_time_s, local_envs)

    def colocated_infer_cap(self, local_envs: int) -> int:
        """Largest profiled batch satisfying the static/peak overlap bound."""
        env_mem = memory_at(self.env_memory, local_envs)
        limit = self.gpu_limit_mib["edge"]
        feasible = []
        for batch in sorted(self.infer_time_s["edge"]):
            infer_mem = memory_at(self.infer_memory["edge"], batch)
            overlap_peak = max(
                env_mem.static_mib + infer_mem.peak_mib,
                env_mem.peak_mib + infer_mem.static_mib,
            )
            if overlap_peak < limit:
                feasible.append(batch)
        return max(feasible, default=0)

    def actor_time(self, actor_gpus: int) -> float:
        if self.actor_scaling == "inverse_batch_interpolation":
            return inverse_batch_interpolate(self.actor_total_time_s, actor_gpus)
        if self.actor_scaling == "inverse_from_reference":
            reference = self.actor_reference_gpus
            if reference not in self.actor_total_time_s:
                raise ValueError(
                    f"actor reference size {reference} is missing from total_time_s"
                )
            return self.actor_total_time_s[reference] * reference / actor_gpus
        if self.actor_total_time_s:
            points = sorted(self.actor_total_time_s)
            if actor_gpus < points[0]:
                reference = points[0]
                return self.actor_total_time_s[reference] * reference / actor_gpus
            if actor_gpus > points[-1]:
                reference = points[-1]
                return self.actor_total_time_s[reference] * reference / actor_gpus
            return interpolate(self.actor_total_time_s, actor_gpus)
        return self.actor_single_gpu_compute_s / actor_gpus + interpolate(
            self.actor_communication_s, actor_gpus
        )

    def actor_fits(self, actor_gpus: int) -> bool:
        if self.actor_scaling == "inverse_batch_interpolation":
            return max(self.actor_peak_mib.values()) < self.gpu_limit_mib["cloud"]
        if self.actor_scaling == "inverse_from_reference":
            peak = self.actor_peak_mib[self.actor_reference_gpus]
            return peak < self.gpu_limit_mib["cloud"]
        return (
            interpolate(self.actor_peak_mib, actor_gpus) < self.gpu_limit_mib["cloud"]
        )
