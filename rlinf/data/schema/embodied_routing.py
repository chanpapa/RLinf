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

"""Profile-based batch allocation and outstanding rollout reservations."""

import json
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def payload_nbytes(value: Any) -> int:
    """Estimate payload bytes, including compressed buffers, without serializing."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if is_dataclass(value):
        return sum(payload_nbytes(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, dict):
        return sum(payload_nbytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(payload_nbytes(v) for v in value)
    return 0 if value is None else 8


@dataclass(frozen=True)
class InferenceProfile:
    """Seconds per physical inference batch, bounded by a safe batch capacity."""

    time_s: dict[int, float]
    max_batch_size: int

    def duration(self, size: int) -> float:
        """Interpolate batch latency; extrapolate proportionally above the table."""
        if not size:
            return 0.0
        points = sorted(self.time_s)
        if size <= points[0]:
            return self.time_s[points[0]]
        for lower, upper in zip(points, points[1:]):
            if size <= upper:
                return self.time_s[lower] + (
                    self.time_s[upper] - self.time_s[lower]
                ) * (size - lower) / (upper - lower)
        return self.time_s[points[-1]] * size / points[-1]


@dataclass
class _Reservation:
    ready_at: float
    duration: float
    finish_at: float


class LoadAwareRouter:
    """Minimize estimated cohort completion using profiled batches and link load.

    Traffic follows Channel -> rollout -> Channel. Rollouts in the same remote
    site share its outbound serialization time. Small action payloads contribute
    return propagation only. Reservations model this channel's traffic and work.
    """

    def __init__(
        self,
        profiles: dict[str, InferenceProfile],
        rollout_sites: dict[str, str],
        channel_site: str,
        links: dict[tuple[str, str], tuple[float, float]],
    ):
        self.profiles = profiles
        self.rollout_sites = rollout_sites
        self.channel_site = channel_site
        self.links = links  # (bandwidth in bytes/s, one-way propagation in seconds).
        self._pending: dict[str, dict[int, _Reservation]] = {
            consumer: {} for consumer in profiles
        }
        self._link_free_at: dict[str, float] = defaultdict(float)

    @classmethod
    def from_config(
        cls, options: Any, network: Any, rollout_infos: list, node: int
    ) -> "LoadAwareRouter":
        """Load inference profiles on the driver before channel launch."""
        raw = json.loads(Path(options["profile_path"]).expanduser().read_text())
        node_sites = {
            rank: site
            for site, group in network["sites"].items()
            for rank in group["node_ranks"]
        }
        channel_site = node_sites[node]
        by_site = options.get("profile_by_site", {})
        profiles, sites = {}, {}
        for info in rollout_infos:
            consumer = info.address.get_name()
            sites[consumer] = node_sites[info.cluster_node_rank]
            label = by_site.get(sites[consumer], info.accelerator_model)
            entry = raw["inference"][label]
            table = {int(k): float(v) for k, v in entry["time_s"].items()}
            if not table or any(k <= 0 or v <= 0 for k, v in table.items()):
                raise ValueError(
                    f"Expected positive inference profile points for {label}"
                )
            profiles[consumer] = InferenceProfile(
                table, int(entry.get("max_batch_size", max(table)))
            )
        links = {}
        for link in network.get("links", []):
            if link["bandwidth_mbps"] <= 0 or link["latency_ms"] < 0:
                raise ValueError(
                    "Network bandwidth must be positive and latency nonnegative"
                )
            links[link["src"], link["dst"]] = (
                link["bandwidth_mbps"] * 1e6 / 8,
                link["latency_ms"] / 1000,
            )
        for site in set(sites.values()) - {channel_site}:
            for pair in ((channel_site, site), (site, channel_site)):
                if pair not in links:
                    raise ValueError(f"Missing cluster.network.links entry for {pair}")
        return cls(profiles, sites, channel_site, links)

    def _estimate(self, sizes: dict, wire_bytes: dict, now: float):
        site_bytes = defaultdict(float)
        for consumer, size in sizes.items():
            if size:
                site_bytes[self.rollout_sites[consumer]] += wire_bytes[consumer]
        arrivals, transfers, returns = {}, {}, {}
        for site, nbytes in site_bytes.items():
            if site == self.channel_site:
                arrivals[site], returns[site] = now, 0.0
            else:
                bandwidth, latency = self.links[self.channel_site, site]
                transfers[site] = (
                    max(now, self._link_free_at[site]) + nbytes / bandwidth
                )
                arrivals[site] = transfers[site] + latency
                returns[site] = self.links[site, self.channel_site][1]
        reservations, completion = {}, []
        for consumer, size in sizes.items():
            if not size:
                continue
            site = self.rollout_sites[consumer]
            pending = self._pending[consumer]
            queue = iter(pending.values())
            first = next(queue, None)
            free_at = max(now, first.finish_at) if first else now
            for queued in queue:
                free_at = max(free_at, queued.ready_at) + queued.duration
            duration = self.profiles[consumer].duration(size)
            finish = max(free_at, arrivals[site]) + duration
            reservations[consumer] = _Reservation(arrivals[site], duration, finish)
            completion.append(finish + returns[site] - now)
        return (max(completion, default=0.0), sum(completion)), reservations, transfers

    def assign(
        self,
        sizes: list[int],
        wire_bytes: list[float],
        available: list[str],
        now: float,
    ) -> list[str]:
        """Greedily allocate minimum shards, then refine against the full batch cost."""
        counts = dict.fromkeys(available, 0)
        volumes = dict.fromkeys(available, 0.0)
        targets = [None] * len(sizes)

        def place(index):
            old = targets[index]
            if old is not None:
                counts[old] -= sizes[index]
                volumes[old] -= wire_bytes[index]
            best, best_cost = None, (float("inf"), float("inf"))
            # Keep the current assignment on ties during refinement.
            order = ([old] if old else []) + [c for c in available if c != old]
            for consumer in order:
                if (
                    counts[consumer] + sizes[index]
                    > self.profiles[consumer].max_batch_size
                ):
                    continue
                counts[consumer] += sizes[index]
                volumes[consumer] += wire_bytes[index]
                cost, _, _ = self._estimate(counts, volumes, now)
                counts[consumer] -= sizes[index]
                volumes[consumer] -= wire_bytes[index]
                if cost < best_cost:
                    best, best_cost = consumer, cost
            if best is None:
                raise ValueError(
                    "Cohort exceeds available profiled batch capacities; reduce "
                    "max_env_requests or profile a larger safe max_batch_size"
                )
            targets[index] = best
            counts[best] += sizes[index]
            volumes[best] += wire_bytes[index]

        order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
        for index in order:
            place(index)
        for _ in range(2):
            for index in order:
                place(index)
        return targets

    def reserve(
        self,
        call_ids: dict[str, int],
        sizes: dict[str, int],
        wire_bytes: dict[str, float],
        now: float,
    ) -> None:
        """Charge the selected physical calls before another cohort is assigned."""
        _, reservations, transfers = self._estimate(sizes, wire_bytes, now)
        self._link_free_at.update(transfers)
        for consumer, reservation in reservations.items():
            self._pending[consumer][call_ids[consumer]] = reservation

    def complete(self, consumer: str, call_id: int, now: float) -> None:
        """Release a completed call and rebase the remaining queue on its feedback."""
        pending = self._pending[consumer]
        del pending[call_id]
        finish = now
        for reservation in pending.values():
            finish = max(finish, reservation.ready_at) + reservation.duration
            reservation.finish_at = finish
        site = self.rollout_sites[consumer]
        if not any(
            self._pending[c] for c, s in self.rollout_sites.items() if s == site
        ):
            self._link_free_at[site] = min(self._link_free_at[site], now)
