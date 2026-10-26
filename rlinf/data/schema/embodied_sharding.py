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

"""Observation sharding and GPU availability for embodied rollout."""

from collections import defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Iterable

import numpy as np
import torch

from rlinf.data.schema.embodied_routing import LoadAwareRouter, payload_nbytes
from rlinf.data.schema.embodied_types import EnvPart, PolicyInput, TrajectorySource
from rlinf.scheduler.channel import ChannelContext, Collector, Dispatcher
from rlinf.utils.obs_compression import compress_obs, decompress_obs

INFERENCE_KEY = "rolink/inference"
EPOCH_END_KEY = "rolink/epoch_end"


def action_key(stage_id: int) -> str:
    """Return the action queue for one environment pipeline stage."""
    return f"rolink/actions/{stage_id}"


def sharding_enabled(cfg: Any) -> bool:
    """Return whether observation sharding is enabled for training."""
    return bool(cfg.runner.get("rolink", {}).get("enable_sharding", False))


@dataclass(frozen=True)
class ShardTopology:
    """Rollout consumers and the GPU-using env ranks sharing their devices.

    An empty blocker set denotes a dedicated rollout or CPU-only environments.
    Device sharing is determined from placement, independently of worker ranks.
    """

    env_group: str
    env_ranks: tuple[int, ...]
    stages: int
    blockers: dict[str, frozenset[int]]

    @classmethod
    def from_workers(
        cls,
        env_group: str,
        env_infos: list,
        rollout_infos: list,
        stages: int,
        requires_gpu: bool,
    ) -> "ShardTopology":
        """Build the sharing relation from registered WorkerInfo objects."""

        def devices(info):
            return (
                {(info.cluster_node_rank, info.accelerator_rank)}
                if info.accelerator_rank >= 0
                else set()
            )

        if requires_gpu and any(not devices(info) for info in env_infos):
            raise ValueError(
                "env.train.requires_gpu=true requires GPU placement for every env worker"
            )
        return cls(
            env_group=env_group,
            env_ranks=tuple(info.rank for info in env_infos),
            stages=stages,
            blockers={
                info.address.get_name(): frozenset(
                    env.rank
                    for env in env_infos
                    if requires_gpu and devices(info) & devices(env)
                )
                for info in rollout_infos
            },
        )


@dataclass
class ShardCall:
    """A coalesced inference call with source offsets preserved."""

    call_id: int
    policy_input: PolicyInput


@dataclass
class ShardResult:
    """Completion after inference and attached terminal forwards have finished."""

    call_id: int
    actions: torch.Tensor | None


@dataclass(frozen=True)
class ShardEpochEnd:
    """Acknowledge final environment parts at a rollout boundary."""

    step_id: int
    epoch_id: int


@dataclass
class ShardDelivery:
    """A message reserved for a specific channel consumer."""

    consumer: str
    payload: Any


class ShardDispatcher(Dispatcher):
    """Deliver the collector's uniform or load-aware reservations.

    The collector reserves calls before enqueueing them, so returning an action
    can wait for all inference already assigned to the same GPU.
    """

    @staticmethod
    def assign(sizes: list[int], available: list[str], cursor: int) -> list[str]:
        """Assign balanced, contiguous runs of shards, rotating the remainder."""
        if not available:
            raise ValueError("Cannot assign shards without an available rollout")
        order = (
            available[cursor % len(available) :] + available[: cursor % len(available)]
        )
        count, remainder = divmod(len(sizes), len(order))
        return [
            consumer
            for index, consumer in enumerate(order)
            for _ in range(count + (index < remainder))
        ]

    def route(self, item: Any, key: Any) -> str | None:
        """Deliver a message to its reserved consumer."""
        if not isinstance(item, ShardDelivery):
            raise TypeError(f"Expected ShardDelivery, got {type(item)}")
        return item.consumer


@dataclass
class _Request:
    policy_input: PolicyInput
    wire_bytes: int = 0
    queued_at: float = field(default_factory=monotonic)
    pending: int = 0
    scheduled: bool = False
    pieces: list[tuple[list[int], torch.Tensor]] = field(default_factory=list)


class ShardCollector(Collector):
    """Own availability, shard reservations, and action assembly in one channel.

    Requests form bounded cohorts before routing. Shared GPUs serve their
    cohort while colocated envs wait; actions return after each env's request
    completes and inference on its shared GPU drains.
    """

    def setup(self, ctx: ChannelContext) -> None:
        """Load placement and the minimum shard size from channel options."""
        self.topology: ShardTopology = ctx.options["topology"]
        self._compression = ctx.options.get("obs_compression")
        self._router: LoadAwareRouter | None = ctx.options.get("router")
        self.min_shard = ctx.options.get("min_shard", 1)
        self.max_env_requests = ctx.options.get("max_env_requests", 1)
        self.batch_timeout_s = ctx.options.get("batch_timeout_ms", 100) / 1000
        if self.max_env_requests < 1 or self.batch_timeout_s < 0:
            raise ValueError(
                "max_env_requests must be positive and timeout nonnegative"
            )
        if type(self.min_shard) is not int or self.min_shard < 1:
            raise ValueError("runner.rolink.min_shard must be a positive integer")
        if (
            not self.topology.blockers
            or not self.topology.env_ranks
            or self.topology.stages < 1
        ):
            raise ValueError(
                "Sharding requires env workers, rollout workers, and positive stages"
            )
        self._slots = {
            (rank, stage)
            for rank in self.topology.env_ranks
            for stage in range(self.topology.stages)
        }
        if any(
            not ranks <= set(self.topology.env_ranks)
            for ranks in self.topology.blockers.values()
        ):
            raise ValueError("Rollout blockers must reference configured env ranks")
        self._waiting: set[tuple[int, int]] = set()
        self._requests: dict[tuple[int, int], _Request] = {}
        self._calls: dict[int, tuple[str, list[TrajectorySource]]] = {}
        self._outstanding = dict.fromkeys(self.topology.blockers, 0)
        self._finished: dict[int, set[int]] = defaultdict(set)
        self._env_ends: dict[int, ShardEpochEnd] = {}
        self._epochs: dict[ShardEpochEnd, set[int]] = defaultdict(set)
        self._last_identity: dict[tuple[int, int], tuple] = {}
        self._next_call_id = 0
        self._cursor = 0
        self._flush_at: float | None = None

    def available_rollouts(self, env_ranks: set[int]) -> tuple[str, ...]:
        """Return dedicated rollouts and unreserved GPUs belonging to this cohort."""
        return tuple(
            consumer
            for consumer, blockers in self.topology.blockers.items()
            if blockers <= env_ranks
            and all(
                (rank, stage) in self._waiting
                for rank in blockers
                for stage in range(self.topology.stages)
            )
            and not any(
                request.scheduled
                for (rank, _), request in self._requests.items()
                if rank in blockers
            )
        )

    def next_flush_at(self) -> float | None:
        """Return the next partial cohort's batching deadline."""
        return self._flush_at

    def flush(self) -> Iterable[tuple[Any, Any]]:
        """Dispatch partial cohorts whose oldest request has timed out."""
        return self._schedule()

    @staticmethod
    def _batch_key(item: PolicyInput) -> tuple:
        def shape(value):
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return type(value), str(value.dtype), tuple(value.shape[1:])
            if isinstance(value, dict):
                return tuple((key, shape(value[key])) for key in sorted(value))
            if isinstance(value, list):
                return list
            return value

        return (
            item.is_last,
            item.requires_inference,
            shape(item.obs),
            shape(item.external_actions),
        )

    def collect(self, item: Any, key: Any) -> Iterable[tuple[Any, Any]]:
        """Process an observation or completion and emit all newly ready messages."""
        if isinstance(item, PolicyInput):
            self._observe(item)
        elif isinstance(item, ShardResult):
            self._complete(item)
        else:
            raise TypeError(f"Expected PolicyInput or ShardResult, got {type(item)}")
        # Complete all state transitions before ChannelWorker awaits enqueue.
        output = self._release_actions()
        output.extend(self._schedule())
        return output

    def _observe(self, item: PolicyInput) -> None:
        if len(item.sources) != 1 or item.sources[0].offset != 0:
            raise ValueError("Expected one complete env batch before sharding")
        source = item.sources[0]
        key = source.key
        slot = (key.env_rank, key.stage_id)
        identity = (key.step_id, key.epoch_id, key.chunk_id, item.is_last)
        if (
            slot not in self._slots
            or slot in self._requests
            or key.stage_id in self._finished.get(key.env_rank, set())
        ):
            raise ValueError(f"Unexpected or duplicate observation for {slot}")
        if identity <= self._last_identity.get(slot, (-1, -1, -1, False)):
            raise ValueError(f"Stale observation for {slot}: {identity}")
        if source.size < 1:
            raise ValueError("Observation batches must be nonempty")
        wire_bytes = payload_nbytes(item) if self._router else 0
        item.obs = decompress_obs(item.obs)
        for part in item.env_parts:
            if part is not None and part.next_obs is not None:
                part.next_obs = decompress_obs(part.next_obs)
        self._last_identity[slot] = identity
        self._requests[slot] = _Request(item, wire_bytes)
        self._waiting.add(slot)

    def _schedule(self) -> list:
        self._flush_at = None
        pending = defaultdict(list)
        for slot, request in self._requests.items():
            if not request.scheduled:
                pending[self._batch_key(request.policy_input)].append(slot)
        output = []
        for slots in pending.values():
            for start in range(0, len(slots), self.max_env_requests):
                cohort = slots[start : start + self.max_env_requests]
                first = self._requests[cohort[0]]
                deadline = first.queued_at + self.batch_timeout_s
                if (
                    len(cohort) < self.max_env_requests
                    and not first.policy_input.is_last
                    and monotonic() < deadline
                ):
                    self._flush_at = (
                        deadline
                        if self._flush_at is None
                        else min(self._flush_at, deadline)
                    )
                    continue
                available = list(self.available_rollouts({rank for rank, _ in cohort}))
                if available:
                    output.extend(self._dispatch(cohort, available))
        return output

    def _dispatch(self, cohort: list[tuple[int, int]], available: list[str]) -> list:
        shards = []
        sizes = []
        wire_bytes = []
        for slot in cohort:
            policy_input = self._requests[slot].policy_input
            size = policy_input.sources[0].size
            split_sizes = [
                min(self.min_shard, size - offset)
                for offset in range(0, size, self.min_shard)
            ]
            shards.extend(policy_input.split(split_sizes))
            sizes.extend(split_sizes)
            wire_bytes.extend(
                self._requests[slot].wire_bytes * n / size for n in split_sizes
            )
        now = monotonic()
        targets = (
            self._router.assign(sizes, wire_bytes, available, now)
            if self._router
            else ShardDispatcher.assign(sizes, available, self._cursor)
        )
        self._cursor += len(sizes)
        grouped = defaultdict(list)
        for shard, target in zip(shards, targets):
            grouped[target].append(shard)
        output = []
        call_ids = {}
        target_sizes = defaultdict(int)
        target_bytes = defaultdict(float)
        for target, size, nbytes in zip(targets, sizes, wire_bytes):
            target_sizes[target] += size
            target_bytes[target] += nbytes
        for target, target_shards in grouped.items():
            requests = []
            runs = []
            for shard in target_shards:
                source = shard.sources[0]
                previous = runs[-1][-1].sources[-1] if runs else None
                if (
                    previous is not None
                    and previous.key == source.key
                    and previous.offset + previous.size == source.offset
                ):
                    runs[-1].append(shard)
                else:
                    runs.append([shard])
            for source_shards in runs:
                request = PolicyInput.merge(source_shards)
                request.env_parts = [
                    EnvPart.merge(request.env_parts)
                    if request.env_parts[0] is not None
                    else None
                ]
                request.request_sizes = [sum(source.size for source in request.sources)]
                requests.append(request)
            policy_input = PolicyInput.merge(requests)
            call_id = self._next_call_id
            self._next_call_id += 1
            call_ids[target] = call_id
            self._calls[call_id] = (target, policy_input.sources)
            self._outstanding[target] += 1
            for source in policy_input.sources:
                self._requests[(source.key.env_rank, source.key.stage_id)].pending += 1
            policy_input.obs = compress_obs(policy_input.obs, self._compression)
            for part in policy_input.env_parts:
                if part is not None and part.next_obs is not None:
                    part.next_obs = compress_obs(part.next_obs, self._compression)
            output.append(
                (INFERENCE_KEY, ShardDelivery(target, ShardCall(call_id, policy_input)))
            )
        if self._router:
            self._router.reserve(call_ids, target_sizes, target_bytes, monotonic())
        for slot in cohort:
            self._requests[slot].scheduled = True
        return output

    def _complete(self, result: ShardResult) -> None:
        if result.call_id not in self._calls:
            raise ValueError(
                f"Unknown or duplicate inference completion {result.call_id}"
            )
        consumer, sources = self._calls[result.call_id]
        first = sources[0].key
        is_last = self._requests[(first.env_rank, first.stage_id)].policy_input.is_last
        size = sum(source.size for source in sources)
        if is_last:
            if result.actions is not None:
                raise ValueError("Final inference completion must not contain actions")
        elif result.actions is None or result.actions.shape[0] != size:
            raise ValueError(f"Expected {size} actions for call {result.call_id}")
        del self._calls[result.call_id]
        if self._router:
            self._router.complete(consumer, result.call_id, monotonic())
        self._outstanding[consumer] -= 1
        offset = 0
        for source in sources:
            slot = (source.key.env_rank, source.key.stage_id)
            request = self._requests[slot]
            if not is_last:
                indices = list(range(source.offset, source.offset + source.size))
                request.pieces.append(
                    (indices, result.actions[offset : offset + source.size])
                )
            offset += source.size
            request.pending -= 1
            if not request.pending:
                if is_last:
                    rank, stage = slot
                    self._finished[rank].add(stage)
                    if len(self._finished[rank]) == self.topology.stages:
                        key = source.key
                        self._env_ends[rank] = ShardEpochEnd(key.step_id, key.epoch_id)
                        self._waiting.difference_update(
                            (rank, i) for i in range(self.topology.stages)
                        )
                    del self._requests[slot]
                else:
                    self._waiting.remove(slot)

    def _release_actions(self) -> list:
        output = []
        for slot, request in list(self._requests.items()):
            if not request.scheduled or request.pending:
                continue
            env_rank, stage = slot
            if self._has_pending_inference(env_rank):
                continue
            positions = [index for indices, _ in request.pieces for index in indices]
            actions = torch.cat([piece for _, piece in request.pieces])
            actions = actions[torch.tensor(positions).argsort()].contiguous()
            output.append(
                (
                    action_key(stage),
                    ShardDelivery(f"{self.topology.env_group}:{env_rank}", actions),
                )
            )
            del self._requests[slot]
        for rank, end in list(self._env_ends.items()):
            if self._has_pending_inference(rank):
                continue
            output.append(
                (EPOCH_END_KEY, ShardDelivery(f"{self.topology.env_group}:{rank}", end))
            )
            del self._env_ends[rank]
            del self._finished[rank]
            self._epochs[end].add(rank)
            if len(self._epochs[end]) == len(self.topology.env_ranks):
                output.extend(
                    (INFERENCE_KEY, ShardDelivery(consumer, end))
                    for consumer in self.topology.blockers
                )
                del self._epochs[end]
        return output

    def _has_pending_inference(self, env_rank: int) -> bool:
        return any(
            self._outstanding[consumer]
            for consumer, blockers in self.topology.blockers.items()
            if env_rank in blockers
        )


def create_shard_channel(cfg: Any, env: Any, rollout: Any):
    """Create the shared observation/completion channel from actual placement."""
    from rlinf.scheduler import Channel, WorkerAddress
    from rlinf.scheduler.manager import WorkerManager

    if cfg.rollout.get("enable_cuda_graph", False):
        raise ValueError(
            "RoLink variable inference batches require rollout.enable_cuda_graph=false"
        )
    requires_gpu = cfg.env.train.get("requires_gpu")
    if type(requires_gpu) is not bool:
        raise ValueError(
            "Set env.train.requires_gpu to true or false for RoLink sharding"
        )
    manager = WorkerManager.get_proxy()

    def infos(group):
        result = [
            manager.get_worker_info(WorkerAddress(group.worker_group_name, worker.rank))
            for worker in group.worker_info_list
        ]
        if any(info is None for info in result):
            raise ValueError(
                "All env and rollout workers must be launched before creating the shard channel"
            )
        return result

    env_infos, rollout_infos = infos(env), infos(rollout)
    node_rank = env_infos[0].cluster_node_rank
    options = cfg.runner.rolink
    routing = options.get("routing", "uniform")
    if routing not in ("uniform", "load_aware"):
        raise ValueError(f"Expected uniform or load_aware routing, got {routing}")
    router = (
        LoadAwareRouter.from_config(
            options, cfg.cluster.network, rollout_infos, node_rank
        )
        if routing == "load_aware"
        else None
    )
    topology = ShardTopology.from_workers(
        env.worker_group_name,
        env_infos,
        rollout_infos,
        cfg.rollout.pipeline_stage_num,
        requires_gpu,
    )
    return Channel.create(
        "RoLink",
        node_rank=node_rank,
        collector=ShardCollector,
        dispatcher=ShardDispatcher,
        producers=[env, rollout],
        consumers=[env, rollout],
        cfg=cfg,
        hook_options={
            "topology": topology,
            "router": router,
            "min_shard": cfg.runner.rolink.get("min_shard", 1),
            "max_env_requests": cfg.runner.rolink.get("max_env_requests", 1),
            "batch_timeout_ms": cfg.runner.rolink.get("batch_timeout_ms", 100),
            "obs_compression": cfg.env.get("obs_compression"),
        },
    )
