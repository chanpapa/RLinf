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

"""Search two-site RoLink placements and export a complete training config."""

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_routing import LoadAwareRouter
from rlinf.data.schema.embodied_sharding import ShardTopology
from rlinf.scheduler import WorkerAddress
from rlinf.utils.logging import get_logger
from toolkits.auto_placement.rolink_profile import (
    OpenPIProfile,
    interpolate,
    memory_at,
)


@dataclass(frozen=True)
class PlacementPlan:
    """GPU allocation, channel grouping, and estimated seconds per update."""

    env_workers: int
    edge_rollout_gpus: int
    cloud_rollout_gpus: int
    actor_gpus: int
    max_env_requests: int
    edge_batch_cap: int
    cloud_batch_cap: int
    rollout_samples_per_cycle: tuple[int, ...]
    predicted_s: float
    rollout_s: float
    actor_s: float
    weight_sync_s: float
    network_s: float
    mode: str
    observation_bytes: float
    edge_trajectory_bytes: float
    trajectory_tail_s: float
    data_ready_s: float


def _infos(group: str, node: int, devices: list[int]) -> list:
    return [
        SimpleNamespace(
            rank=rank,
            address=WorkerAddress(group, rank),
            cluster_node_rank=node,
            accelerator_rank=device,
            accelerator_model="",
        )
        for rank, device in enumerate(devices)
    ]


def _evaluate(
    cfg: Any,
    profile: OpenPIProfile,
    profile_path: Path,
    edge_gpus: int,
    env_workers: int,
    edge_rollouts: int,
    cloud_rollouts: int,
    actor_gpus: int,
    max_requests: int,
    mode: str,
    env_step_s: float | None,
) -> PlacementPlan | None:
    train = cfg.env.train
    local_envs = train.total_num_envs // env_workers
    requires_gpu = train.requires_gpu
    edge_devices = list(range(edge_gpus - edge_rollouts, edge_gpus))
    shared = requires_gpu and any(gpu < env_workers for gpu in edge_devices)
    if (
        requires_gpu
        and memory_at(profile.env_memory, local_envs).peak_mib
        >= profile.gpu_limit_mib["edge"]
    ):
        return None
    caps = {}
    for site in ("edge", "cloud"):
        actor_memory = (
            interpolate(profile.actor_peak_mib, actor_gpus)
            if mode == "sync" and site == "cloud" and cloud_rollouts
            else 0.0
        )
        caps[site] = max(
            (
                size
                for size in profile.infer_time_s[site]
                if memory_at(profile.infer_memory[site], size).peak_mib + actor_memory
                < profile.gpu_limit_mib[site]
            ),
            default=0,
        )
    if shared:
        caps["edge"] = min(caps["edge"], profile.colocated_infer_cap(local_envs))
    if (edge_rollouts and not caps["edge"]) or (cloud_rollouts and not caps["cloud"]):
        return None

    edge_node = cfg.cluster.network.sites.edge.node_ranks[0]
    cloud_node = cfg.cluster.network.sites.cloud.node_ranks[0]
    env_infos = _infos(
        "env",
        edge_node,
        list(range(env_workers)) if requires_gpu else [-1] * env_workers,
    )
    rollout_infos = _infos("rollout", edge_node, edge_devices)
    for info in _infos("rollout", cloud_node, list(range(cloud_rollouts))):
        info.rank += edge_rollouts
        info.address = WorkerAddress("rollout", info.rank)
        rollout_infos.append(info)
    topology = ShardTopology.from_workers(
        "env", env_infos, rollout_infos, 1, requires_gpu
    )
    options = {
        "profile_path": str(profile_path),
        "profile_by_site": {"edge": "edge", "cloud": "cloud"},
    }
    router = LoadAwareRouter.from_config(
        options, cfg.cluster.network, rollout_infos, edge_node
    )
    for consumer, site in router.rollout_sites.items():
        router.profiles[consumer] = type(router.profiles[consumer])(
            profile.infer_time_s[site], caps[site]
        )

    min_shard = cfg.runner.rolink.get("min_shard", 1)
    per_env_sizes = [
        min(min_shard, local_envs - i) for i in range(0, local_envs, min_shard)
    ]
    observation_bytes = profile.observation_mib * 1024**2
    gpu_demand = defaultdict(float)
    batch_totals = Counter()
    max_service, network_bytes = 0.0, 0.0
    for start in range(0, env_workers, max_requests):
        cohort = set(range(start, min(start + max_requests, env_workers)))
        available = [
            consumer
            for consumer, blockers in topology.blockers.items()
            if blockers <= cohort
        ]
        sizes = per_env_sizes * len(cohort)
        if not available or sum(
            router.profiles[c].max_batch_size for c in available
        ) < sum(sizes):
            return None
        volumes = [size * observation_bytes for size in sizes]
        try:
            targets = router.assign(sizes, volumes, available, now=0.0)
        except ValueError:
            # A minimum shard may not fit the remaining capacity of any GPU.
            return None
        counts, wire_bytes = Counter(), defaultdict(float)
        for consumer, size, nbytes in zip(targets, sizes, volumes):
            counts[consumer] += size
            wire_bytes[consumer] += nbytes
        (service, _), _, _ = router._estimate(counts, wire_bytes, now=0.0)
        if len(cohort) < max_requests:
            service += cfg.runner.rolink.get("batch_timeout_ms", 100) / 1000
        max_service = max(max_service, service)
        batch_totals.update(counts)
        for consumer, size in counts.items():
            gpu_demand[consumer] += router.profiles[consumer].duration(size)
            if router.rollout_sites[consumer] == "cloud":
                network_bytes += wire_bytes[consumer]

    env_s = profile.env_time(local_envs) if requires_gpu else env_step_s
    for consumer, blockers in topology.blockers.items():
        if blockers:
            gpu_demand[consumer] += env_s
    network_s = (
        network_bytes / router.links["edge", "cloud"][0] if network_bytes else 0.0
    )
    cycle_s = max(env_s + max_service, network_s, max(gpu_demand.values(), default=0.0))
    chunks = train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
    rollout_s = (chunks + 1) * train.rollout_epoch * cycle_s
    observation_volume = network_bytes * (chunks + 1) * train.rollout_epoch
    edge_samples = sum(
        count
        for consumer, count in batch_totals.items()
        if router.rollout_sites[consumer] == "edge"
    )
    trajectory_volume = (
        edge_samples * chunks * train.rollout_epoch * profile.trajectory_mib * 1024**2
    )
    # Observations take priority; trajectories use the remaining link capacity.
    bandwidth = router.links["edge", "cloud"][0]
    overlap_bytes = max(0.0, rollout_s * bandwidth - observation_volume)
    tail_s = max(0.0, trajectory_volume - overlap_bytes) / bandwidth
    data_ready_s = rollout_s + tail_s
    actor_s = profile.actor_time(actor_gpus)
    sync_s = interpolate(profile.weight_sync_s, edge_rollouts)
    predicted = (
        data_ready_s + actor_s + sync_s
        if mode == "sync"
        else max(data_ready_s, actor_s + sync_s)
    )
    return PlacementPlan(
        env_workers,
        edge_rollouts,
        cloud_rollouts,
        actor_gpus,
        max_requests,
        caps["edge"],
        caps["cloud"],
        tuple(batch_totals[info.address.get_name()] for info in rollout_infos),
        predicted,
        rollout_s,
        actor_s,
        sync_s,
        network_s,
        mode,
        observation_volume,
        trajectory_volume,
        tail_s,
        data_ready_s,
    )


def search(
    cfg: Any,
    profile_path: Path,
    *,
    edge_gpus: int,
    cloud_gpus: int,
    queue_sizes: tuple[int, ...] = (1, 2),
    env_workers: tuple[int, ...] | None = None,
    min_actor_gpus: int = 1,
    mode: str = "sync",
    env_step_s: float | None = None,
    rollout_scope: str = "any",
) -> list[PlacementPlan]:
    """Enumerate GPU placements and channel cohorts under profile memory limits.

    Env uses edge GPUs from the beginning, rollout uses edge GPUs from the end
    and cloud GPUs from the beginning. Synchronous actors reuse all cloud GPUs;
    asynchronous actors use the remaining cloud GPUs.
    """
    if mode not in ("sync", "async"):
        raise ValueError(f"Expected sync or async mode, got {mode}")
    if edge_gpus < 1 or cloud_gpus < 1 or min_actor_gpus < 1:
        raise ValueError("GPU budgets and min_actor_gpus must be positive")
    if cfg.rollout.pipeline_stage_num != 1:
        raise ValueError("RoLink placement search currently models one pipeline stage")
    if any(
        len(cfg.cluster.network.sites[s].node_ranks) != 1 for s in ("edge", "cloud")
    ):
        raise ValueError(
            "RoLink placement search requires one edge node and one cloud node"
        )
    groups = {group.label: str(group.node_ranks) for group in cfg.cluster.node_groups}
    for site in ("edge", "cloud"):
        node = str(cfg.cluster.network.sites[site].node_ranks[0])
        if groups.get(site) not in (node, f"{node}-{node}"):
            raise ValueError(f"Node group {site} must match network site {site}")
    if not cfg.env.train.requires_gpu and (env_step_s is None or env_step_s <= 0):
        raise ValueError(
            "CPU environments require a positive --env-step-s per action chunk"
        )
    profile = OpenPIProfile.load(profile_path)
    candidates = []
    env_counts = env_workers or tuple(range(1, edge_gpus + 1))
    for count in env_counts:
        if count < 1 or cfg.env.train.total_num_envs % count:
            continue
        if cfg.env.train.requires_gpu and count > edge_gpus:
            continue
        if (cfg.env.train.total_num_envs // count) % cfg.algorithm.group_size:
            continue
        for cloud_rollouts in range(cloud_gpus + (mode == "sync")):
            actors = cloud_gpus if mode == "sync" else cloud_gpus - cloud_rollouts
            if actors < min_actor_gpus or cfg.actor.global_batch_size % (
                cfg.actor.micro_batch_size * actors
            ):
                continue
            if not profile.actor_fits(actors):
                continue
            for edge_rollouts in range(edge_gpus + 1):
                rollouts = edge_rollouts + cloud_rollouts
                if not rollouts:
                    continue
                if (
                    (rollout_scope == "edge" and cloud_rollouts)
                    or (rollout_scope == "cloud" and edge_rollouts)
                    or (
                        rollout_scope == "hybrid"
                        and not (edge_rollouts and cloud_rollouts)
                    )
                ):
                    continue
                if cfg.runner.get("val_check_interval", -1) > 0:
                    eval_envs = cfg.env.eval.total_num_envs
                    if eval_envs % rollouts or eval_envs % (
                        count * cfg.algorithm.group_size
                    ):
                        continue
                for size in queue_sizes:
                    if not 1 <= size <= count:
                        continue
                    candidate = _evaluate(
                        cfg,
                        profile,
                        profile_path,
                        edge_gpus,
                        count,
                        edge_rollouts,
                        cloud_rollouts,
                        actors,
                        size,
                        mode,
                        env_step_s,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return sorted(
        candidates, key=lambda p: (p.predicted_s, p.max_env_requests, p.env_workers)
    )


def export_config(
    cfg: Any, plan: PlacementPlan, profile_path: Path, edge_gpus: int, output: Path
) -> None:
    """Write the selected placement and capped runtime profile without mutating cfg."""
    raw = copy.deepcopy(OmegaConf.to_container(cfg, resolve=True))
    profile = json.loads(profile_path.read_text())
    runtime = {"source": profile.get("source", "unknown"), "inference": {}}
    for site, cap in (("edge", plan.edge_batch_cap), ("cloud", plan.cloud_batch_cap)):
        runtime["inference"][site] = {
            "time_s": profile["inference"][site]["time_s"],
            "max_batch_size": cap,
        }
    profile_output = output.with_suffix(".profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.write_text(json.dumps(runtime, indent=2) + "\n")
    cloud = plan.cloud_rollout_gpus
    rollout_devices = list(range(edge_gpus - plan.edge_rollout_gpus, edge_gpus)) + list(
        range(edge_gpus, edge_gpus + cloud)
    )
    placement = raw["cluster"]["component_placement"]
    placement["rollout"] = {
        "node_group": "edge,cloud",
        "placement": ",".join(map(str, rollout_devices)),
    }
    actor_start = 0 if plan.mode == "sync" else cloud
    placement["actor"] = {
        "node_group": "cloud",
        "placement": f"{actor_start}-{actor_start + plan.actor_gpus - 1}",
    }
    if raw["env"]["train"]["requires_gpu"]:
        placement["env"] = {
            "node_group": "edge",
            "placement": f"0-{plan.env_workers - 1}",
        }
    else:
        edge_node = raw["cluster"]["network"]["sites"]["edge"]["node_ranks"][0]
        placement["env"] = {
            "node_group": "node",
            "placement": f"{edge_node}:0-{plan.env_workers - 1}",
        }
    raw["runner"]["rolink"].update(
        {
            "enable_sharding": True,
            "routing": "load_aware",
            "max_env_requests": plan.max_env_requests,
            "profile_path": str(profile_output),
            "profile_by_site": {"edge": "edge", "cloud": "cloud"},
        }
    )
    raw["rollout"]["enable_cuda_graph"] = False
    OmegaConf.save(OmegaConf.create(raw), output)


def main() -> None:
    """Search a composed Hydra config and optionally export its best candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--edge-gpus", type=int, required=True)
    parser.add_argument("--cloud-gpus", type=int, required=True)
    parser.add_argument("--queue-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--env-workers", type=int, nargs="+")
    parser.add_argument("--min-actor-gpus", type=int, default=1)
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument(
        "--rollout-scope", choices=["any", "edge", "cloud", "hybrid"], default="any"
    )
    parser.add_argument(
        "--env-step-s", type=float, help="CPU env time per action chunk"
    )
    parser.add_argument(
        "--output", type=Path, help="write best config and its runtime profile"
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault(
        "EMBODIED_PATH",
        str(Path(__file__).resolve().parents[2] / "examples/embodiment"),
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(args.config.resolve().parent)
    ):
        cfg = compose(config_name=args.config.stem)
    candidates = search(
        cfg,
        args.profile,
        edge_gpus=args.edge_gpus,
        cloud_gpus=args.cloud_gpus,
        queue_sizes=tuple(args.queue_sizes),
        env_workers=args.env_workers,
        min_actor_gpus=args.min_actor_gpus,
        mode=args.mode,
        env_step_s=args.env_step_s,
        rollout_scope=args.rollout_scope,
    )
    if not candidates:
        parser.error(
            "No feasible placement under the configured workload and profile limits"
        )
    report = {
        "profile_source": OpenPIProfile.load(args.profile).source,
        "cost_model": "profile-based dependency and resource-demand estimate",
        "candidate_count": len(candidates),
        "plans": [asdict(p) for p in candidates[: args.top]],
    }
    if args.output:
        export_config(cfg, candidates[0], args.profile, args.edge_gpus, args.output)
        args.output.with_suffix(".plan.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    get_logger().info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
