# RoLink Profiling and Placement

[中文](README.zh-CN.md)

Use this guide to prepare component profiles, search a RoLink placement, and export a training configuration. The current search targets OpenPI with one edge node, one cloud node, and `rollout.pipeline_stage_num: 1`.

Profiling is a separate preparation step. The RoLink command reads measurements from JSON; it does not launch profiling jobs. Reuse a profile while its hardware and workload remain the same, and remeasure the affected components when they change.

```text
Target hardware + model + training configuration
    → Measure components and save profile JSON
    → Search with training YAML + profile JSON + GPU budgets
    → Export placement YAML + runtime inference profile + plan report
    → Run a short training check, then the full experiment
```

## Prepare the Inputs

Run the commands below from the repository root in your RLinf environment. Training requires the [embodied installation](../../docs/source-en/rst_source/start/installation.rst) and [multi-node setup](../../docs/source-en/rst_source/guides/multi_node.rst).

| Input | What it controls |
| --- | --- |
| [Training YAML](../../examples/embodiment/config/maniskill_ppo_openpi_rolink.yaml) | Model, env count, action chunks, training batch, GPU requirement, and network topology. Hydra resolves its `defaults`. |
| [Full profile JSON](../../examples/embodiment/config/rolink_openpi_profile.json) | Env, inference, actor, weight-sync, memory, and observation-size measurements used by placement search. |
| CLI GPU budgets | Available GPUs on the edge and cloud nodes. |
| `--queue-sizes` | Candidate numbers of env requests to group in the Channel before assigning shards to GPUs. |

> **Reference data:** The bundled full profile has `source: partial-distributed-profile` and includes measurements, calibration, and extrapolation from earlier experiments. Use it to exercise the search. Replace its values with target-machine measurements before selecting a production placement.

Copy the inputs to give the new experiment its own files:

```bash
cp examples/embodiment/config/maniskill_ppo_openpi_rolink.yaml \
  examples/embodiment/config/maniskill_ppo_openpi_rolink_target.yaml
cp examples/embodiment/config/rolink_openpi_profile.json \
  examples/embodiment/config/rolink_target_profile.json
```

Edit the copied YAML's model paths and workload. Keep `edge` and `cloud` in both `cluster.node_groups` and `cluster.network.sites`, with one node each. Set `env.train.requires_gpu` according to GPU use by simulation or rendering. The supplied example uses node 0 for edge and node 1 for cloud.

## Measure and Save the Profile

Measure the same operations that training will run. Warm up each configuration, synchronize the accelerator at timing boundaries, and collect repeated wall-clock samples. Save a representative statistic, such as the median, in the JSON; retain raw samples and the statistic used alongside it. Exclude initialization and checkpoint I/O from steady-state component times.

| JSON field | Measurement and table key |
| --- | --- |
| `env.time_s` | Seconds for one env worker to execute one **action chunk**, including simulation/rendering and preparing the next observation. Key: local env count. A chunk can contain several simulator steps. |
| `env.memory_mib` | Resident (`static_mib`) and maximum (`peak_mib`) GPU memory at each local env count. |
| `inference.edge` / `inference.cloud` | `time_s` and `memory_mib` for one complete rollout prediction, keyed by actual inference batch size. Measure each site's hardware, including required action/logprob/value outputs, with `enable_cuda_graph: false`. |
| `actor.total_time_s` / `actor.peak_mib` | Complete training-phase seconds and maximum per-rank GPU memory, keyed by actor GPU count. Keep global batch, micro batch, `algorithm.update_epoch`, precision, sharding, and offload identical to the intended run. |
| `weight_sync.time_s` | Seconds to synchronize weights to edge rollout replicas. Key: **edge rollout replica count**, following the current cost model. |
| `network.observation_mib` | Representative transferred payload per env per request, including attached fields and the configured compression. Unit: MiB. |
| `network.trajectory_mib` | Training trajectory bytes per env per action chunk, in MiB. Only edge rollout trajectories cross the WAN. Defaults to observation size when omitted. |
| `gpu_memory_limit_mib` | Usable memory budget for `edge` and `cloud`, leaving room for runtime overhead. Unit: MiB. |

Use seconds for time tables, MiB (`1024²` bytes) for memory/payload, and stringified counts for JSON keys. For example, `inference.edge.time_s["16"]` holds the seconds for one prediction of 16 observations; it is not time per observation.

Sweep inference batch sizes such as 1, 2, 4, 8, 16, 32, and 64 up to the usable capacity. Cover the local env counts and actor GPU counts that the search can select. The planner interpolates missing points and can extrapolate outside the tables, so measure any extrapolated winning candidate before using it.

For a measured actor table, use `actor.scaling: "profile_table"` and populate `total_time_s`. The loader also requires `single_gpu_compute_s` and `communication_s`; these fallback fields are unused when `total_time_s` is populated. Preserve the full schema when editing the supplied file. Inference-only JSON is sufficient for runtime routing, but placement search needs the full profile.

For GPU colocation, the planner bounds memory by:

```text
max(env_static + inference_peak, env_peak + inference_static) < GPU budget
```

Measure resident and peak memory consistently, including allocations outside PyTorch where applicable. Check the selected colocated configuration on the real machine, because allocator caches and simulator allocations can change the observed peak.

Store the resolved training YAML, GPU/software versions, model revision, raw measurements, and measurement method with the profile. Set `source` to identify that experiment. The loader does not automatically check that the profile matches the training YAML. RLinf's [profiling and tracing guide](../../docs/source-en/rst_source/guides/profile.rst) helps inspect component timings; conversion into this JSON is currently a separate step.

## Configure the Network and Reuse Measurements

Put pairwise bandwidth and latency under `cluster.network.links` in the training YAML. `sites` only maps node ranks to locations. Links are directed; use the same values in both directions for a symmetric connection.

`bandwidth_mbps` is effective bandwidth in megabits/s. `latency_ms` is **one-way** propagation delay; use RTT/2 only when assuming symmetry. Measure with representative payloads and the intended transport. These values guide routing and do not emulate or throttle traffic.

Set `runner.trajectory_channel_node_rank` to the training node rank (for example, `1`) to assemble trajectories on that node. Rollout workers then send their chunks there during collection. When omitted, the Channel uses its default placement near the first producer. This setting controls placement; it does not disable streaming.

| Change | Measurements to refresh |
| --- | --- |
| GPU type, model, precision, input image size, or action generation settings | Affected inference timing/memory; actor timing/memory where applicable. |
| Simulator, scene, rendering, or action chunk length | Env timing/memory and observation payload size; inference if inputs change. |
| Global/micro batch, update epochs, sharding, or offload | Actor timing/memory. |
| Link bandwidth/latency or weight-sync protocol | YAML links and weight-sync timing. |
| GPU budget or Channel grouping only | Reuse matching tables, cover newly selected sizes/counts, and rerun search. |

## Search and Run

After replacing the copied profile values, search the two-node budget and export the best candidate:

```bash
python -m examples.embodiment.rolink_auto_placement \
  --config examples/embodiment/config/maniskill_ppo_openpi_rolink_target.yaml \
  --profile examples/embodiment/config/rolink_target_profile.json \
  --edge-gpus 4 --cloud-gpus 4 \
  --queue-sizes 1 2 --mode sync --top 5 \
  --output examples/embodiment/config/maniskill_ppo_openpi_rolink_auto.yaml
```

The command enumerates env workers, edge/cloud rollout GPUs, remaining cloud actor GPUs, and Channel grouping. It checks memory and workload divisibility, ranks feasible candidates by estimated time, and creates:

| Output | Use |
| --- | --- |
| `maniskill_ppo_openpi_rolink_auto.yaml` | Resolved training configuration with the selected placement and Channel settings. The input YAML stays unchanged. |
| `maniskill_ppo_openpi_rolink_auto.profile.json` | Inference tables with selected batch capacity limits, referenced by the generated YAML. |
| `maniskill_ppo_openpi_rolink_auto.plan.json` | Top candidates, resource allocations, and estimated timing. |

`--queue-sizes 1 2` searches groups of one or two env requests, not inference batch sizes. The input `batch_timeout_ms: 100` remains the timeout for partial groups. `--rollout-scope hybrid` requires rollout on both sites; `edge`, `cloud`, and `any` are also accepted. Use `--env-workers` to restrict env worker counts and `--min-actor-gpus` to set the actor minimum.

For CPU environments, set `requires_gpu: false` and pass `--env-step-s` with the measured time per action chunk. Physical robot bindings still need their normal configuration.

Launch a short synchronous check from the repository root after starting Ray on both nodes:

```bash
STEPS=3 bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_rolink_auto
```

This checks the generated placement through rollout, updates, and weight synchronization. Compare measured component times and peak memory with the plan before launching the full experiment without `STEPS=3`. Keep the generated profile accessible at its configured path on the launching node.

For asynchronous training, supply the corresponding async YAML, search with `--mode async`, and launch through `examples/embodiment/run_async.sh`. `--mode` selects the planner's cost formula; it does not switch the training entry point.

## Interpret the Results

Treat the report as a placement estimate to validate on hardware. Let `R` be time to finish environment interaction, excluding the trajectory upload tail, `B` average uplink bandwidth, `O` cloud-bound observation bytes, and `T` edge trajectory bytes. The estimate assumes observations take priority and trajectories use spare capacity:

```text
overlap_bytes = max(0, R * B - O)
trajectory_tail = max(0, T - overlap_bytes) / B
data_ready = R + trajectory_tail = max(R, (O + T) / B)
sync: data_ready + actor + weight_sync
async: max(data_ready, actor + weight_sync)
```

This aggregate model does not replay when each chunk becomes available. Different observation and trajectory representations can make `O + T` depend on placement. Reports retain both byte totals and the remaining transfer time.

If every training observation uses the same representation and crosses the WAN exactly once, either before cloud inference or with an edge trajectory, its total traffic is independent of placement. Extra terminal observations, representation differences, and other metadata fall outside that statement. The current model measures the two payload types separately.

`rollout_samples_per_cycle` counts samples assigned to each rollout over one complete env cycle. It can include multiple calls and is not a single inference batch size or a fixed runtime mapping. Runtime routing continues to assign shards using current queue estimates.

Runtime routing uses the configured network links and tracks its outstanding requests. The search supports two sites with one node each; see the [search implementation](rolink.py) and [profile loader](rolink_profile.py) for the schema.
