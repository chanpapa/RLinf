# RoLink Profile 与 Placement 使用说明

[English](README.md)

本文说明如何准备组件 profile、搜索 RoLink placement，并导出训练配置。当前搜索面向 OpenPI，支持一个边缘节点、一个云端节点，要求 `rollout.pipeline_stage_num: 1`。

Profile 需要提前单独测量。RoLink 命令读取 JSON 中的数据，目前不会自动启动测量任务。硬件和工作负载不变时可以复用 profile；配置变化后，重新测量受影响的部分。

```text
目标机器 + 模型 + 训练配置
    → 测量各组件，保存 profile JSON
    → 输入训练 YAML + profile JSON + GPU 数量，搜索 placement
    → 导出 placement YAML + 运行时推理 profile + 搜索报告
    → 短跑检查，再开始完整实验
```

## 准备输入

以下命令都在仓库根目录、RLinf 环境中执行。训练前先完成[具身环境安装](../../docs/source-zh/rst_source/start/installation.rst)和[多节点配置](../../docs/source-zh/rst_source/guides/multi_node.rst)。

| 输入 | 用途 |
| --- | --- |
| [训练 YAML](../../examples/embodiment/config/maniskill_ppo_openpi_rolink.yaml) | 模型、env 数量、action chunk、训练 batch、GPU 占用声明及网络拓扑。Hydra 会解析其中的 `defaults`。 |
| [完整 profile JSON](../../examples/embodiment/config/rolink_openpi_profile.json) | 搜索需要的 env、推理、actor、权重同步、显存及 observation 大小数据。 |
| CLI GPU 数量 | 边缘节点和云端节点各可使用多少张 GPU。 |
| `--queue-sizes` | Channel 在分配 shard 前，尝试合并多少个 env 请求。 |

> **参考数据：** 仓库中的完整 profile 标记为 `source: partial-distributed-profile`，包含旧实验的测量、校准和外推数据，可用于试运行搜索。正式选择 placement 前，应替换成目标机器上的测量数据。

先复制两份输入，为当前实验单独保存配置：

```bash
cp examples/embodiment/config/maniskill_ppo_openpi_rolink.yaml \
  examples/embodiment/config/maniskill_ppo_openpi_rolink_target.yaml
cp examples/embodiment/config/rolink_openpi_profile.json \
  examples/embodiment/config/rolink_target_profile.json
```

修改新 YAML 中的模型路径和训练参数。`cluster.node_groups` 与 `cluster.network.sites` 都保留 `edge`、`cloud`，各对应一个节点。按仿真或渲染是否使用 GPU 设置 `env.train.requires_gpu`。示例中 node 0 是边缘，node 1 是云端。

## 测量并保存 Profile

按训练时实际执行的操作测量。每个配置先预热，在计时边界同步 accelerator，重复采集 wall-clock 耗时。JSON 中保存统一的统计量，例如中位数，并在旁边保留原始数据和统计方法。稳态组件耗时不计初始化和 checkpoint I/O。

| JSON 字段 | 测什么、按什么建表 |
| --- | --- |
| `env.time_s` | 一个 env worker 执行一个 **action chunk** 的秒数，包含仿真、渲染和准备下一次 observation。键是该 worker 的 env 数量。一个 chunk 可能包含多个仿真 step。 |
| `env.memory_mib` | 各 env 数量下的常驻显存 `static_mib` 和最高显存 `peak_mib`。 |
| `inference.edge` / `inference.cloud` | 一次完整 rollout prediction 的 `time_s` 和 `memory_mib`，键是实际推理 batch size。分别在两地机器上测量，包含训练所需的 action、logprob、value 输出，设置 `enable_cuda_graph: false`。 |
| `actor.total_time_s` / `actor.peak_mib` | 完整训练阶段的秒数及各 rank 中最高显存，键是 actor GPU 数量。global batch、micro batch、`algorithm.update_epoch`、精度、sharding 和 offload 与目标训练保持一致。 |
| `weight_sync.time_s` | 权重同步到边缘 rollout 副本的秒数。按当前模型约定，键是 **边缘 rollout 副本数**。 |
| `network.observation_mib` | 每个 env 每次请求的代表性传输大小，包含附带字段，并使用实际压缩配置。单位 MiB。 |
| `network.trajectory_mib` | 每个 env 每个 action chunk 的训练轨迹大小。只有边缘 rollout 的轨迹计入跨域上传；省略时按 observation 大小估计。单位 MiB。 |
| `gpu_memory_limit_mib` | `edge`、`cloud` 各自可用的显存预算，为运行时开销留出空间。单位 MiB。 |

时间表使用秒，显存和 payload 使用 MiB（`1024²` 字节），JSON 的键使用数量的字符串形式。例如 `inference.edge.time_s["16"]` 表示一次处理 16 个 observation 的总秒数。

推理可按 1、2、4、8、16、32、64 等 batch size 测到可用容量上限；env 数量和 actor GPU 数量也要覆盖搜索可能选中的范围。缺少的点会经过插值，表外还可能外推，因此正式采用一个依赖外推的候选前，应补测该点。

Actor 有完整实测表时，使用 `actor.scaling: "profile_table"` 并填写 `total_time_s`。Loader 还要求保留 `single_gpu_compute_s` 和 `communication_s`；`total_time_s` 有数据时不会使用这两个 fallback 字段。修改示例文件时保留完整结构。运行时路由只需要推理表，placement 搜索则需要完整 profile。

Env 和推理共置时，planner 使用下面的显存约束：

```text
max(env_static + inference_peak, env_peak + inference_static) < GPU budget
```

常驻和峰值显存应采用一致口径，并计入适用的 PyTorch 之外的分配。选中后仍要在真实共置配置下检查，因为 allocator 缓存和仿真器分配会影响实际峰值。

将解析后的训练 YAML、GPU 和软件版本、模型 revision、原始数据和测量方法与 profile 一起保存，`source` 写明实验来源。Loader 目前不会自动检查 profile 与训练 YAML 是否匹配。可参考 RLinf 的 [profiling 与 tracing 说明](../../docs/source-zh/rst_source/guides/profile.rst)查看组件耗时；整理成这里的 JSON 仍是独立步骤。

## 配置网络与复用测量

站点之间的带宽和延迟写在训练 YAML 的 `cluster.network.links` 中，`sites` 只负责节点分组。Link 有方向；对称链路在两个方向填写相同数值。

`bandwidth_mbps` 是有效带宽，单位 megabits/s；`latency_ms` 是 **单向** 传播延迟，只有假设对称时才能用 RTT/2。测量时使用有代表性的 payload 和实际传输方式。这些配置参与路由计算，不会模拟或限速网络。

将 `runner.trajectory_channel_node_rank` 设为训练节点 rank（例如 `1`），轨迹就会在该节点组装，rollout worker 在收集过程中将 chunk 发到那里。省略时沿用 Channel 靠近首个 producer 的默认 placement。这个设置控制位置，不用于关闭 streaming。

| 变化 | 需要更新的数据 |
| --- | --- |
| GPU 型号、模型、精度、输入图像大小或 action 生成配置 | 受影响的推理耗时和显存；适用时更新 actor 数据。 |
| 仿真器、场景、渲染或 action chunk 长度 | Env 耗时、显存和 observation 大小；输入变化时更新推理数据。 |
| Global/micro batch、update epoch、sharding 或 offload | Actor 耗时和显存。 |
| 链路带宽、延迟或权重同步协议 | YAML links 和权重同步耗时。 |
| 只改 GPU 数量或 Channel 组批数量 | 复用匹配的表，补齐新涉及的 size/count，再搜索。 |

## 搜索并运行

替换新 profile 中的数据后，搜索两节点资源并导出最佳候选：

```bash
python -m examples.embodiment.rolink_auto_placement \
  --config examples/embodiment/config/maniskill_ppo_openpi_rolink_target.yaml \
  --profile examples/embodiment/config/rolink_target_profile.json \
  --edge-gpus 4 --cloud-gpus 4 \
  --queue-sizes 1 2 --mode sync --top 5 \
  --output examples/embodiment/config/maniskill_ppo_openpi_rolink_auto.yaml
```

命令枚举 env worker 数、边缘/云端 rollout GPU 数、剩余云端 actor GPU 数和 Channel 组批数量，检查显存与 workload 整除约束，按预计耗时排序，并生成：

| 输出 | 用途 |
| --- | --- |
| `maniskill_ppo_openpi_rolink_auto.yaml` | 解析后的完整训练配置，写入选中的 placement 和 Channel 参数。输入 YAML 保持原样。 |
| `maniskill_ppo_openpi_rolink_auto.profile.json` | 带推理 batch 容量限制的表，由生成的 YAML 引用。 |
| `maniskill_ppo_openpi_rolink_auto.plan.json` | 排名前列的候选、资源分配和预计耗时。 |

`--queue-sizes 1 2` 表示尝试合并一或两个 env 请求，与推理 batch size 是不同维度。输入的 `batch_timeout_ms: 100` 继续控制未凑满组时的等待时间。`--rollout-scope hybrid` 要求两地都有 rollout，也可以选择 `edge`、`cloud` 或 `any`。`--env-workers` 限制 env worker 数的候选，`--min-actor-gpus` 设置 actor 最少卡数。

CPU 环境设置 `requires_gpu: false`，并通过 `--env-step-s` 传入一个 action chunk 的实测秒数。真实机器人的物理设备绑定仍需按原有方式配置。

两节点启动 Ray 后，在仓库根目录启动同步短跑：

```bash
STEPS=3 bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_rolink_auto
```

短跑检查生成配置下的 rollout、更新和权重同步。将实测组件时间、峰值显存与计划对照后，去掉 `STEPS=3` 开始完整实验。启动节点需要能访问 YAML 所引用的生成 profile 路径。

异步训练使用对应的异步 YAML 作为输入，搜索时传 `--mode async`，通过 `examples/embodiment/run_async.sh` 启动。`--mode` 只选择 planner 的代价公式，训练入口仍需对应选择。

## 理解搜索结果

报告中的耗时是需要上机验证的估计值。令 `R` 为完成环境交互的 rollout 时间（不含 trajectory 尾部上传等待）、`B` 为平均上行带宽、`O` 为上云 observation 字节数、`T` 为边缘 trajectory 字节数。估计模型假设 observation 优先、trajectory 使用剩余带宽：

```text
overlap_bytes = max(0, R * B - O)
trajectory_tail = max(0, T - overlap_bytes) / B
data_ready = R + trajectory_tail = max(R, (O + T) / B)
同步：data_ready + actor + weight_sync
异步：max(data_ready, actor + weight_sync)
```

这是按字节总量估计重叠的模型，并未逐条回放数据产生的时刻。Observation 和训练轨迹的编码、图像分辨率或附带字段不同时，`O + T` 会随 placement 改变，不能直接假定总跨域字节数恒定。报告保留两类字节数和尾部传输时间。

如果每个训练 observation 使用相同表示，且只跨域一次（云端推理前上传，或边缘推理后随轨迹上传），那么这部分总量确实与 placement 无关。这一结论不包含额外 terminal observation、不同表示的大小差异和其他元数据；当前实现按两类实测载荷分别计算。

`rollout_samples_per_cycle` 表示所有 env 推进一个周期时，各 rollout 总共处理的样本数，可能包含多次调用。它不是单次推理 batch size，也不会作为固定映射写入运行时；运行时继续根据队列负载估计分配 shard。

运行时路由使用配置中的网络链路参数，并跟踪尚未完成的请求。搜索范围为两地、各一个节点；完整字段可查看[搜索实现](rolink.py)和 [profile loader](rolink_profile.py)。
