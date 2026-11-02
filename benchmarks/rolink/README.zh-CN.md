# Baseline 与 RoLink 对照

[English](README.md)

用两台各 4 张 GPU 的节点、256 个环境，对比异步 PPO。两组使用相同模型、seed=0、global batch=4096，运行 10 次更新，只记录训练 reward 和步耗时，关闭评测、checkpoint 和网络仿真。

| 方案 | Rollout GPU | 轨迹传输 |
| --- | --- | --- |
| Baseline | 边缘 4 卡 | Env 攒完整轮轨迹，再上传 actor |
| RoLink | 边缘 4 卡 + 云端 4 卡 | Channel 分片 observation，rollout 逐 chunk 上传 actor |

两组的 env 都在边缘、actor 都在云端。RoLink 的 queue size=1，云端 rollout 与 actor 并行共用 GPU。这是两套完整配置的对照。

先准备[具身运行环境](../../docs/source-zh/rst_source/start/installation.rst)和[两节点 Ray 集群](../../docs/source-zh/rst_source/guides/multi_node.rst)，两台机器使用相同的模型路径。在主节点的仓库根目录运行 RoLink：

```bash
export ROLINK_MODEL_PATH=/path/to/RLinf-Pi0-ManiSkill-25Main-SFT
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python examples/embodiment/train_async.py \
  --config-dir "$EMBODIED_PATH/config" \
  --config-path "$PWD/benchmarks/rolink/configs" --config-name rolink
```

Baseline 使用单独的 checkout，应用 env-buffered patch，沿用相同的模型和训练实现，将轨迹改为在 env 端积累后上传：

```bash
git worktree add --detach benchmarks/rolink/baselines/env-buffered HEAD
cd benchmarks/rolink/baselines/env-buffered
git apply --check ../../env-buffered.patch
git apply ../../env-buffered.patch
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD"
python examples/embodiment/train_async.py \
  --config-dir "$EMBODIED_PATH/config" \
  --config-path "$PWD/benchmarks/rolink/configs" --config-name baseline
```

两组依次运行，每次在两台节点同步对应 checkout，并重新启动 Ray 集群。对比各 checkout 下 `logs/rolink`、`logs/baseline` 中 TensorBoard 的 `time/step` 和训练 reward。`configs/baseline.yaml` 继承 `configs/rolink.yaml` 的公共训练参数。
