# Baseline vs RoLink

[中文](README.zh-CN.md)

Compare asynchronous PPO on two nodes with four GPUs each and 256 environments.
Both variants use the same model, seed 0, global batch 4096, and 10 updates.
Record training reward and step time; evaluation, checkpoints, and network emulation are disabled.

| Variant | Rollout GPUs | Trajectories |
| --- | --- | --- |
| Baseline | Four edge GPUs | Env accumulates a complete rollout, then uploads to actor |
| RoLink | Four edge + four cloud GPUs | Channel shards observations; rollout streams chunks to actor |

Both place env on the edge and actor on the cloud. RoLink uses queue size 1;
cloud rollout and actor share GPUs concurrently. This compares the complete setups.

Prepare the [embodied environment](../../docs/source-en/rst_source/start/installation.rst)
and [two-node Ray cluster](../../docs/source-en/rst_source/guides/multi_node.rst).
Use the same model path on both nodes. From this repository's root on the head node:

```bash
export ROLINK_MODEL_PATH=/path/to/RLinf-Pi0-ManiSkill-25Main-SFT
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python examples/embodiment/train_async.py \
  --config-dir "$EMBODIED_PATH/config" \
  --config-path "$PWD/benchmarks/rolink/configs" --config-name rolink
```

For the baseline, create a separate checkout and apply the env-buffered patch.
It keeps the same model and training code and changes where trajectories accumulate.

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

Run the two variants separately, with the matching checkout available on both
nodes and a fresh Ray cluster per run. Compare TensorBoard `time/step` and the
training reward metrics in each checkout's `logs/rolink` or `logs/baseline`.
`configs/baseline.yaml` inherits the shared settings from `configs/rolink.yaml`.
