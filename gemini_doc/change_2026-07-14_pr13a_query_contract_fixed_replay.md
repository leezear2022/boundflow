# 2026-07-14：PR-13A Query/State Contract 与真实固定流 Replay

## 阶段目标与判定

PR-12 已以 `VALIDATED-REDUCED` 关闭。本切片正式启动 PR-13，但只完成第一层语义基础：把现有
host-side BaB solver 发出的节点 bound call 表示为带稳定身份、状态版本和 capability 的查询，
并在不改变 solver branch/queue/termination 的前提下记录和重放真实查询流。

当前判定：

```text
PR-13A Query/State Contract:       PASS
PR-13A fixed-stream smoke replay:  PASS
PR-13B dynamic BatchManager:       NOT STARTED
PR-13 overall:                     IN PROGRESS
```

这里的 `PASS` 是 contract/replay foundation，不是 BaB 性能或 non-toy workload 结论。

## 实现

新增 `boundflow/runtime/bab_query.py`：

- `BoundQuery`：query/parent ID、sequence、model/weight/input/spec/split hash、method/stage、
  α/β/cuts version、dtype/device/numeric policy、requested outputs；
- `QueryCompatibilityKey`：完整 batch/compiled-plan 等价键；αβ/split 查询的 capability 固定为
  `alpha_beta_dense_split`，不得误选 PR-12 plain-CROWN fused TIR；
- `BoundQueryPayload`：把动态 tensor 与可序列化 identity 分离，并在录制时 detached clone；
- `StateValidityManager`：显式返回 `EXACT_REUSE`、`CONDITIONAL_REUSE`、
  `WARM_START_ONLY` 或 `INVALIDATE`；
- `FixedBabQueryRecorder`：拒绝 duplicate ID、unknown/duplicate result、sequence gap 和 incomplete
  query；
- `QueryBatch` reference contract：只接收完整 key 相同的 request，拒绝 mixed capability/key，
  并在 pack/unpack reference execution 中恢复声明顺序；
- `execute_bound_query` / `replay_fixed_query_trace`：通过原有 α/αβ per-node executor 重放。

`solve_bab_mlp` 新增可选 `query_recorder` observer。每个实际进入节点 bound evaluation 的请求
使用稳定 `bab-e{example}-n{node}` ID，并携带真实 parent link、split 和 warm-start 版本；默认
`None` 时不改变原 solver 行为。子节点新增 `parent_node_id`，同时修复 batch fallback 中潜在的
未定义 `_consume_live_node` 调用，统一走现有 `_decrement_live_node`。

## 状态有效性不变量

- 图结构和权重相关预处理在 model/weight version 相同下可 exact reuse；
- compiled module 只有完整 compatibility key 相同时 exact reuse；
- 父 α 和 intermediate bounds 只能作为 child warm start；
- β 与 final bounds 在父→子 split 变化时必须 invalidate；
- weight version 变化首先使所有状态 reuse 判定失效；
- dynamic tensor 值不同但 shape/method/numeric/capability 不同的 query 不得合批。

## 测试与工件

新增：

- `tests/test_phase7a_pr13a_query_contract.py`；
- `tests/test_phase7a_pr13a_fixed_replay.py`；
- `scripts/run_phase7a_pr13a_fixed_replay.py`；
- `artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714/`。

工件由现有 `solve_bab_mlp` 真正展开搜索树，不是复制 synthetic batch。确定性 two-ReLU MLP
smoke 产生 8 个 query：

```text
query_count:         8
replay_passed:       8
replay_failed:       0
max_abs_diff:        0.0
query_loss:          0
duplicate_query_ids: 0
```

运行命令：

```bash
python scripts/run_phase7a_pr13a_fixed_replay.py \
  --out-dir artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714
pytest -q tests/test_phase7a_pr13a_query_contract.py \
  tests/test_phase7a_pr13a_fixed_replay.py
```

首轮相关回归为 12 passed；PR-13A 新增专项为 4 passed。最终全量测试和静态检查在提交前
另行记录。

## 明确限制与下一门禁

- 当前工件是 CPU/two-ReLU smoke，不能支撑 throughput、tail latency 或 time-to-verify claim；
- 尚无 `QueryBatch`、deadline/timeout flush、memory-budget packing 或 OOM split/retry；
- 尚未把 PR-12 compile-aware multi-backend Planner 接入 BaB αβ query；
- PR-13A 只证明 logical batch 的 pack/unpack 顺序与结果不变；真正 batched executor、动态
  packing 和 fill/latency 将由 PR-13B 关闭；
- 当前 solver result 为该人为 threshold 下的 `unsafe` smoke，不用于论文 solver-quality 结论；
- third-party 源码和 PR-12 TIR/schedule 均未修改。

下一阶段唯一主线是 PR-13B dynamic `BatchManager`：compatibility 分桶、显存预算、partial
flush、deadline、OOM 拆批、结果顺序恢复，以及 queue-wait/fill/latency 观测。PR-13C same-solver
adapter 和性能实验不得越过这一 correctness 门禁。
