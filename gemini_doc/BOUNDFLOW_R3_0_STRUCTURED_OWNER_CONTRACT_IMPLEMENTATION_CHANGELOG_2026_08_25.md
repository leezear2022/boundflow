---
status: implemented-pending-clean-source-formal
updated: 2026-08-25T04:05:00+08:00
type: changelog
topic: boundflow
slug: r3-0-structured-owner-contract-implementation
stage: s01
---

# BoundFlow R3-0 Structured Owner Contract 修改记录

## Summary

R3-0 已实现 first-class lower-region DAG、Template/Instance 分离、closure/fanout/bias ownership、
bounded scratch liveness、dense escape、context tensor reachability、saved-tensor ledger 与 fail-closed
receipt。该阶段仍是 contract-only：没有 production 接入、custom backward、CUDA kernel、timing 或
performance claim。clean-source formal artifact/replay 通过前不开放 R3-1。

## Changes

- `boundflow/ir/structured_lower_region.py`
  - 冻结九类节点：SpecSeed、ReluLowerTransform、LinearRight、Conv2dRight、Add、Reshape、Slice、
    BiasSplit、InputConcretize；
  - node field/arity/typed-attribute、拓扑序、consumer count、root closure、escaped consumer、
    `node_count <= 4x source_op_count` 全部 fail closed；
  - BiasSplit 用整数 fraction witness 证明子 token 之和恰为父 token；
  - scratch 仅允许 slot 0/1，拒绝越界与同 slot live interval overlap；
  - `StructuredCoefficientHandleV1.to_dense()` 永远抛 `StructuredDenseEscapeError`；
  - Instance 只保存 tensor identity metadata，不持有 Tensor；只有 α/β 可 `requires_grad=True`，
    empty beta `(6,0)` 合法，scratch count `<=2`；
  - saved-tensor ledger 独立重算 logical/unique storage bytes，并拒绝 coefficient lineage；
  - recursive context checker 可穿透 dataclass/container/对象属性并拒绝任何 Tensor；
  - R3-0 receipt 强制 production/timing/performance=`false`，dense/context/coefficient=`0`。
- `boundflow/runtime/r3_structured_owner_contract.py`
  - 构造固定 P-anchor contract-only DAG、fanout witness、两 scratch liveness、empty beta、saved-state
    预算和 receipt；
  - semantic replay 逐层 parse exact fields、重算 template/instance/bundle/summary hashes，并与冻结
    bundle 比较。
- `scripts/run_r3_structured_owner_artifact.py`
  - clean committed source、code blob、atomic generate、manifest 与 replay；
  - replay 读取 recorded commit blob，不依赖当前工作树源码内容。
  - formal preflight 首次运行发现通用 git helper 的 `.strip()` 会破坏 porcelain leading status，令
    `.docops/ev.jsonl` 被误解析为 `docops/ev.jsonl`；已改为保留原始 stdout 的专用 parser，并新增
    dot-path/rename destination 测试。该失败发生在 artifact 创建和合同执行前，没有实验结果可采信。
- `scripts/probe_r3_structured_owner_tamper.py`
  - 12 类 topology/state/ownership/liveness/claim 全重签 mutation 探针。
- tests：新增 40 个合同/负向测试，覆盖 topology、closure、source expansion、bias ownership、scratch、
  dense escape、recursive context、saved ledger、instance identity 和 forbidden claims。

## Validation

- R3-0 targeted：`40 passed`；
- mypy：IR/runtime clean；
- Black：pass；Pylint：目标 `10.00/10`；
- formal artifact 尚未运行；必须先提交 clean source，再由 artifact runner 从该 commit 生成。

## Claim Boundary

- 当前状态仅为 `IMPLEMENTED-R3-0-PENDING-CLEAN-SOURCE-FORMAL`；
- `production_connected=false`、`timing_recorded=false`、`performance_claimed=false`；
- 不声称 custom VJP、dα/dβ correctness、saved-state 实测、memory improvement 或 speedup；
- R3-1 在 formal artifact/replay 与 tamper 全通过前保持关闭。

## Next

1. 提交 R3-0 contract source；
2. 从 clean commit 生成 artifact，replay，并运行 12 类 fully re-signed tamper；
3. 同步 claims/memo/status；若通过，才开放 R3-1 P-anchor mandatory custom backward correctness。

## Links

- design：`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`
- R1 reprioritization：`BOUNDFLOW_CIBC_R1_A_FORMAL_NO_GO_CLOSURE_2026_08_25.md`
