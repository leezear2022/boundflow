---
status: validated
updated: 2026-08-25T01:26:00+08:00
type: changelog
topic: boundflow
slug: cibc-r1-0-contract-clock-topology
stage: s01
---

# BoundFlow CIBC R1-0 Contract/Clock/Topology 修改记录

## Summary

- 完成 R1-0 的 scope target、source topology、CUPTI↔host clock、event owner、四种时间口径与
  query-local Amdahl route 的 fail-closed v1 合同。
- 本轮只证明 schema/derivation/replay 机制；`performance_claimed=false`，未开放 R1-A formal share、
  R1-B same-solver admission 或 R2 优化。

## Changes

- 新增 `boundflow/runtime/cibc_r1_attribution.py`：
  - 固定 complete-query `1.00x/1.15x` 与 queue `1.20x` 三项目标，拒绝跨 scope 代入；
  - 从 `BoundTask` 与运行前 shape/dtype/device metadata 构造 ordinal/topology/marker identity；
  - 保存 trace 前后各 64 个原始 host/CUPTI triplet，并从 raw 重算 affine fit、bracket、residual、
    slope/anchor drift 与 Nsight anchor admission；
  - event owner 只接受 correlation parent、graph node 或显式 runtime scope，unowned 与 temporal fallback
    必须为零；
  - 分离 `kernel_sum`、exclusive wall、critical path、overlap-adjusted wall，单 stream/无 overlap 时强制
    退化一致；
  - complete-query projection 只接受 query-local `G_query,k`；无 candidate 强制 `G=1`，独立 graph
    `2.45631x` 不得进入 query route；
  - 所有 serialized receipt 均有 exact-field parser，从 primitive raw 重算 hash、派生量与 verdict，
    不信任外层重签摘要。
- CUPTI collector 固定 1 ms monotonic-raw 采样基线，避免连续瞬时采样使 slope fit 数值病态；不修改
  预注册的 100 ppm/2 us/10 us 门槛。
- 新增 `tests/test_cibc_r1_attribution.py`，覆盖 target/scope、topology、clock、owner、timing、route 与
  更新外层 digest 后的篡改拒绝。

## Validation

- `pytest -q tests/test_cibc_r1_attribution.py`：`24 passed`。
- R1 + CIBC interval/CUDA Graph + FSG3/B3 same-solver artifact 组合：`40 passed`。
- `mypy --follow-imports=skip`（R1 module + tests）：clean。
- `pylint`（R1 module + tests）：`10.00/10`。
- `black --check --target-version py312` 与 `git diff --check`：通过。
- RTX 4060 Laptop 上真实 `/opt/cuda/lib64/libcupti.so` smoke 共 3 次：2 次
  `cupti_admitted=true`（p95 `1162/1333 ns`、max `3236/3396 ns`、residual `512/586 ns`、
  slope drift `1.66/0.52 ppm`）；1 次因 OS 抢占出现 max bracket `45004 ns`、residual `20997 ns`，
  被原门禁拒绝。三次均因缺 Nsight anchor 而 `formal_admitted=false`。

## Decisions

- 不放宽 calibration threshold，也不删除失败样本；fresh process 若遇 OS 抢占就使该 worker fail closed。
- 本机没有 `nsys`，因此当前 live smoke 只能证明 CUPTI 合同可执行，不能形成 formal attribution claim。
- source topology 在 profile 前冻结；kernel shape 只允许从 correlation parent 恢复，不提供 temporal
  fallback。

## Follow-Ups

- 实现 R1-A additive/opt-in marker 与 control/profile worker；先保留 smoke-only 输出。
- 实现 canonical raw artifact/replay 与 worker inventory/order 门禁。
- 在具备 Nsight Systems 的 qualified host 上生成至少 3 个 NVTX/CUPTI anchor 后，才允许 R1-A formal。

## Links

- plan: `gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- roadmap:
