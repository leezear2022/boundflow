---
status: validated-no-go
updated: 2026-08-04T08:28:00Z
type: changelog
topic: boundflow
slug: per-child-objective-refinement-v1
stage: s01
---

# BoundFlow Per-Child Objective Refinement V1 Changelog

## Summary

- NRIR-21 已完成：per-child exact-state refinement 的 IR/control、queue lineage、replay 与
  packed execution 全部成立；固定 ResNet clauses 0/1 的最差 depth-2 leaf lower 均弱于
  root-global，故该 tightness 策略以 `VALIDATED-NO-GO` 关闭。

## Changes

- `native_optimized_relu_split_bab_runtime.py` 新增 opt-in per-child objective refinement：每个
  node 依据自身 split state 独立执行 forward IBP、objective influence、target selection、selected
  CROWN、intersection 与 propagation；之后才拼为 optimizer domain batch。
- 新增 `NativePerChildRefinementTrace`，并把 node split、refinement Plan/Task/Schedule、去 timing
  semantic trace、initial/final intermediate bounds 与 target count 一一绑定到 evaluation。
- parent alpha/beta 仅作 `monotonic_split_refinement` warm initialization；parent refined bounds
  不作为 child exact state。默认关闭时 queue/evaluation 不序列化新字段，旧 NRIR-20 payload/hash
  保持兼容。
- 新增真实 generate/replay runner、digest-bound evidence、四类 artifact/tamper tests；工件位于
  `artifacts/per-child-objective-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`。
- claims map、execution memo、current status、master plan、README 与总 change log 已同步。

## Validation

- focused：`17 passed`。
- fresh generate：closure=`validated_no_go`，evidence hash=
  `976adc4a50c53592ce5bc011b36d0ee4d1e09927c6c0bccf662aee8eded3d310`。
- fresh replay：四组 semantic result hash 为
  `1059f5f35432726cc08ed19f35e954038dc3666d4e639d6d97766e85f30fbcc8`、
  `beaecfa46cf4b0feefa21971ae608b241ab58f7169e733d19dcceb24f5a07da3`、
  `864e225c2bc6551fb6d5d210d75bc076d6066002fd7cd3243407e6d87a24ecb9`、
  `380f9f577edbd95b8c92cf199ca62e25c063cdf98ae9e98a9853df93a4e6af77`。
- 旧 NRIR-20 current-code probe：clause 0 objective queue trace 与冻结 payload 精确相等，hash=
  `f4e54eb05c8517b771bd99747ad055a0f5c2f093d167878437d3c18f6ea98e2c`。
- 全量回归：`749 passed, 37 skipped`；37 个 skip 均为 CUDA/TVM 环境边界。
- Black、targeted Mypy、Pylint `10.00/10`、`git diff --check` 通过。

## Decisions

- 不把扩大 root target 数量当作 per-child 替代：NRIR-20 的 32/64-target sensitivity 虽继续
  改善，但 clauses 0/1 root lower 仍显著为负。
- 保持 packed optimizer batch；先逐 node 生成正确的 refined semantics，再拼接 batch。
- fixed clauses `0/1` 的 root-global/per-child root lower 均精确等于
  `-417.292480/-602.551392`；root-global worst leaf 为
  `-413.739044/-591.944275`，per-child 为 `-414.587006/-592.880920`，delta=
  `-0.847961/-0.936646`。结果没有被选择性隐藏。
- 原因归因：independent child recomputation 虽使用 exact split，却丢弃了 root/parent 已证明的
  selected-CROWN tightening；相同 shortlist 预算不能保证覆盖祖先选中的有效约束。

## Follow-Ups

- 下一分支实现 ancestral-constraint carry-forward refinement：child initial state 必须是 local
  exact-split forward 与 parent proven refined constraints 的单调交集/propagation，再做 child
  objective target selection；继续禁止把 parent bounds 冒充 child exact result。
- 仍需完成 DocOps closure、PR 与 merge；不启动 CUDA timing 或扩大 tree 掩盖当前 No-Go。

## Links

- plan: `gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_PLAN_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
