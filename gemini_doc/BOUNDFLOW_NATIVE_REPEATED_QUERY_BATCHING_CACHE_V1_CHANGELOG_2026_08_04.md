---
status: completed
updated: 2026-08-04T07:00:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1
stage: s01
---

# Native Repeated-Query Batching and Cache v1 Changelog

## Summary

- NRIR-7 在 NRIR-6 合并后启动，首次把 native real-network compiler 放入多 query stream，目标
  是从单 query 内 slicing 进入真实 query formation/cache/结果恢复。

## Changes

- 用 DocOps 在 `gemini_doc/` 创建 plan/changelog并冻结 v1 边界。
- 新增 typed query spec/range/layout、compile result、per-query result、packed/serial traces 与
  exact in-process compilation cache。
- cache key 包含 workload/state/input/intermediate-bound、ordered query contents、budget/policy/
  batch configuration；首次 miss、完全相同第二次 hit，objective/order/state 变化均 miss。
- packed path 将 9 条 property queries 组成 9-spec source，并以 size-3 执行 3 个 child；serial
  reference 在同一 source representation/storage policy 下独立执行 9 条 query。
- packed aggregate 按 exact ranges 恢复 9 个 query IDs/results；新增真实 artifact replay 与
  query/cache/result/semantic/gate/claim tamper tests。

## Validation

- toy：4 query packed-2 vs serial-4，dense/structured 两种 policy 均逐 query 一致；cache 与
  objective/order/state tamper 门禁通过。
- fixed ResNet：9 个不同 property objectives 显式成为 9 queries；packed 3 child、serial 9
  child；first miss/second hit，9/9 lineage 恢复。
- packed/cached max diff `0`；packed/serial max diff `3.2186508178710938e-06`；packed/external
  `1.9073486328125e-06`；serial/external `3.2186508178710938e-06`；均 allclose、sign 9/9。
- artifact generate/replay exit 0；聚焦 `121 passed`；全量
  `540 passed, 37 skipped, 7 warnings in 130.24s`。
- Black/Mypy clean、Pylint 10.00/10、`git diff --check` 通过。
- DocOps change `ev001253`、validation pass `ev001254` 已记录；`dol validate` 与
  `dol lint --soft` 均通过。

## Decisions

- 9 个不同 property objectives 是 9 条 query，不再把它们只描述为一个匿名 spec tensor。
- child-count reduction 是机制计数，不是性能证据。
- cache 只允许 exact workload/state/query/policy identity reuse。

## Follow-Ups

- 发布并合并 NRIR-7。
- 下一切片进入 BaB parent/child domain batch：显式 state validity、不同 input boxes、domain-axis
  packing/restore；在此之前不得把 property packing 等同于 BaB runtime。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
