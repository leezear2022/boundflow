---
status: completed
updated: 2026-08-03T22:04:06Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1
stage: s01
---

# Native BaB Domain Batching v1 Changelog

## Summary

- NRIR-8 在 NRIR-7 合并后启动，从同域 property queries 推进到不同 input boxes 与显式
  parent/child state validity 的 domain batching。

## Changes

- 用 DocOps 创建 plan/changelog并冻结 input-box-only、no-performance 边界。
- 新增 domain BatchCandidate variant compiler；full 与 size-4 保持 domain/spec/sample 三轴独立，
  query-time `max_domain_batch_size` 进入通用 selector 与 PlanInstance identity。
- 新增 typed leaf/parent query、exact-state、source Schedule→child IR binding、packed/serial execution
  trace；parent state 强制 `warm_start_only` 且 `consumed_as_exact=false`。
- 每个 leaf box 独立运行 forward IBP；child InputSpec、interval/ReLU state 与 objective 按 domain
  axis 组成真实 batched payload，执行结果按 leaf order 恢复。
- 新增 fixed ResNet generate/replay artifact；packed-4/full-8/serial-1 三路径分别执行 2/1/8 个
  native child stacks，8×1 lower/upper bitwise equal。
- 同步 execution memo、claims map、current status、master plan、README 与 change log。

## Validation

- fixed ResNet artifact generate/replay：8 项 gate 全过；packed/full/serial max diff `0.0`。
- focused runtime/artifact/tamper tests：`19 passed`。
- Black、Mypy clean、Pylint `10.00/10`、`git diff --check` 已通过。
- 全量 pytest：`559 passed, 37 skipped`（7 条已知环境/依赖 warning）。
- DocOps validate/lint closure 结果见本轮 `.docops` change/validation events。

## Decisions

- parent intermediate state 只能 warm-start；每个 child execution 必须消费其重新计算的 exact state。
- domain child-count reduction 不是完整 BaB 或 speedup 证据。

## Follow-Ups

- 下一阶段实现 native ReLU-split state、priority queue/branch/prune control flow；在此之前不把
  input-box split 写成完整 BaB，也不以 2 vs 8 child stacks 声称 speedup。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
