---
status: completed
updated: 2026-08-04T06:13:29Z
type: changelog
topic: boundflow
slug: objective-directed-intermediate-refinement-v1
stage: s01
---

# BoundFlow Objective-Directed Intermediate Refinement V1 Changelog

## Summary

- NRIR-20 已完成：objective-directed intermediate target selection 已成为可哈希、可执行、
  可语义重放的 Plan/Task/Schedule；fixed ResNet 双子句 same-budget 根界均改善。

## Changes

- 新增 `objective_influence_width_per_relu_v1`，以 clause-sensitive backward coefficient
  influence × ambiguous width 排序。
- Plan 冻结 objective hash；target 冻结 influence 与 selection score；objective influence 成为
  SELECT_TARGETS 的显式 Task/Schedule 输入，运行时不再旁路读取未声明依赖。
- CROWN backward 可返回每个 ReLU 的 upper/lower coefficient 最大绝对影响力；旧调用路径显式
  关闭采集，保持默认行为。
- 默认 width policy 的 payload/hash 保持兼容，NRIR-19 artifact replay 已通过。
- 新增 fresh-process 双子句/双 policy runner、manifest/log digest、source-to-IR semantic replay
  与 coherent relink tamper tests。
- NRIR-19 replay 遇到新代码 revision 时不再在语义检查前退出：同时披露 artifact/replay
  revision 与 match flag，并要求旧 width Plan/Task/Schedule 从冻结 source 精确重编译匹配。

## Validation

- focused：`16 passed`；CROWN/ResNet/refinement 扩展 focused：`27 passed`；全量
  `739 passed, 37 skipped`。
- ResNet same-budget 根节点探针：clause 0 width/objective lower
  `-473.221222/-417.292480`，objective 再改善 `+55.928741`；clause 1
  `-628.780334/-602.551392`，再改善 `+26.228943`。
- 两个 clause 均为 width/objective 各 96 targets；target overlap 分别 `16/96`、`27/96`。
- artifact fresh semantic replay hash=
  `8fce1c7c3e5c63adb14a7ab5b9f23407e4a7a1406353750e4f150ee745b4e88e`；Black、targeted
  Mypy、Pylint 10.00/10 与 `git diff --check` 通过。
- NRIR-19 跨 revision semantic replay 通过，`native_code_revision_match=false` 被显式披露，
  三 workload 的旧 width Plan/Task/Schedule hashes 全部精确恢复。
- 16-target 第二 pass 与 32/64-target 开发敏感性探针继续改善但 root lower 仍负；只用于选择
  下一路线，不写成冻结性能结论。

## Decisions

- 继续 objective-directed 路线并生成 artifact；不因探针 timing 形成 speedup claim。
- 每个性质子句独立编译 objective-aware refinement Plan；不把多子句 max influence 冒充
  clause-sensitive selection。
- 本轮先验证 root-global same-budget tightness；只有仍不足时才进入 per-child refinement。
- 本轮以 objective-directed IR/control + fixed-root tightness `VALIDATED-REDUCED` 关闭；
  `performance_claimed=false`、ASPLOS-ready=NO。

## Follow-Ups

- 下一分支实现 per-child objective-directed refinement：每个 child 按 exact split state 重算
  intermediate bounds、influence 与 Plan/Task/Schedule；parent 结果只允许 warm-start。

## Links

- plan: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_PLAN_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
