---
status: completed
updated: 2026-08-04T04:55:03Z
type: changelog
topic: boundflow
slug: native-intermediate-bound-refinement-v1
stage: s01
---

# BoundFlow Native Intermediate-Bound Refinement V1 Changelog

## Summary

- NRIR-19 已完成。selected plain-CROWN intermediate refinement 已成为 native
  Plan/Task/Schedule，并在三类真实 VNN-COMP workload 上做 same-policy baseline/refined 对照。

## Changes

- 建立独立 plan/changelog；冻结 IR-first、native provenance、selected CROWN 与三 workload
  baseline/refined 对照边界。
- plain CROWN 支持任意已产生中间张量和显式 selected-row objective；卷积张量不再要求先物化
  整层 identity matrix。
- 新增 top ambiguous-width per-ReLU target policy、分块 backward、单调 intersection、逐 pass
  forward propagation，以及 action/pass trace。
- 新增 `native_refined` intermediate provenance，并贯穿 optimizer、selected native Bound IR 与
  BaB child batch；不复用或冒充 `external_verifier`。
- 新增六 fresh-process generate/replay runner、source-to-IR 重编译、manifest/log digest 与 tamper tests。

## Validation

- MNISTFC：unresolved `{3,7,8}→{8}`，关闭 clauses `3/7`，nodes `31→21`；九个 root
  lower 全改善 `+0.096706` 至 `+0.125973`。
- ResNet2B：状态仍 unknown，unresolved/pending 不变；两个已完成 root lower 从
  `-543.717/-789.331` 改为 `-473.221/-628.780`，改善 `+70.496/+160.551`。
- OVAL21：`unknown→verified`，关闭 clause `8`，nodes `15→11`；clause 8 root lower
  `-0.173439→-0.002876`。
- refinement selected/tightened neuron：MNISTFC `99/137`、ResNet `96/2591`、OVAL21
  `107/217`；worker 内 refinement 分别约 `21.8/114.3/32.1 ms`，只作诊断。
- artifact fresh replay hash=
  `f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；focused
  `9 passed`；全量 `732 passed, 37 skipped`；Black、Mypy、Pylint 10.00/10 全过。

## Decisions

- 不复用 `external_verifier` provenance 表示 native refinement。
- 不先做 CUDA timing；先回答中间层收紧是否改善真实 hard clauses。
- 以 native refinement IR/control 与 multiworkload tightness `VALIDATED-REDUCED` 关闭；不是
  3/3 complete closure、GPU 或 performance claim，ASPLOS-ready 仍为 NO。

## Follow-Ups

- ResNet 仍是主 blocker；当时冻结的 objective-directed intermediate target selection 已由
  NRIR-20 完成，same-budget root tightness 改善但未闭合；当前下一路线为 per-child refinement。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
