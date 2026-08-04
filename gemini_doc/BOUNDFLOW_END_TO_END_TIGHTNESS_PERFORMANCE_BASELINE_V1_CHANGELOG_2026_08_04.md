---
status: completed
updated: 2026-08-04T02:41:43Z
type: changelog
topic: boundflow
slug: end-to-end-tightness-performance-baseline-v1
stage: s01
---

# End-to-End Tightness and Performance Baseline v1 Changelog

## Summary

- NRIR-15 已启动。首轮只读探针确认：当前 9/9 unknown 的主嫌疑不是 optimizer steps 太少，
  而是 optimized queue 未继承 NRIR-1/RVIR 已验证的 external intermediate semantics；同时
  audit-mode 单 clause CPU wall time 约 6.4—7.2 秒，不能作为 production 性能路径。

## Changes

- 新建 standalone plan/changelog，冻结 tightness、phase、execution-mode 与重复测量边界。
- optimizer/queue/query 新增 typed external-intermediate opt-in。external ReLU bounds 与
  `EXTERNAL_VERIFIER` provenance 必须同时出现；ordinal/key/shape/dtype/device/finiteness/
  lower≤upper 任一不符即 fail closed。child batch 会先 repeat frozen root intervals，再与
  active/inactive split 相交；parent warm scope 使用 parent split 后的同一外部语义。
- `NativeAlphaBetaOptimizerPolicy` 新增 `adaptive` α 初始化；positive=1、negative=0、ambiguous
  按 `upper > -lower` 选择 1/0。默认 `constant` 不写入 canonical payload，因此 NRIR-10—14
  历史 policy hash 保持不变。
- 新增 `run_end_to_end_tightness_performance_baseline.py` 与 frozen artifact。runner 记录
  local-constant / external-constant / external-adaptive 三种 audit queue，1 warmup、3 个轮换
  group、raw/median/p90，并把 setup/candidate/queue/verdict 分离。

## Validation

- frozen external initial lower：
  `[-0.54294, 4.32518, -0.52819, 0.83001, -0.59845, 0.10807, 0.67722, 2.72719, 3.10527]`，
  initial-CROWN 已可直接证明 6/9 clauses。
- current local-intermediate optimized root lower（steps=1）：约 `-408.01…-863.19`，0/9；
  与 external 的差距为数百量级。
- optimizer steps `0/1/2/4` 的九子句总 wall time分别约
  `58.67/59.16/59.13/59.38` 秒；steps=4 虽改善下界，仍为约 `-383.69…-830.18`，0/9。
  步数增加几乎不改变总时长，表明固定 compile/validation 成本占主导且单纯加步数不能修复语义差距。
- 正式 external-adaptive 完整查询把 fixed ResNet 从 local `0/9` 提升为 `6/9 verified`；仅
  clauses `0/2/4` 保持 unknown。九个 lower 对 frozen external initial 均无退化，最大改善
  `0.0072252750`，sign agreement `9/9`；fresh replay 得到相同 evidence hash
  `14c3b9dc2e5376156be1f33f3e8804ec21f60e11096bd3bdc95225b7e1474376`。
- clause 0 的三组 CPU audit queue median：local-constant `6.7178 s`、external-constant
  `6.7969 s`、external-adaptive `6.7317 s`；candidate search `3.612 ms`，三种 verdict
  `3.899/3.939/3.922 ms`。语义/初始化对 audit wall time 无实质影响，固定 compiler/hash/
  selected-native re-execution 是当前主耗时。
- external child-batch 首轮探针曾因 root interval 未叠加 split constraint 被 fail-closed 拒绝；
  修复为 external interval 与 per-node split 相交后，真实多层 child warm-start 运行通过。
- focused runtime/artifact/tamper 回归为 `35 passed`；全量 `684 passed, 37 skipped`；Mypy
  clean、Pylint 10.00/10、Black/diff check 通过。

## Decisions

- external bridge 已关闭主要语义断层，但只证明 6/9；clauses 0/2/4 需要后续 branching/
  stronger bound，不能靠增加固定 α steps 冒充闭合。
- 现有 queue 是 audit/validation path：scheduled optimizer 后又 compile/execute selected native
  stack 并比较；正式性能基线必须单列，不能冒充 production fast path。
- 下一单一工程优先级确定为 prepared production fast path：复用已验证 compilation/program，
  production 执行不做 selected-native 双执行；先建立与 audit path 数值/状态一致门禁，再重新
  计时。该优化关闭后再对三个 hard clauses 启动外部语义下的 branching/tightness 路线。

## Follow-Ups

- 实现 prepared production execution capsule，分离一次性 compile/validate 与 repeated execute；
  使用相同三组轮换协议比较 audit/production，仍不得升级为跨 workload 或竞品 speedup claim。

## Links

- plan: `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
- artifact: `artifacts/end-to-end-tightness-performance/vnncomp21-resnet2b-prop0-cpu-v1/`
