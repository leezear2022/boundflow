---
status: implemented-pending-run
updated: 2026-08-24T12:40:00+08:00
type: changelog
topic: boundflow
slug: cibc-ibp-horizontal-formal-protocol
stage: s01
---

# CIBC IBP Horizontal Formal Protocol Changelog

## 目标

把CIBC水平融合从单进程诊断推进到可独立重放的正式operator与whole-model证据。该协议独立于已经
NO-GO的B4纵向alpha-CROWN路线，不继承其claim。

## 冻结协议

- workload固定为VNN-COMP 2021 ResNet2B property 0生产capture与ONNX digest；
- operator层覆盖真实图中6个Conv ordinal `0/2/4/5/8/10`；
- schedule候选在运行前固定为`64/128/256` threads，每个候选由独立进程测量6个signature；
- 每个operator为30组×500次，baseline/candidate顺序按`BC/CB/BC`冻结；
- schedule只按6算子speedup geomean选择；不允许看whole-model结果后改候选；
- whole-model使用6个独立进程、`BC/CB`交替，每组100次、每进程30组；
- baseline与candidate都用CUDA Graph；每次replay均包含lower/upper输入copy；
- 正确性检查覆盖全部中间interval和最终logit，要求absolute diff≤`3e-4`且sign exact；
- performance gate为operator geomean≥`2x`、operator worst≥`1.2x`、whole-model geomean≥`1.5x`、
  whole-model worst≥`1.2x`；
- 目标硬件固定RTX 4060 Laptop、compute capability `8.9`；raw、protocol、summary、manifest均有
  canonical SHA256，root replay从raw重算全部统计量。

## Claim边界

正式运行完成前`performance_claimed=false`。即使通过，也只成立“该GPU、该ResNet2B、IBP
forward-bound、6 Conv水平融合”的reduced claim；不能外推到alpha-CROWN/BaB、其他GPU、显存收益或
ASPLOS完整系统结论。显存需要baseline/candidate分进程的独立artifact，不从本轮两计划共驻留推断。

## 验证范围

新增artifact contract tests覆盖schedule选择、全重签timing派生篡改和Conv coverage篡改；正式提交后
才可生成artifact，保证`source_git_head`和8个code blob可由git历史重建。
