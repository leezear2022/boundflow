---
status: validated-reduced
updated: 2026-08-05T02:24:45Z
type: changelog
topic: boundflow
slug: prepared-intermediate-refinement-capsule-v1
stage: s01
---

# BoundFlow Prepared Intermediate Refinement Capsule v1 Changelog

## Summary

- NRIR-45 Phase A/B 已按预注册门禁完成，以 fixed ResNet2B property 0 CPU8 internal admission
  `VALIDATED-REDUCED` 关闭；final 仍 9/9 unknown，`performance_claimed=false`。

## Changes

- 冻结 base=`main@b6eb697` 与 NRIR-44 Phase A/B source hashes；
- 唯一变量为 intermediate refinement 的 prepare-once validation ownership；
- 冻结 Phase A per-clause 与 Phase B global correctness/work/timing 门禁；
- 明确首次完整验证、artifact full replay 和 stale/mutation fail-closed 不能删除。
- 新增 prepared capsule、5-stage Task/Schedule/Trace、prepared Program/Execution receipt、additive
  per-child/shared queue 与 projected-floor global composition；frozen NRIR-42/44 路径不改；
- runtime receipt 用 exact owner、容器成员身份与 Tensor mutation version fail closed；full replay 显式绕过
  fast path 重跑历史完整验证，prepared hash token 只缓存已完整准入的 Plan/Task/Schedule digest。

## Validation

- Phase A 六组 exact：control/prepared 的 target-selection=`246→98`、full Program validation=`186→38`、
  full hash=`217→39`；每条 prepared queue 有 30 capsules 并逐一 full replay；
- clause 2 control/prepared median=`12.981239/9.444103 s`、ratio=`0.727519`；clause 3=
  `13.122778/9.666283 s`、ratio=`0.736603`；改善均大于 pooled MAD；Phase-A formal hash=
  `be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`；
- Phase B floor=`8.625022/8.583826/8.628565 s`，whole trace=
  `31.262521/31.319772/31.470078 s`，measured wall=`36.396631/36.513683/36.611709 s`；相对 NRIR-44
  trace/measured median ratio=`0.710268/0.615738`，每轮 selected `[2,3]`、nodes `[31,31]`、
  prepared capsules/full replay=`60/60`；Phase-B payload hash=
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；
- 两阶段 artifact replay、typed payload reconstruction、outer-rehash tamper、Black、mypy、Pylint
  `10.00/10` 与全量 `984 passed, 37 skipped` 通过。

## Decisions

- NRIR-44 已把 floor 降至约 8.6 秒；下一最大可控成本是 top-2 queue 的 child refinement；
- NRIR-43 scorer batching CPU NO-GO 不重开；NRIR-45 不改算法、预算或 policy；
- Phase A 与 Phase B 全部门禁成立；NRIR-45 取代 NRIR-44 作为固定内部 production 路径，但不形成
  公平竞品、多 workload、GPU、property closure 或 ASPLOS-ready claim；
- 下一步先做 NRIR-45 residual phase attribution，冻结新的单变量；优先审计 remaining target selection、
  selected-CROWN 与 queue/aggregate 开销，不重开 NRIR-43 已否决的 CPU scorer batching。

## Follow-Ups

- 发布当前功能/证据分支并接受外部审计；
- 基于最终 `~31.3 s` trace 做 residual attribution 后再预注册 NRIR-46，不事后扫 policy/budget。

## Links

- plan: `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
