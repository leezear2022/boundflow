---
status: validated-reduced
updated: 2026-08-05T02:24:45Z
type: plan
topic: boundflow
slug: prepared-intermediate-refinement-capsule-v1
stage: s01
---

# BoundFlow Prepared Intermediate Refinement Capsule v1

## Goal

- 把 production queue 中 per-child intermediate refinement 的完整 validate/hash/target-selection 从“每次
  被引用都递归重算”改为“每个 exact program 在 prepare 时完整准入一次，runtime 消费 immutable
  capsule/token”；
- 保持 child refinement target、selected CROWN、bounds、optimizer、objective branch、queue、31/depth4、
  NRIR-44 projected floor 与 global-60s deadline 完全不变；
- fixed ResNet2B property 0 CPU8 上，先使 clauses 2/3 的 31-node queue median ratio 均 `<=0.80`，
  再使 whole execution trace `<=40 s`、measured end-to-end `<=50 s`；
- prepared capsule 必须 fail closed，不能把“少验证”实现成无条件信任可变 Tensor 或跳过首次准入。

## Scope

### 基线与诊断

- integration base=`main@b6eb697`；NRIR-44 merge=`f194034`、feature=`437680e`；
- frozen NRIR-44 Phase A formal hash=
  `ecb553d88be065054abb0a480b79086ae12cec55a84e5c0ba537572e904ff0fe`；
- frozen NRIR-44 Phase B payload hash=
  `2f22d44fe9f57f233c8a853b66f67f404b03a087d097451e10f663ee257272d9`；
- NRIR-44 whole trace=`43.571040/44.144990/44.095736 s`，measured wall=
  `58.666217/59.332377/59.300634 s`；floor 仅约 8.6 秒，余量主要在两条 production queues；
- 路线前 cProfile 只读诊断：clause 2 的 31-node queue 在 profiler 下为 26.224 秒、87,638,150 calls；
  `_execute_per_child_refinements` cumulative 14.463 秒，30 次 execute/compile 分别 8.059/5.499 秒；
  `_select_targets` 共 246 次，其中 186 次由重复 `Program.validate()` 触发，语义 compile/runtime 各 30；
- “每个 exact program/Execution 仍完整验证一次”的只读 ceiling probe 将 clause 3 queue trace 从
  约 `12.85 s` 降至 `9.761678 s`，31 nodes 与 worst-active lower `-30.258447647094727` 不变。
  以上 probe 只用于选路线，不是正式 claim。

### 唯一变量

只改变 intermediate-refinement validation ownership：

- prepare：完整验证 Program/Task/Schedule、source lineage、split/objective/input/intermediate hashes、
  target table 与 execution result，生成 typed immutable capsule；
- runtime：Schedule 显式消费 capsule/Plan-owned targets 和 semantic token；保留每次 execute 恰好一次
  target confirmation，但不再为同一 exact object 递归重跑 forward、target selection、JSON
  serialization 与 stable hash；
- replay/audit：仍能从 artifact 重建 capsule 并重新执行完整验证，不能只核对外层 token；
- mutation/stale source/wrong target/wrong split/wrong result 必须 fail closed。

冻结不改：NRIR-44 projected floor、rank/top-2、child refinement policy/pass/cap、selected-CROWN math、
optimizer steps/state、objective branch candidate/score、31/depth4、queue order、cache、dtype、threads、
workload、threshold、global deadline 与 final aggregation。

### IR 所有权

- `PreparedIntermediateRefinementCapsuleIR` 绑定 source Plan/Task/Schedule、input/split/objective/source
  refinement、target table、result bounds 与首次完整 validation receipt；
- prepared Program/Execution wrapper 只接受 capsule 中 exact owner，不允许 object-ID-only 或裸 bool cache；
- prepared Task/Schedule 将 `ADMIT_EXACT → CONSUME_PLAN_TARGETS → EXECUTE_SELECTED_CROWN →
  COMMIT_RESULT → EMIT_RECEIPT` 固定为显式阶段；
- 兼容投影只在 additive NRIR-45 production runtime 内使用，frozen NRIR-31/42/44 文件不改。

## Tasks

### A. Prepared refinement IR/runtime

- [x] 实现 capsule/receipt Plan、Task、Schedule 与 deterministic hash；
- [x] 首次完整 validate 后封装 exact Program/Execution；
- [x] prepared execute 每次只保留一次 target confirmation，不因 validate/hash 再次 `_select_targets`；
- [x] stale/mutation/source/split/objective/target/result tamper fail closed。

### B. Phase A per-clause formal

- [x] clauses 2/3 three fresh counterbalanced control/prepared 31-node queues；
- [x] queue/branch/score/state/refinement/selected-CROWN/worst lower exact；
- [x] full Program validation 与 `_select_targets` 调用按 ownership 收敛，无隐藏重算；
- [x] 两条 prepared/control queue median ratio 均 `<=0.80` 且改善大于 pooled MAD；
- [x] typed replay 与 synchronized outer-rehash tamper 通过。

### C. Phase B whole query

- [x] 仅 Phase A 全过后接 NRIR-44 projected global runtime；
- [x] three fresh global-60s queries，floor/rank/selected 与 `[31,31]` nodes exact；
- [x] 每轮 execution trace `<=40 s`、measured wall `<=50 s`；
- [x] trace median ratio vs NRIR-44 `<=0.90`、measured median ratio `<=0.85`，改善大于 pooled MAD；
- [x] final 9/9 unresolved、replay/tamper/full pytest/Black/mypy/Pylint 全过；DocOps 随关闭提交记录。

## Validation

Phase A 任一 correctness/ownership/timing gate 失败即 `VALIDATED-NO-GO`，不得启动 Phase B。Phase B
只有全部门禁通过才允许 fixed ResNet2B property 0 CPU8 prepared-refinement production admission
`VALIDATED-REDUCED`；`performance_claimed=false`。本阶段不构成公平竞品 speedup、GPU、多 workload、
property closure 或 ASPLOS-ready。

正式判定：Phase A/B 全部门禁通过，以 fixed ResNet2B property 0 CPU8 internal admission
`VALIDATED-REDUCED` 关闭；精确结果与 hash 见配套 changelog。final 仍为 9/9 unknown，且
`performance_claimed=false`，不升级为公平竞品、GPU、多 workload、property closure 或 ASPLOS-ready。

## Rollback

- 全部实现 additive；失败时继续使用 NRIR-44 runtime；
- 不修改 frozen refinement policy 或 source artifact，不以降低 target cap/pass、减少 nodes、放宽门禁、
  跳过首次验证或 object-ID cache 换取收益；
- 若 Phase A 正确但 timing 未过，保留 prepared ownership IR 机制但不得接 production。

## Links

- changelog: `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_CHANGELOG_2026_08_05.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
