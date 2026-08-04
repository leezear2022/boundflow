---
status: active
updated: 2026-08-04T07:06:25Z
type: plan
topic: boundflow
slug: ancestral-constraint-refinement-v1
stage: s01
---

# BoundFlow Ancestral Constraint Refinement V1 Plan

## Goal

- 修复 NRIR-21 的结构性退化：independent child refinement 从 local exact-split forward 重新开始，
  会丢失 root/parent selected-CROWN 已证明的 tightening。NRIR-22 要让 child 以 local forward 与
  parent proven refined constraints 的单调交集为初始状态，再执行 objective-directed refinement。
- 将 ancestral constraint source 作为 refinement Plan/Task/Schedule 的显式输入与 hash，而不是
  runtime 隐式 side channel；继续保持 parent alpha/beta warm-only 与 bound constraint provenance
  两条语义分离。

## Scope

- `NativeIntermediateRefinementPlanIR` 新增可选 source-constraint、source refinement Plan 与
  source semantic-trace 三哈希；启用时 materialize-forward Task/Schedule 必须显式消费 constraint
  value，Program/Execution 必须保存并验证原始 mapping 与 producer identity。
- queue 新增显式 strategy：`independent_exact_split_v1` 保留 NRIR-21；
  `ancestral_constraint_carry_v1` 的 root 无 source，每个 child source hash 必须等于 parent final
  refinement hash。source 是 sound constraint，不是 child exact bounds，相关 false claim 字段保持。
- 不改变默认 width/objective root refinement、默认 optimized queue 或 NRIR-21 independent payload。
- 不包含 CUDA、latency/speedup、无限树、完整 property closure 或 ASPLOS-ready claim。

## Tasks

1. 扩展 refinement Plan/Task/Schedule 和 execution trace，绑定 optional source constraints。
2. 增加 compile/execute fail-closed admission：identity、shape/dtype/device、finite、lower≤upper、
   infeasible intersection、task dependency 和 source hash 任一漂移必须拒绝。
3. 在 optimized queue 中实现 ancestral strategy：从实际 parent node refinement result 提取约束，
   child 仍按自身 exact split 重新 forward/select/refine；optimizer batch 只接收 child final bounds。
4. 增加 local→constrained-initial→final 双重单调、parent lineage、tamper、packed/serial、旧 hash
   compatibility tests。
5. 生成 fixed ResNet clauses 0/1 的 root-global、independent-per-child、ancestral-carry 三模式
   7-node/depth-2 same-policy artifact，并 source-to-IR replay。
6. 更新 claims/status/memo/index，运行 full/static/DocOps 门禁，发布 PR。

## Validation

- Plan 的 source constraints/producer Plan/producer semantic trace 必须与 validated parent
  execution 精确相等；materialize Task 的输入集合必须在有/无 source 两种模式下确定且不可伪造。
- 对每个 child：source hash=`parent.final_intermediate_bounds_hash`；constrained initial 必须相对
  local exact-split forward 单调收紧；final 必须相对 constrained initial 再单调收紧。
- parent constraint consumption 必须记录为 `sound_constraint_only`，不得设置为 exact-state reuse；
  alpha/beta 仍由既有 `monotonic_refinement_initialization_only` 规则单独约束。
- packed/serial 必须匹配 logical queue、node split、per-node refinement semantic IR/bounds；旧
  independent/root-global/default payload/hash 不变。
- fixed artifact 首要门禁：ancestral-carry 的 worst depth-limit leaf lower 必须严格优于 independent，
  且不得弱于 root-global。否则 `VALIDATED-NO-GO` 并继续分析更强 relaxation/optimizer 方法。

## Rollback

- 新 Plan 字段、Task 输入和 queue strategy 全部 opt-in 条件序列化；移除 strategy/constraint 参数
  即恢复 NRIR-21。旧 artifacts 必须可 replay，禁止改写历史证据。

## Links

- changelog: `gemini_doc/BOUNDFLOW_ANCESTRAL_CONSTRAINT_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
