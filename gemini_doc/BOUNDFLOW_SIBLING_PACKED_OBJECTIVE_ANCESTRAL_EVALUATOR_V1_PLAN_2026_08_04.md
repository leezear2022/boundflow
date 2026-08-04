---
status: completed
updated: 2026-08-04T18:05:00Z
type: plan
topic: boundflow
slug: sibling-packed-objective-ancestral-evaluator-v1
stage: s01
---

# Sibling-Packed Objective-Ancestral Evaluator v1 Plan

## Goal

- 保持 NRIR-32/33 的 cap128 objective-ancestral semantics、31/depth4 与 60 秒 deadline，消除每对
  siblings 重复进行 optimizer/native evaluator 编译和执行的开销，使 bound-tightness 转化为严格更多
  committed nodes，并最终检验 ResNet hard-clause/full-query closure。

## Scope

- feasibility 固定 ResNet2B property 0 clause 0 的 root-selected 第一对 children；root source、split、
  child refinement Plan 与 optimizer policy exact。
- serial mode 逐 child 执行 refinement + optimizer + selected-native query；packed mode 仍逐 child编译
  exact ancestral refinement，但把两个不同 split domains 合成一个 optimizer/native batch。
- phase profiler 只包裹 child 区间并分别累计 refinement compile/execute、optimizer compile/execute、
  selected-native compile/execute；root source/eval 单独记录。
- feasibility 运行顺序固定 serial→packed；只用于机制/瓶颈定位，不形成 performance claim。

## Tasks

1. [x] 新增只读 first-sibling-pair profiler，输出 exact input/Plan/split/parent-lineage、逐 phase timing、
   lower/upper/branch/state semantics 和 serial↔packed comparison。
2. [x] feasibility gate：两个 child lower/upper max diff `<=1e-5`、split exact、parent source exact；
   packed child elapsed 必须严格小于 serial，并至少减少一个 optimizer/native compile+execute group。
3. [x] gate 通过后新增 sibling-group Plan/Task/Schedule 与 atomic pair commit；late packed pair 整体丢弃，
   不得出现半个 sibling 进入 frontier。
4. [x] 运行 serial-vs-packed 三组交替 fresh repeats，固定 cap128/31/depth4/60 s；packed 每轮 accepted
   nodes 必须严格大于 serial 7，common logical domains bounds/state parity。
5. [x] 正式 gate 通过后回接 ResNet 9 clauses/full query；只在 original verdict/ordinal sound 且新增
   coverage 或可复核 same-algorithm E2E 收益时升级 claim。

## Validation

- packed batch 必须包含同一 parent 生成的 exact `(-1,+1)` siblings；任意不同 parent、缺 child、重复
  branch 或 split/source hash drift fail closed。
- 每个 child refinement 仍独占 Plan/semantic/final-bound lineage；batch 只能共享 optimizer/native
  compiler execution，不得把两个 child 的 intermediate bounds 混成同一 source。
- formal performance 至少三 fresh repeats，并同时报告 raw phase/whole timings、node/depth/frontier、
  deadline discard；pilot timing 不进入 claims map。

## Rollback

- 新增文件 additive；删除 NRIR-34 IR/runtime/tests/scripts/docs/artifact 即回到 `main@45d2ea6`。
  NRIR-32/33 replay 必须继续 PASS。

## Closure

- first-pair feasibility：serial `13.291550 s`、packed `7.018038 s`，diagnostic speedup
  `1.893913×`；optimizer/native group 均由 `2→1`，bounds exact，split/source exact。
- 三 fresh repeats：serial accepted nodes=`[7,7,7]`，packed=`[15,15,15]`；minimum gain=`+8`，
  common-node lower/upper max diff 均为 `7.62939453125e-06`，alpha/beta max diff 分别为
  `1.0728836059570312e-04`/`8.940696716308594e-08`。
- packed max depth=`3`，worst active lower=`-76.07719421386719`；serial 为 depth `2`、
  `-104.76541137695312`。formal hash=
  `9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`。
- 9-clause global-60s integration 保持 original ordinal 与 sound `unknown`，完成 clause 0 的
  13-node/6-group atomic queue，clauses `1..8` 明确 pending；evidence hash=
  `dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`。
- 关闭等级为 single-hard-clause same-algorithm deadline coverage `VALIDATED-REDUCED`。没有 property
  closure、GPU、competitor 或 ASPLOS-ready 升级；下一门禁是跨 clause 的 objective/root/compiler
  共享和 anytime budget，而不是延长 deadline。

## Links

- changelog: `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
