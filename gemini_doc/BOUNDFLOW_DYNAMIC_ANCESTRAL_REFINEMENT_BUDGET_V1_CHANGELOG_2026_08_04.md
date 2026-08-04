# BoundFlow Dynamic Ancestral Refinement Budget v1 修改记录（NRIR-25）

日期：2026-08-04
分支：`feat/dynamic-ancestral-refinement-budget-v1`
状态：`VALIDATED-REDUCED`

## 当前记录

- PR #35 已合并为 `main@47ca159`，从该基线建立 NRIR-25 分支；
- 根据 NRIR-24 未闭合但持续改善的曲线，停止把纯 fixed-depth 扩展作为主路线；
- 冻结 `fixed16` 与 generated-batch 内 parent-lower 风险分配 `dynamic8_24` 的同 31-node/depth-4、
  同 planned target-cap 对照；
- 冻结 first-class budget policy/decision IR、Plan/Task/Schedule lowering、逐组守恒、六 shard fresh
  replay 与 fail-closed 门禁；
- 明确两 mode 可生成不同 logical tree，报告 split-state overlap；CPU timing 不是性能 claim。

## 实现进展

- 新增 `NativeIntermediateRefinementBudgetPolicyIR` 与
  `NativeIntermediateRefinementBudgetDecisionIR`，冻结 parent-lower risk allocation、8/16/24 cap、
  tie tolerance、group conservation、node/split/parent scope 与 stable hash；
- optimized queue 在每个 evaluation group 生成 typed decisions，派生的 cap 进入实际 refinement
  Plan policy；record/evaluation/queue trace 绑定 decision，execution validator 反查 Plan policy；
- trace 对 group semantic hash、成员数、逐 node assigned cap 和 group total 做闭环，旧无 dynamic
  policy 的 payload 保持条件兼容；
- 新增 toy root/base/high/low、全树守恒、Plan lowering、admission 与 group-hash tamper tests；dynamic
  targeted `2 passed`，既有 refinement/runtime `29 passed`，Mypy clean、Pylint `10.00/10`。

## 固定证据

- 六个 fresh-process shards 全部生成并通过静态语义校验；artifact evidence hash=
  `85d9f274c6e17614bcbf318bdbfea18219b03876024be16aea3329ee4d3c56bd`；
- clauses `0/2/4` 的 fixed16 worst terminal lower 分别为
  `-0.2823597193/-0.4018449783/-0.4599394798`；dynamic8_24 分别为
  `-0.2819737196/-0.4016119838/-0.4596676826`；
- dynamic delta 为 `+0.0003859997/+0.0002329946/+0.0002717972`，三条均不弱且均严格改善，
  通过预注册 `VALIDATED-REDUCED` 门禁；
- 每个 mode 的 planned target cap 均为 `31×16=496`，实际 selected target count 都为 `2976`；
  clause `0` 的 logical-domain overlap/union=`29/33`，clauses `2/4` 均为 `31/31`；
- 三条动态树仍为 unknown，proof deficits 为 `0.2819737196/0.4016119838/0.4596676826`；没有
  complete property、CUDA、multi-workload、multi-pass、competitor 或 performance claim。

## Artifact 与测试

- runner：`scripts/run_dynamic_ancestral_refinement_budget_artifact.py`；支持六分片 atomic checkpoint、
  strict resume、aggregate 与 fresh-process semantic replay；
- artifact：
  `artifacts/dynamic-ancestral-refinement-budget/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；
- artifact tests 固定 digest、三 clause 数值、逐组守恒、decision→Plan lowering、claim/tamper 与
  checkpoint fail-closed；
- fresh-process semantic replay 6/6 通过，最终输出
  `{"evidence_hash":"85d9f274...56bd","status":"ok"}`；
- focused `34 passed`；全量 `778 passed, 37 skipped`（skip 均为 CUDA/环境门禁）；Black、targeted
  Mypy、Pylint `10.00/10` 与 `git diff --check` 通过；
- artifact 单测曾因 tamper fixture 改到本来就是 16-cap 的节点而失败；fixture 改为按 typed decision
  定位 24-cap 节点后通过，未修改实现或冻结 evidence。

## 下一门禁

- 本轮证明 parent-lower dynamic cap 在固定范围内有一致但很小的正向 tightness 信号，不足以关闭
  hard clauses；下一单一路线应冻结 typed multi-pass termination/reallocation，对第二 pass 是否把
  预算投向第一 pass 后仍高 influence×width 的 targets 做同预算/增量预算对照；
- 在 multi-pass 形成一等 Plan/Task/Schedule、pass-to-pass lineage 与 fresh replay 前，不进入 CUDA
  性能宣称，也不把本结果写成 complete verifier/ASPLOS-ready。
