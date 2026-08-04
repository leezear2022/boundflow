---
status: completed
updated: 2026-08-04T04:55:03Z
type: plan
topic: boundflow
slug: native-intermediate-bound-refinement-v1
stage: s01
---

# BoundFlow Native Intermediate-Bound Refinement V1 Plan

## Goal

- 把 native 中间层 pre-activation bound refinement 从隐式 Python 辅助逻辑提升为可哈希、
  fail-closed、可 replay 的 Plan/Task/Schedule IR，并验证它能否缩小 NRIR-18 三类真实网络的
  root gap 与 unresolved-clause 集合。

## Scope

- 允许 plain CROWN 从任意已产生的中间值反向传播，并用稀疏 one-hot objective 只精化选中的
  ReLU pre-activation neuron；不物化整层 identity matrix。
- 新增 native-refined intermediate provenance、精化策略/目标选择/执行 trace，以及逐 pass
  backward -> intersect -> forward-propagate 的确定性 Schedule。
- 在 MNISTFC、CIFAR ResNet2B、OVAL21 上以同一 frozen source/property 和相同 complete-verifier
  policy 对比 local-forward baseline 与 native-refined 结果。
- 本轮不做 GPU/speedup claim；CPU 时间只报告 refinement cost 和诊断性 E2E。

## Tasks

- [x] 冻结 refinement Plan/Task/Schedule schema、stable hash、依赖与 fail-closed 门禁。
- [x] 实现任意中间输出的 selected CROWN bound，并用小网络枚举角点验证 soundness。
- [x] 实现 top ambiguous-width per-ReLU 选择、分块 backward、单调 intersection 和逐 pass
  forward propagation。
- [x] 将 native-refined provenance 接入 optimizer/BaB 消费路径，禁止冒充 external bounds。
- [x] 增加正向、负向、tamper、单调性与 schedule-driven execution 测试。
- [x] 生成三 workload baseline/refined replay artifact；更新 claims/status/memo/master/README。

## Validation

- IR contract/tamper focused tests 全过；Schedule action 数量和执行 trace 一一对应。
- selected intermediate bounds 包含角点真值，且 refined lower 不小于/upper 不大于原 IBP；
  不允许 infeasible intersection、身份/shape/dtype/device/hash 漂移。
- 三 workload 报告每层 selected/tightened 数、最大/累计收紧量、refinement wall time、root lower、
  verified/unresolved/pending clauses；只在实际数据成立时升级 claim。
- 全量 pytest、Black、Mypy、Pylint、artifact replay、`git diff --check`、DocOps validate/lint。

最终结果：MNISTFC unresolved `3→1`、关闭 clauses `3/7`、nodes `31→21`；OVAL21
`unknown→verified`、关闭 clause `8`、nodes `15→11`；ResNet 状态仍 unknown，但两个已完成
clauses 的 root lower 改善 `+70.496/+160.551`。artifact hash=
`f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；focused
`9 passed`，全量 `732 passed, 37 skipped`，静态门禁全过。

## Rollback

- 新能力默认关闭；不提供 refinement policy 时保持 NRIR-18 local-forward 行为与旧 trace 语义。
- 若真实 workload 无稳定增益，以 `VALIDATED-NO-GO` 保存证据，不把 external αβ-CROWN bounds
  混入 native claim，也不回退到堆固定树深伪造进展。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
