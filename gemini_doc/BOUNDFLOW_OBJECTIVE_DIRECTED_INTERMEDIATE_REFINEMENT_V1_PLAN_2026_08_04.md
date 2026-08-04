---
status: completed
updated: 2026-08-04T06:13:29Z
type: plan
topic: boundflow
slug: objective-directed-intermediate-refinement-v1
stage: s01
---

# BoundFlow Objective-Directed Intermediate Refinement V1 Plan

## Goal

- 将 NRIR-19 的纯区间宽度 shortlist 升级为由当前性质子句驱动的 intermediate target
  selection：以 ReLU 后向系数影响力乘 pre-activation width 排序，并把 objective identity、
  score、Task/Schedule 数据依赖冻结为可哈希、可拒绝篡改的一等 IR。
- 在固定 VNN-COMP 2021 ResNet2B property 0 的两个已执行 hard clauses 上，用相同
  `16 targets/ReLU` 成本预算比较 width 与 objective-directed refinement，回答它是否继续提高
  根节点 bound tightness。

## Scope

- 包含：plain CROWN 后向过程中逐 ReLU 捕获 clause-sensitive coefficient influence；
  objective hash 与每个 target 的 influence/selection score；objective-aware Plan/Task/Schedule；
  native execution trace；固定 CPU artifact、digest、source-to-IR replay 与负向测试。
- 选择分数固定为 `ambiguous_width * max(abs(A_u), abs(A_l))`。选择是 heuristic；soundness
  仍来自 selected plain-CROWN 与 interval intersection，不依赖排序本身正确。
- 不包含：per-child 重算、扩大 BaB tree、CUDA、latency/speedup、竞品公平性能或 ASPLOS-ready
  claim。所有 timing 仅诊断，`performance_claimed=false`。

## Tasks

1. 扩展 refinement Plan/target policy，同时保持 NRIR-19 width policy 的 canonical payload/hash
   兼容。
2. 从当前 scalar objective 的 CROWN backward state 提取逐 ReLU influence，并对 shape/dtype/
   device/finite/nonnegative fail closed。
3. 让 objective influence 只经声明的 Task/Schedule value dependency 进入 target selection；
   program validation 重算 objective hash 和 selected targets。
4. 新增 width/objective admission、selection divergence、objective tamper、旧 artifact replay 测试。
5. 生成 ResNet 双子句 baseline/width/objective artifact；固定相同 target count 和 verifier policy，
   冻结 root bounds、target overlap、Plan/Task/Schedule/trace hashes。
6. 更新 ASPLOS memo、claims map、current status、README 和本 changelog，再走 DocOps 验证与交接。

## Validation

- focused pytest 覆盖 IR linkage、数值 soundness、policy admission、objective identity/tamper、
  artifact replay/tamper；NRIR-19 artifact 必须继续 replay。
- fixed ResNet 两个 clause 的 objective-directed target count 必须与 width policy 完全相同；
  refined intermediate bounds 必须单调包含于初始 bounds；fixed artifact 上 objective root lower
  必须高于 width root lower，否则以 no-go 关闭该 heuristic。
- 运行 Black、Mypy、Pylint、全量 `pytest tests`，记录 CUDA skips，不升级性能或设备 claim。
- artifact replay 必须先验校验所有文件 digest，再从冻结 ONNX/VNNLIB 重编译 objective-aware
  Plan/Task/Schedule，并比较 canonical hashes；同步更新 digest 后的语义篡改仍须失败。

## Rollback

- 新策略用独立 `candidate_policy_id` 准入；默认仍是 NRIR-19 width policy。若 fixed ResNet
  不改善或引入不可接受的内存/正确性问题，删除 objective policy 路径即可，旧 payload/hash 与
  artifact 保持可重放。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
