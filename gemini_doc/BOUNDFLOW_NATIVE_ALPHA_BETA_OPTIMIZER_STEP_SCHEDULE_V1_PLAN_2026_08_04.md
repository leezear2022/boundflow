---
status: completed
updated: 2026-08-04T08:10:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1
stage: s01
---

# Native Alpha/Beta Optimizer-Step Schedule v1 Plan

## Goal

- 关闭 NRIR-10 的明确边界：不再由一个 opaque runtime call 隐藏 evaluate/reduce/backward/Adam/
  project/best-select 控制流；为 frozen optimized Bound stack 建立 first-class optimizer Plan、Task、
  Schedule 和逐 action state-transition trace。

## Scope

- v1 将固定步数 optimizer schedule 静态展开，绑定 NRIR-10 source Bound/Plan/Schedule hash、state
  scope、optimizer policy、ReLU state IDs 和 step budget。
- gradient bound evaluation 复用现有 dense αβ-CROWN autograd semantics；最终 selected state 仍必须
  经 NRIR-10 native Bound/Plan/Task/Schedule stack bitwise/容差校验。
- warm start 只接受 NRIR-10 exact/refinement classifier；Schedule 不得自行放宽 state validity。
- CPU correctness/control ownership only；无完整 BaB、property verdict、CUDA、latency或 speedup claim。

## Tasks

1. 新增 typed optimizer Plan/Task/Schedule IR 与 stable hash、cross-layer verifier。
2. lower 固定 step policy 为 evaluate/reduce/backward/update/project/select-best task/action序列。
3. 实现 schedule-driven executor 和 replay-grade tensor/bound/metric/state transition trace。
4. 与 legacy `run_alpha_beta_crown_mlp` 对齐，添加 order/linkage/action/state tamper tests。
5. 在 fixed ResNet 上生成 artifact/replay并同步 ASPLOS claims/limitations。

## Acceptance criteria

- 所有 optimizer action 必须按 schedule 执行一次；step、dependency、state version、source compiler
  hash 任一篡改 fail closed。
- toy 与 fixed ResNet selected bounds/alpha/beta 必须与 legacy optimizer 对齐；最终 state 再经 native
  frozen-state compiler执行对齐。
- runtime trace 必须证明 beta gradient/update 非空、alpha/beta projection 合法、best state 来自已评估
  iteration，而不是事后伪造。
- artifact generate/replay 与同步重哈希后的 schedule/task/transition/claim tamper probes 通过。

## Completion boundary

- 只关闭 fixed-step optimizer control IR；dynamic early stop、multi-node BaB integration、完整 verdict 和
  性能证据仍 pending。

## Result

- `NativeOptimizerPlanIR` 绑定 NRIR-10 的 10 个 source compiler hash、initial state、scope、policy、
  ReLU keys、warm-start kind 与固定 step budget。
- policy 被静态 lower 为 `evaluate → reduce → backward → Adam → project` 的逐步 Task/Action，末尾
  `select-best`；2-step toy 为 13 actions，1-step fixed ResNet 为 8 actions。
- executor 严格按 Schedule 运行，并记录逐 action 输入/输出 hash、alpha/beta gradient、projection、
  evaluation 和 best iteration；跨层 identity、顺序或 hash-chain 篡改 fail closed。
- fixed ResNet Schedule/legacy/final native compiler lower/upper max diff 全为 `0.0`；alpha/beta state
  hash 相同；gradient L1 为 `169.23175295069814/12.862210273742676`。
- artifact generate/replay hash 为
  `31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`；聚焦
  `35 passed`，全量 `612 passed, 37 skipped`，Black/Mypy/Pylint/diff 门禁全过。

状态为 fixed-step optimizer control ownership `VALIDATED-REDUCED`。下一门禁是把该 Schedule 接回
native ReLU-split BaB queue，同时保持 parent→child warm-start initialization-only 与逐节点最终
state native re-execution。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_CHANGELOG_2026_08_04.md`
