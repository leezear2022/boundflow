---
status: completed
updated: 2026-08-04T08:05:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1
stage: s01
---

# Native Alpha/Beta Optimization State v1 Plan

## Goal

- 将 legacy runtime 中的 per-ReLU alpha/beta 张量、split beta constraint 和 parent-to-child
  warm start 提升为可哈希、可验证、fail-closed 的 native state contract；冻结状态必须被实际
  Bound/Plan/Task/Schedule stack 消费，而不只是写入 capability metadata。

## Scope

- v1 复用现有 `run_alpha_beta_crown_mlp` 产生优化后的 alpha/beta 张量；优化迭代控制流仍由
  runtime adapter 所有，不宣称 Adam loop 已 lower 为 Task/Schedule IR。
- Bound IR 为每个 ReLU 显式绑定 split、alpha、beta 三类 typed graph input；alpha 只替换
  ambiguous lower slope，beta 只以 `-beta * split` 进入 lower dual coefficient。
- state identity 至少绑定 primal graph、input region、objective、intermediate bounds、split payload、
  tensor payload 和 optimizer policy；任一漂移均拒绝 exact reuse。
- warm start 只允许 exact same-scope 或 monotonic split refinement。refinement 可作为初始化，但
  parent state 绝不冒充 child exact state；split 回退、改写既有 branch、model/spec/domain 漂移均拒绝。
- 先覆盖 deterministic toy 与 fixed ResNet 的有界 state/IR replay；无完整搜索/property verdict、
  CUDA、latency、memory 或 speedup claim。

## Tasks

1. 新增 native alpha/beta state identity、tensor digest、warm-start classification 与 tamper gates。
2. 扩展 ReLU BoundOp typed ports/attrs/value contract 和 optimized-CROWN lowering。
3. 扩展 reference interpreter、Task/Schedule adapter，实际消费 alpha slope 与 beta constraint。
4. 接入 legacy optimizer output，证明 native frozen-state execution 与 legacy dense oracle 对齐。
5. 添加 exact/refinement/rejected warm-start tests、fixed artifact/replay/tamper tests并同步 claims。

## Acceptance criteria

- alpha key/shape/dtype/device/range/hash 与 beta key/shape/dtype/device/nonnegative/hash 任一错误均
  fail closed；split/alpha/beta 必须属于同一 ReLU 和 domain batch。
- beta 非零时 native lower bound 与 legacy alpha-beta dense reference 对齐，并与去掉 beta 的结果
  形成可观察差异；不能只证明 metadata 存在。
- exact scope 可重放；单调新增 split 仅得到 `initialization_only`；branch reversal、split removal、
  objective/input/model/intermediate-bound drift 全部拒绝。
- Bound、Plan、Task、Schedule hash 均随优化状态变化；Schedule launch 实际覆盖全部 Bound ops。
- fixed artifact generate/replay 和同步重哈希后的 payload/lineage/claim tamper probes 全部通过。

## Validation

- toy optimized state：native Bound/Plan/Task/Schedule lower/upper 与 legacy αβ oracle bitwise 相同；
  非零 beta 的 lower-dual 结果严格优于 zero-beta 对照。
- fixed ResNet：6 个 ReLU、19 个 query inputs、6 个 optimized ReLU ops、21 Task/Launch；native/
  legacy lower/upper max diff 均为 `0.0`，beta lower improvement 为 `0.34039306640625`。
- parent zero-split state 到 child active-split state 被分类为
  `monotonic_split_refinement`、`initialization_only`、`exact_state_reuse_allowed=false`；reversal/
  removal 与 model/input/objective/policy drift fail closed。
- artifact generate/replay 证据 hash
  `302f536685885e75248582698589d49f667d7709ca3258c043310e02278e6884`；聚焦 `50 passed`；
  全量 `591 passed, 37 skipped`（7 条既有依赖/环境 warning）。
- Black、Mypy 7 source files clean、Pylint `10.00/10`、`git diff --check` 通过。

## Rollback

- additive optimized-state entry；默认 plain-CROWN 和 NRIR-9 split queue API/hash 不因未传 alpha/beta
  而变化。

## Completion boundary

- v1 只关闭 frozen optimized-state ownership、beta constraint execution 和 warm-start validity。
- optimizer step/gradient/Adam control flow 未进入 IR、BaB 未完成且无公平系统证据时，不升级完整
  alpha-beta-CROWN verifier、C3 或 performance claim。
- 下一门禁为 `native alpha/beta optimizer-step Task/Schedule control v1`；不得直接跳到性能结论。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
