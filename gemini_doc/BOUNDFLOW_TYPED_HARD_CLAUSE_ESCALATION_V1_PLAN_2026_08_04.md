---
status: completed
updated: 2026-08-04T13:34:00Z
type: plan
topic: boundflow
slug: typed-hard-clause-escalation-v1
stage: s01
---

# Typed Hard-Clause Escalation v1 Plan

## Goal

- 把 NRIR-29 的“纯扩树饱和”转成真正的编译/调度决策：先用 local-forward parametric
  `7 nodes/depth 2` 跑完整 query，只把 baseline unresolved clauses 投影到第二阶段；第二阶段先
  编译一次 shared native selected-CROWN intermediate refinement，再以 `31 nodes/depth 4` 重跑
  投影后的 hard clauses，最后按原 clause ordinal sound aggregate。
- 将 stage policy、admission decision、objective projection、refinement、escalated query、fallback 与
  aggregate 全部纳入 first-class Plan/Task/Schedule/trace；不得以 runner 中的 if/for 日志冒充 IR。

## Scope

- 固定 NRIR-29 三个真实 VNN-COMP workload、CPU、torch threads=8、60 秒 whole-query deadline、
  5-step optimizer、4-step candidate search、parametric production v2、batch `2/4`。baseline budget
  固定 `7/2`，escalation 固定 `31/4`。
- shared refinement 固定一 pass、每 ReLU 最多 128 个 top-width ambiguous targets、backward chunk
  32、`selected_plain_crown_v1`；不读取正式结果后按 workload 改 cap，不使用 external verifier seed。
- admission 必须恰等于 baseline unresolved ordinals；baseline verified/unsafe clause 不得重跑。
  refinement/escalation 超时或失败时保留 baseline verdict 并标记 fail-closed fallback，不得伪造 closure。
- 只声明 typed staged control 和 fixed-deadline search/property coverage；不声明跨 stage speedup、GPU、
  αβ-CROWN parity/超越、完整 benchmark suite 或 ASPLOS-ready。

## Tasks

1. [x] 定义 Escalation Plan、Decision、Task、Schedule IR；冻结 stage policy、原始/投影 ordinal、
   deadline ownership、guarded action 与 aggregate 规则。
2. [x] 实现 additive runtime orchestration：baseline→decision→shared refinement→hard-clause projection→
   parametric escalation→original-ordinal aggregate，并输出 action trace 与 semantic signature。
3. [x] 覆盖 no-unresolved、unsafe、deadline/refinement fallback、ordinal tamper、verified regression、
   Plan/Task/Schedule hash 和 compiler-cache binding 的正负测试。
4. [x] 先做三 workload 单次 pilot；接口/门禁通过后执行三 fresh repeats，并保存 source、IR、trace、
   raw timing、manifest 和 fresh replay artifact。
5. [x] 运行 focused/full pytest、Black、Mypy、Pylint、NRIR-28/29/30 replay、diff 与 DocOps gate，
   更新 claims/status/memo/changelog 后提交 PR。

## Validation

- baseline stage 必须与 NRIR-29 n7d2 的 completed/pending/verified ordinals 和 common root lower
  对齐；decision escalated ordinals 必须逐项等于 baseline unresolved ordinals，且 original↔projected
  mapping 双射。
- final verified set 必须包含 baseline verified set；任何 escalation unknown/failure 只能保留 baseline，
  不能回退或升级。所有 Plan/Task/Schedule/action/result hash、deadline accounting 和 clause ordinal
  在 replay 中重算。
- 三个 workload 各三次 fresh process；只有所有 repeat 无 soundness/accounting 回退，且至少一个
  workload 从 unknown 变 verified，才以 property-coverage `VALIDATED-REDUCED` 关闭；若仅增加
  verified clauses 但无完整 workload closure，则只作 diagnostic reduced；完全无新增则 NO-GO。
- 资源数保留 raw/median/p90，但不同 stage/预算不计算 speedup；60 秒是 baseline+refinement+
  escalation 的 whole-query deadline，不给每 stage 各自 60 秒。

## Rollback

- 只新增 escalation IR/runtime/runner/tests/artifact，不修改 NRIR-28/29 frozen compiler/runner。
  删除新增文件即可回到 `main@a170429`；任何 admission、ordinal、deadline、refinement provenance 或
  aggregate gate 失败都拒绝新 artifact，并保留 baseline verdict。

## Result

- 三 workload × 三 fresh repeats 全部完成且 `fallback=none`；baseline 与 NRIR-29 n7d2 的 clause
  accounting、root bounds、evaluated nodes 在 `1e-5` 内对齐，admission 恰等于 baseline unresolved。
- OVAL21 三次均由 `8/9 unknown` 变为 `9/9 verified`；MNISTFC 三次均由 `6/9` 提升为 `8/9`，
  ResNet2B 三次均保持 `0/9`。median whole-stage execution 为 `2.208/2.974/20.146 s`（按
  OVAL/MNIST/ResNet），均未触发 60 秒 fallback。
- artifact evidence hash=
  `df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`；按预注册门禁以
  property-coverage `VALIDATED-REDUCED` 关闭。完整 benchmark/GPU/competitor/ASPLOS-ready 不升级。
- 下一门禁是把 shared top-width refinement 改为 exact scalar-clause objective-directed Plan；只对
  remaining ordinals 编译 per-clause influence/targets，并与本阶段相同 deadline/cap 比较。

## Links

- changelog: `gemini_doc/BOUNDFLOW_TYPED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_WALL_CLOCK_PARAMETRIC_BAB_SCALING_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
