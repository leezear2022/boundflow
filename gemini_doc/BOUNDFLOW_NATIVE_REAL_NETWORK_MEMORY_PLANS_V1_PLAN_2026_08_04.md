---
status: completed
updated: 2026-08-04T03:48:56+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1
stage: s01
---

# Native Real-Network Memory Plans v1 计划

## Goal

- 在 NRIR-1 已 native lower 的固定 VNN-COMP ResNet2B main CROWN backward 上，建立至少
  两个真实、合法且可执行的 StorageCandidate。
- 让同一 `BFBoundModule` 与同一 `PlanTemplate` 在预算阈值两侧选择不同
  `PlanInstance.storage_decision`，并 lower 为不同 Schedule arena。
- 让低内存计划不止改变 JSON/哈希：Task IR 执行时按选中 binding 的 `live_to_op_id`
  释放实际 Python/Torch runtime value 引用，并生成独立可重放 trace。
- 保持 NRIR-1 的冻结 schema、artifact 与五层 hash 可继续 replay。

## Scope

- 固定输入不变：VNN-COMP 2021 `resnet_2b.onnx`、prop0、αβ-CROWN
  `e5c7e17...`、6 组 external intermediate bounds 及 aggregate digest
  `d51615b0...8cf1`。
- 高预算候选 `storage:native-retain-all-v1`：每个值使用独立对齐区间，生命周期延长到
  final op，复现既有 eager session 保留全部中间值的行为。
- 低预算候选 `storage:native-lifetime-reuse-v1`：保留 compiler-derived exact last-use，
  用确定性 first-gap allocator 仅在生命周期不重叠时复用 arena byte range。
- `StoragePlanRuntime` 在每个 Task 前验证输入仍 resident；Task 后按已验证生命周期删除
  `session.env` 引用；graph outputs 与 state-store source 被 pin 到消费完成。
- 新 artifact 只证明 CPU correctness、budget admission、Schedule arena ownership 与
  runtime last-use release，不测 latency，不声称 CUDA allocator peak 或 OOM rescue。
- 本切片不加入假的 structured/batch candidate。审计确认当前 Plan representation decision
  尚不能驱动 Bound IR rewrite；现有 `MaterializeAction` reference executor 也只记账。因此
  representation/materialization 与真正 sliced batch execution 留待独立 bridge。

## Tasks

- [x] 审计 representation/materialization/backend dispatch，确认不能直接复用当前 metadata
  伪造 real-graph structured plan。
- [x] 从 NRIR-1 dense baseline 派生 retain-all 与 lifetime-reuse 两个 verified storage plan。
- [x] 为 lifetime-reuse 实现 deterministic aligned interval allocation，并复用 Plan IR 已有
  physical alias safety verifier。
- [x] 实现 runtime last-use enforcement、use-before-release rejection 与 canonical storage trace。
- [x] 新增 memory-aware native compile/execute 入口，保持 NRIR-1 入口与 artifact 不变。
- [x] 在 small residual contract 上验证预算切换、数值一致、提前释放和低于最小预算拒绝。
- [x] 在固定真实 ResNet 上生成双计划 artifact，并进行 digest + semantic recompute replay。
- [x] 完成全量回归、静态检查与 DocOps closure 记录；分支发布在代码提交后单独记录。

## Validation

- 真实图：17 Primal ops、21 Bound ops、21 Task units；两计划共享同一 Bound hash
  `16e27f31...80fb` 与 PlanTemplate hash `359ee68f...43f3`。
- retain-all：Schedule/observed peak 均为 `1,860,912` bytes。
- lifetime-reuse：Schedule/observed peak 均为 `442,656` bytes，较 retain-all 的逻辑 arena
  减少 `1,418,256` bytes（`76.213%`）；这是确定性 compiler/runtime byte ledger，非设备
  allocator 性能测量。
- 低内存计划存在 386 对生命周期不重叠的 physical alias；85 个值在最终 Task 前释放。
- 预算 `442,656` 选择 lifetime-reuse；`442,655` 以
  `memory_budget_exceeded` fail closed。
- 两计划 lower/upper bitwise 相同；相对 external lower max diff
  `7.152557373046875e-07`，allclose `2e-4/2e-4`，sign `9/9`。
- NRIR-1 artifact replay 保持原五层 hash，不发生 schema/identity 漂移。
- focused contract/artifact tests：`7 passed`；全量回归：`473 passed, 37 skipped`，37 个
  skip 均为既有 CUDA/环境边界。
- Black、`git diff --check`、Mypy 5 files clean、Pylint 5 files `10.00/10`。
- artifact：
  `artifacts/native-real-network-memory-plans/vnncomp21-resnet2b-prop0-cpu-v1/`。

## Rollback

- 新 memory compile/execute 入口与 `StoragePlanRuntime` 均为显式 opt-in；删除新增模块、runner、
  tests 与 artifact 即可回到 NRIR-1，不需要修改旧 artifact。
- `execute_task_ir_semantics` 的新增 runtime hook 默认为 `None`，旧执行与 trace hash 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_CHANGELOG_2026_08_04.md`
- prior: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
- PR: `https://github.com/leezear2022/boundflow/pull/13`
