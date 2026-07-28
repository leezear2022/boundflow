# 变更记录：Plan IR v1 schema、跨决策 verifier 与旧计划迁移

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`47eb5ee`（IR-1 Bound IR reference semantic closure）
> 状态：IR-2A schema/verifier/replay/migration foundation validated；IR-2 尚未完成

## 1. 目标

IR-1 关闭后，下一缺口不是再扩展 Bound IR，而是消除现有计划事实的分散状态：

- `MaterializationPlan` 只决定单点 dense/structured/reduce-batch；
- `MaterializationPlacementPlan` 只保存已选 barrier placement；
- `ExecutionCandidate` 把 placement/backend/batch/schedule 字段放在一条记录里，但不与其他
  计划对象交叉验证；
- `StoragePlan` 没有 Bound IR value lifetime；
- `FusedCrownExecutionStep` 已是执行片段，不应倒灌为 Plan IR；
- `PlanBundle.meta/lowering_plan` 是 untyped dict，不能承载新计划语义。

本轮建立 `PlanTemplate`（静态候选空间）与 `PlanInstance`（动态完整选择）的第一套一等合同，并
为上述旧对象给出代码级 adapter 或显式 unsupported 结论。

## 2. Plan IR v1 schema

新增 `boundflow/ir/plan.py`，核心模块不依赖旧 planner、runtime 或 PyTorch。

### 2.1 静态输入

- `HardwareProfile`：device、总显存、dtype、alignment、backend capability IDs；
- `WorkloadProfile`：method、grad/α/β/split、static shape、domain/spec/sample 三轴、numeric
  policy；
- `BackendCapabilitySpec`：method/op/representation/dtype/device/state capability；
- `PlanCost`：latency、peak bytes、compile/setup cost、confidence、risk tags。

### 2.2 `PlanTemplate`

候选空间分为：

- `RegionCandidate`：partition/fusion 和 Bound IR boundary；
- `RepresentationCandidate`：dense/structured/chunked 及所需 transition IDs；
- `MaterializationCandidate`：cast/materialize 的 source value 和 before-op；
- `BackendCandidate`：backend、capability、兼容 representation、artifact key；
- `BatchCandidate`：domain/spec/sample 三轴；
- `StorageCandidate`：value→arena/offset/physical bytes/lifetime；
- `StateCandidate`：cache/recompute/evict 与 state version。

所有 candidate ID 在整个 template 内全局唯一，不能只在各自局部表中唯一。

### 2.3 `PlanInstance`

实例显式包含 region、representation、materialization、backend、batch、storage、state
decision，并要求：

- 每个 template candidate 必须恰好是 selected 或 rejected；
- 每个 rejected candidate 都有非空理由；
- 不允许 selected/rejected 重叠；
- query bucket、available memory、memory budget、deadline 都是一等字段；
- cost summary 不能低报 selected latency/compile/setup/risk，也不能高报 confidence。

## 3. 跨决策 verifier

Verifier 不只分别调用局部 `validate()`，还检查：

1. template 的 `bound_module_hash` 与当前 `BFBoundModule.stable_hash()` 一致；
2. selected region partition 对 Bound IR ops 恰好覆盖一次，无 gap/overlap；
3. region input/output boundary 与真实 use-def 一致；
4. representation 所需 transitions 必须全部选择，且不得选择无用 transition；
5. backend capability 与 method/state/op/representation/dtype/device/static-shape 一致；
6. 每个 selected region 恰有一个 representation 和 backend；
7. storage 与 selected batch/representation compatibility 一致；
8. batch domain/spec/sample 不超过 workload bucket；
9. storage size、hardware alignment、producer/user lifetime 和 physical alias 不冲突；
10. state version 与 Bound IR value `state_version` 一致；
11. selected storage peak 与 cost summary 一致，且不超过 hardware/available/config budget。

## 4. Deterministic dump/hash/replay

`PlanTemplate` 和 `PlanInstance` 均有 canonical JSON 与 SHA-256 stable hash。

`PlanInstance.from_canonical_json()`：

- 严格解析字段集合和标量类型；
- 重新执行完整 template/bound/capability/memory verifier；
- 拒绝 unknown candidate、tampered decision；
- 拒绝字段相同但格式非 canonical 的 JSON；
- replay 后对象与原实例完全相等。

当前 replay 假定 template 由相同 Bound IR/hardware/workload/config 确定性重建并通过
`template_hash` 锁定。Template artifact 独立加载器留给 IR-2B artifact assembly。

## 5. PR-11/12 迁移

新增 `boundflow/planner/plan_ir_legacy.py`。迁移结果只有三种状态：

- `ADAPTED`：字段和证据完整；
- `PARTIAL`：已生成 typed candidates，同时保留每个缺口；
- `UNSUPPORTED`：不生成候选，只返回明确原因。

| 旧对象 | 结果 |
|---|---|
| `MaterializationPlan` | dense/structured → RepresentationCandidate；缺失 latency 必须外部补证并标记 partial；reduce-batch 标记 runtime replan gap |
| `MaterializationPlacementPlan` | 已选 barrier → representation candidates；因没有未选候选表，固定为 partial |
| `ExecutionCandidate` | 拆为 representation/backend/batch；`schedule_id` 明确延后到 IR-3，因此 partial |
| `StoragePlan` | 只有显式提供 Bound value/lifetime/representation mapping 才生成 StorageCandidate；否则 unsupported |
| `FusedCrownExecutionStep` | unsupported in Plan IR；明确属于 IR-3 Task/Schedule lowering |
| `PlanBundle.meta/lowering_plan` | 禁止迁移语义；只可保留为 debug provenance 或由专属 adapter 处理 |

Adapter 不会把旧记录缺失的 latency、lifetime、offset、schedule 语义静默猜出来。

## 6. 测试与结果

新增：

- `tests/test_plan_ir_v1.py`
- `tests/test_plan_ir_v1_legacy_adapter.py`

专属 12 项覆盖：

- template/instance deterministic dump/hash；
- canonical JSON replay；
- noncanonical/tampered replay；
- candidate 全量 selected/rejected 记账；
- backend×representation conflict；
- memory budget；
- storage alias、under-allocation、lifetime；
- Bound hash、partition gap；
- 六类旧对象迁移状态和 source hash；
- Plan IR core 无 legacy planner/runtime/torch import。

相邻回归：

```bash
pytest -q \
  tests/test_plan_ir_v1.py \
  tests/test_plan_ir_v1_legacy_adapter.py \
  tests/test_bound_ir_v1.py \
  tests/test_bound_ir_v1_plain_crown.py \
  tests/test_phase7a_pr11_materialization_planner.py \
  tests/test_phase7a_pr11_materialization_placement.py \
  tests/test_phase7a_pr12_execution_candidate.py \
  tests/test_phase5b_pr3_buffer_reuse.py \
  tests/test_env.py
```

结果：`88 passed`，1 个既有 PyTorch deprecation warning。

全量：

```bash
pytest -q tests
```

结果：`409 passed, 1 skipped, 6 warnings`，用时 46.01 s。

静态门禁：

- Black：clean；
- Mypy：0 issues；
- Pylint：10.00/10。

## 7. 边界与下一门禁

本轮只关闭 **IR-2A schema/verifier/replay/migration foundation**，不能宣称 IR-2 完成。

尚缺：

1. 从 Bound IR + profile/capability/cost evidence 自动构建完整 `PlanTemplate`；
2. runtime query bucket + available memory + deadline → `PlanInstance` selector；
3. 多预算下产生不同、合法且完整记账的实例；
4. 将旧 adapter 输出组装进同一 template，而不产生重复/孤立 candidate；
5. template/instance artifact 文件与独立 replay 命令；
6. 旧 PR-11/12 每项决策在真实记录上的批量 migration report。

下一切片是 **IR-2B reference template builder + selector**。完成后才能进入 IR-2C 的旧 artifact
批量迁移与 closure 审计。
