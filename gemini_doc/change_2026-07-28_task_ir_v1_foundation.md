# 变更记录：Task IR v1 schema、Plan lowering 与 Schedule linkage

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`523de97`（IR-3B Schedule control/executor/trace）
> 状态：IR-3C typed Task IR foundation validated；IR-3 尚未完成

## 1. 为什么不能继续复用旧 `ir/task.py`

旧 `BFTaskModule` 是早期 IBP/runtime 容器：

- `TaskKind` 只有 `INTERVAL_IBP`；
- `TaskOp.attrs`、`BoundTask.memory_plan`、bindings 含 `Any/dict`；
- task 与 Bound region、Plan backend、Schedule launch 没有稳定双向引用；
- 参数、runtime state、memory effect 和 reference implementation ID 不完整。

因此本轮新增独立 `boundflow/ir/task_v1.py`，不修改旧对象含义，也不把它直接改名包装为新 IR。

## 2. Typed Task IR v1

核心对象：

- `TaskIRModule`：锁定 Bound/Template/Instance 三重 hash；
- `TaskIRUnit`：一个 selected Plan region 对应一个 backend-callable task；
- `TaskOpRef`：只引用 Bound IR op ID/kind，数学 attrs 仍由 Bound IR 所有；
- `TaskMemoryEffect`：boundary value 的 READ/WRITE；
- `TaskStateDependency`：value/state-version/access；
- `TaskExternalDependency`：OBJECTIVE、PERTURBATION、PREACTIVATION_BOUND；
- `TaskBackendBinding`：backend candidate、capability、compiled artifact、reference implementation。

Task kind 已区分：

- `BOUND_BINDING`；
- `PLAIN_CROWN_REGION`；
- `CONCRETIZATION`；
- `STATE_UPDATE`（schema 预留，独立 state task lowering 尚未启用）。

核心模块不导入旧 `ir/task.py`、runtime 或 `Any/Dict`。

## 3. PlanInstance → Task IR lowering

`lower_plan_instance_to_task_ir()`：

1. 对 selected regions 按 Bound topology 排序；
2. 每个 region 生成稳定 `task:<region_id>`；
3. op refs 与 region op IDs/kinds 精确一致；
4. input/output 与 Plan region boundary 精确一致；
5. 从 typed Bound attrs 提取 weight/bias/constant 参数；
6. 提取 objective/perturbation/ReLU preactivation 外部依赖；
7. 从 boundary values 提取 state version；
8. 从 value use-def 生成 task dependency edges；
9. 绑定 selected backend capability/artifact；
10. 生成 module canonical JSON/stable hash 后再次完整验证。

Verifier 会拒绝：

- region/op/kind/boundary 不一致；
- 参数、external/state dependency 缺失；
- memory effect 不完整；
- task dependency use-before-task；
- backend 与 PlanInstance 不一致；
- entry/output task 集不正确。

## 4. Task ↔ Schedule linkage

`TaskIRModule.validate_schedule_linkage()` 要求：

- 每个 Task IR task 恰好一个 Schedule `LaunchAction`；
- 不允许重复 launch 或未绑定 task；
- task/region/backend/artifact/input/output 逐字段一致。

Schedule artifact 现新增：

- `task_module.json`；
- `task_trace.json`；
- `task_module_hash`；
- `task_trace_hash`。

fresh-process generate/replay 同时核对 TaskModule、TaskTrace、Schedule、ScheduleTrace。

## 5. Reference task dispatch trace

新增 `boundflow/runtime/task_ir_executor.py`：

- 按 Schedule launch 顺序 dispatch typed tasks；
- 每次 dispatch 前检查 task dependency 已完成；
- trace 固定 task/region/op IDs、dependencies、backend 和 reference implementation；
- task 不得重复或遗漏；
- canonical JSON/stable hash。

当前 executor 证明 typed task dispatch/linkage，不宣称已逐 task 执行 CROWN 数学语义。数学结果
仍由 whole-Bound interpreter oracle 提供。

## 6. 验证结果

- Task IR 专属：`4 passed`；
- Task+Schedule 专属：`14 passed`；
- 相邻 Bound/Plan/PR-11/12/storage/env：`111 passed`；
- 全量：`432 passed, 1 skipped, 6 warnings`，50.54 s；
- Black clean；
- Mypy 0 issues；
- Pylint 10.00/10；
- `git diff --check` 通过。

## 7. 下一门禁

IR-3 仍缺 **IR-3D per-task semantic executor + closure**：

1. 把 Bound reference interpreter 拆成可持续 env 的 region/task stepping；
2. 每个 TaskIRUnit 只执行自己的 Bound ops；
3. materialization/state action 与 task env 明确交接；
4. task-by-task final result 与 whole-Bound interpreter 对齐；
5. MLP/CNN/residual/concat 多图覆盖；
6. Task/Schedule artifact replay 包含真实 per-task semantic outputs；
7. IR-3 closure audit 逐条核对 dependency、peak、OOM、stream、query 和 topo-loop 对照门禁。

在上述完成前，IR-3 仍是 foundation，不升级为 complete。
