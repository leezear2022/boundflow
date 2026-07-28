# 变更记录：Schedule IR v1 schema、reference lowering 与 verifier foundation

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`a6e6a77`（IR-2 Plan IR validated-reduced closure）
> 状态：IR-3A schema/lowering/verifier foundation validated；IR-3 尚未完成

## 1. 目标

IR-2 已经能选择完整 `PlanInstance`，但执行仍没有一等 IR。现有
`runtime/scheduler.py` 只是 `TaskGraph.topo_sort()` 后调用 executor，无法表达或验证：

- PlanInstance 的 backend/representation/storage/state decision；
- allocation/free 与 peak-memory ledger；
- materialization 的发生位置；
- query result 的完整恢复；
- stream/event、batch、retry/fallback/state/replan。

本轮建立 Schedule IR v1 的第一层同步 reference contract，不复用或重命名旧 topo loop。

## 2. 新增 schema

新增 `boundflow/ir/schedule.py`，核心对象为 `ScheduleModule`：

- 锁定 Bound IR、PlanTemplate、PlanInstance 三个 stable hash；
- 显式 `query_ids`；
- `ScheduleBuffer` 完整携带 selected `StorageBinding`；
- action 序列使用全局唯一 action ID；
- canonical JSON 与 SHA-256 stable hash。

IR-3A 已实现的 action：

- `CheckBudgetAction`；
- `AllocateAction`；
- `MaterializeAction`；
- `LaunchAction`；
- `EmitResultAction`；
- `FreeAction`。

stream 在本轮只允许 `"sync"`。Record/Wait event、BatchLoop、StateLoad/Store/Invalidate、
Fallback/Retry/RequestReplan 保留到 IR-3B/3C，不能宣称 Schedule IR 已完整。

## 3. PlanInstance → Schedule lowering

`lower_plan_instance_to_reference_schedule()`：

1. 重新验证 PlanInstance；
2. 从 selected StorageCandidate 生成 buffer/arena ledger；
3. 先 `CheckBudget`，再为每个 arena 分配精确 bytes；
4. 按 Bound IR op topology 排序 selected regions；
5. 在 consumer region launch 前发出 selected transition materialization；
6. 每个 selected region 恰好发出一个 backend-linked launch；
7. 一次性按原 query 顺序 emit 全部 Bound IR outputs；
8. emit 后释放全部 arena；
9. 对 action IDs、instance hash、query IDs 生成稳定 schedule identity。

## 4. 跨层 verifier

当前 verifier 已检查：

- Bound/Template/Instance hash 与 supplied typed objects 精确一致；
- buffers 与 selected storage bindings 完全相等；
- arena allocation size、静态 peak 与 PlanInstance cost/budget 相等；
- budget check 必须在 allocation/launch 前且只出现一次；
- arena 不得重复 allocate、提前 free 或最终泄漏；
- materialization 必须来自 selected transition、source 已定义且早于 consumer；
- launch 的 region/backend/artifact/input/output 必须与 PlanInstance 一致；
- launch input 无 use-before-def；
- selected region 与 transition 全覆盖、无重复；
- emit outputs 等于 Bound IR outputs；
- emit query IDs 与 ScheduleModule query IDs 完全相等，无丢失/重复；
- 同步 reference path 不接受伪装的其他 stream。

## 5. 验证结果

- Schedule IR 专属：`3 passed`；
- 相邻 Bound/Plan/PR-11/12/storage/env：`100 passed`；
- 全量：`421 passed, 1 skipped, 6 warnings`，47.94 s；
- Mypy 0 issues；
- Pylint 10.00/10；
- Black clean；
- `git diff --check` 通过。

## 6. 边界与下一步

本轮只关闭 **IR-3A schema + synchronous lowering + static verifier foundation**。下一切片必须补：

1. typed BatchLoop 与 query slice/merge accounting；
2. bounded Retry/Fallback/OOM semantics；
3. RecordEvent/WaitEvent happens-before；
4. StateLoad/Store/Invalidate 与 Plan state-validity；
5. reference executor、runtime trace、canonical replay；
6. reference executor 输出与 Bound IR interpreter 对齐。

在这些完成前，IR-3、C1/C2 和 IR-driven E2E 均不能升级为 complete。
