# 变更记录：Task/Schedule IR v1 逐任务语义闭环

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`9aa81bc`（IR-3C typed Task IR foundation）
> 状态：IR-3 synchronous reference closure validated-reduced；下一阶段为 IR-4

## 1. 本轮关闭的问题

IR-3C 只证明了 Task schema、Plan region lowering 和 Schedule launch linkage，Task dispatch
本身没有执行 Bound IR 数学语义。本轮把 reference path 改为真实的逐 Task 执行：

```text
Bound IR + PlanInstance
  -> TaskIRModule + ScheduleModule
  -> Schedule LaunchAction order
  -> PlainCrownBoundIRSession.execute_task(exact op partition)
  -> per-task boundary value hashes
  -> final lower/upper
```

旧 whole-Bound interpreter 继续作为数值对照；旧 `runtime/scheduler.py` 只服务历史路径，
不被新的 Task/Schedule reference path 导入。

## 2. Stateful Bound IR stepping

`PlainCrownBoundIRSession`：

- 初始化并验证 Bound module、旧 primal task/parameter fingerprint、objective payload；
- 保留跨 Task 的 runtime env；
- 每次只接受拓扑中的下一个连续 op slice；
- 拒绝 skip、reorder、repeat 和提前读取最终结果；
- 执行 SPEC_BIND、cast/materialize、compose、linear/conv/ReLU、reshape、add、concat、
  concretize 的 reference semantics；
- 为每个 Task boundary value 生成确定性 SHA-256。

逐 Task executor 按 Schedule `LaunchAction` 次序执行对应 `TaskIRUnit.op_refs`，在 launch 前检查
Task dependency，并输出 canonical `TaskExecutionTrace`。

## 3. Task/Schedule 契约补项

closure audit 对照冻结契约后补了两个遗漏：

1. `TaskValueConstraint` 把每个 task input/output 的 `BoundTensorType` 显式带入 Task IR，
   包含 static/dynamic shape、dtype、layout、device 和 sample/spec/domain batch axes；
2. Schedule action set 新增 typed `TransferAction`/`TransferDirection`，reference verifier 会检查
   value definition、方向、目标设备和 stream，executor trace 会记录该 action。

当前 IR-3 范围仍是单 host/单 device、同步 reference launch；多设备 D2D 和真实异步数据搬运
不在本轮完成声明内。

## 4. 数值与图覆盖

逐 Task final bounds 与原 whole-Bound interpreter 对齐：

- 两层 MLP；
- chain CNN（conv/relu/flatten/linear）；
- residual/fanout DAG；
- concat/fanout DAG；
- structured MLP，显式 `REPRESENTATION_CAST`/`MATERIALIZE` ops 均出现在 Task trace。

此外，fresh-process Schedule artifact 升级为
`boundflow.schedule-ir-artifact/v2`：

- `task_trace.json` 含每个 Task 的真实 output value hashes；
- `bound_result.json` 含最终 lower/upper 的 dtype、shape、content hash；
- manifest 锁定上述 payload/hash；
- replay 在新进程重建 semantic inputs 并重新逐 Task 执行；
- 篡改 final result 会 fail closed。

## 5. IR-3 closure audit

| 冻结门禁 | 证据 | 判定 |
|---|---|---|
| Task dependency/use-before-def | Task verifier + semantic executor dependency check | 通过 |
| peak ledger 与 runtime trace 对齐 | Schedule verifier/executor + Plan peak equality | 通过 |
| OOM retry 有界且不丢 query | declared retry/fallback ladder、exhaustion/query tests | 通过 |
| custom stream/event happens-before | missing wait rejection + record/wait regression | 通过 |
| batch query 无丢失/重复/错序 | BatchLoop/EmitResult accounting tests | 通过 |
| state validity | exact state-version Plan decision → StateLoad/Store/Invalidate verifier | reference contract 通过 |
| topo loop 仅作对照 | 新 Task/Schedule executor 不导入 legacy scheduler | 通过 |
| per-task semantic outputs/replay | graph-family equality + artifact v2 fresh-process replay | 通过 |

因此 IR-3 可以关闭为 **synchronous reference closure validated-reduced**。这不等于生产
backend/runtime 已迁移，也不升级 ASPLOS C1/C2 headline claim。

## 6. 验证结果

- Task/Schedule/Artifact 专属：`24 passed`；
- 全量：`442 passed, 1 skipped, 6 warnings`，55.07 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black clean；
- `git diff --check` 通过。

## 7. 保留边界与下一步

仍未完成：

- production PyTorch dense/chunked/structured 和 TVM fused/unfused backend 只消费新 Task IR；
- compile cache/capability rejection/state payload 由新 hash/id 驱动；
- PR-13 query runtime lower 到 PlanInstance/Task/Schedule；
- 多 GPU、异步编译、真实 D2D transfer；
- PR-14 external mismatch 仍是显式 No-Go。

下一阶段严格进入契约 **IR-4：现有 backend/runtime 迁移**。第一切片应建立 typed backend
dispatch adapter 和 compile-cache key，从 PyTorch dense reference candidate 开始，不直接跳到
TVM 性能调优。
