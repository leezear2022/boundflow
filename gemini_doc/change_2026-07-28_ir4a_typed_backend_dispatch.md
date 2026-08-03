# 变更记录：IR-4A typed backend dispatch 与 cache identity

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`1a56445`（IR-3 Task/Schedule synchronous reference closure）
> 状态：IR-4A PyTorch reference dispatch foundation validated；IR-4 尚未完成

## 1. 动机

IR-3 已让 Task/Schedule reference path 真实执行逐 Task Bound semantics，但 backend 调用仍直接
落到 `PlainCrownBoundIRSession.execute_task()`。历史 PyTorch/TVM 路径还分别使用旧
`BFTaskModule`、`FusedReluAffineRequest`、graph fingerprint 或 ad-hoc shape tuple 作为 cache
identity，不能满足 IR-4 的“backend 只消费 typed Task/Schedule IR”门禁。

本轮先迁移最窄的 PyTorch reference candidate，冻结后续 backend 共用的 dispatch/cache 合同。

## 2. BackendDispatchKey

新增 `boundflow/runtime/task_backend_dispatch.py`。`BackendDispatchKey` 锁定：

- Bound module hash；
- PlanTemplate hash；
- PlanInstance hash；
- TaskIRModule hash；
- task ID；
- selected backend candidate/capability ID；
- compiled artifact key；
- reference implementation ID。

key 使用 canonical JSON + SHA-256，不从 runtime tensor shape 临时猜测。构造前会完整验证
TaskModule 与 Bound/Plan/Instance 的跨层引用。

## 3. Typed PyTorch reference adapter

`PyTorchReferenceTaskBackend`：

- 入口只接收 `TaskIRUnit`、`BackendDispatchKey`、typed Plan capability 与 Bound session；
- 拒绝非 `REFERENCE`、非 static-legal、capability 不匹配或 op kind 不支持的 candidate；
- 检查 key 的 Bound/Template hash 与当前 session/template 一致；
- 用完整 dispatch hash 缓存 prepared task partition；
- cache hit 仍核对 task/op/output/backend/capability，拒绝 collision/stale payload；
- 最终只执行该 Task 的 op refs 和 output boundary。

`TaskExecutionEvent` 新增 `backend_dispatch_key`，因此 artifact v2 会把实际 backend dispatch
identity 一并锁定和重放。

## 4. 验证

- 同一 workload 重复执行：第一次每个 task miss，第二次每个 task hit；
- 两次 final bounds 与 Task trace 完全一致；
- 每个 task 的 dispatch key 唯一且为 64 位 SHA-256；
- 篡改 Bound hash：dispatch 前 fail closed；
- selected candidate 改为 static-illegal：capability dispatch 前 fail closed；
- Task/Artifact 定向：`15 passed`；
- 全量：`443 passed, 1 skipped, 6 warnings`，56.12 s；
- Black、Mypy、Pylint、diff-check 门禁见提交前验证。

## 5. 边界与下一步

本轮只完成 **PyTorch reference dispatch/cache foundation**，没有迁移：

- PyTorch chunked/structured candidate；
- TVM fused/unfused candidate 与 disk compile cache；
- PR-13 QueryBatch → PlanInstance/Task/Schedule；
- runtime state payload load/store；
- CUDA stream 上的真实 transfer/launch driver。

下一切片应把 `TorchChunkedFusedCrownExecutor` 和 structured reference candidate 接到同一
`TypedTaskBackend` registry，再单独迁移 TVM compile cache。IR-4 在上述路径完成前不能关闭。
