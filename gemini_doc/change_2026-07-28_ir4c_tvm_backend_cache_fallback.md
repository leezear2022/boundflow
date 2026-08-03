# 变更记录：IR-4C TVM typed backend、cache namespace 与 semantic fallback

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`12b064a`（IR-4B PyTorch typed backend registry）
> 状态：TVM typed backend/cache + semantic fallback validated；IR-4 尚未关闭

## 1. TVM typed Task registry

新增 `TVMTaskBackendRegistry`：

- reference regions委托 typed reference adapter；
- `TVM_FUSED_TIR` 只接受连续的 ReLU→Linear/Conv fused Task；
- `TVM_TIR_UNFUSED` 使用相同 typed fused request，但执行显式 scaled-A workspace baseline；
- non-fused、错误 op pattern、错误 capability/backend 均 fail closed；
- composite `TypedTaskBackendRegistry` 可在 PyTorch/TVM/reference candidate 之间按 typed
  backend candidate 路由。

MLP/CNN 的 selected fused partition 由 Plan selector 的 exact-cover search 产生；其余 binding、
reshape、concretize regions 保持 reference candidate，TVM 不会错误消费不支持的 Task。

## 2. Typed dispatch-namespaced TVM cache

旧 fused cache key 只包含 kernel signature、target、code schema 和 TVM ABI，两个不同
PlanInstance/TaskModule 可能共享同一 cache identity。

本轮升级为 `boundflow.fused_crown_cache/v2`：

- `FusedCrownModuleCache.get()` 接收可选的 64 位 `backend_dispatch_key`；
- canonical cache payload/manifest 包含该 dispatch key；
- memory key 也包含 dispatch namespace；
- typed TVM registry 用 Task event 的完整 `BackendDispatchKey.stable_hash()` 绑定 cache；
- library SHA-256、manifest payload、schema 和 cache key 仍全部 fail closed；
- legacy direct executor 使用 `None` namespace，旧测试接口保持兼容，但 v1 disk entry 会失效重编。

TVM unfused baseline 原本没有 disk cache，本轮只迁移 typed execution，不虚构 unfused disk-hit
声明。

## 3. Schedule semantic OOM fallback

`execute_task_ir_semantics()` 现在读取 Schedule 的 `RetryAction/FallbackAction`：

1. 先按 selected backend dispatch；
2. 捕获 `ScheduleOutOfMemoryError` 或 `torch.OutOfMemoryError`；
3. 按 Schedule 声明的有界 fallback ladder，构造同 region 的临时 typed backend binding；
4. 使用保留原 PlanInstance/TaskModule hash、但锁定 fallback candidate/capability/artifact 的新
   dispatch key；
5. 真实执行 fallback backend；
6. 全部失败后抛 `ScheduleRetryExhausted`。

`TaskExecutionEvent` 新增 `attempted_backend_candidate_ids`，并要求最后一个 attempt 等于实际
成功 backend。artifact v2 会序列化并重新验证该字段。

## 4. 验证证据

新增/扩展门禁：

- CUDA typed TVM fused MLP、CNN 与 whole-Bound 对齐；
- CUDA typed TVM unfused MLP、CNN 与 whole-Bound 对齐；
- fused cache manifest 为 v2，包含 64 位 backend dispatch namespace；
- 同一 typed workload 两个 cache object 为 miss→disk_hit；
- 两个独立 Python 进程为 miss→disk_hit，cache key 完全一致；
- selected PyTorch dense backend 注入 semantic OOM 后，真实切换 reference fallback；
- Task trace 记录 `(selected, fallback)`，最终 bounds 对齐且 Schedule query IDs 不丢失；
- 2×2 TVM Conv descriptor 保持 capability rejection；测试改用 backend v1 合法的 3×3 CNN。

结果：

- IR-4C 新增专属：`7 passed`；
- IR-4/Plan/Task/Schedule/TVM 相邻：`43 passed`；
- 全量：`455 passed, 1 skipped, 6 warnings`，63.76 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black、`git diff --check` clean。

## 5. 尚未完成

IR-4 当前剩余：

- PR-13 `BoundQuery/QueryBatch` 仍未 lower 为 PlanInstance→Task/Schedule 执行；
- runtime cached state payload 尚未实际 load/store/skip computation；
- 旧 solver-facing PyTorch/TVM entry points 仍保留，尚未全部改为 typed registry；
- PR-14 external whole-query mismatch 继续为显式 No-Go。

下一切片为 **IR-4D Query Runtime + state payload migration**。完成后必须逐条审计 IR-4 冻结门禁，
不能只凭 backend registry 关闭 IR-4。
