# 变更记录：IR-4B PyTorch typed backend registry

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`bbaa5ff`（IR-4A typed backend dispatch）
> 状态：PyTorch dense/structured/chunked typed migration validated；IR-4 尚未完成

## 1. 审计结论

Plan IR selector 已经能在候选 region 中枚举 exact non-overlap partition，因此 fused region 不需要
再造一套 planner。真正缺口有两个：

1. Task lowering 把所有 backend 的 implementation ID 都错误写成
   `bound_ir_region_reference/v1`；
2. Bound session 只能逐 op stepping，无法把一个 ReLU→Affine fused Task 交给现有
   `TorchChunkedFusedCrownExecutor`。

本轮修复这两个跨层断点，没有用 registry 包装 reference 执行来冒充 backend 迁移。

## 2. Backend-specific Task identity

Task lowering 现在按 `BackendKind` 生成稳定 implementation ID：

- `REFERENCE` → `bound_ir_region_reference/v1`；
- `PYTORCH_DENSE` → `pytorch_dense_bound_region/v1`；
- `PYTORCH_STRUCTURED` → `pytorch_structured_bound_region/v1`；
- `PYTORCH_CHUNKED` → `pytorch_chunked_fused_relu_affine/v1`；
- TorchCompile/TVM 三类也有预留的独立 ID。

Task verifier 使用同一映射反向核验，不能再把 selected chunked/TVM candidate 伪装为 reference。

## 3. 真实 fused Task execution

`PlainCrownBoundIRSession.execute_fused_relu_affine_task()`：

- 只接受拓扑中连续的两个 op；
- 严格要求 `RELU_RELAXATION → LINEAR_BACKWARD/CONV2D_BACKWARD`；
- 从 typed Bound attrs、runtime parameter binding 和 ReLU preactivation bounds 构造
  `FusedReluAffineDescriptor/Request`；
- 计算与原 CROWN path 相同的 αu/αl/βu/βl；
- Linear/Conv 分别携带 weight、bias、shape、stride/padding/dilation/groups/output_padding；
- 调用 backend 的 `supports_descriptor`、`supports`、`run`；
- 把 fused result 写回 session env，并一次推进两个 Bound ops；
- 拒绝 non-contiguous、wrong-pattern、unsupported descriptor/request。

因此 chunked 路径会真实调用原 `TorchChunkedFusedCrownExecutor`，不是 sequential reference
stepping。

## 4. PyTorchTaskBackendRegistry

registry 支持：

- `REFERENCE`：IR-4A reference adapter；
- `PYTORCH_DENSE`：fused region 调用 `TorchDenseFusedCrownReference`，其他 typed region
  逐 op reference；
- `PYTORCH_STRUCTURED`：执行显式 cast/materialize/operator Bound IR；
- `PYTORCH_CHUNKED`：只接受 fused ReLU→Affine Task，调用 chunked executor；
- 其他 backend fail closed。

每个 prepared entry 仍由完整 `BackendDispatchKey` 锁定，并核对 backend/task/op/output/capability。

## 5. 验证证据

专属 `tests/test_task_backend_dispatch_v1.py`：

- MLP fused ReLU→Linear：dense typed backend 与 whole-Bound 对齐；
- CNN fused ReLU→Conv：dense typed backend 与 whole-Bound 对齐；
- structured MLP：显式 operator IR 经 structured registry 与 whole-Bound 对齐；
- CUDA MLP：chunk rows=2 的真实 chunked executor 与 whole-Bound 对齐；
- 把 chunked backend 绑定到 non-fused task：fail closed。

结果：

- IR-4B 专属：`5 passed`；
- Plan/Task/Schedule/Artifact/IR-4B 相邻：`42 passed`；
- 全量：`448 passed, 1 skipped, 6 warnings`，56.81 s；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black 与 `git diff --check` clean。

## 6. 未完成边界

IR-4 仍未关闭：

- TVM fused/unfused backend 尚未接入 `TypedTaskBackend`；
- TVM disk compile cache 尚未改用完整 dispatch key；
- Schedule OOM retry 目前只验证 control driver，尚未驱动 semantic backend fallback；
- PR-13 QueryBatch 尚未 lower 到 PlanInstance/Task/Schedule；
- state payload 的实际 load/store/skip execution 尚未迁移。

下一切片为 **IR-4C TVM typed backend + compile cache migration**。必须先完成 fresh-process
compile/cache replay 和 Python/TVM fused/unfused 对齐，再进入 Query Runtime 迁移。
