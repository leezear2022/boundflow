# 变更记录：IR-4D typed compiler query 与精确 state runtime

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`e511966`（IR-4C TVM typed backend/cache + semantic fallback）
> 状态：IR-4D validated；IR-4 closure audit 与 legacy same-solver 路径处理仍待完成

## 1. 审计结论与入口边界

PR-13 `BoundQuery/BoundQueryPayload` 只携带 α/β/split BaB 查询身份、input/spec 与 solver
state，不携带重建 plain-CROWN typed compiler pipeline 所需的 `BFBoundModule`、
`PlanTemplate`、`relu_pre` 等完整输入。当前两类 capability 也分别是
`alpha_dense` 和 `alpha_beta_dense_split`。

因此本轮没有把 PR-13 请求偷偷降级成 plain CROWN，而是新增
`TypedCompilerQueryRequest/Payload`：

- payload 显式持有 legacy primal task、Bound IR、PlanTemplate、input/spec 和 ReLU
  pre-activation runtime binding；
- 入口只接受 CROWN、无 grad/alpha/beta/split 的已验证子集；
- runtime 按 Bound/Template/shape/dtype/device/perturbation identity 复用
  PlanInstance 与 TaskIRModule；
- 每个请求 lower 独立 Schedule，结果严格保持调用者 query 顺序；
- 不宣称跨查询 physical batching；该字段在 audit 中固定为 `false`；
- legacy PR-13 α/β 请求在 compiler 入口抛 `CompilerQueryCapabilityError`，错误文本保留
  PR-14 whole-query `NO-GO`，禁止 plain-CROWN fallback。

## 2. State payload 与计算跳过

新增 `BoundRuntimeStatePayload/Store`，将一个 runtime value 绑定到：

- `state_id`；
- `source_value_id`；
- Bound IR `state_version`；
- 完整 `bound_module_hash`；
- tensor content SHA-256。

v1 只接受 static-shape dense tensor。结构化 `LinearOperator` 尚无稳定序列化契约，因此
fail closed。

Schedule semantic execution现在真实执行：

1. `StateLoad` 查找 module/value/version 完全一致的 payload；
2. session 再核对 shape/dtype/device/state_version 与 content hash；
3. 只有当一个 Task 的全部 boundary outputs 均由 exact load 覆盖时，Schedule 才省略
   Launch，session 才推进并跳过该 Task 的计算；
4. `StateStore` 从已计算 session 导出 owned dense tensor并写入精确身份；
5. `StateInvalidate` 删除对应 payload；
6. stale/missing payload 不会产生复用计划，若 Schedule 已声明 load 但 payload 不存在则
   明确报错。

Task trace 对跳过任务记录 `backend_candidate_id=state-reuse`、真实 output hashes 和由全部
payload identity 组成的 SHA-256。

## 3. 验证证据

专属测试覆盖：

- 两个 typed query 保持输入顺序，Plan cache 为 1 miss→1 hit；
- typed query final bounds 与 whole-Bound reference 完全对齐；
- 首次 4 个 middle-region outputs cache，第二次 4 个 exact load；
- 第二次只跳过对应 middle Task，final lower/upper 不变；
- stale state 不进入 reuse plan，并由 cache 重算覆盖；
- payload 创建后被原地篡改时 content hash 校验失败；
- PR-13 alpha-CROWN / alpha-beta-CROWN 均在 compiler 入口显式 No-Go；
- 两个独立进程 generate/replay 同一 artifact，query order、plan/state audit、trace 和
  final bound hashes 全部一致。

当前结果：

- IR-4D + Plan/Task/Schedule/Backend 相邻：`42 passed`；
- Mypy：0 issues；
- Pylint：10.00/10；
- Black、`git diff --check` clean。

- 全量：`462 passed, 1 skipped, 6 warnings`，65.65 s。

## 4. 尚未关闭的 IR-4 问题

`SameSolverQueryRuntime` 的 α/β executor 仍直接执行旧 solver runtime，而不是消费 typed
Task/Schedule IR。IR-4 契约要求旧 PR-13 机制通过正式适配器消费新 IR，同时又要求 PR-14
external mismatch 不被 fallback 隐藏。因此 IR-4D 只关闭“可证明等价的 typed query +
state runtime”：

- 不声称 α/β Task IR 已实现；
- 不声称 PR-13 external workload 已 compiler-migrated；
- 不把显式 No-Go 写成“迁移成功”；
- 下一步必须执行 IR-4 closure audit，并对 legacy same-solver 路径给出实现、退役或
  validated-reduced 边界，之后才能决定是否进入 IR-5。
