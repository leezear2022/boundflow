# auto_LiRPA vs BoundFlow Executors 对比（IBP/Interval 视角）

本文对比三个“算 bounds 的执行路径”：

- **auto_LiRPA**：外部 ground truth（成熟实现，算法/工程复杂度更高）
- **BoundFlow `PythonTaskExecutor`**：reference backend（正确性优先、可调试）
- **BoundFlow `TVMTaskExecutor`**：编译后端 demo（Python driver + TVM kernel，逐步扩大覆盖）

默认语境：**IBP（Interval Bound Propagation）**，输入扰动为 `L∞`（`center ± eps`）。

---

## 1. 三者在 BoundFlow 体系里的角色

### auto_LiRPA
- 作用：**外部基准/对齐对象**（ground truth）。
- 入口：`BoundedModule.compute_bounds(x=(BoundedTensor,), method="IBP", C=...)`。
- 特点：算子覆盖多、带大量工程优化（缓存、中间界、loss fusion、不同 method 路径等）。

### PythonTaskExecutor（BoundFlow）
- 作用：**reference backend**，用于：
  - 作为 TVM lowering 的语义参考（TVM 输出应与它一致）
  - 快速调试 Task/Planner/attrs/shape 问题
- 入口：`PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(...))`。
- 特点：实现直接、可读；不追求性能。

### TVMTaskExecutor（BoundFlow）
- 作用：**Phase 4C v0 的编译后端 demo**，用于证明：
  - “同一个 BFTaskModule 可以被编译并执行”
  - “TVM kernel 输出与 reference backend 一致”
- 入口：`TVMTaskExecutor().run_ibp(task_module, LinfInputSpec(...))`。
- 特点：目前是 **Python driver** 顺序执行 TaskOp；其中部分算子走 TVM kernel（v0 先从 `linear` 开始），其余算子 fallback 到 torch。

---

## 2. 执行链路对比（从模型到 bounds）

### BoundFlow（两种 executor 共用）
1) Torch 导入：`import_torch(model, (x0,), export_mode="export", normalize=True)`  
2) Planner：`plan_interval_ibp_v0(program)` 生成 `BFTaskModule`（TaskOp 序列 + StoragePlan）  
3) Executor：`PythonTaskExecutor` 或 `TVMTaskExecutor` 执行 TaskOp，输出 `IntervalState(lower, upper)`

对应代码：
- Torch 前端：`boundflow/frontends/pytorch/frontend.py`
- Planner v0：`boundflow/planner/interval_v0.py`
- Task executor（reference）：`boundflow/runtime/task_executor.py`
- TVM executor（demo）：`boundflow/runtime/tvm_executor.py`

### auto_LiRPA
- 直接 wrap `nn.Module`：`BoundedModule(model, example_input)`
- 输入扰动：`BoundedTensor(x0, PerturbationLpNorm(norm=inf, eps=eps))`
- `compute_bounds` 内部构建/遍历 bound graph，对每个 op 调用自己的 `interval_propagate`

---

## 3. 语义对齐：我们怎么“知道对不对”

### A) BoundFlow vs auto_LiRPA（语义正确性）
这是“最强背书”的对齐：同模型/同输入/同 eps，输出 bounds 必须 `allclose`。

现有测试：
- IBP（MLP）：`tests/test_phase4_task_pipeline_against_auto_lirpa.py`
- IBP（MNIST CNN）：`tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`
- IBP + Property（C 矩阵）：`tests/test_phase4b2_margin_c_against_auto_lirpa.py`

> 注意：IBP + `C` 并不是简单“先算 logits interval 再乘 C”就能稳定对齐。我们实现了与 auto_LiRPA 对齐的策略（优先把 `C` 融合进最后一层 linear），见 `boundflow/planner/interval_v1.py`。

### B) TVMTaskExecutor vs PythonTaskExecutor（lowering 正确性）
这是“后端正确性”的对齐：同一个 `BFTaskModule` 输入相同，TVM 输出必须与 Python reference 一致。

现有测试：
- `tests/test_phase4c_tvmexecutor_matches_python.py`

---

## 4. 对比表（工程与性能视角）

| 维度 | auto_LiRPA | PythonTaskExecutor | TVMTaskExecutor |
|---|---|---|---|
| 目标 | 完整 verifier/训练工具 | reference/ground truth | 编译后端 demo |
| 输入表示 | BoundedTensor + Perturbation | BFTaskModule + LinfInputSpec | BFTaskModule + LinfInputSpec |
| 图/调度 | 内部 bound graph + 复杂选项 | TaskOp 顺序执行 | TaskOp 顺序执行（部分 op 用 TVM kernel） |
| 编译 | 无（纯 torch） | 无 | 有（首次遇到某 shape/dtype/target 编译） |
| 缓存 | 多层缓存/优化 | 无特别缓存 | 进程内 `lru_cache`（按 shape/dtype/target） |
| 覆盖范围 | 很广 | 当前只覆盖 v0.1 primitive 子集 | v0 仅加速 `linear`（2D weight）；其它 fallback |
| 用途 | ground truth / 对比 | 作为 TVM 的对齐基准 | 证明可编译与后端通路正确 |

---

## 5. “TVMTaskExecutor 真的用到 TVM kernel 吗？”

是的，但目前只在 `linear` 上：

- 编译：`boundflow/backends/tvm/interval_linear.py` 的 `build_interval_linear_module(...)`
- 调用：`boundflow/runtime/tvm_executor.py` 的 `func(...)`

现阶段它更像“按需编译的 JIT（进程内缓存）”：
- 同一进程内，同 shape/dtype/target 的 `linear` 只编译一次
- 重启进程会重新编译（尚未做磁盘缓存）

---

## 6. 现阶段限制（与后续演进方向）

- TVMTaskExecutor v0 只加速 `linear` 的 interval affine（且仅 2D weight 的 case），下一步会扩到：
  - `conv2d` 的 interval affine kernel
  - 更粗粒度的 fused task（减少 Python driver 调度与数据搬运）
  - 磁盘缓存（跨进程复用 build 产物）
- 更一般的 LiRPA（CROWN/α-CROWN）不在这份对比范围内，需要引入 `lirpa_linear` DomainState 与 backward bound（Phase 5）。

