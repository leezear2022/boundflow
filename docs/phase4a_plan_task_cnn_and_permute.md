# Phase 4A 计划：Task pipeline CNN 对齐 + permute 支持补齐

## 1. 动机与目标

Phase 4 已经把执行路径切换为 **Planner → BFTaskModule → TaskExecutor**，但目前 Task 执行覆盖还偏窄：

- 只有 MLP 的 task pipeline 对齐测试（MLP 走 task 路径已对齐 auto_LiRPA）
- CNN 虽然在 Phase 3（非 task）已对齐 auto_LiRPA，但 Phase 4（task）尚未覆盖 CNN 结构
- `transpose`/`permute` 仍是“占位支持”，缺少 permute 维度信息，导致更真实的模型（ResNet/ONNX 导入等）会卡住

本 Phase 4A 的目标是：

1. **补齐 Phase 4 的 CNN 对齐测试**：MNIST 风格 CNN（Conv→ReLU→Conv→ReLU→Flatten→Linear→ReLU→Linear）走 `plan_interval_ibp_v0 + PythonTaskExecutor`，与 auto_LiRPA `IBP` 完全一致。
2. **实现 permute/transpose 的真实语义**：从 Torch 前端提取 `aten.permute` 的 dims，并在 task 执行时调用 `torch.permute`（同时更新 shape meta 的一致性约束）。

> 验收导向：一方面用 auto_LiRPA 当 ground truth，另一方面把 task pipeline 覆盖从 MLP 扩到 CNN，并消灭 transpose 的占位实现。

## 2. 现状与缺口

### 2.1 现状

- Torch 前端会把 `aten.permute.default` 映射为 `transpose`（`boundflow/frontends/pytorch/frontend.py`）
- Task pipeline v0 能执行：`linear/conv2d/relu/add/mul/flatten/reshape`（`boundflow/runtime/task_executor.py`）
- `transpose` 在 executor 中仍为“identity 占位”，缺 dims 信息

### 2.2 缺口

- 缺少 Phase 4 的 CNN 对齐测试：目前只有 `tests/test_phase4_task_pipeline_against_auto_lirpa.py`（MLP）
- 缺少 permute dims 的提取与落地：
  - 前端未在 `transpose` 的 attrs 中记录 dims
  - planner 未把 dims 带入 task op
  - task executor 未执行真正的 permute

## 3. 修改计划（按步骤可验收）

### Step 1：补 Phase 4 CNN 对齐测试

- 新增测试文件：`tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`
- 测试内容：
  - 构造 `auto_LiRPA.utils.Flatten` 版本的 MNIST CNN（与 Phase 3 的 CNN 对齐用例一致）
  - `program = import_torch(..., normalize=True)`
  - `task_module = plan_interval_ibp_v0(program)`
  - `bf_out = PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(...))`
  - `auto_LiRPA` 侧 `BoundedModule.compute_bounds(method="IBP")` 得到 `lb/ub`
  - `allclose` 对齐 `lower/upper`

**验收**：该测试在 `conda run -n boundflow python -m pytest -q ...` 下通过。

### Step 2：前端提取 permute dims（IR/Task 可携带）

- 修改 `boundflow/frontends/pytorch/frontend.py`：
  - 在 `_extract_attrs_for_call_function` 中识别 `op_type == "transpose"` 的情况：
    - `aten.permute.default(input, dims)`：提取 dims（list/tuple），写入 `attrs["dims"]`
  - 确保 dims 为 `List[int]`

**验收**：构造一个简单 `x.permute(...)` 的模型导入后，节点 `op_type=="transpose"` 且 `node.attrs["dims"]` 正确。

### Step 3：Planner v0 将 dims 透传到 TaskOp

- 修改 `boundflow/planner/interval_v0.py`：
  - TaskOp.attrs 已复制自 Node.attrs，本步只需确认不丢失，并在必要时对 `transpose` 的 dims 做规范化（例如 tuple→list）

**验收**：`plan_interval_ibp_v0` 输出的 task 中，`transpose` 的 op.attrs 含有 dims。

### Step 4：TaskExecutor 执行真实 permute

- 修改 `boundflow/runtime/task_executor.py`：
  - 增加 `op_type == "transpose"` 的分支：
    - 读取 `dims = op.attrs["dims"]`
    - `lower = x.lower.permute(*dims)`，`upper = x.upper.permute(*dims)`
  - 若 dims 缺失：抛出明确错误（不再 silent identity）

**验收**：新增一个小网络（含 permute）在 task 执行中形状正确且不报错。

### Step 5：回归与记录

- 运行回归测试：
  - `tests/test_phase4_task_pipeline_against_auto_lirpa.py`（MLP）
  - `tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`（新增 CNN）
  - 相关前端/IR 单测（如已有）
- 更新 `docs/change_log.md` 追加 Phase 4A 记录（动机、改动点、验证命令）

## 4. 风险点与规避

- **FX/Export 图中 permute 表达差异**：不同 PyTorch 版本可能产生 `call_method` 或 `call_function` 形式；v0.1 先覆盖 `aten.permute.default` 的 `call_function`，后续再补 `call_method::permute`。
- **dims 常量性**：若 dims 不是常量（极少见），直接报错并提示 normalize/前端需要先常量折叠。

## 5. 完成定义（DoD）

- 新增 CNN task pipeline 对齐测试通过
- `transpose` 不再是占位：在 task executor 中真实执行 permute；缺 dims 时显式失败
- 更新 `docs/change_log.md` 记录本阶段变更

