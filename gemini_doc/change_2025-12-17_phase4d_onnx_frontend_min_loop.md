# 变更记录：Phase 4D（ONNX 前端最小闭环）

## 动机

为了避免 Phase 5 出现“两套前端两套语义/normalize”的分叉，Torch-export 与 ONNX-import 必须尽早统一到：

- 同一 Primal IR
- 同一 planner（`plan_interval_ibp_v0`）
- 同一 executor（`PythonTaskExecutor` / `TVMTaskExecutor`）

本次 Phase 4D 的目标是：**在算子子集内形成 ONNX → Primal IR → Task → IBP 的闭环**，并用测试证明其与 Torch-import 一致。

## 本次改动

### 1) ONNX frontend 实现

- 文件：`boundflow/frontends/onnx/frontend.py`
- 能力：
  - 支持 `onnx.shape_inference.infer_shapes`
  - 支持 ONNX → Primal IR 映射（闭环子集）：
    - `Gemm` → `linear`
    - `MatMul`（weight 为常量）→ `linear`（无 bias）
    - `Conv` → `conv2d`
    - `Relu` → `relu`
    - `Add`/`Mul` → `add`/`mul`
    - `Flatten` → `flatten(start_dim=axis, end_dim=-1)`
    - `Reshape` → `reshape(shape=...)`（shape 需为常量）
    - `Transpose` → `permute(dims=...)`
    - `Identity`：alias 消除
    - `Constant`：写入 `program.params` 并标记为 const
  - `Reshape` 的第二输入（shape）必须是 initializer/Constant：在 v0 直接固化到 `attrs["shape"]`，避免引入 shape 计算子图。

### 2) 端到端对齐测试（Torch-import vs ONNX-import）

- 文件：`tests/test_phase4d_onnx_frontend_matches_torch.py`
- 覆盖：
  - MLP：`Linear → ReLU → Linear`
  - MNIST-style CNN：`Conv → ReLU → Conv → ReLU → Flatten → Linear → ReLU → Linear`
- 验收方式：
  - 对同一模型/输入/eps，比较 `import_torch(...)` 与 `import_onnx(...)` 进入同一 planner/executor 后的 IBP 输出上下界（allclose）

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase4d_onnx_frontend_matches_torch.py
conda run -n boundflow python -m pytest -q
```

注：测试中会用 `torch.onnx.export` 临时导出 ONNX；它会打印 PyTorch 的 exporter deprecation warning，但不影响正确性。

