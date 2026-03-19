# Phase 4 总结：Task/Planner/Executor 闭环（IBP）+ Spec(C) 对齐 + TVM 后端 demo + ONNX 前端最小闭环

> 本文总结 Phase 4 的“目标 → 里程碑 → 代码落点 → 回归钉子 → 已知限制”，用于：
>
> - 论文/答辩叙事：解释为什么 Phase 4 是 Phase 5/6 的必要地基；
> - 工程接手：快速定位 Task IR/Planner/Executor/TVM/ONNX 的入口与验证方式；
> - 复现回归：知道哪些测试需要可选依赖（auto_LiRPA/tvm/onnx），以及如何跑。
>
> 主要依据：
> - `docs/change_log.md`（Phase 4 总账）
> - `gemini_doc/change_2025-12-17_phase4*.md`（Phase 4B.3/4C/4D 的分 PR 记录）
> - Phase 4→5 过渡计划：`gemini_doc/next_plan_after_phase4c.md`

---

## 1. Phase 4 解决什么问题（目标与边界）

Phase 4 的核心目标是把 Phase 3 的 “Primal IR + PythonInterpreter（逐节点解释执行）” 升级成一个可扩展的系统骨架：

1. **把执行抽象成 Task IR**：让算子语义由 `TaskOp` 显式携带（而不是 Domain 通过 tensor rank 猜测）。
2. **让 Planner 输出可执行模块**：`BFTaskModule`（v0 先单 task），并提供 `validate()`。
3. **提供两条执行路径**：
   - `PythonTaskExecutor`：reference 执行器（正确性优先、便于调试）
   - `TVMTaskExecutor`：后端 demo（证明“同一 Task 能下沉到 TVM kernel 且输出与 reference 一致”）
4. **补齐 property/spec（C 矩阵）口径**：对齐 auto_LiRPA 的 `compute_bounds(C=..., method='IBP')` 语义。
5. **避免前端分叉**：ONNX-import 与 Torch-export 必须统一到同一套 Primal IR + planner + executor。

Phase 4 的边界（刻意不做）：

- 不引入 TaskGraph/scheduler/cache/batching（留给 Phase 5 系统化）；
- 不引入 CROWN/αβ/BaB（留给 Phase 6 方法族与 solver）。

---

## 2. Phase 4 的完成定义（Done Definition）

Phase 4 的“完成”不以性能为准，而以接口稳定与三方对齐为准：

1. **Planner+Task pipeline 对齐 auto_LiRPA（IBP）**（MLP 与 CNN 子集）。
2. **Spec(C) 对齐 auto_LiRPA**：IBP + C 的输出不因“先算 logits 再乘 C”而额外松弛。
3. **TVM 后端 demo 正确性**：`TVMTaskExecutor == PythonTaskExecutor`，并且两者都对齐 auto_LiRPA（可选依赖）。
4. **ONNX 前端最小闭环**：同一模型 Torch-import 与 ONNX-import 进入同一 planner/executor 后输出一致。
5. **工程化收口**：layout-only `permute` 在 planner 层有最小可优化语义（合并/消除），避免 Phase 5 被无意义重排拖垮。

---

## 3. 里程碑回顾：4 → 4A → 4B → 4C → 4D

### 3.1 Phase 4：Task/Planner v0（把 IBP 解释执行抽象成任务）

**核心改动**

- Task IR：`boundflow/ir/task.py`
  - `TaskOp`（可执行算子表示）
  - `TaskKind.INTERVAL_IBP`
  - `BFTaskModule(entry_task_id, validate)`
- Planner v0：`boundflow/planner/interval_v0.py`
  - `plan_interval_ibp_v0(program)`：Primal Graph → 单 task 的 `BFTaskModule`
  - 对 `linear/conv2d/reshape` 写入必要 attrs（op/shape）
- Runtime：`boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp(...)`：执行 TaskOp 序列（reference）
- 兼容层：`boundflow/runtime/executor.py`
  - 保留 Phase 3 API（输入 `BFPrimalProgram`），内部改为 planner+task executor（平滑迁移）

**关键回归**

- `tests/test_phase4_task_pipeline_against_auto_lirpa.py`

### 3.2 Phase 4A：CNN 覆盖 + permute/transpose 语义补齐

**核心改动**

- Torch frontend：`boundflow/frontends/pytorch/frontend.py`
  - `aten.permute.default` 映射为 `transpose`/`permute` 并写入 dims
- Task executor：`boundflow/runtime/task_executor.py`
  - 真实执行 `permute(*dims)`，dims 缺失显式报错（避免 silent wrong）

**关键回归**

- `tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`

### 3.3 Phase 4B.0：引入 StoragePlan（Memory/Storage 抽象 schema 钉住）

**核心改动**

- `boundflow/ir/task.py`：`BufferSpec` + `StoragePlan`，并让 `BFTaskModule` 携带 `storage_plan`
- `boundflow/planner/interval_v0.py`：默认填充 “一值一 buffer”

**关键回归**

- `tests/test_phase4b_storage_plan.py`

### 3.4 Phase 4B.2：Spec/Property（LinearSpec(C)）对齐 auto_LiRPA

**核心改动**

- Spec IR：`boundflow/ir/spec.py`（`LinearSpec(C)`）
- Planner：`boundflow/planner/interval_v1.py`
  - 优先把 `C` 融合进最后线性层（`W' = C@W, b' = C@b`）以对齐 auto_LiRPA 的 IBP+C 语义
- Executor：`boundflow/runtime/task_executor.py`
  - 支持 batched weight（`w:[B,O,I]`）以执行融合后的 property

**关键回归**

- `tests/test_phase4b2_margin_c_against_auto_lirpa.py`

### 3.5 Phase 4B.3：layout-only `permute` 简化 pass（合并/消除）

**核心改动**

- Planner pass：`boundflow/planner/passes/layout_only.py`
  - 连续 `permute` 合并、identity `permute` 消除（通过 value alias 重写）

**关键回归**

- `tests/test_phase4b3_layout_permutes.py`

对应记录：`gemini_doc/change_2025-12-17_phase4b3_layout_only_permute_pass.md`。

### 3.6 Phase 4C：TVMTaskExecutor v0（Python driver + TVM kernel demo）

**核心改动**

- TVM kernels（IBP）：
  - `boundflow/backends/tvm/interval_linear.py`
  - `boundflow/backends/tvm/interval_conv2d.py`（后续补齐）
- Executor：
  - `boundflow/runtime/tvm_executor.py`：`TVMTaskExecutor`（优先走 TVM kernel，不支持则 fallback）
  - `last_stats`：统计 tvm_ops/fallback_ops、kernel cache 等（提升可观测性）
- bench：
  - `scripts/bench_phase4c_tvmexecutor.py`（可选）

**关键回归**

- `tests/test_phase4c_tvmexecutor_matches_python.py`
- `tests/test_phase4c_tvmexecutor_matches_python_cnn.py`
- 三方对齐（可选依赖）：`tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

对应记录：`gemini_doc/change_2025-12-17_phase4c_triple_alignment_test.md`、`gemini_doc/change_2025-12-17_phase4c_interval_conv2d_tvm.md`。

### 3.7 Phase 4D：ONNX 前端最小闭环（shape_infer + Primal IR 映射）

**核心改动**

- `boundflow/frontends/onnx/frontend.py`
  - `onnx.shape_inference.infer_shapes`
  - ONNX 子集 op 映射到 Primal IR（Gemm/Conv/Relu/Add/Mul/Flatten/Reshape/Transpose/Constant/Identity）
  - `Reshape` 的 shape 必须常量（v0 直接固化到 attrs，避免引入 shape 计算子图）
- 对齐测试：
  - Torch-import vs ONNX-import 进入同一 planner/executor 后 IBP 输出一致

**关键回归**

- `tests/test_phase4d_onnx_frontend_matches_torch.py`

对应记录：`gemini_doc/change_2025-12-17_phase4d_onnx_frontend_min_loop.md`。

---

## 4. 代码落点与“从哪看起”

### 4.1 Task/Planner/Executor 主链路

- `boundflow/ir/task.py`：Task IR + `BFTaskModule` + `StoragePlan`
- `boundflow/planner/interval_v0.py`：Primal → Task（v0）
- `boundflow/planner/interval_v1.py`：IBP + `LinearSpec(C)`（property 融合）
- `boundflow/runtime/task_executor.py`：Python reference 执行 Task（IBP）
- `boundflow/runtime/tvm_executor.py`：TVM 执行路径（部分算子下沉 + fallback）

### 4.2 前端统一

- `boundflow/frontends/pytorch/frontend.py`：torch.export → Primal IR（含 permute dims）
- `boundflow/frontends/onnx/frontend.py`：ONNX → Primal IR（Phase 4D 子集闭环）

### 4.3 TVM kernels

- `boundflow/backends/tvm/interval_linear.py`
- `boundflow/backends/tvm/interval_conv2d.py`

---

## 5. 回归与验证（建议命令）

> 注意：Phase 4 的部分对齐/后端/ONNX 测试依赖可选包（`auto_LiRPA/tvm/onnx`）。缺失时测试应当 skip（不会在 collection 阶段崩）。

```bash
# Phase 4：Task pipeline（MLP/CNN）与 StoragePlan
python -m pytest -q \
  tests/test_phase4_task_pipeline_against_auto_lirpa.py \
  tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py \
  tests/test_phase4b_storage_plan.py \
  tests/test_phase4b3_layout_permutes.py

# Phase 4B.2：Spec(C) 对齐 auto_LiRPA（需要 auto_LiRPA）
python -m pytest -q tests/test_phase4b2_margin_c_against_auto_lirpa.py

# Phase 4C：TVM executor（需要 tvm）
python -m pytest -q \
  tests/test_phase4c_tvmexecutor_matches_python.py \
  tests/test_phase4c_tvmexecutor_matches_python_cnn.py

# 三方对齐（需要 tvm + auto_LiRPA）
python -m pytest -q tests/test_phase4c_tvmexecutor_against_auto_lirpa.py

# Phase 4D：ONNX 前端闭环（需要 onnx）
python -m pytest -q tests/test_phase4d_onnx_frontend_matches_torch.py
```

---

## 6. 已知限制（Phase 4 为 Phase 5/6 留的口）

- Planner 仍以 “单 task” 为主（TaskGraph/scheduler 属于 Phase 5）。
- TVMTaskExecutor 覆盖的算子是增量式 demo（依赖 kernel 覆盖度与 target 可用性）。
- ONNX 前端仅覆盖子集，且 reshape shape 需常量（避免引入 shape 子图与新的 IR 复杂度）。
- Spec(C) 目前聚焦 IBP 语义对齐；更强方法族（CROWN/αβ/BaB）属于 Phase 6。

---

## 7. Phase 4 对 Phase 5/6 的意义（一句话）

Phase 4 把“算法语义（IBP）”与“系统载体（Task/Planner/Executor/Backend/Frontend）”分离并钉住了对齐口径，使得：

- Phase 5 可以在不返工算子语义的前提下引入 TaskGraph/scheduler/cache/bench JSONL；
- Phase 6 可以在既有 `InputSpec/Spec(C)` 与 Task 执行接口上演进到 CROWN/αβ/BaB，而不是重新组织整个执行栈。

