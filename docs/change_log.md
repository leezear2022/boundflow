# BoundFlow 修改记录（Change Log）

约定：
- 记录按“自然批次”追加（一次明确目标的修改算一条），每条包含目的、改动点、影响面、验证方式。
- 默认在 conda 环境 `boundflow` 下验证：`conda activate boundflow`。

---

## 2025-12-17：Phase 0/1 首次止血与 IR 加固

**动机**
- 清理重复包结构，支持 `pip install -e .` 的标准安装路径。
- 将 Primal IR 升级为 Node/Value 双层结构并加入一致性校验，避免后续 planner/runtime 返工。

**主要改动**
- 工程化：新增 `pyproject.toml`，支持 `python -m pip install -e .`。
- 清理结构：删除重复/空壳目录 `boundflow/boundflow/`（避免 `boundflow.boundflow.*` 迷惑路径）。
- IR：重写 `boundflow/ir/primal.py` 为 `Node`/`Value`/`TensorType`，并提供 `BFPrimalGraph.validate()`。
- 前端壳子对齐：更新 `boundflow/frontends/pytorch/frontend.py`、`boundflow/frontends/onnx/frontend.py` 以适配新的 `BFPrimalGraph()` 构造方式；`boundflow/frontends/normalize.py` 增加 `graph.validate()`。
- 测试：新增 `tests/test_ir_primal_validate.py`；`tests/test_env.py` 增加 BoundFlow import，并打印 `CONDA_DEFAULT_ENV`（非 `boundflow` 环境提示如何运行）。
- 仓库忽略：新增 `.gitignore`（忽略 `__pycache__/`、`*.egg-info/`、`.pytest_cache/` 等）。
- 文档：新增/更新 `docs/strategy_a_refactor_plan.md`，并吸收 `docs/strategy_response_and_suggestions.md` 的观点（Node/Value、Executor、Relax、Multi-Task）。

**验证**
- 在 `boundflow` 环境：`conda run -n boundflow python tests/test_env.py`。
- IR 单测：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py`。

---

## 2025-12-17：Phase 2 TorchFrontend 最小可用（torch.export → Primal IR）

**动机**
- 让仓库从“只有 IR 草图”变成“能实际导入一个 Torch 模型并得到可校验的 Primal IR”，为后续 Bound IR/Interpreter/Planner 打基础。

**主要改动**
- Torch 前端：实现 `boundflow/frontends/pytorch/frontend.py`：
  - `export_mode="export"`：使用 `torch.export.export()` 获取 FX Graph，并转换为 Primal IR（Node/Value、shape/dtype、inputs/outputs、参数占位符映射）。
  - 将常见 `aten.*` 映射到 v0.1 primitive 名称（`linear/relu/add/...`），未知 op 保留原名便于 debug。
  - 将参数 placeholder 名映射到 `ExportedProgram.state_dict`，填充 `BFPrimalProgram.params`。
- Normalizer：`boundflow/frontends/normalize.py` 增加最小规范化（`call_method::*` → `reshape/transpose` 等），并在入口处 `validate()`。
- 测试：新增 `tests/test_torch_frontend_import.py`，验证小 MLP 的 torch.export 导入、primitive 映射、输入/参数 kind、以及图校验。

**验证**
- Torch 前端单测：`conda run -n boundflow python -m pytest -q tests/test_torch_frontend_import.py`
- 回归：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py`
- 环境 smoke：`conda run -n boundflow python tests/test_env.py`

---

## 2025-12-17：Phase 3 Interval IBP + PythonInterpreter（对齐 auto_LiRPA）

**动机**
- 提供一个“正确性优先、可调试”的 reference executor（fallback backend），并且用 auto_LiRPA 的 IBP 作为 ground truth 对齐，确保后续扩算子/接 TVM 时不漂。

**主要改动**
- Interval 域：新增 `boundflow/domains/interval.py`：
  - `IntervalState(lower, upper)`（torch.Tensor）
  - `IntervalDomain` 支持 `linear/relu/add/mul` 的 IBP 规则（v0.1 先覆盖 MLP 子集）
- Runtime：新增 `boundflow/runtime/executor.py`：
  - `LinfInputSpec(value_name, center, eps)`（L∞ 输入扰动）
  - `PythonInterpreter.run_ibp(program, input_spec)`：对 Primal IR 顺序执行，输出最终 output value 的 interval
  - `boundflow/runtime/__init__.py` 导出 `PythonInterpreter/LinfInputSpec`
- 测试：新增 `tests/test_phase3_ibp_against_auto_lirpa.py`：
  - 用一个小 MLP（Linear→ReLU→Linear）在同一输入与 eps 下对齐 `auto_LiRPA` 的 `compute_bounds(method='IBP')`

**验证**
- 对齐测试：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_against_auto_lirpa.py`
- 回归：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py tests/test_torch_frontend_import.py`

---

## 2025-12-17：Phase 3 扩展 Conv2d/Flatten（对齐 auto_LiRPA MNIST CNN 的 IBP）

**动机**
- 对齐 `auto_LiRPA/examples/vision/simple_verification.py` 的最简单 CNN 路径，使 v0.1 的 IBP reference 能覆盖 Conv2d，并能作为后续 TVM lowering/Planner 的 ground truth。

**主要改动**
- Torch 前端：`boundflow/frontends/pytorch/frontend.py`
  - 增加 op 映射：`aten.conv2d.default`→`conv2d`，`aten.flatten.using_ints`→`flatten`
  - 为 `conv2d/flatten` 提取常量 attrs（stride/padding/dilation/groups、start_dim/end_dim）
- Interval 域：`boundflow/domains/interval.py`
  - `affine_transformer` 增加 Conv2d IBP：权重正负分解后用 `torch.nn.functional.conv2d` 计算上下界
- Reference executor：`boundflow/runtime/executor.py`
  - `PythonInterpreter` 增加 `conv2d/flatten` 支持
  - `reshape` 由“占位”改为按 output meta shape 执行真实 `reshape`（用于 Flatten 后的线性层输入形状对齐）
- 测试：新增 `tests/test_phase3_ibp_cnn_against_auto_lirpa.py`
  - MNIST 风格 CNN（Conv→ReLU→Conv→ReLU→Flatten→Linear→ReLU→Linear）下，对齐 auto_LiRPA 的 `IBP` bounds

**验证**
- CNN 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_cnn_against_auto_lirpa.py`
- MLP+CNN 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_against_auto_lirpa.py tests/test_phase3_ibp_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4 Task/Planner v0（把 IBP 解释执行抽象成任务）

**动机**
- 让执行路径从“逐节点解释器”过渡到“任务（Task）”形态：为后续 fusion/batching/reuse、以及 TVM Relax/TIR lowering 建立稳定接口。
- 同时避免 Domain 通过 tensor rank 猜测算子类型：由任务/算子显式携带 op 信息（例如 linear vs conv2d）。

**主要改动**
- Task IR：更新 `boundflow/ir/task.py`
  - 新增 `TaskOp`（可执行算子表示）与 `TaskKind.INTERVAL_IBP`
  - `BFTaskModule` 引入 `entry_task_id` 与 `validate()`，支持 Multi-Task 容器（v0 仍是单任务）
- Planner v0：新增 `boundflow/planner/interval_v0.py`
  - `plan_interval_ibp_v0(program)`：把整张 Primal Graph 打包成一个 `ibp_task0`
  - 对 `linear/conv2d` 的 TaskOp 写入 `attrs["op"]`，对 `reshape` 写入 `attrs["shape"]`（来自 output meta）
- Task Runtime：新增 `boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp(BFTaskModule, LinfInputSpec)`：执行 TaskOp 序列（reference backend）
- 兼容层：重写 `boundflow/runtime/executor.py`
  - `PythonInterpreter` 保持 Phase 3 API（输入 `BFPrimalProgram`），内部改为 `plan_interval_ibp_v0` + `PythonTaskExecutor`
- Domain：`boundflow/domains/interval.py` 的 `affine_transformer` 支持 `attrs["op"]` 显式分派（并保留旧的 rank fallback）
- 测试：新增 `tests/test_phase4_task_pipeline_against_auto_lirpa.py`
  - 走 Phase 4 的 planner+task executor 路径，对齐 auto_LiRPA 的 `IBP` bounds

**验证**
- Task pipeline 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4A Task pipeline 覆盖 CNN + permute/transpose 语义补齐

**动机**
- Phase 4 的 Task pipeline 需要覆盖从 MLP 扩展到 CNN，才能作为后续优化与 lowering 的稳定 ground truth。
- `transpose` 之前缺少维度信息，属于占位实现；真实模型中常见的 `permute` 需要被正确执行。

**主要改动**
- 新增 Phase 4 CNN 对齐测试：`tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`
  - MNIST 风格 CNN 走 `plan_interval_ibp_v0 + PythonTaskExecutor`，对齐 auto_LiRPA `IBP` 输出。
- Torch 前端提取 permute dims：`boundflow/frontends/pytorch/frontend.py`
  - `aten.permute.default`（映射为 `transpose`）现在会写入 `attrs["dims"]`。
- Task executor 执行真实 transpose：`boundflow/runtime/task_executor.py`
  - `op_type == "transpose"` 时对 interval 的 lower/upper 执行 `permute(*dims)`，dims 缺失则显式报错。

**验证**
- Phase 4（MLP+CNN）：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4B.0 引入 StoragePlan（Memory/Storage 抽象接口钉住）

**动机**
- `docs/stage_4_critical_review.md` 指出：没有 Memory/Storage 抽象，Global Planner 很难名副其实；后续 aliasing/复用/显存峰值控制也会返工。
- v0.1 先把 schema 与默认填充方式钉住，保持现有对齐测试不受影响。

**主要改动**
- Task IR：`boundflow/ir/task.py`
  - 新增 `BufferSpec` 与 `StoragePlan`，并在 `BFTaskModule` 增加 `storage_plan` 字段与校验。
- Planner：`boundflow/planner/interval_v0.py`
  - `plan_interval_ibp_v0` 现在会填充默认 `StoragePlan`（一值一 buffer，`buf_<value_name>`）。
- 测试：新增 `tests/test_phase4b_storage_plan.py`
  - 验证 planner 生成的 module 含非空 `storage_plan`，且映射关系自洽。

**验证**
- StoragePlan 单测：`conda run -n boundflow python -m pytest -q tests/test_phase4b_storage_plan.py`
- Phase 4 对齐回归：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4B.2 Spec/Property（C 矩阵）对齐 auto_LiRPA

**动机**
- Phase 4 的计划要求补齐 Spec/Property，尤其是对齐 auto_LiRPA 的 `compute_bounds(C=..., method=...)` 语义。
- 注意：对 IBP 来说，`C` 不是简单“把 logits interval 再乘一次 C”就能对齐 auto_LiRPA；auto_LiRPA 会在最后线性层将 `C` 融合进权重/偏置，从而避免对 logits 各维独立化造成的额外松弛。

**主要改动**
- 新增 SpecIR：`boundflow/ir/spec.py`
  - `LinearSpec(C)`：C shape `[B,S,O]`，输出 shape `[B,S]`
- 新增 Planner v1：`boundflow/planner/interval_v1.py`
  - `plan_interval_ibp_with_linear_spec(program, spec)`：
    - 优先将 `C` 融合进最后 `linear`（`W' = C@W`, `b' = C@b`）以对齐 auto_LiRPA 的 IBP + C 行为
    - fallback：无法融合时追加 `spec_linear` op（语义正确但可能更松）
- Task executor：`boundflow/runtime/task_executor.py`
  - 支持 batched linear 权重（`w` rank-3 `[B,O,I]`）以执行融合后的 property
  - 保留 `spec_linear`（直接对 logits 做 C 线性组合）的执行支持
- 导出：`boundflow/planner/__init__.py`

**测试**
- 新增对齐测试：`tests/test_phase4b2_margin_c_against_auto_lirpa.py`
  - 同一模型/输入/eps 下，BoundFlow(task+spec) 输出 == auto_LiRPA `compute_bounds(C=C, method='IBP')`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4b2_margin_c_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4C v0 TVMExecutor（Python driver + TVM kernel demo）

**动机**
- 参考 `docs/phase4_plan.md`：在不引入复杂 Relax orchestration 的前提下，先打通 TVM lowering/执行通路，证明“同一个 Task：Python reference vs TVM backend 输出一致”。

**主要改动**
- TVM kernel（interval linear）：`boundflow/backends/tvm/interval_linear.py`
  - 基于 TE → `te.create_prim_func` → `tvm.build` 生成 `interval_linear_ibp`（输入 `x_l/x_u/w/b`，输出 `y_l/y_u`）
  - 注意：本仓库的 TVM runtime 张量类型是 `tvm.runtime.Tensor`（不是 `tvm.nd.NDArray`），因此 executor 使用 `tvm.runtime._tensor.tensor/empty` 分配与拷贝
- TVM executor：`boundflow/runtime/tvm_executor.py`
  - `TVMTaskExecutor`：Python driver 顺序执行 TaskOp，v0 仅加速 `linear`（2D weight），其它 op fallback 到 torch
- 测试：`tests/test_phase4c_tvmexecutor_matches_python.py`
  - 验证 MLP 下 `TVMTaskExecutor` 输出与 `PythonTaskExecutor` 完全一致（允许浮点误差）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py`

---

## 2025-12-17：Phase 4B/4C 小修：permute 命名与 StoragePlan 字段占位

**动机**
- 吸收 `docs/stage_4_critical_review.md` 的建议：`permute` 不应误叫 `transpose`，否则后续 layout 分析/优化容易混淆；同时 StoragePlan 需要尽早预留后端关键字段避免返工。

**主要改动**
- 前端/规范化：`aten.permute.default` 现在映射为 `permute`（并保留旧 `transpose` 的 backward-compat）
  - `boundflow/frontends/pytorch/frontend.py`
  - `boundflow/frontends/normalize.py`
- Runtime：task executors 对 `permute/transpose` 统一执行真实 `permute(*dims)`
  - `boundflow/runtime/task_executor.py`
  - `boundflow/runtime/tvm_executor.py`
- StoragePlan schema：`BufferSpec` 增加占位字段 `device/layout/strides/alignment/alias_group`
  - `boundflow/ir/task.py`
  - 默认 planner 填充 `scope`（param/const/global）与 `layout`
  - `boundflow/planner/interval_v0.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py tests/test_phase4b_storage_plan.py tests/test_phase4b2_margin_c_against_auto_lirpa.py tests/test_phase4c_tvmexecutor_matches_python.py`

---

## 2025-12-17：Phase 4C 增补：auto_LiRPA vs PythonTaskExecutor vs TVMTaskExecutor 三方对齐测试

**动机**
- `tests/test_phase4c_tvmexecutor_matches_python.py` 只验证了 TVMExecutor 对齐 Python reference，但没有把 auto_LiRPA 拉进同一条链路里做端到端 sanity check。

**主要改动**
- 新增测试：`tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`
  - 断言 `PythonTaskExecutor` 的 IBP 输出与 auto_LiRPA `compute_bounds(method="IBP")` 一致
  - 断言 `TVMTaskExecutor` 的输出与 `PythonTaskExecutor` 一致（从而形成三方闭环）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

---

## 2025-12-17：测试体验修复：避免收集 3rdparty 测试 & 修正 test_env.py 的 pytest 行为

**动机**
- `pytest` 默认会递归收集 `boundflow/3rdparty/*` 下的 upstream 测试，导致大量 collection error（这些不属于 BoundFlow 的回归范围）。
- `tests/test_env.py` 原先是脚本式写法（import 时 `sys.exit`），会导致 `pytest tests` 直接在 collection 阶段失败。

**主要改动**
- 新增 `pytest.ini`
  - 将默认 `testpaths` 限制在 `tests/`
  - `norecursedirs` 排除 `boundflow/3rdparty`
- 重写 `tests/test_env.py`
  - 提供 `test_env_smoke_imports()` 作为 pytest 测试（不再在 import 时退出）
  - 保留 `python tests/test_env.py` 的脚本用法（通过 `main()` + `__main__`）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_env.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-17：Phase 4C 增补：TVM interval conv2d kernel + CNN 对齐测试 + 运行统计

**动机**
- 之前 `TVMTaskExecutor` 仅加速 `linear(w:2D)`，无法覆盖 CNN 的主要算子（`conv2d`），也不易判断到底哪些 op 走了 TVM。

**主要改动**
- TVM kernel：`boundflow/backends/tvm/interval_conv2d.py`
  - 新增 `interval_conv2d_ibp`（NCHW）用于 IBP：输入 `x_l/x_u/w/b` 输出 `y_l/y_u`
  - v0 限制：仅支持 `groups==1`（其余走 fallback）
- TVM executor：`boundflow/runtime/tvm_executor.py`
  - `conv2d` 优先走 TVM kernel（不满足条件则 fallback 到 `IntervalDomain`）
  - 新增 `last_stats`（记录本次 run 中走 TVM 的 op、fallback 的 op、以及 kernel 编译缓存命中信息）
- 导出：`boundflow/backends/tvm/__init__.py`

**测试 / 基准**
- 新增测试：`tests/test_phase4c_tvmexecutor_matches_python_cnn.py`
  - MNIST CNN 下 `TVMTaskExecutor` 输出与 `PythonTaskExecutor` 对齐，并断言至少一次 `conv2d` 走 TVM
- 新增基准脚本：`scripts/bench_phase4c_tvmexecutor.py`
  - 对比 `PythonTaskExecutor` vs `TVMTaskExecutor` 的运行耗时（以 IBP 为目标）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python_cnn.py`

---

## 2025-12-17：Phase 4B.3：layout-only `permute` 简化 pass（合并/消除）

**动机**
- `permute` 属于 layout-only op，Phase 5 做 transpose sinking/elimination 之前，需要先把“能确定消去的情况”在 planner 层钉住，避免无意义重排在后端固化成 kernel。

**主要改动**
- 新增 planner pass：`boundflow/planner/passes/layout_only.py`
  - 连续 `permute` 做组合：`permute(p1) -> permute(p2)` 合成一个 `permute(compose(p1,p2))`
  - identity `permute` 直接消除，并通过 value alias 重写后续输入/输出
  - 统一把 `transpose` 视为 `permute`（向后兼容）
- `plan_interval_ibp_v0` 默认启用该 pass：`boundflow/planner/interval_v0.py`

**测试**
- 新增：`tests/test_phase4b3_layout_permutes.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4b3_layout_permutes.py`

---

## 2025-12-17：Phase 4D：ONNX 前端最小闭环（shape_infer + Primal IR 映射）

**动机**
- Phase 5 之前需要避免“前端分叉”：Torch-export 与 ONNX-import 必须统一到同一套 Primal IR + planner/executor，才能稳定做后续优化与对齐。

**主要改动**
- ONNX frontend：`boundflow/frontends/onnx/frontend.py`
  - 支持 `onnx.shape_inference.infer_shapes`
  - 将 ONNX Graph 映射到 Primal IR（覆盖闭环子集）：`Gemm/MatMul/Conv/Relu/Add/Mul/Flatten/Reshape/Transpose/Identity/Constant`
  - `Reshape` 的 shape 必须是常量（initializer/Constant），并被固化到 `attrs["shape"]`（避免引入 shape 计算子图）
  - initializers/Constant 进入 `program.params`，并建立 `Value` meta（shape/dtype）
- 新增对齐测试：`tests/test_phase4d_onnx_frontend_matches_torch.py`
  - MLP 与 MNIST-style CNN：`Torch import` 与 `ONNX import` 在 IBP 输出上对齐（allclose）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4d_onnx_frontend_matches_torch.py`

---

## 2025-12-17：TVM 后端更新：默认改为 Relax 算子实现（不再手写 TE/TIR）

**动机**
- TE 已逐步不再作为 TVM 推荐的“上层入口”；希望用 Relax op 表达 kernel 逻辑，由 TVM 自行 legalize/lower（内部仍会生成 TIR，但不需要我们手写）。
- 当前仓库的 TVM runtime 没有 `tvm.nd`（使用 `tvm.runtime.Tensor`），但 Relax VM 在该 fork 下是可用的，适合做“先不用手写 TIR”的阶段性实现。

**主要改动**
- 新增 Relax kernel builder：
  - `boundflow/backends/tvm/relax_interval_linear.py`
  - `boundflow/backends/tvm/relax_interval_conv2d.py`
- `TVMTaskExecutor` 默认使用 Relax VM（可通过 `TVMExecutorOptions(kernel_style=\"te\")` 退回旧 TE demo）：
  - `boundflow/runtime/tvm_executor.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py tests/test_phase4c_tvmexecutor_matches_python_cnn.py tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

---

## 2025-12-18：Phase 5A PR#1：TaskGraph + PlanBundle/PlannerPass 骨架 + 串行 scheduler

**动机**
- 进入 Phase 5 需要把 “整图单 task” 的执行模型升级为可调度的 Task DAG（为后续 cache/reuse/batching/部分 TVM 做地基）。
- 同时需要一个可扩展的 planner 输出容器（PlanBundle）与 pass pipeline 骨架，便于系统化消融与科研迭代。

**主要改动**
- TaskGraph IR：`boundflow/ir/task_graph.py`
- Planner skeleton：`boundflow/planner/core.py`（`PlannerConfig` / `PlanBundle` / `PlannerPass`）
- BFTaskModule 扩展：`boundflow/ir/task.py`
  - 增加 `task_graph` 字段与 `get_task()`
- Scheduler：`boundflow/runtime/scheduler.py`
  - 支持按 TaskGraph topo 顺序串行执行
- PythonTaskExecutor：`boundflow/runtime/task_executor.py`
  - 增加 `run_ibp_task()`（task 级执行单元，为 scheduler 提供基础能力）

**关键调整（避免 Phase5B/5E 返工）**
- TaskGraph edge 升级为 **buffer 级依赖**（携带 `src/dst value + buffer_id`，并对齐 `StoragePlan.value_to_buffer`）
- Scheduler/env 升级为 **buffer_id -> IntervalState**（TaskIO contract 明确化）

**测试**
- `tests/test_phase5a_pr1_taskgraph_and_scheduler.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5a_pr1_taskgraph_and_scheduler.py`

---

## 2025-12-18：Phase 5A PR#2：interval_v2 最小 partition（多 task DAG）+ 等价回归

**动机**
- 在不引入 cost model 的前提下，先让 planner 能输出“多 task + TaskGraph”，并验证它与 Phase 4 的单 task 行为完全等价。

**主要改动**
- 新增 v2 planner：`boundflow/planner/interval_v2.py`
  - 复用 v0 lowering（稳定的 TaskOp + StoragePlan）
  - baseline partition：layout-only（permute）单独成段，其余算子作为 compute 段；若仍不足 `min_tasks` 则按 op 数量二分
  - 生成多 `BoundTask` + `TaskGraph`（buffer 级依赖）
  - 每个 task 显式填充 TaskIO：`input_buffers` / `output_buffers`（对齐 StoragePlan）
- planner 导出：`boundflow/planner/__init__.py`
- scheduler 默认输出推断增强：`boundflow/runtime/scheduler.py`
  - 当 `output_value` 为空时，尝试根据 task_graph 推断唯一 sink task 的唯一输出；否则要求显式指定 `output_value`

**测试**
- 新增：`tests/test_phase5a_pr2_partition_multitask_equivalence.py`
  - MLP/CNN：`plan_interval_ibp_v2 + run_ibp_scheduled` 输出 == `plan_interval_ibp_v0 + PythonTaskExecutor.run_ibp`
  - 手工构造 branch+merge primal graph：确保 cross-segment use/def 在 buffer 级正确连边

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5a_pr2_partition_multitask_equivalence.py`

---

## 2025-12-18：Phase 5B PR#3：task 粒度 liveness + physical buffer reuse（v0）

**动机**
- Phase 5A 已有 multi-task DAG（TaskGraph）与 buffer 级 TaskIO contract；Phase 5B 需要把“生命周期/复用”升级为 planner 产物，为后续 cache/reuse/Relax lowering 签名做地基。

**主要改动**
- StoragePlan 支持 logical vs physical：
  - `boundflow/ir/task.py`（新增 `physical_buffers` / `logical_to_physical` / `to_physical()`）
- Liveness IR + 计算：
  - `boundflow/ir/liveness.py`（task 粒度、保守）
- Planner passes（骨架 + 可复用 helper）：
  - `boundflow/planner/passes/liveness_pass.py`
  - `boundflow/planner/passes/buffer_reuse_pass.py`
- Runtime env 改为 physical buffer id：
  - `boundflow/runtime/scheduler.py`
  - `boundflow/runtime/task_executor.py`
- interval_v2 可选开启复用（默认关闭）：
  - `boundflow/planner/interval_v2.py`（`enable_storage_reuse`）

**测试**
- `tests/test_phase5b_pr3_buffer_reuse.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5b_pr3_buffer_reuse.py`

---

## 2025-12-18：Phase 5B PR#3.1：Correctness hardening（edge-driven last_use + physical env 断言）

**动机**
- 强化“env 只认 physical buffer id”与“跨 task last_use 以 TaskGraph 为准”的不变量，避免后续引入更激进复用/cache/Relax lowering 时出现隐式别名错误。

**主要改动**
- `boundflow/ir/liveness.py`：跨 task last_use 由 `TaskGraph.edges` 驱动更新
- `boundflow/runtime/scheduler.py`、`boundflow/runtime/task_executor.py`：当存在 `physical_buffers` 时强校验 env key 必须是 physical id
- `boundflow/ir/task.py`：`TaskOp.memory_effect` 占位字段（未来 alias/memory-effect 模型用）
- `boundflow/planner/passes/buffer_reuse_pass.py`：预留 `ReusePolicyFn` hook（默认 LIFO 不变）

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5B PR#4A：PlannerConfig 复用配置 + ReuseStats 可观测性

**动机**
- 在进入 5B.2（放宽 key / policy 消融 / bench）前，先把“复用参数与统计”挂到 PlannerConfig/PlanBundle，保证实验可复现并能解释 miss 原因。

**主要改动**
- `boundflow/planner/storage_reuse.py`：`StorageReuseOptions`、`ReuseKeyMode/ReusePolicy`、`BufferReuseStats`、`estimate_bytes_saved()`
- `boundflow/planner/core.py`：`PlannerConfig.storage_reuse`
- `boundflow/planner/passes/buffer_reuse_pass.py`：输出 `reuse_stats` 到 `PlanBundle.meta`
- `boundflow/planner/interval_v2.py`：透传 `reuse_key_mode/reuse_policy`（默认 STRICT/LIFO，不改变默认行为）
- `scripts/bench_storage_reuse.py`：bench/统计脚本（不放进 pytest 阈值）

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5B PR#4B：memory_effect Enum + bench 输出 + 更细 miss reasons

**动机**
- 让 `memory_effect` 类型更稳（避免字符串拼写导致的隐式分支爆炸），并把复用统计输出落地到 bench（CSV/JSON），方便后续 5B.2/5F 做消融与画图。

**主要改动**
- `boundflow/ir/task.py`：新增 `MemoryEffect` enum，`TaskOp.memory_effect` 改为 `Optional[MemoryEffect]`
- `boundflow/planner/storage_reuse.py`：新增 `StorageReuseOptions.respect_memory_effect`（占位）与 `ReuseMissReason.KEY_MISMATCH`
- `boundflow/planner/passes/buffer_reuse_pass.py`：miss reason 更细分（NO_FREE/KEY_MISMATCH/LIFETIME_OVERLAP），并记录 overlap 阻塞者 task topK 与 pool 碎片度统计
- `scripts/bench_storage_reuse.py`：支持 `--format text|json|csv` 与 `--out`，并输出 `git_commit`、版本/env vars 白名单、DAG stats 与 why-not-reused/key 分布 topK

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：工程环境：默认禁用 TVM-FFI 可选 torch-c-dlpack JIT（避免 tvm import/pytest 卡住）

**动机**
- `tvm` import 会触发 `tvm-ffi` 的可选 torch-c-dlpack 扩展 JIT 编译；在部分环境下会显著拖慢甚至卡住 import，导致 pytest 超时。该扩展对当前阶段不是必需，因此默认禁用并把 cache/tmp 放入 repo。

**主要改动**
- `env.sh`
  - 默认 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`（可通过设为 0 覆盖）
  - 默认 `TVM_FFI_CACHE_DIR=$BOUNDFLOW_ROOT/.cache/tvm-ffi`
  - 默认 `TMPDIR=$BOUNDFLOW_ROOT/.tmp`

**验证**
- `conda activate boundflow && source env.sh && python -c "import tvm; print('tvm_ok')"`
- `pytest -q`
