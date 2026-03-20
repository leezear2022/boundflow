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

---

## 2025-12-18：Phase 5C PR#5：Planner Pipeline 统一入口 + config_dump 可复现消融

**动机**
- Phase 5 的 planner 消融需要“统一入口 + 可序列化配置快照”，否则实验不可比、不可复现。

**主要改动**
- `boundflow/planner/options.py`：结构化选项（partition/lifetime/layout/debug，占位但稳定）
- `boundflow/planner/pipeline.py`：`plan()` 统一入口，输出 `PlanBundle.meta["config_dump"]` 与 `planner_steps`
- `scripts/bench_planner_pipeline.py`：最小 pipeline bench（输出 config_dump）
- `tests/test_phase5c_pr5_pipeline_config_dump.py`：不同 config 下 task 数变化但输出等价

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr5_pipeline_config_dump.py`

---

## 2025-12-18：Phase 5C PR#6：Invariant Verifiers + Pipeline Instrument（pass contract）

**动机**
- PR#5 已统一 planner 入口与 config_dump；PR#6 进一步钉住“每一步产物是否仍合法”的 pass contract，避免后续 Relax/cache/CROWN 扩展时出现 silent wrong。

**主要改动**
- `boundflow/planner/verify.py`：TaskGraph/StoragePlan/Liveness+Reuse 三类核心不变式 verifier
- `boundflow/planner/instrument.py`：timing + verify instrument（before/after step hooks）
- `boundflow/planner/pipeline.py`：`validate_after_each_pass=True` 时每步后运行 verifier 并写入 `PlanBundle.meta["verify"]`
- `tests/test_phase5c_pr6_validators.py`：负例覆盖 broken edge/mapping/overlap + pipeline verify 记录

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr6_validators.py`

---

## 2025-12-18：Phase 5C PR#7：Determinism + DumpPlanInstrument + 结构化 VerifyError

**动机**
- 进一步钉住 planner 的可复现性（determinism）与可观测性：避免 topo 顺序/统计在不同运行漂移，并提供 step 级 JSON snapshot 便于定位 verifier 报错与 silent wrong。

**主要改动**
- `boundflow/ir/task_graph.py`：`topo_sort()` 改为 heapq 驱动的确定性顺序（按 task_id 字典序）
- `boundflow/planner/passes/buffer_reuse_pass.py`：overlap blocker 选择增加稳定 tie-break（避免 set 迭代顺序影响）
- `boundflow/planner/verify.py`：`VerifyError` 增加 `where`，关键错误填充定位信息
- `boundflow/planner/instrument.py`：`PlannerInstrument.should_run()` 预留 + `DumpPlanInstrument`（step 后 dump JSON）+ verify 输出包含 `where`
- `boundflow/planner/options.py`：`PlannerDebugOptions` 增加 `dump_plan/dump_plan_dir/dump_plan_run_id`
- `boundflow/planner/pipeline.py`：debug 开启后自动启用 DumpPlanInstrument；hook 调用统一走 should_run
- `tests/test_phase5c_pr7_determinism_and_dump.py`：新增 determinism 与 dump 回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr7_determinism_and_dump.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#8：Task → Relax IRModule lowering skeleton（interval linear）

**动机**
- 在 5C 的 pipeline contract/determinism/dump 就位后，开始补齐编译后端链路：把 interval-IBP task lower 成 Relax IRModule，为 PR#9（TVMExecutor compile+execute）做准备。

**主要改动**
- `boundflow/backends/tvm/relax_task_lowering.py`：新增 task-level lowering（`RELAX_OPS` 与 `CALL_TIR` 两种模式），v0 仅支持 single-op `linear`
- `boundflow/backends/tvm/interval_linear.py`：新增 `build_interval_linear_primfunc()`（给 `relax.call_tir` 使用）
- `tests/test_phase5d_pr8_relax_lowering_skeleton.py`：IRModule 可构建 + `relax.build(..., target="llvm")` 可编译回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr8_relax_lowering_skeleton.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#9：TVMTaskExecutor：compile cache + run_ibp_task（scheduler 对齐）

**动机**
- PR#8 已证明 Relax IRModule 可构建/可编译；PR#9 将其接到 runtime，通过 scheduler 的 physical env contract 跑通执行闭环，并与 Python reference allclose 对齐。

**主要改动**
- `boundflow/backends/tvm/relax_task_lowering.py`：新增 key 驱动的 `build_interval_linear_relax_ir_module()`（并修复 CALL_TIR 下 global_symbol 重复问题）
- `boundflow/runtime/tvm_executor.py`：实现 `run_ibp_task()`（physical env），并为 interval linear 引入编译缓存（支持 `kernel_style=relax|call_tir`）
- `tests/test_phase5d_pr9_tvm_executor_linear_equiv.py`：scheduler 下 Python vs TVM allclose 回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr9_tvm_executor_linear_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#10：TVM 编译侧可观测性（PassTimingInstrument + DumpIR）

**动机**
- planner 侧已经有 `timings_ms/config_dump/verify`；为了系统消融需要把 TVM compile 侧的 per-pass timing 与 IR dump 也纳入可观测数据，拆清楚 compile vs run 开销。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 compile-side 选项（pass timing / dump ir / cache tag），并在 `relax.build` 外包 `tvm.transform.PassContext(instruments=[...])`；将 compile stats 暴露为 json-able 数据
- `tests/test_phase5d_pr10_tvm_compile_instruments.py`：回归测试（timing 与 dump 落盘）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr10_tvm_compile_instruments.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11A：compute task 全 TVM（RELAX_OPS：linear+relu(+add/mul)）

**动机**
- 减少 task 内 per-op 的 Python↔TVM 往返与解释器开销；为后续 PR#11B（fusion 使 call_tir 数量下降）提供 reference 路径。

**主要改动**
- `boundflow/backends/tvm/relax_interval_task_ops.py`：新增 task-level RELAX_OPS lowering（lane 拆分契约，输出扁平 tuple）
- `boundflow/runtime/tvm_executor.py`：新增 `enable_task_relax_ops` 开关（默认 False）；开启后尝试整 task 编译/执行，失败则回退到 per-op 执行
- `tests/test_phase5d_pr11a_task_relax_ops_equiv.py`：scheduler 下 TVMTaskExecutor(RELAX_OPS) vs Python allclose

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11B：可控 fusion pipeline + call_tir 数量统计

**动机**
- 将 “RELAX_OPS → legalize/fuse” 变成可控编译开关，并把 `call_tir` 数量等 IR 结构统计落到 compile_stats，支撑后续论文级消融（调用次数下降 vs runtime）。

**主要改动**
- `boundflow/backends/tvm/relax_analysis.py`：新增 `call_tir` 计数与 IR stats
- `boundflow/runtime/tvm_executor.py`：task-level compile 支持 fusion pipeline（LegalizeOps/Annotate/FuseOps/FuseTIR），并在 `compile_stats["ir_stats"]` 记录各阶段统计
- `tests/test_phase5d_pr11a_task_relax_ops_equiv.py`：开启 fusion pipeline 回归并检查 `call_tir` 单调性（best-effort）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11C：降低 Relax VM 调用开销（VM/PackedFunc 缓存 + VM-level passes 插槽）

**动机**
- 在 task-level RELAX_OPS + fusion pipeline 之后，进一步减少 VM/dispatch 开销，并预留可插拔的 VM-level pass 插槽，方便后续做 tuple 展开/删无用参数/inline 等消融而不改架构。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 VM/PackedFunc 缓存（按 `(cache_key_hash, dev.type, dev.index)`），并加入 `task_vm_opt_passes` 插槽
- `tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py`：开启缓存与 pass 插槽后仍与 Python reference allclose

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#11C.1：save_function bench + task-level pipeline 修复

**动机**
- 增加 save_function closure micro-bench（对比 VM 调用方式），并修复 task-level 自定义 relax_pipeline：必须组合 TVM 官方 `default_build_pipeline()`，否则可能触发 VM codegen 对 `alloc_tensor` 的不支持错误。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：task-level compile pipeline 改为 `pre-pass + default_build_pipeline()`；fusion 统计链补上 `ConvertToDataflow`
- `boundflow/backends/tvm/relax_interval_task_ops.py`：从 StoragePlan.scope 推断 param/const，避免把 param 当作 interval 输入
- `scripts/bench_relax_vm_overhead.py`：save_function micro-bench（JSON 输出）
- `tests/test_phase5d_pr11c1_save_function_closure.py`：save_function 输出一致性回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c1_save_function_closure.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：补充大模型协作工作流摘要

**动机**
- 将“输入计划 → 修正测试 → 总结 → 下一步计划”的交流流程固定成简明摘要，方便复用与对齐。

**主要改动**
- 更新 `gemini_doc/llm_collaboration_workflow.md`，新增“快速版：对话工作流摘要”与 6 步流程。
- 更新 `AGENTS.md` 的关键文档索引，标注工作流摘要入口。

**验证**
- 无（文档更新）

---

## 2025-12-20：Phase 5D PR#12：StaticPlanBlockMemory baseline × BoundFlow reuse（开关 + memory stats + 四象限 bench）

**动机**
- 将 TVM Relax 的 `StaticPlanBlockMemory` 作为 “intra-function 静态内存规划” baseline，与 BoundFlow 的 “inter-task logical→physical reuse” 放到同一张四象限表里，支撑后续系统消融与论文叙事。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 `MemoryPlanMode` 与 `TVMExecutorOptions.memory_plan_mode`，并在 task-level pipeline 中可选择跳过 `StaticPlanBlockMemory`；`compile_stats` 增加 `memory_plan_mode/memory_stats`。
- `boundflow/backends/tvm/relax_analysis.py`：新增 `collect_relax_memory_stats`，统计 `relax.vm.alloc_storage/alloc_tensor` 以及 `alloc_storage_total_bytes/max_bytes`（IR 侧估算）。
- `tests/test_phase5d_pr12_static_plan_modes.py`：DEFAULT vs DISABLE_STATIC_PLAN 两种模式下仍与 Python reference allclose。
- `scripts/bench_static_plan_baseline.py`：四象限 bench（reuse ON/OFF × static plan ON/OFF），输出 JSON 字段用于表格/画图。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_static_plan_modes.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#12.1：memory estimator 对照 + DEFAULT pipeline 边界 + tir_var_upper_bound 占位

**动机**
- 为 PR#12 的 memory planning baseline 增加 TVM 官方 `estimate_memory_usage` 口径佐证；同时让 `MemoryPlanMode.DEFAULT` 更贴近 TVM 官方默认 pipeline，并预留 dynamic shape 上界变量（避免后续引入导致历史数据不可比）。

**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `MemoryPlanMode.DEFAULT/FORCE_STATIC_PLAN` 使用 `tvm.relax.pipeline.default_build_pipeline()`；`DISABLE_STATIC_PLAN` 仍用“等价默认但移除 StaticPlanBlockMemory”的自定义 pass 列表
  - `compile_stats["memory_stats"]` 结构调整为 `{by_scan, by_tvm_estimator}`
  - 预留 `TVMExecutorOptions.tir_var_upper_bound` 并纳入 cache key 与 compile_stats
- `tests/test_phase5d_pr12_static_plan_modes.py`：适配 `memory_stats` 新结构
- `scripts/bench_static_plan_baseline.py`：输出字段增加 `tir_var_upper_bound`，汇总字段使用 `compile_ms_total`

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#12.2：estimator_stage 固定 + 写入 tir_var_upper_bound attrs + dynamic 回归用例

**动机**
- 固定 `estimate_memory_usage` 的调用阶段并记录，避免未来 pipeline 调整导致数据漂移；同时让 `tir_var_upper_bound` 从“仅记录 options”升级为“真实写入 Relax function attrs 并可观测到效果”。
**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `compile_stats["memory_stats"]` 增加 `by_tvm_estimator_stage`（固定为 `pre_static_plan`）
  - 当 `TVMExecutorOptions.tir_var_upper_bound` 非空时，把它写入 task-level `main` Relax function 的 `tir_var_upper_bound` attrs（best-effort）
- `tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
  - 构造带动态维度 `n` 的 Relax module，使用 TVM `default_build_pipeline()` lowering 后对比 `collect_relax_memory_stats`：有 upper bound 时可折算出常量 bytes 且 nonconst bytes 下降

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#13A：Ablation matrix bench（统一 JSONL schema + 最小矩阵）

**动机**
- 在 Phase 5D 进入论文/系统化消融阶段前，先钉死统一的实验矩阵与输出 schema（JSONL/一行一条 run），避免后续每次加变量都返工 bench 字段。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 提供 `partition/reuse/static_plan/fusion` 的 2×2×2×2 默认矩阵输出（JSONL）
  - 可选 `--matrix small` 跑单点，用于 CI/快速排查
  - 支持可选 auto_LiRPA baseline timing（默认开启，可用 `--no-auto-lirpa` 关闭）
- `tests/test_phase5d_pr13_ablation_matrix_smoke.py`
  - smoke 测试：跑 `--matrix small` 并断言输出 JSONL schema

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#13B：bench 计时公平性/可解释性增强 + env.sh stdout 清洁

**动机**
- 系统化消融阶段要求：stdout 可机器解析（JSONL/CSV 不被环境提示污染）、compile vs run 明确拆分、baseline 的 setup/compute 拆分、并记录差异幅度便于 debug。

**主要改动**
- `env.sh`
  - 提示信息默认写入 stderr（不污染 stdout），并支持 `BOUNDFLOW_QUIET=1` 静默。
- `boundflow/runtime/tvm_executor.py`
  - 增加 task-level compile cache 统计：`get_task_compile_cache_stats()` 返回 hit/miss/fail（用于 bench 公平性解释）。
- `scripts/bench_ablation_matrix.py`
  - 增加 `compile_first_run_ms`（首次运行/含编译触发）并保留 steady-state `run_ms_*`
  - 输出 compile cache stats；auto_LiRPA baseline 增加 `setup_ms`；correctness 增加 max abs diff 指标
- `tests/test_env_sh_quiet_stdout.py`
  - 回归：`env.sh` 默认不写 stdout，且可用 `BOUNDFLOW_QUIET=1` 静默

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_env_sh_quiet_stdout.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --output /tmp/boundflow_ablation.jsonl`

---

## 2025-12-20：Phase 5D PR#13C：JSONL schema 文档 + schema_version/time_utc + cache delta + rel diff

**动机**
- 系统化消融进入“多人协作 + 论文画图”阶段后，需要固定 bench 的 JSONL schema，并补齐去歧义字段（schema 版本、UTC 时间、cache 增量、相对误差）。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 顶层增加 `schema_version`；`meta` 增加 `time_utc`
  - 增加 `compile_cache_stats_delta_compile_first_run`（首次运行/编译触发阶段的 cache 增量）
  - correctness 增加 `*_max_rel_diff_*`（相对误差）
- `docs/bench_jsonl_schema.md`
  - 新增 JSONL 字段/口径说明，固定输出协议（stdout 纯 payload）
- `AGENTS.md`
  - 在关键文档索引中注册 `docs/bench_jsonl_schema.md`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --output /tmp/boundflow_ablation.jsonl`

---

## 2025-12-20：Phase 5D PR#13D：JSONL schema contract test

**动机**
- 将 JSONL schema 从“约定”升级为 CI 级契约：逐行可解析、关键字段存在且类型/范围合理，防止后续字段漂移导致画图/后处理阶段才发现输出断裂。

**主要改动**
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 运行 `bench_ablation_matrix --matrix small` 并逐行解析 JSONL，校验 schema_version/time_utc/runtime/correctness/compile cache delta 等关键字段与类型。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`

---

## 2025-12-20：Phase 5D PR#13E：postprocess 产线（JSONL → CSV/表格/图）

**动机**
- 将 bench 产物从 JSONL 进一步变成“可直接画图/做表”的数据与产线脚本，面向论文/AE 的复现与后处理。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - JSONL 扁平化导出 `out/phase5d/ablation.csv`
  - 最小汇总表 `out/phase5d/tables/ablation_summary.csv`
  -（可选）示例图 `out/phase5d/figures/cache_miss_vs_compile_first_run.png`（若 matplotlib 可用）
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 合成最小 JSONL 样例，验证 postprocess 输出 CSV/表格落盘
- `docs/bench_jsonl_schema.md`
  - 增加后处理脚本说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`

---

## 2025-12-20：Phase 5D PR#13E.1：postprocess hardening（缺失值/分组/流式读取/enum 修复）

**动机**
- 修复后处理脚本的 4 类静默口径错误：missing correctness 不应当 0、group key 需包含 eps/input_shape/domain/spec、防大 JSONL 内存峰值、修正 enum repr 解析。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - JSONL 流式读取（按行迭代）
  - 修正 enum 解析正则（`:\s*'value'`）
  - group key 纳入 `input_shape/eps/domain/spec` 防止混组
  - summary 不再把缺失 correctness 当 0，并输出 `python_vs_tvm_rel_diff_missing`
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 新增缺失 correctness 与 group key 分组回归
- `tests/test_postprocess_enum_normalization.py`
  - 覆盖 enum repr value 解析
- `docs/bench_jsonl_schema.md`
  - 补充缺失值约定说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`
- `conda run -n boundflow python -m pytest -q tests/test_postprocess_enum_normalization.py`

---

## 2025-12-20：Phase 5D PR#13F：bench 支持 eps/batch 覆盖（用于分组验证）

**动机**
- 为了用真实 bench 输出验证 postprocess 的 group key 不混组（eps/input_shape 变化），并为后续消融扩展预留入口，给 `bench_ablation_matrix.py` 增加最小旋钮 `--eps/--batch`。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 新增 `--eps`（覆盖 Linf eps）与 `--batch`（覆盖输入 batch size；当前仅支持 `workload=mlp`）
- `docs/bench_jsonl_schema.md`
  - 增加“Workload 参数化（用于分组验证）”说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`

---

## 2025-12-20：Phase 5D PR#13E.2：MANIFEST 换行修复 + --no-check 结构稳定

**动机**
- 修复 MANIFEST.txt 的可读性（避免字面量 `\\n`），并让 `--no-check` 时 JSONL 的 correctness diff 字段结构稳定（输出为 null），减少下游处理分支与口径歧义。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - MANIFEST 使用真实换行符写入
- `scripts/bench_ablation_matrix.py`
  - `--no-check` 下 diff keys 仍存在（值为 null）
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 回归：MANIFEST 不含字面量 `\\n`
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 回归：`--no-check` 仍输出 diff keys（为 null）
- `docs/bench_jsonl_schema.md`
  - 补充 `--no-check` 的字段口径说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`

---

## 2025-12-22：忽略 artifacts/out 运行产物目录

**动机**
- `artifacts/` 与 `out/` 是运行产物目录（JSONL/CSV/图/manifest），体积与内容随实验增长/变化，不适合纳入 git 版本控制；复现应由 runner/bench 重新生成。

**主要改动**
- `.gitignore`
  - 新增忽略：`artifacts/`、`out/`

**验证**
- `git status --porcelain`

---

## 2025-12-22：PR#15A/15B：baseline 外提预计算 + schema_version 冻结为 1.0

**动机**
- auto_LiRPA baseline 不依赖矩阵旋钮；将 baseline 计算外提到矩阵循环外，减少重复开销并避免点内触发带来的边界条件。
- Phase 5D 的 JSONL 字段与计时/分组口径已稳定，冻结为 `schema_version=1.0`，降低后续 Phase 6 扩展时“口径被冲掉”的风险。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - baseline 在进入矩阵循环前预计算，并在每行 JSONL 直接附加（点内不再触发 compute_bounds）。
  - `schema_version` 从 `0.1` 升级为 `1.0`。
- `docs/bench_jsonl_schema.md`
  - 更新当前版本为 `1.0`，并说明 1.0 为 Phase 5D 冻结口径。
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 适配 `schema_version=1.0`。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`

---

## 2025-12-22：PR#15C：TVM task-level compile cache 落盘（跨进程复用）

**动机**
- AE/大矩阵多次运行时，进程内编译缓存无法跨进程复用，可能导致重复编译耗时与算力浪费。
- 增加可选 `--tvm-cache-dir`：将 task-level RELAX_OPS 的编译产物落盘，并在下次运行直接加载，缩短 cold-start。

**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions` 新增：`compile_cache_dir`、`compile_cache_refresh`
  - task-level 编译：支持从磁盘 cache（`task_<hash>.so` + `task_<hash>.spec.json`）加载（best-effort）。
- `scripts/bench_ablation_matrix.py`
  - 新增 CLI：`--tvm-cache-dir`、`--tvm-cache-refresh`
  - TVM executor options 透传上述参数，并默认将 `compile_cache_tag` 设为当前 git commit（降低跨版本误命中风险）。
- `docs/bench_jsonl_schema.md`
  - 补充 `compile_cache_dir` 字段说明
- `tests/test_phase5d_pr15c_tvm_disk_cache.py`
  - 回归：同一 cache_dir 下二次运行应避免 compile miss（delta miss=0）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr15c_tvm_disk_cache.py`

---

## 2025-12-22：Phase 5 完成声明 + 全流程文档更新

**动机**
- Phase 5 已完成工程收口（schema/产线/基线/可观测性），需要一份面向论文/AE 的“完成声明”与固定复现入口。
- 同步修订全流程总览文档，避免与最新实现（artifact runner、workload 支持、schema_version=1.0、可选 tvm 落盘 cache）不一致。

**主要改动**
- 新增：`docs/phase5_done.md`
  - Phase 5 完成声明（覆盖范围、复现入口、DoD、已知限制、Phase 6 边界）。
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 修正 Non-goals/TODO，反映 runner/workload/口径冻结的最新状态。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增 Quick Restart（IBP 边界快速复跑）

**动机**
- 目前 BoundFlow 已支持 interval IBP 的端到端路径（reference + 可选 TVM + 可选 auto_LiRPA 对照），需要一个“重新上手/快速复跑”的最短指南，降低新同学/AE 复现成本。

**主要改动**
- 新增：`gemini_doc/quick_restart_ibp.md`
  - 环境启动、自检、bench 一条命令出 JSONL、artifact runner、最小 Python API 示例。
- 更新：`gemini_doc/README.md`
  - 增加 Quick Restart 文档索引入口。

**验证**
- 文档变更（以脚本 `--help` 口径校验参数存在）。

---

## 2025-12-22：重写 Phase 5「实验产线与系统化消融」总结

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 中 Phase 5 小节原先偏“组件罗列”，对论文/系统结构视角的“为什么这样组织实验产线、如何支持系统化消融”表达不足。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 Phase 5 从“5D/5E 列表”重写为“产线叙事”：Phase 4 knob → 实验矩阵；bench→JSONL（schema 冻结）→postprocess→artifact runner；并强调 compile/run 口径分解、baseline 证据链与 contract tests。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：重写 Phase 0~6 路线图与验收标准（风格统一）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 的 Phase 0~6 小节内部风格不统一（尤其 Phase 5 的叙事风格与其它 Phase 不一致），且“系统结构解读”小节缺少可引用的序号。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 为“系统结构解读（学术视角：分层、可验证、可复现实验）”补充 `3.1` 序号。
  - 重写 Phase 0~6：统一为“目标/实现状态/关键实现/验收标准/主要产物”的模板，并对缺失证据部分以 TODO 明示。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：gemini_doc 总导引 + AGENTS.md 索引注册

**动机**
- `gemini_doc/` 内文档数量较多，需要一个目录级导引，便于交接与快速定位关键文档入口。
- 在 `AGENTS.md` 注册导引，方便后续大模型/新贡献者按索引阅读与遵循维护规则。

**主要改动**
- 新增：`gemini_doc/README.md`
  - gemini_doc 目录导引（阅读路径、关键文档、维护规则）
- 更新：`AGENTS.md`
  - 关键文档索引新增 `gemini_doc/README.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：继续优化 full pipeline 文档元信息（v2）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 顶部元信息需要更便于引用与追溯（schema 版本、代码版本），且系统架构图已包含 Normalize 节点但“对应实现入口”列表缺少该索引。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 元信息改为列表，并补充 `docs/bench_jsonl_schema.md` 链接与 git short SHA。
  - “对应实现入口”补充 `normalize_primal_graph` 的实现入口（`boundflow/frontends/normalize.py`）。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补齐论文辩护（为何不端到端用 auto_LiRPA / 为何不直接用 TVM）

**动机**
- 论文里需要把“为什么不用 auto_LiRPA”和“为什么不直接只用 TVM”讲成 reviewer-proof 的系统分层论证，并能指向仓库证据。

**主要改动**
- 更新：`gemini_doc/why_boundflow_not_auto_lirpa_or_tvm.md`
  - 将 TVM 的定位改为“张量编译/codegen”，强调 BoundFlow 的缺失中间层贡献（verification-aware IR/Planner/Scheduler + 复现/消融/对齐契约）。
  - 新增“仓库证据索引”，便于论文/AE 引用实现与测试。
- 更新：`gemini_doc/README.md`
  - 将辩护文档加入关键索引。
- 新增：`gemini_doc/change_2025-12-22_paper_defense_why_boundflow_not_auto_lirpa_or_tvm.md`
  - 记录本次论文辩护表述的调整与证据索引补齐。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增多范数输入扰动（L∞/L2/L1/L0）设计文档

**动机**
- 需要把不同扰动集合下的线性算子上/下界公式讲清楚，并给出 BoundFlow 的可扩展落地设计（不破坏现有 interval IBP 管线）。

**主要改动**
- 新增：`gemini_doc/perturbation_support_design.md`
  - `PerturbationSet` + support function 的统一设计与公式汇总（Lp 对偶范数、L0 top-k）。
  - 给出与现有 pipeline 的最小侵入式对接点与迭代路线（conv2d 可先降级再 tighten）。
- 新增：`gemini_doc/change_2025-12-22_perturbation_support_design.md`
  - 记录本次设计文档新增的动机与内容摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增 IBP/CROWN/αβ-CROWN/BaB 的统一设计文档

**动机**
- Phase 6 将引入更强的 LiRPA 方法族与可选 BaB；需要一个三轴解耦的系统设计，避免接口与实现路径爆炸，并明确“新增扰动/新增方法”的工作量边界。

**主要改动**
- 新增：`gemini_doc/bound_methods_and_solvers_design.md`
  - 三轴解耦：`PerturbationSet × BoundMethod/DomainState × Solver(BaB)`。
  - 方法族以可组合 stages 表达（forward/relax/backward/optimize），并给出 cache/batching 统一策略与工程原则（控制流留在 Python，TVM 专注张量核）。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_and_solvers_design.md`
  - 记录本次设计文档新增的动机与内容摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补充 bound_methods 设计中的 `concretize` 实现模式

**动机**
- `concretize` 的职责边界容易混淆；需要明确 Interval/Linear 两类状态在输入处数值化的统一模式，避免把 concretize 散落进各个 Domain 子类导致重复与组合爆炸。

**主要改动**
- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 在 §6 增补 `concretize` 实现模式（Interval 直接返回；Linear 在输入处调用 `PerturbationSet.concretize(A, x0)`）。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_concretize_section.md`
  - 记录本次补充的动机与摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补强 bound_methods 设计的“落地避坑清单”

**动机**
- 把“方向正确但落地会撞墙”的关键风险点固化成接口约束/DoD：`A` 不强制显式化、task contract、αβ warm-start 与 cache key、subproblem 约束结构、CachePlan 粒度、逐阶段对齐测试等。

**主要改动**
- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 新增 `§7 落地避坑清单（建议作为 Phase 6 的接口约束/DoD）`。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_landing_pitfalls_checklist.md`
  - 记录本次补强的动机与摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：Phase 6A 起步——引入 InputSpec + LpBallPerturbation（L∞/L2）与线性 concretize

**动机**
- 将“输入扰动集合”从 Phase 5 的 `LinfInputSpec` 固化形式中解耦出来，为 Phase 6 的 CROWN/αβ-CROWN/BaB 打基础，并先实现最小可行的 `L2` 线性层 concretize。

**主要改动**
- 新增：`boundflow/runtime/perturbation.py`
  - `PerturbationSet`、`LpBallPerturbation`、`InputPerturbationState`。
- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec`，`PythonTaskExecutor.run_ibp` 支持 `L2` 输入在线性层用对偶范数公式 concretize。
- 更新：`boundflow/runtime/scheduler.py`、`boundflow/runtime/tvm_executor.py`、`boundflow/runtime/executor.py`
  - 接口允许透传 `InputSpec`，但 scheduler/TVM 当前仍仅支持 `L∞`（非 L∞ 明确报错）。
- 新增：`tests/test_phase6a_inputspec_lpball_linear.py`

**验证**
- `python -m pytest -q tests/test_phase6a_inputspec_lpball_linear.py`

---

## 2025-12-23：Phase 6B 起步——最小 CROWN-IBP（MLP: Linear+ReLU）

**动机**
- 在 Phase 6A 的扰动解耦基础上，跑通 “IBP forward + CROWN backward” 的最小闭环（先覆盖 MLP 的 `Linear+ReLU`），为后续 multi-spec batching 与 α/β/BaB 打基础。

**主要改动**
- 更新：`boundflow/runtime/perturbation.py`
  - 新增 `concretize_affine(center, A, b)`（显式张量 `A` 的最小实现）。
- 新增：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)` 最小 CROWN-IBP 执行器（single-task，`linear/relu` 子集）。
- 新增：`tests/test_phase6b_crown_ibp_mlp.py`
  - `L∞`/`L2` 采样 soundness，以及 `L∞` 下 upper bound 不劣于 IBP。
- 新增：`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_minimal.md`
  - 记录本次 Phase 6B 起步的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`

---

## 2025-12-23：Phase 6B 补强——L1 测试 + brute-force + multi-spec 入口

**动机**
- 用测试把 CROWN-IBP 的关键语义（对偶范数、ReLU 符号选择）钉牢，并为下一步 multi-spec batching 提供最小入口（`linear_spec_C`）。

**主要改动**
- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec.l1(...)`。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., linear_spec_C=...)` 多目标入口（`C: [B,S,O]`）。
- 更新：`tests/test_phase6b_crown_ibp_mlp.py`
  - 新增 `L1` 采样 soundness 与 `L∞` brute-force 网格测试。
- 新增：`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_hardening.md`
  - 记录本次补强的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`

---

## 2025-12-23：Phase 6C（CROWN-IBP MLP）multi-spec 真 batch——吞吐 microbench + forward 复用回归

**动机**
- Phase 6B 已将 CROWN-IBP（MLP: Linear+ReLU）的 correctness 风险钉死；Phase 6C 需要开始验证 “multi-spec 真 batch” 的系统收益，并用回归测试保证 forward IBP 不随 spec 维度重复计算。

**主要改动**
- 新增：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 比较 `run_crown_ibp_mlp(..., C:[B,S,O])`（batch）与循环 `C[:,s:s+1,:]`（serial）的 p50 耗时，并输出 JSON payload。
- 新增：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - forward 复用回归：统计 `IntervalDomain` 的 forward transformer 调用次数，断言 `S=1` 与 `S=32` 时次数相同。
- 新增：`gemini_doc/change_2025-12-23_phase6c_multispec_true_batch_microbench_and_reuse_test.md`
  - 记录本次 Phase 6C 的动机、改动与验证口径。

**验证**
- `python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py`
- `python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --specs-list 1,4,16,64`

---

## 2025-12-23：Phase 6C microbench 稳态增强——元信息/计时后端/串行口径说明

**动机**
- 为避免吞吐 microbench 的复现实验常见质疑点，进一步把计时与输出口径工程化钉死（warmup/iters、可选计时后端、串行 baseline 口径透明、CUDA 同步等）。

**主要改动**
- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 增加 `meta` 输出（torch 版本、计时参数、串行口径说明等），并支持 `--timer torch_benchmark`。
- 更新：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - 统一用 `torch.inference_mode()` 包裹，减少上下文差异引入的波动。
- 新增：`gemini_doc/change_2025-12-23_phase6c_microbench_stability_metadata_and_timer.md`
  - 记录本次 microbench 稳态增强的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py`
- `python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --timer torch_benchmark --specs-list 1,4,16,64`

---

## 2025-12-23：Phase 6D（α-CROWN MLP）起步——ReLU 下界 α 参数 + K-step 优化 + warm-start

**动机**
- 在 CROWN-IBP 的 correctness 与 multi-spec 系统收益稳定后，引入可优化的 ReLU lower relaxation（α），并用 autograd 形成最小优化闭环，为后续 BaB 节点 warm-start 做接口准备。

**主要改动**
- 新增：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(...)`：最小 α 优化循环（best-of，支持 warm-start）。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_alpha=...)`：不稳定 ReLU 的 lower slope 支持按节点输入 α 参数化。
- 新增：`tests/test_phase6d_alpha_crown_mlp.py`
  - 优化不回退、warm-start 不劣、采样 soundness、梯度链路钉子。
- 新增：`scripts/bench_phase6d_alpha_opt_convergence.py`
  - 输出 α 优化轨迹（stdout JSON）。
- 新增：`gemini_doc/change_2025-12-23_phase6d_alpha_crown_mlp_alpha_opt_and_warm_start.md`

**验证**
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`

---

## 2025-12-23：Phase 6E（BaB MLP）起步——split state + α-CROWN bound oracle + priority-queue driver

**动机**
- 把 split state 作为运行时一等公民接入 BaB 控制流，跑通 “节点评估 → 分支 → 剪枝/终止” 的最小闭环（MLP 链式子集）。

**主要改动**
- 新增：`boundflow/runtime/bab.py`
  - `ReluSplitState` + `solve_bab_mlp(...)`：最小 BaB driver（控制流在 Python）。
- 更新：`boundflow/runtime/crown_ibp.py`
  - forward IBP 支持 `relu_split_state` 做 best-effort 的 pre-activation 区间收缩。
- 新增：`tests/test_phase6e_bab_mlp.py`
  - split 约束收紧回归、toy complete 演示（注：该演示依赖 1D Linf 的输入域收缩补丁）。
- 新增：`gemini_doc/change_2025-12-23_phase6e_bab_mlp_split_state_and_driver.md`

**验证**
- `python -m pytest -q tests/test_phase6e_bab_mlp.py`

---

## 2025-12-23：Phase 6F PR-1（β/αβ-CROWN MLP）——αβ oracle 闭环 + feasibility 接口 + β 梯度钉子 + 非平凡空域（pairwise）

**动机**
- 先把 αβ oracle 的接口/可微闭环/可剪枝形态落地：让 BaB 能通过 `feasibility` 将空域当成一等公民剪枝，并用 β 梯度钉子避免 silent bug。

**主要改动**
- 新增：`boundflow/runtime/alpha_beta_crown.py`
  - PR-1：β 以 conservative 占位符进入计算图，并提供 `feasibility`。
- 更新：`boundflow/runtime/bab.py`
  - `BabConfig.oracle={"alpha","alpha_beta"}`：允许切换 oracle，并在 `infeasible` 时 prune。
- 新增：`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - β 梯度钉子、pairwise 的非平凡空域回归。
- 新增：`gemini_doc/change_2025-12-23_phase6f_pr1_alpha_beta_oracle_beta_grad_and_infeasible.md`

**验证**
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`

---

## 2025-12-23：Phase 6F PR-2（β/αβ-CROWN MLP）——β 真实 split-constraint encoding + 可证空域（非 pairwise）+ BaB 1D patch 降级

**动机**
- 将 β 从 PR-1 的占位符升级为真实的 split-constraint encoding，使 BaB 的 complete 支点回到 β 本身，并加强空域识别（不仅限于 pairwise）。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_pre_add_coeff_*)`：提供对 pre-activation 的线性系数注入槽位。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - PR-2：β 以 Lagrangian 形式注入 split 约束（`s*z>=0` ⇒ `-β*s*z`），并新增 convex-combo infeasible 证书搜索（first-layer）。
- 更新：`boundflow/runtime/bab.py`
  - 1D Linf 输入域收缩补丁改为可选开关 `use_1d_linf_input_restriction_patch=False`（默认关闭）。
- 更新：`tests/test_phase6e_bab_mlp.py`、`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - 回归/DoD：非 pairwise 空域证书、BaB 在不启用 patch 时由 αβ 恢复 complete。
- 新增：`gemini_doc/change_2025-12-23_phase6f_pr2_beta_split_constraint_encoding_and_bab_patch_demoted.md`

**验证**
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`

---

## 2025-12-24：Phase 6G PR-1（αβ oracle）——multi-spec 真 batch 回归 + `spec_reduce` 口径固化

**动机**
- 在 Phase 6F 把 β 语义闭环钉死后，Phase 6G 开始做“系统化收益”。PR-1 先在 oracle 层把 multi-spec 真 batch 的语义与优化目标口径（mean vs worst）固定下来，避免后续性能化/缓存化返工。

**主要改动**
- 更新：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(..., spec_reduce={"mean","min","softmin"}, soft_tau=...)`：固化多 spec 的目标聚合口径（默认 mean 保持兼容）。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp(..., spec_reduce=..., soft_tau=...)`：同步支持；并对 infeasible 检测增加 `m==1` 快速路径。
- 新增：`tests/test_phase6g_alpha_beta_multispec_batch.py`
  - batch vs serial 一致性、forward 复用计数（S=1 vs S=32）、multi-spec 梯度链路回归。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr1_alpha_beta_multispec_true_batch_and_spec_reduce.md`

**验证**
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2025-12-24：Phase 6G PR-2（BaB）——node-batch（batch pick + batch eval）回归 + 梯度隔离钉子 + 吞吐 microbench

**动机**
- 在 Phase 6F 把 β 编码语义闭环钉死后，Phase 6G PR-2 只做系统化收益：把 BaB 从 “一次评估 1 个节点” 升级到 “一次评估 K 个节点”，让 oracle 吃满 batch，提高吞吐，并用测试钉死正确性与 autograd 语义。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - node-batch 支持：`relu_split_state:[B,H]` 与 `beta:[B,H]` 的 β 注入（`_beta_to_relu_pre_add_coeff`）。
  - infeasible detector 仅对 `B==1` 启用；node-batch（`B>1`）跳过该 best-effort 检测以避免错误口径。
- 新增：`tests/test_phase6g_bab_node_batch.py`
  - `node_batch_size=1` vs `node_batch_size=4` 在 1D toy 上 verdict 一致（`proven`）。
- 新增：`tests/test_phase6g_node_batch_grad_isolation.py`
  - node-batch 梯度隔离：只对 node0 的 loss 反传时，其它 node 的梯度为 0（防串味）。
- 新增：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - microbench：对同一组 split states，对比 batched node-eval vs serial node-eval 的 p50 耗时与 speedup（stdout JSON）。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr2_bab_node_batch_eval.md`

**验证**
- `python -m pytest -q tests/test_phase6g_bab_node_batch.py`
- `python -m pytest -q tests/test_phase6g_node_batch_grad_isolation.py`

---

## 2025-12-24：Phase 6G PR-3A（BaB/αβ）——NodeEvalCache（split+config+spec）与命中回归钉子

**动机**
- 在 PR-2 完成 node-batch 后，最稳的系统化收益来自“减少重复评估”：同一 split pattern 在同一 run 内被重复 tighten/重复触达时，不应重复调用 oracle。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - 新增 `NodeEvalCache`：以 `(module,input_spec,C,oracle_config,split_state)` 为 key 的进程内缓存。
  - 新增 `eval_bab_alpha_beta_node(...)`：封装 “cache→oracle→writeback” 的统一入口。
  - `solve_bab_mlp` 接入 cache，并在 node-batch 下支持 partial hit（batch 内部分节点直接复用）。
- 新增：`tests/test_phase6g_bab_node_eval_cache.py`
  - cache hit/miss 回归钉子（同 node 不重算；改一处 split 必 miss）。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3a_node_eval_cache.md`

**验证**
- `python -m pytest -q tests/test_phase6g_bab_node_eval_cache.py`

---

## 2025-12-26：Phase 6G PR-4（BaB/αβ）——microbench 开关矩阵 + 计数器口径固化（可归因收益）

**动机**
- 将 Phase 6G 的系统化收益拆成可归因对照（cache / branch hint / infeasible prune），避免仅用 p50 计时导致“收益来源不清”的 reviewer 质疑。

**主要改动**
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - 引入开关矩阵：`enable_node_eval_cache/use_branch_hint/enable_batch_infeasible_prune`。
  - stdout JSON 固化输出关键计数器：`oracle_calls/forward_trace_calls/cache_hits/cache_misses/cache_hit_rate/pruned_infeasible_count/evaluated_nodes_count`。
  - 输出结构：`rows`（每个开关组合一行）+ `meta`（设备/形状/计时口径/版本等）。
- 新增：`gemini_doc/change_2025-12-26_phase6g_pr4_microbench_switch_matrix_and_counters.md`

**验证**
- `python scripts/bench_phase6g_bab_node_batch_throughput.py --device cpu --nodes 32 --node-batch-size 8 --specs 16 --steps 0 --warmup 1 --iters 3`

---

## 2025-12-29：Phase 6H PR-1（BaB/αβ）——端到端 time-to-verify 基准（开关矩阵 + JSONL 工件）

**动机**
- 将 Phase 6G 的“node-eval throughput”归因口径升级为端到端 “time-to-verify” 闭环证据链（verdict/节点数/队列行为/计数器），并产出可直接用于论文复现实验的 JSON/JSONL 工件。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - `BabConfig` 新增 `use_branch_hint`（默认 True），用于 E2E ablation。
  - `BabResult` 补齐端到端统计字段：`nodes_evaluated/nodes_expanded/batch_rounds/avg_batch_fill_rate`（不改变语义，仅增强可观测性）。
- 新增：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 端到端 BaB bench：对 `enable_node_eval_cache/use_branch_hint/enable_batch_infeasible_prune` 做 2×2×2 开关矩阵；
  - 输出 `{meta, rows}` JSON，且支持 `--jsonl-out` 追加写入 JSONL 工件。
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr1_e2e_bab_time_to_verify_bench.md`

**验证**
- `python scripts/bench_phase6h_bab_e2e_time_to_verify.py --device cpu --workload 1d_relu --oracle alpha_beta --steps 0 --max-nodes 256 --node-batch-size 32 --warmup 1 --iters 3`

---

## 2025-12-29：Phase 6H PR-1 DoD 补强——meta schema 固化 + 可比性标注 + bench schema 钉子

**动机**
- 钉死 E2E bench 的复现口径与可比性，避免 verdict 不一致时误解 speedup；并用测试长期守住输出 schema。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `rows` 增加 `comparable/note`；`batch_stats/serial_stats` 增加 `popped_nodes_total/queue_peak` 别名；`meta` 增加 `git_sha/device_name/spec_reduce/torch_num_threads`。
- 新增：`tests/test_phase6h_bench_e2e_schema.py`
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr1_dod_hardening_meta_comparable_schema_test.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py`

---

## 2025-12-29：Phase 6H PR-2——sweep 汇总 + 出图出表（JSONL → CSV/fig），闭环“可发表工件”

**动机**
- 将 6H PR-1 的 JSONL 工件升级为可批量 sweep、可汇总、可出图/出表的流水线，形成论文可直接引用的证据链。

**主要改动**
- 新增：`scripts/sweep_phase6h_e2e.py`
  - 批量运行 E2E bench 并追加 JSONL（每 run 一行 `{meta,rows}`）。
- 新增：`scripts/report_phase6h_e2e.py`
  - JSONL 展平为 CSV（switch 组合 × {batch,serial}）并生成 `summary.md`（仅以 comparable 行为主表）。
- 新增：`scripts/plot_phase6h_e2e.py`
  - 从 JSONL 自动出图（speedup、counters 对照、fill-rate 散点；需要 `matplotlib`）。
- 新增：`tests/test_phase6h_report_csv_schema.py`
- 新增：`tests/test_phase6h_plot_smoke.py`（无 `matplotlib` 则 skip）
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr2_sweep_report_plot_pipeline.md`

**验证**
- `python -m pytest -q tests/test_phase6h_report_csv_schema.py`
- `python -m pytest -q tests/test_phase6h_plot_smoke.py`

---

## 2025-12-29：Phase 6H PR-3（AE/论文工件准备）——一键 runner + meta 补全 + sweep 失败记录

**动机**
- 让“复现实验”更接近 AE 交付形态：一键运行主结果；meta 补齐 OS/Python；sweep 失败不静默且可审计。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta` 增加 `python_version/platform`。
- 更新：`scripts/sweep_phase6h_e2e.py`
  - 增加 `--fail-fast`；失败时写入 JSONL 失败记录（`meta.run_status=error` + stderr_tail），并在结束返回非 0。
- 更新：`scripts/report_phase6h_e2e.py`
  - `summary.md` 增加失败运行区块。
- 新增：`scripts/run_phase6h_artifact.sh`
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr3_artifact_runner_meta_and_sweep_failure_records.md`

**验证**
- `bash scripts/run_phase6h_artifact.sh`

---

## 2025-12-30：Phase 6H PR-4（AE 打包准备）——AE README + claims 映射 + schema_version + 环境审计

**动机**
- 将 6H 的可复现流水线升级为 AE 友好的交付口径：Kick-the-tires（≤30min）+ Claim→产物映射；并固化 `schema_version` 与 runner 环境审计信息，降低 “fresh machine” 排错成本。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version="phase6h_e2e_v1"`。
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - `meta.schema_version="phase6g_node_eval_v1"`。
- 更新：`scripts/run_phase6h_artifact.sh`
  - 输出 `env.txt/pip_freeze.txt/conda_list.txt` 环境审计信息（best-effort）。
- 新增：`gemini_doc/ae_readme_phase6h.md`
- 新增：`gemini_doc/change_2025-12-30_phase6h_pr4_ae_readme_schema_version_and_env_audit.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py`
- `bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run`

---

## 2025-12-31：Phase 6H PR-5（不动语义）——扩展 E2E bench workload suite（小型非 toy MLP）

**动机**
- 在不改 runtime 语义的前提下，把 time-to-verify 的主图从 toy 扩展到小型非 toy MLP case，使图表更接近论文主结果。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 新增 `--workload`：`mlp2d_2x16`、`mlp3d_3x32`。
  - 新增 `_make_chain_mlp(...)`：按 seed 构造可复现链式 MLP（Linear+ReLU）。
- 新增：`tests/test_phase6h_workload_suite_smoke.py`
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr5_workload_suite_mlp_small.md`

**验证**
- `python -m pytest -q tests/test_phase6h_workload_suite_smoke.py`

---

## 2025-12-31：Phase 6H PR-4 补丁——runner 支持 workloads 覆盖（kick-the-tires 默认不变）

**动机**
- 同时满足 kick-the-tires（默认快）与更像论文主结果的 workload suite（可选扩展），避免强制拉长 AE 路径。

**主要改动**
- 更新：`scripts/run_phase6h_artifact.sh`
  - 增加第二参数/环境变量 `PHASE6H_WORKLOADS` 覆盖 workload 列表，默认仍为 `1d_relu`。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - 增加 workload 覆盖用法说明。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr4_runner_workloads_override.md`

**验证**
- `bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run "1d_relu,mlp2d_2x16"`

---

## 2025-12-31：Phase 6 收尾 PR（测试收集卫生）——可选依赖 onnx/tvm 不再导致 collection 崩溃

**动机**
- 让 `pytest -q tests` 在缺少 `onnx/tvm` 等大依赖时仍可收集与出报告（相关用例可 skip，但不能在 collection 阶段崩溃），满足 AE/CI 的硬门槛。

**主要改动**
- 更新：`tests/test_phase4d_onnx_frontend_matches_torch.py`
  - 将 `import_onnx` 延后到 `importorskip("onnx")` 之后导入，避免无 onnx 时 collection 失败。
- 更新：`tests/test_phase5d_pr8_relax_lowering_skeleton.py`、`tests/test_phase5d_pr11c1_save_function_closure.py`、`tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
  - 模块级 `pytest.importorskip("tvm")` + `llvm` backend gate，避免无 tvm 时 collection 失败。
- 更新：`tests/test_env.py`
  - core imports 仅要求 `torch/boundflow`；`tvm/auto_LiRPA` 缺失则 skip。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - 增加 optional deps/skip 说明。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr6_test_collection_hygiene_optional_deps.md`

**验证**
- `pytest -q tests`

---

## 2025-12-31：Phase 6 收尾 PR（E2E 统计口径加固）——p90/p99 tail latency + timeout 计数 + schema v2

**动机**
- Phase 6 已具备 AE/论文交付形态，但 reviewer/AE 常追问 “是否只报 p50 掩盖尾部？” 与 “异常样本/timeout 是否被透明记录？”。
- 同时修复 `torch.utils.benchmark` 在 PyTorch 2.8+ 下的字段差异（`Measurement.number` 不存在）。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version` 升级为 `phase6h_e2e_v2`。
  - 每个开关组合输出 `batch/serial` 的 `p50/p90/p99`、`runs/valid_runs/timeouts` 与 `speedup_p90/speedup_p99`。
  - 新增 `--timeout-s`（perf_counter best-effort）与 `--torch-benchmark-repeats`（估计 p90/p99）。
- 更新：`scripts/report_phase6h_e2e.py`
  - CSV/summary 同步新增 `p90/p99` 与 run 计数字段；summary 主表展示 `p50+p90`。
  - 修复：sweep 失败记录（`rows=[]`）也能在 `summary.md` 显示。
- 更新：`scripts/plot_phase6h_e2e.py`
  - 新增 `*_speedup_p90.png`（p90 speedup 图）。
- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`、`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - `torch_benchmark` 计时元信息改用 `Measurement.number_per_run`（兼容 PyTorch 2.8+）。
- 更新：`tests/test_phase6h_bench_e2e_schema.py`、`tests/test_phase6h_report_csv_schema.py`
  - schema 钉子同步覆盖新增字段。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - Claim/产物映射同步纳入 `p90` 与 `*_speedup_p90.png`。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr7_e2e_tail_latency_p90_timeout_schema_v2.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py tests/test_phase6h_report_csv_schema.py tests/test_phase6h_plot_smoke.py`

---

## 2025-12-31：Phase 6 收官文档——新增 Phase 6 总结（phase6_summary.md）

**动机**
- Phase 6 的细节已分散记录在多份 `gemini_doc/change_*.md` 与 `docs/change_log.md` 中，但缺少一份“横向串联”的收官总结，便于论文/答辩叙事与研发接手。

**主要改动**
- 新增：`gemini_doc/phase6_summary.md`
  - 汇总 Phase 6 的目标/计划基线（三轴解耦 + stage pipeline）、6A→6H 里程碑、关键代码落点、DoD/回归钉子、可复现工件链与已知限制。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase6_summary.md` 纳入“关键交付文档”索引。
- 新增：`gemini_doc/change_2025-12-31_phase6_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 4 收官文档——新增 Phase 4 总结（phase4_summary.md）

**动机**
- Phase 4 的细节已记录在 `docs/change_log.md` 与 `gemini_doc/change_2025-12-17_phase4*.md`，但缺少一份“横向串联”的总结文档，便于论文/答辩与工程接手。

**主要改动**
- 新增：`gemini_doc/phase4_summary.md`
  - 汇总 Phase 4 的目标/完成定义、4/4A/4B/4C/4D 里程碑、关键代码落点与回归钉子（含可选依赖说明）。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase4_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase4_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 5 收官文档——新增 Phase 5 总结（phase5_summary.md）

**动机**
- Phase 5 的交付重点是“可复现评测产线 + schema_version=1.0 冻结”，细节分散在 `docs/phase5_done.md`、`docs/bench_jsonl_schema.md`、`gemini_doc/artifact_claims_phase5d.md` 与 `docs/change_log.md` 的 Phase 5 系列条目中；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase5_summary.md`
  - 总结 Phase 5 的 bench→JSONL→postprocess→artifact 产线闭环、TVM/Relax 可观测性与消融矩阵、schema contract tests、复现入口与已知限制。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase5_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase5_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 3 收官文档——新增 Phase 3 总结（phase3_summary.md）

**动机**
- Phase 3 的交付核心是 “Interval IBP reference + auto_LiRPA 对齐（MLP/CNN）”，作为后续 Phase 4/5/6 的 correctness 地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase3_summary.md`
  - 总结 Phase 3 的目标/完成定义、里程碑（MLP→CNN 扩展）、关键代码落点与回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase3_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase3_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 2 收官文档——新增 Phase 2 总结（phase2_summary.md）

**动机**
- Phase 2 的交付核心是 “TorchFrontend：torch.export → Primal IR + 最小 normalize 起步”，作为后续 Phase 3/4/5/6 的前端地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase2_summary.md`
  - 总结 Phase 2 的目标/完成定义、关键代码落点（`frontends/pytorch/frontend.py`、`frontends/normalize.py`）与回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase2_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase2_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 1 收官文档——新增 Phase 1 总结（phase1_summary.md）

**动机**
- Phase 1（总账中以 Phase 0/1 合并记录）的交付核心是 “工程止血 + Primal IR 加固（Node/Value + validate）”，作为后续 Phase 2/3/4/5/6 的工程/IR 地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase1_summary.md`
  - 总结工程止血（editable install/包结构清理）与 Primal IR 加固（Node/Value + validate）以及最小回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase1_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase1_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 0 收官文档——新增 Phase 0 总结（phase0_summary.md）

**动机**
- Phase 0（工程止血：editable install/包结构清理/最小 smoke）在总账中与 Phase 1 合并记录；补一份独立总结以保持 phase 总结文档体系一致。

**主要改动**
- 新增：`gemini_doc/phase0_summary.md`
  - 聚焦工程止血与最小可用开发基线（不展开 IR 设计细节）。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase0_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase0_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-01-03：更新全流程总览文档（从 claims 到 AE，对齐 Phase 0~6）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 原版本停留在 Phase 5（schema 1.0）视角，并把 Phase 6（αβ-CROWN + BaB + E2E 工件链）写成 TODO；随着 Phase 6 收官，需要把“从研究主张到 AE 交付”的总导览升级为覆盖 Phase 0~6 的现状。

**主要改动**
- 重写：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 版本升级为 v2.0，明确两条可复现主线：
    - Phase 5：interval IBP + TVM（bench JSONL `schema_version=1.0`）
    - Phase 6：αβ oracle + BaB（E2E JSON `schema_version=phase6h_e2e_v2`）
  - 补齐两套工件链的 Mermaid 流水线图，并更新 Phase 0~6 的导航入口（链接到 `gemini_doc/phase0_summary.md`~`gemini_doc/phase6_summary.md`）。
- 新增：`gemini_doc/change_2026-01-03_update_full_pipeline_from_claims_to_ae_v2.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-01-03：修正文档 Mermaid 兼容性（全流程总览改用纯文本流水线图）

**动机**
- 部分 Markdown 渲染器不支持 Mermaid，导致 `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 中的图无法正常显示；为保证在任意 Markdown 环境可读，改为纯文本图。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 Phase 5/Phase 6 的 Mermaid 流水线图替换为 `text` 纯文本流水线图。
- 新增：`gemini_doc/change_2026-01-03_fix_mermaid_in_full_pipeline_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-24：Phase 6G PR-3B（BaB/αβ）——消除分支选择二次 forward（复用 forward trace / branch hint）

**动机**
- 节点评估（oracle）已跑 forward IBP，但分支选择 `_pick_branch` 仍会再次调用 `_forward_ibp_trace_mlp`，导致每个节点额外一次 forward，吞吐收益被抵消；PR-3B 目标是复用 node eval 的 forward trace/分支提示，避免二次 forward。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - 新增 `run_crown_ibp_mlp_from_forward_trace(...)`：给定 `interval_env/relu_pre` 的 backward-only 入口，用于在优化循环/分支选择中复用 forward trace。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp`：forward trace 只算一次，优化循环内复用；并在 `AlphaBetaCrownStats` 暴露 `branch_choices`（每个 batch/node 的分支提示）。
- 更新：`boundflow/runtime/bab.py`
  - `NodeEvalCacheValue` 增加 `branch`；`eval_bab_alpha_beta_node` 返回 `branch_hint`；`solve_bab_mlp` 分支阶段优先使用 hint（无 hint 才回退 `_pick_branch`）。
- 新增：`tests/test_phase6g_branch_pick_reuses_forward_trace.py`
  - monkeypatch 统计 `_forward_ibp_trace_mlp` 调用次数：oracle=1 次，branch pick=0 次。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3b_branch_pick_reuse_forward_trace.md`

**验证**
- `python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py`

---

## 2025-12-24：Phase 6G PR-3C（BaB/αβ）——per-node infeasible/reason/witness + node-batch partial prune（first-layer）

**动机**
- node-batch（`B>1`）场景下 infeasible detector 被跳过会降低剪枝效率；同时 `feasible/reason/witness` 必须作为 per-node 元信息进入 cache/reuse（不影响 soundness，仅影响效率与可解释性）。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 新增 `check_first_layer_infeasible_split(...)`：对 first-layer split halfspaces 做 best-effort infeasible 检测并返回 `AlphaBetaCrownStats`（含 witness）。
- 更新：`boundflow/runtime/bab.py`
  - 新增 `BabConfig.enable_batch_infeasible_prune: bool=False`（默认关闭）。
  - 新增 `prune_infeasible_first_layer_items(...)`：batch 内逐 node 检测并 partial prune，且把 infeasible 元信息写入 `NodeEvalCache`。
- 新增：`tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - 混合 infeasible/feasible 节点的 prune 回归，并验证 infeasible 元信息进入 cache。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3c_per_node_infeasible_and_partial_prune.md`

**验证**
- `python -m pytest -q tests/test_phase6g_node_batch_partial_infeasible_prune.py`

## 2025-12-22：新增 Phase 6 评审备忘（无外链版）

**动机**
- 将三轴解耦 + stage pipeline 的评审建议改为只引用仓库内证据、统一指代、面向工程落地的版本，便于长期维护与引用。

**主要改动**
- 新增：`gemini_doc/phase6_review_three_axis_stage_pipeline.md`
  - 无外链版评审备忘（优势/避坑/落地顺序），并对齐 `docs/stage_4_critical_review.md`、`docs/p4_p5.md`、`docs/bench_jsonl_schema.md`。
- 新增：`gemini_doc/change_2025-12-22_phase6_review_notes_no_external_links.md`
  - 记录本次新增评审备忘的动机与摘要。
- 更新：`gemini_doc/change_2025-12-22_phase6a_inputspec_lpball_perturbation.md`
  - 补充指向评审备忘的链接。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-03-15：新增本机 AI CLI 更新脚本

**动机**
- 本机已安装 `gemini`、`claude`、`codex` 三个 AI CLI，且均以 npm 全局包形式存在；手动逐个更新不便于统一检查当前版本、最新版本与 PATH 中实际命令位置。

**主要改动**
- 新增脚本：`scripts/update_ai_clis.sh`
  - 内置三组映射：
    - `gemini` -> `@google/gemini-cli`
    - `claude` -> `@anthropic-ai/claude-code`
    - `codex` -> `@openai/codex`
  - 支持 `--check`：只读显示当前版本、最新版本、命令路径和状态。
  - 支持指定目标更新：如 `bash scripts/update_ai_clis.sh codex`。
  - 缺失包默认跳过，`--install-missing` 时才安装。
  - 更新结束后回显最终状态，便于确认。
- 新增变更记录：`gemini_doc/change_2026-03-15_add_ai_cli_update_script.md`

**验证**
- `bash -n scripts/update_ai_clis.sh`
- `bash scripts/update_ai_clis.sh --check`

---

## 2026-03-15：新增研发脉络总览文档（project_evolution_overview）

**动机**
- 现有文档已经覆盖总账、阶段总结、全流程与设计评审，但缺少一篇面向“项目演化主线”的总整理文档，导致接手者需要在多份文档之间来回跳转。

**主要改动**
- 新增：`gemini_doc/project_evolution_overview.md`
  - 从项目目标、阶段推进、代码落点、现有记录分工与下一步路线五个维度整理 Phase 0~6 的研发主线。
- 更新：`gemini_doc/README.md`
  - 新增“研发演化/接手视角”阅读路径，并将 `project_evolution_overview.md` 纳入长期有效文档索引。
- 新增：`gemini_doc/change_2026-03-15_add_project_evolution_overview.md`

**影响面**
- 仅影响文档导航与接手效率，无代码语义变更，无 Python API、CLI、schema 变化。

**验证**
- 检查 `gemini_doc/project_evolution_overview.md` 内引用路径存在。
- 检查 `gemini_doc/README.md` 索引可读且定位准确。
- 确认新文档与 `phase*_summary.md` 形成互补，而非重复替代关系。

---

## 2026-03-15：Phase 7A PR-1——LinearOperator 输入侧基础抽象（concretize_affine foundation）

**动机**
- Phase 6 的 CROWN/alpha-beta 路径已能在链式 MLP 子集上跑通，但输入处 `concretize_affine(...)` 仍要求显式稠密 `A:[B,K,I]`。如果直接继续扩 Conv 或一般图，会把算子覆盖问题和线性形式表达问题耦合在一起，返工风险高。

**主要改动**
- 新增：`boundflow/runtime/linear_operator.py`
  - 提供运行时内部 `LinearOperator` 协议与 `DenseLinearOperator` 实现。
- 更新：`boundflow/runtime/perturbation.py`
  - `PerturbationSet.concretize_affine(...)` 改为接受 `torch.Tensor | LinearOperator`，但保持 tensor 路径数值行为不变。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)` 与 `run_crown_ibp_mlp_from_forward_trace(...)` 的最终输入 concretize 统一走 `DenseLinearOperator`。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - first-layer infeasible helper 同步走 operator 路径。
- 新增：`tests/test_phase7a_linear_operator_concretize.py`
- 新增：`gemini_doc/change_2026-03-15_phase7a_pr1_linear_operator_concretize_foundation.md`

**影响面**
- 仅引入 runtime 内部基础抽象，不改公开 CLI、schema、artifact 口径。
- 不把 `LinearOperator` 混入 IR/Task/Planner，当前 backward 中间态仍保持 dense tensor。

**验证**
- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-15：新增 BoundFlow 工作流 skill

**动机**
- 仓库里已经有 `gemini_doc/llm_collaboration_workflow.md` 这份协作文档，但它还只是文档，不是一个可被 Codex 直接触发和复用的 skill。需要把这套长期使用的 PR-by-PR 工作流固化成真正的 skill。

**主要改动**
- 新增 skill：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
  - 触发范围限定在 BoundFlow 仓库内的工程迭代任务。
  - 固化执行顺序：读 `AGENTS.md` 与 `gemini_doc/llm_collaboration_workflow.md`、先做 DoD、实现最小闭环、跑定向测试、写 `gemini_doc/change_*.md`、追加 `docs/change_log.md`。
- 新增：`gemini_doc/change_2026-03-15_add_boundflow_workflow_skill.md`

**影响面**
- 不改仓库代码语义，不改 CLI、schema、artifact 口径。
- 将现有文档化工作流提升为可调用的 skill。

**验证**
- 检查 `~/.codex/skills/boundflow-workflow/SKILL.md` 已创建且内容覆盖 BoundFlow 工作流要点。
- 检查仓库中已记录本次变更，并追加总账。

---

## 2026-03-16：Phase 7A PR-2——线性 MLP backward A 状态 operator 化

**动机**
- Phase 7A PR-1 已经让输入边界 `concretize_affine(...)` 支持 `LinearOperator`，但 CROWN/alpha-beta 主路径里的 backward `A_u/A_l` 仍是全程 dense tensor。继续直接做 Conv 或一般图会把表达抽象和算子覆盖耦合在一起，返工风险高。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - 扩展 `LinearOperator` 内部能力，新增 `contract_last_dim(...)` 与 `matmul_right(...)`。
  - 新增 `RightMatmulLinearOperator`，用于表达 linear backward 的 lazy 右乘组合。
- 更新：`boundflow/runtime/crown_ibp.py`
  - 新增共享 `AffineBackwardState` 与 `_run_crown_backward_from_trace(...)`，收敛 `run_crown_ibp_mlp(...)` 和 `run_crown_ibp_mlp_from_forward_trace(...)` 的 backward 逻辑。
  - linear backward 改为 operator 路径；ReLU backward 保持显式 dense barrier。
  - 最终输入 concretize 直接接收 `LinearOperator`，不再只在最后包装 `DenseLinearOperator`。
- 新增：`tests/test_phase7a_pr2_linear_operator_backward_state.py`
  - 覆盖 lazy operator 数值等价、嵌套融合、非法输入防御，以及主 backward 路径确实调用 `matmul_right(...)`。
- 更新：`tests/test_phase7a_linear_operator_concretize.py`
  - 将对 CROWN 主路径的断言从 `DenseLinearOperator` 放宽到 `LinearOperator`。
- 新增：`gemini_doc/change_2026-03-16_phase7a_pr2_operatorize_linear_backward_state.md`

**影响面**
- 不改 CLI、schema、artifact、IR、planner。
- `alpha_crown.py` 与 `alpha_beta_crown.py` 通过共享 CROWN backward 自动继承这一改动，无需改变公开语义。
- 这一步的重点是表达层解耦，不承诺立即带来性能收益。

**验证**
- `python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py`
- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-16：为 Codex 全局安装 Superpowers，并补跨主机复用文档

**动机**
- 需要把 `superpowers` 装成主机级 Codex skill，使这台机器上的所有工程都能复用。
- 还需要留下一份可跨主机复用的安装文档，让另一台机器上的 Codex 读完后能自动完成安装，而不是死写某个固定路径。

**主要改动**
- 主机级配置：
  - clone `superpowers` 到 `~/.codex/superpowers`
  - 建立软链接：`~/.codex/skills/superpowers -> ~/.codex/superpowers/skills`
  - 更新 `~/.codex/config.toml`，启用：
    - `[features]`
    - `collab = true`
- 新增：`gemini_doc/codex_superpowers_global_install.md`
  - 记录目录探测规则、主机级安装步骤、验证/更新/卸载命令，以及一段可直接发给 Codex 的执行指令。
- 新增：`gemini_doc/change_2026-03-16_install_codex_superpowers_global.md`
- 更新：`gemini_doc/README.md`

**影响面**
- 不改 BoundFlow 代码语义，不改测试、schema、artifact 口径。
- 重启 Codex 后，这台机器上的所有工程都可发现 `superpowers`。

**验证**
- `ls -la ~/.codex/skills/superpowers`
- `git -C ~/.codex/superpowers rev-parse --short HEAD`
- `rg -n '^\[features\]|^collab = true$' ~/.codex/config.toml`

---

## 2026-03-16：Phase 7A PR-3——原生 NCHW contract 与 Conv-ready CROWN-IBP

**动机**
- PR-1/PR-2 已经把输入 concretize 和线性 MLP backward `A` 状态 operator 化，但 runtime contract 仍然默认输入是扁平 `[B,I]`。如果继续做 Conv 扩展而不升级 contract，后续会在 flatten/unflatten 适配上反复返工。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - `LinearOperator` 协议新增 `input_shape/input_numel/spec_dim/contract_input/reshape_input/conv2d_right`。
  - `DenseLinearOperator` 现在显式携带 `input_shape`。
  - 新增 `ReshapeInputLinearOperator` 与 `Conv2dLinearOperator`，让 runtime 内部可以原生表达 NCHW 输入与 `A @ Conv2d(x)`。
- 更新：`boundflow/runtime/perturbation.py`
  - `PerturbationSet.concretize_affine(...)` 现在原生接受 `center:[B,*input_shape]`。
  - tensor `A` 允许 `[B,K,I]` 或 `[B,K,*input_shape]`；operator `A` 则通过 `input_shape` 做显式校验。
- 更新：`boundflow/runtime/crown_ibp.py`
  - plain CROWN-IBP 从 `{linear,relu}` 扩到链式 `{conv2d,relu,flatten,linear}`。
  - forward trace 新支持 `conv2d` 与 `flatten(start_dim=1,end_dim=-1)`。
  - backward 新增 conv2d/flatten 分支；高维 ReLU pre-bound 在 barrier 处临时 flatten，再恢复原始 `input_shape`。
  - `get_crown_ibp_mlp_stats(...)` 现在接受 chain CNN 子集，但继续拒绝 skip/branch 非链式图。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 对含 `conv2d/flatten` 的图显式 fail-fast，避免误把 PR-3 理解成 alpha-CROWN 也支持 CNN。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 对含 `conv2d/flatten` 的图显式 fail-fast；PR-3 只扩 plain CROWN-IBP，不扩 alpha-beta/BaB。
- 新增测试：
  - `tests/test_phase7a_pr3_highdim_concretize.py`
  - `tests/test_phase7a_pr3_conv_linear_operator.py`
  - `tests/test_phase7a_pr3_crown_ibp_cnn.py`
- 新增文档：
  - `gemini_doc/change_2026-03-16_phase7a_pr3_native_nchw_contract_and_conv_crown.md`

**影响面**
- runtime public contract 现在原生支持高维输入，但当前承诺范围只到 rank-2 flat 和 rank-4 `NCHW`。
- 不改 IR、planner、CLI、artifact schema。
- `alpha_crown.py`、`alpha_beta_crown.py`、`bab.py` 仍然是 MLP-only。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_crown_ibp_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase7a_linear_operator_concretize.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-18：Phase 7A PR-4——Conv2dLinearOperator 的 exact lazy row-norm 归约

**动机**
- PR-3 已经把 `Conv2dLinearOperator` 接到高维 `NCHW` contract 和 plain CROWN-IBP 上，但它的 `row_abs_sum / row_l2_norm / row_abs_max` 仍然直接走 `to_dense()` 再归约。语义正确，但过早摊平了 Conv operator 的结构信息。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - 新增 `_materialize_feature_map_rows(...)`，把 operator 递归 materialize 成 `[B*K,C,H,W]` feature-map rows。
  - 新增 `_reduce_feature_map_rows(...)`，直接在 NCHW rows 上做 `l1/l2/linf` 归约。
  - `Conv2dLinearOperator.row_abs_sum / row_l2_norm / row_abs_max` 改为 exact lazy 路径，不再直接调用 `self.to_dense()`。
  - `Conv2dLinearOperator.to_dense()` 本身保持不变，继续作为 debug/reference path。
- 新增测试：
  - `tests/test_phase7a_pr4_conv_lazy_norms.py`
  - 覆盖单层 conv、嵌套 conv、禁止回退到 `Conv2dLinearOperator.to_dense()`、以及 `concretize_affine(...)` 在 `p in {inf,2,1}` 下与 dense reference 完全一致。
- 新增文档：
  - `gemini_doc/change_2026-03-18_phase7a_pr4_conv_lazy_row_norms.md`

**影响面**
- 不改 public API，不改 `LinearOperator` protocol，不改 `PerturbationSet.concretize_affine(...)` 签名。
- 不扩 `CROWN` / `alpha` / `alpha-beta` / `BaB` 的公开语义。
- 这次 PR 的“lazy”指结构化递归归约，而不是零 materialization，也不承诺最坏复杂度优化。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr4_conv_lazy_norms.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-19：Phase 7A PR-5——将 alpha-CROWN 从 MLP 扩到 chain CNN

**动机**
- PR-3/PR-4 已经把 plain CROWN-IBP 和 `Conv2dLinearOperator` 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_crown.py` 仍然是 MLP-only，`run_crown_ibp_mlp(..., relu_alpha=...)` 也还拒绝 rank>2 的 ReLU alpha。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_broadcast_relu_alpha(...)` 从 rank-2 扩到支持高维 ReLU pre bound。
  - 允许 shared alpha 形状：`[]`、`[*S]`、`[I]`、`[1,*S]`、`[1,I]`。
  - 明确拒绝 batch-specific alpha：`[B,*S]`、`[B,I]`。
  - `_forward_ibp_trace_mlp(...)` 对 conv `relu_split_state` 报更清晰的未支持信息。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 删除原来的线性层静态维度推断，改为从 `_forward_ibp_trace_mlp(...)` 的 `relu_pre` 读取逻辑 shape。
  - `AlphaState.alpha_by_relu_input[name]` 现在既支持 MLP 的 `[H]`，也支持 conv ReLU 的 `[C,H,W]`。
  - `run_alpha_crown_mlp(...)` 改成 forward-trace reuse，优化循环内调用 `run_crown_ibp_mlp_from_forward_trace(...)`。
  - warm-start 现在支持高维 shared alpha 的 shape 归一。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 对 conv 图继续 fail-fast，但文案改成“PR5 只扩 alpha-CROWN，alpha-beta-CROWN 仍然是 MLP-only”。
- 更新：`boundflow/runtime/bab.py`
  - 在 `solve_bab_mlp(...)` 入口处显式拒绝 `conv2d/flatten` 图，避免 `run_alpha_crown_mlp(...)` 扩容后误开 BaB。
- 新增测试：
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase7a_pr5_alpha_crown_chain_cnn.md`

**影响面**
- `run_alpha_crown_mlp(...)` 现在支持链式 CNN 子集 `{conv2d,relu,flatten,linear}`。
- `run_crown_ibp_mlp(..., relu_alpha=...)` 现在支持高维 shared alpha。
- `alpha-beta-CROWN` 与 `BaB` 仍然保持 MLP-only。
- conv `relu_split_state` 仍未开放。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py`

---

## 2026-03-19：Phase 7A PR-6——将 alpha-beta-CROWN 从 MLP 扩到 chain CNN

**动机**
- PR-5 已经把 plain CROWN-IBP 和 alpha-CROWN 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_beta_crown.py` 仍然停在 MLP-only：conv split/beta/first-layer detector 都还没打通。

**主要改动**
- 新增：`boundflow/runtime/relu_shape_utils.py`
  - 抽出高维 ReLU 公共 shape/broadcast helper：`shape_numel(...)`、`relu_input_shapes(...)`、`coerce_relu_param_shape(...)`、`broadcast_relu_split_like_pre(...)`。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_apply_relu_split(...)` 从 rank-2 扩到 rank-agnostic，conv `relu_split_state` 现在真正进入 `_forward_ibp_trace_mlp(...)`。
  - `relu_pre_add_coeff_l` 现在允许高维 structured 形状并在 backward 时 flatten 到 `[B,I]`。
  - `relu_alpha` 广播现在接受 per-batch 高维形状，供 alpha-beta oracle 的 `per_batch_params=True` 使用。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 改用 `relu_shape_utils.py`，删除本地重复的高维 alpha shape helper。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 去掉 conv 图 fail-fast，`run_alpha_beta_crown_mlp(...)` 现在支持链式 CNN 子集 `{conv2d,relu,flatten,linear}`。
  - `AlphaState/BetaState` 从“隐藏维度 `H`”切到基于 `relu_pre` 的逻辑 shape `[*S]`。
  - `per_batch_params=False` 时，conv alpha/beta 为 shared `[*S]`；`per_batch_params=True` 时为 `[B,*S]`。
  - `_beta_to_relu_pre_add_coeff(...)` 现在支持高维 split + 高维 beta，统一输出 flat `[B,I]`。
  - `_branch_choices_from_relu_pre(...)` 改成任意 rank flatten 后选最大 gap，继续返回 `(relu_input_name, flat_idx)`。
  - `check_first_layer_infeasible_split(...)` 与 `_collect_first_layer_split_halfspaces(...)` 新增对 direct-input `conv2d -> relu` 的支持。
  - first-layer conv 证书通过 one-hot output row + `DenseLinearOperator(...).conv2d_right(...).to_dense()` 提取 affine row。
  - deeper-than-first-layer conv split 不进入 halfspace 证书，只返回 `ok (no first-layer split halfspaces)`。
- 更新：`boundflow/runtime/bab.py`
  - conv 图继续 fail-fast，但文案改成：`BaB conv graphs not yet supported; PR6 only extends alpha-beta-CROWN oracle`。
- 新增测试：
  - `tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`
- 更新测试：
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
  - 把 PR-5 里“conv split_state 仍不支持”的旧断言改成 PR-6 新行为检查。
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase7a_pr6_alpha_beta_crown_chain_cnn.md`

**影响面**
- `run_alpha_beta_crown_mlp(...)` 现在支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`。
- conv `relu_split_state` 支持 shared + batch-specific 高维形状，并统一到 `[B,I]`。
- conv `alpha/beta` 完全沿用 `per_batch_params` 语义。
- first-layer direct-input conv split 可被 infeasible detector 证伪。
- `AlphaBetaCrownStats.branch_choices` 在 conv 图上继续返回 flat idx。
- conv `BaB` 仍然未开放。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py`

---

## 2026-03-19：Phase 6G 零 split-state detector 短路修正

**动机**
- 在清理并验证剩余未提交改动时，`tests/test_phase6g_branch_pick_reuses_forward_trace.py` 暴露出一个回归：root 节点空 split 也会触发 alpha-beta oracle 的 first-layer infeasible detector，导致多跑一次 `_forward_ibp_trace_mlp(...)`。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 新增 `_has_nonzero_split_state(...)`
  - `_collect_first_layer_split_halfspaces(...)` 在 `relu_split_state` 为空或“全部为 0”时直接返回，不再额外跑 forward trace
  - `run_alpha_beta_crown_mlp(...)` 的 `do_infeasible_check` 改成只在存在非零 split 时开启
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase6g_zero_split_detector_short_circuit.md`

**影响面**
- root node 的 `ReluSplitState.empty(...)` 不再误触发 detector
- `branch_choices` 继续复用 alpha-beta oracle 已有的 forward trace
- Phase 6G 的 branch picking forward reuse 回归恢复通过

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py`
- `conda run -n boundflow python -m pytest -q tests/test_env.py tests/test_phase4d_onnx_frontend_matches_torch.py tests/test_phase5d_pr8_relax_lowering_skeleton.py tests/test_phase5d_pr9_tvm_executor_linear_equiv.py tests/test_phase5d_pr10_tvm_compile_instruments.py tests/test_phase5d_pr11a_task_relax_ops_equiv.py tests/test_phase5d_pr11c1_save_function_closure.py tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py tests/test_phase5d_pr12_static_plan_modes.py tests/test_phase6c_crown_ibp_multispec_batch.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6h_artifact_runner_smoke.py tests/test_phase6h_bench_e2e_schema.py tests/test_phase6h_plot_smoke.py tests/test_phase6h_report_csv_schema.py tests/test_phase6h_workload_suite_smoke.py`

---

## 2026-03-20：Phase 7A PR-7——BaB on chain CNN（含 node-batch，与真实样本 batch 共存）

**动机**
- PR-6 已经把 `alpha-beta-CROWN` oracle 扩到 chain CNN，但 `bab.py` 仍然停在 MLP-only：conv 图 fail-fast、`ReluSplitState` 只支持 `[H]`、node-batch 仍假设输入 `B==1`。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - `solve_bab_mlp(...)` 现在支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`，但仅在 `oracle="alpha_beta"` 时开放。
  - `oracle="alpha"` 在 conv 图上继续 fail-fast，文案改为：`alpha-only BaB does not yet support conv graphs`。
  - `ReluSplitState.empty(...)` 新增 `input_spec=`，改为从 forward trace 的 `relu_pre` 推断高维逻辑 shape。
  - `ReluSplitState.with_split(...)` 继续接收 flat idx，但现在对逻辑 shape 做 flatten 更新后再恢复。
  - `_QueueItem` 新增 `example_idx`，host 侧改成“每样本独立搜索树 + 全局 heap 调度”。
  - node-batch 可以混合不同样本的节点进入一次 `alpha-beta` oracle 调用。
  - `max_nodes` 口径改成每样本独立预算。
  - 新增 `BabPerExampleResult`，`BabResult` 保留旧聚合字段并增加 `per_example`。
  - `_pick_branch(...)` 改成 rank-agnostic，对任意 `relu_pre` flatten 成 `[B,I]` 后返回 flat idx。
  - `prune_infeasible_first_layer_items(...)` 改成按 `item.example_idx` 切样本，并按样本分隔 `NodeEvalCache`。
- 更新脚本：
  - `scripts/bench_phase6g_bab_node_batch_throughput.py`
    - 适配 `_QueueItem.example_idx` 与 `cache_by_example`
  - `scripts/bench_phase6h_bab_e2e_time_to_verify.py`
    - 修正 instrumentation 对 `prune_infeasible_first_layer_items(...)` 新签名的兼容
- 更新测试：
  - `tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
- 新增测试：
  - `tests/test_phase7a_pr7_bab_chain_cnn.py`
  - `tests/test_phase7a_pr7_bab_batch_examples.py`
- 新增文档：
  - `gemini_doc/change_2026-03-20_phase7a_pr7_bab_chain_cnn.md`

**影响面**
- BaB 现在支持 chain CNN，但不扩 skip/branch/general DAG。
- `B>1` 采用混合方案：
  - host/BaB：每样本独立搜索树
  - oracle：batch 维表示一批待评估节点/domain
- 现有 bench/schema 不因为 `BabResult.per_example` bump 版本；旧脚本继续读取聚合字段。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr7_bab_chain_cnn.py tests/test_phase7a_pr7_bab_batch_examples.py tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6e_bab_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase7a_pr6_alpha_beta_crown_cnn.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase6h_bench_e2e_schema.py`
- 结果：`49 passed in 7.35s`
