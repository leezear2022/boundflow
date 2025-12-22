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
