# BoundFlow 项目交接摘要（给其它大模型/合作者）

> 本文用于快速让新合作者理解：**现在项目做到哪了、结构是什么、跑通了什么、下一步要做什么**。默认 conda 环境为 `boundflow`。

## 1. 一句话目标

BoundFlow 旨在把 **LiRPA/CROWN（含 IBP）边界传播**（以及后续 BaB/认证训练）当成一等公民工作负载，通过 **验证感知 IR + 全局 Planner +（后续）TVM 后端**，系统性解决 Python/库式 verifier 的 kernel 碎片、同步与重复计算问题。

当前阶段重点：先把 **IBP（Interval Bound Propagation）** 的正确性路径做成可优化的 **Task pipeline**，并用 auto_LiRPA 作为 ground truth 对齐测试。

## 2. 当前完成度（到 Phase 4B.0）

### 2.1 已经跑通的闭环

- Torch 模型（`torch.export`）→ Primal IR（Node/Value）→ Planner（v0：整图一个任务）→ BFTaskModule → PythonTaskExecutor（reference）→ 输出 interval bounds
- 对齐验证：
  - MLP：Task pipeline 输出 == auto_LiRPA `compute_bounds(method="IBP")`
  - MNIST 风格 CNN：Task pipeline 输出 == auto_LiRPA `compute_bounds(method="IBP")`

### 2.2 已经落地的关键设计点

- **Node/Value 分离的 Primal IR**（Value 承载 shape/dtype/kind）：
  - `boundflow/ir/primal.py`
- **Task 化执行路径**：
  - `BoundTask` 由 `TaskOp` 序列组成，`BFTaskModule` 支持 multi-task（v0 仍是单 task）
  - `boundflow/ir/task.py`
- **Planner v0（先钉接口）**：
  - `plan_interval_ibp_v0(program)`：整图打包成一个 `TaskKind.INTERVAL_IBP` 任务
  - `boundflow/planner/interval_v0.py`
- **Reference backend（PythonTaskExecutor）**：
  - `boundflow/runtime/task_executor.py`
- **兼容入口（PythonInterpreter）**：
  - 接收 `BFPrimalProgram`，内部走 planner + task executor
  - `boundflow/runtime/executor.py`
- **StoragePlan（Memory/Storage 抽象接口钉住）**：
  - `BFTaskModule.storage_plan` 包含 `BufferSpec` 与 `value_to_buffer`
  - planner 默认“一值一 buffer”，后续可做 aliasing/复用优化
  - `boundflow/ir/task.py`，`boundflow/planner/interval_v0.py`

## 3. 目录结构（重要模块）

- `boundflow/ir/`
  - `primal.py`：Primal IR（Node/Value/TensorType），`BFPrimalGraph.validate()`
  - `task.py`：Task IR（TaskKind/TaskOp/BoundTask/BFTaskModule + StoragePlan）
  - `bound.py`：Bound IR 占位（v0.1 未进入主流程）
- `boundflow/frontends/`
  - `pytorch/frontend.py`：`torch.export` → Primal IR；提取 conv2d/flatten/permute 的必要 attrs
  - `normalize.py`：最小规范化（例如 call_method 映射），并强制 `graph.validate()`
- `boundflow/domains/`
  - `interval.py`：IntervalState + IntervalDomain（IBP 规则：linear/conv2d/relu/add/mul）
- `boundflow/planner/`
  - `interval_v0.py`：Planner v0（整图一个 task）+ 默认 StoragePlan
- `boundflow/runtime/`
  - `task_executor.py`：PythonTaskExecutor（reference backend，执行 TaskOp）
  - `executor.py`：PythonInterpreter（兼容入口，内部走 task pipeline）
- `tests/`
  - `test_env.py`：环境导入 smoke（torch/auto_LiRPA/tvm/boundflow）
  - `test_phase3_ibp_*`：非 task 的 IBP 对齐（历史/回归）
  - `test_phase4_task_pipeline_*`：Task pipeline 对齐（MLP/CNN）
  - `test_phase4b_storage_plan.py`：StoragePlan schema 回归
- `docs/`
  - `strategy_a_refactor_plan.md`：策略 A 的总体计划
  - `phase4a_plan_task_cnn_and_permute.md`：Phase 4A 计划
  - `next_plan_after_phase4a.md`：Phase 4A 后总体路线（已吸收批判性审查）
  - `stage_4_critical_review.md`：对 Phase 4 的风险点审查（memory/relax/layout）
  - `change_log.md`：变更记录（按批次追加）
- `boundflow/3rdparty/`
  - `tvm` / `tvm-ffi` / `auto_LiRPA`：submodule（当前已切换为 SSH URL）

## 4. 关键 API（你可以从这里开始读）

### 4.1 Torch 导入

- `from boundflow.frontends.pytorch.frontend import import_torch`
- `program = import_torch(model, (example_input,), export_mode="export", normalize=True)`

### 4.2 规划与执行（Task pipeline）

- `from boundflow.planner import plan_interval_ibp_v0`
- `task_module = plan_interval_ibp_v0(program)`
- `from boundflow.runtime.task_executor import PythonTaskExecutor, LinfInputSpec`
- `out = PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(value_name="input", center=x0, eps=0.1))`
- 输出：`IntervalState(lower, upper)`（torch.Tensor）

## 5. 如何在本地验证（推荐命令）

### 5.1 环境（conda）

- 进入环境：`conda activate boundflow`
- 安装/编译依赖（第一次）：`bash scripts/install_dev.sh`
- 环境导入检查：`python tests/test_env.py`

### 5.2 关键测试（对齐 auto_LiRPA）

- Task pipeline（MLP + CNN）：
  - `python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`
- StoragePlan schema：
  - `python -m pytest -q tests/test_phase4b_storage_plan.py`

## 6. 当前限制（刻意收敛的 v0.1 范围）

- 只实现 **IBP/Interval**；CROWN/α-CROWN（lirpa_linear）尚未进入主流程。
- Spec/Property 目前只有 `LinfInputSpec(center, eps)`；尚未支持 margin `C` 矩阵、输出性质表达、BaB node/spec batching。
- Task executor 覆盖的 primitive 仍有限（够支撑 MLP + MNIST CNN 的 IBP 对齐）；更多算子需要继续补（如 `sub/max_pool2d/batch_norm` 等）。
- TVM lowering/执行（TVMExecutor）尚未开始（下一阶段）。

## 7. 后续计划（推荐优先级）

> 详见 `docs/next_plan_after_phase4a.md`，这里给最短路径版本。

1) **Phase 4B.2：Spec/Property（margin/C 矩阵）**
   - 把“验证性质”表达引入 Task/Module（例如输出 margin bound）
   - 新增与 auto_LiRPA `compute_bounds(C=...)` 的对齐测试

2) **Phase 4B.3：Layout/Transpose 作为可优化对象**
   - 在 IR 中把 layout transform 明确成 hint/一等公民，为 planner 的 transpose elimination / layout propagation 留接口

3) **Phase 4C：TVM Backend v0（先 linear）**
   - 先用 Python driver 调 TVM 编译出的 kernel（不急着全迁移到 Relax VM）
   - 做到：同一 BFTaskModule，PythonTaskExecutor 输出 == TVMExecutor 输出（允许小误差）

4) **Phase 5：最小 CROWN（lirpa_linear）闭环**
   - 引入线性界 DomainState（A,b 或等价表示）
   - 实现最小 backward bound（线性层 + ReLU），对齐 auto_LiRPA `method="CROWN"`（或至少不劣于 IBP）

## 8. 变更记录与历史

- 变更记录：`docs/change_log.md`
- 目前 remote/submodule 已切换为 SSH（repo：`git@github.com:leezear2022/boundflow.git`；依赖同理）。

