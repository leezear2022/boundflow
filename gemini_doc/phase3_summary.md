# Phase 3 总结：Interval IBP reference（PythonInterpreter）+ auto_LiRPA 对齐（MLP/CNN）

> 本文总结 Phase 3 的“目标 → 里程碑 → 代码落点 → 回归钉子 → 已知限制”，用于：
>
> - 解释 Phase 3 为什么是 Phase 4/5/6 的正确性地基；
> - 新同学快速定位：IBP reference 实现在哪里、如何对齐 auto_LiRPA；
> - 复现回归：哪些测试依赖 auto_LiRPA、如何跑。
>
> 主要依据：
> - `docs/change_log.md`（Phase 3 总账）
> - Phase 4 兼容层说明（`PythonInterpreter` API 在 Phase 4 内部被重定向到 task pipeline）

---

## 1. Phase 3 解决什么问题（目标与边界）

Phase 3 的目标非常聚焦：

1. **实现一个“正确性优先、可调试”的 IBP reference 执行器**（作为后端/编译优化之前的 ground truth）。
2. **用 auto_LiRPA 作为外部权威参考实现进行对齐**，把“语义正确”钉死，避免后续扩算子/接 TVM/做 planner 时漂移。

Phase 3 的边界（刻意不做）：

- 不做 Task IR / planner / scheduler（Phase 4/5 的系统工作）；
- 不做 CROWN/αβ/BaB（Phase 6 的方法族与 solver）。

---

## 2. Phase 3 的完成定义（Done Definition）

Phase 3 的“完成”以 correctness gate 为准：

1. **MLP 子集**（Linear/ReLU）下，BoundFlow 的 IBP 上下界与 auto_LiRPA `compute_bounds(method="IBP")` 对齐；
2. **CNN 子集**（Conv2d/Flatten/Linear/ReLU）下，同样对齐；
3. 执行器具备足够的工程可用性：输入扰动用 `LinfInputSpec` 表达，输出为 `IntervalState(lower, upper)`。

---

## 3. 里程碑回顾：Phase 3（MLP）→ Phase 3 扩展（CNN）

### 3.1 Phase 3：Interval IBP + PythonInterpreter（MLP 对齐）

**核心改动（代码落点）**

- Interval 域：`boundflow/domains/interval.py`
  - `IntervalState(lower, upper)`
  - `IntervalDomain` 支持 `linear/relu/add/mul` 的 IBP 规则（先覆盖 MLP 子集）
- Runtime：`boundflow/runtime/executor.py`
  - `LinfInputSpec(value_name, center, eps)`
  - `PythonInterpreter.run_ibp(program, input_spec)`：按 Primal IR 顺序解释执行，输出最终 output value 的 interval
- 导出：`boundflow/runtime/__init__.py`

**关键回归**

- `tests/test_phase3_ibp_against_auto_lirpa.py`（MLP 对齐 auto_LiRPA）

对应总账：`docs/change_log.md` 的 “Phase 3 Interval IBP + PythonInterpreter（对齐 auto_LiRPA）”。

### 3.2 Phase 3 扩展：Conv2d/Flatten（CNN 对齐）

**动机**：对齐 auto_LiRPA 的最小 CNN 验证路径，确保 IBP reference 不只覆盖 MLP。

**核心改动（代码落点）**

- Torch frontend：`boundflow/frontends/pytorch/frontend.py`
  - 增加 op 映射：`aten.conv2d.default → conv2d`、`aten.flatten.using_ints → flatten`
  - 提取 conv2d 与 flatten 的常量 attrs（stride/padding/dilation/groups、start_dim/end_dim）
- Interval 域：`boundflow/domains/interval.py`
  - `conv2d` 的 IBP：权重正负分解后用 `torch.nn.functional.conv2d` 计算上下界
- PythonInterpreter：`boundflow/runtime/executor.py`
  - 增加 `conv2d/flatten` 支持
  - `reshape` 从占位改为按 output meta shape 执行真实 reshape（避免 Flatten 后线性层形状不一致）

**关键回归**

- `tests/test_phase3_ibp_cnn_against_auto_lirpa.py`（CNN 对齐 auto_LiRPA）

对应总账：`docs/change_log.md` 的 “Phase 3 扩展 Conv2d/Flatten（对齐 auto_LiRPA MNIST CNN 的 IBP）”。

---

## 4. 代码入口与怎么用

Phase 3 的核心入口（面向开发/调试）：

- `boundflow/runtime/executor.py`：`PythonInterpreter.run_ibp(program, LinfInputSpec(...))`
- `boundflow/domains/interval.py`：IBP 规则实现（最容易排查单算子错误）

> 注：从 Phase 4 开始，`PythonInterpreter` 作为兼容层仍保留 API，但内部会通过 `plan_interval_ibp_v0 + PythonTaskExecutor` 执行（见 `docs/change_log.md` Phase 4 条目）。这确保 Phase 3 的对外接口不被“系统重构”打断。

---

## 5. 回归与验证（建议命令）

auto_LiRPA 为可选依赖；缺失时相关对齐测试会被 skip（Phase 6 的测试收集卫生已进一步强化此类行为）。

```bash
# MLP 对齐（需要 auto_LiRPA）
python -m pytest -q tests/test_phase3_ibp_against_auto_lirpa.py

# CNN 对齐（需要 auto_LiRPA）
python -m pytest -q tests/test_phase3_ibp_cnn_against_auto_lirpa.py
```

---

## 6. 已知限制与 Phase 4/5/6 的接口意义

Phase 3 的限制是刻意选择的：

- 算子覆盖仅到 “最小闭环子集”（MLP + MNIST-style CNN）；
- 输入扰动只覆盖 `L∞`（`LinfInputSpec`），更一般的 `L2/L1/L0/patch` 在 Phase 6A 引入 `InputSpec + PerturbationSet` 后系统化扩展；
- 执行形态是逐节点解释执行，缺少 Task/Planner/Backend 的系统化插槽（Phase 4/5 负责解决）。

Phase 3 对后续阶段的意义：

- **对 Phase 4**：提供了 Task pipeline 重构前的语义 ground truth，确保“抽象成 Task”不会改算法口径；
- **对 Phase 5**：作为 bench/correctness gate 的历史基线之一（尤其是与 auto_LiRPA 的对齐基础设施）；
- **对 Phase 6**：作为从 IBP 扩展到 CROWN/αβ/BaB 时的 reference 起点（先保证 soundness，再谈 tightness/性能）。

