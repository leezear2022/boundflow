# Phase 2 总结：TorchFrontend 最小可用（torch.export → Primal IR）+ normalize 起步

> 本文总结 Phase 2 的“目标 → 里程碑 → 代码落点 → 回归钉子 → 已知限制”，用于：
>
> - 解释 Phase 2 为什么是后续 Phase 3/4/5/6 的必要地基（没有可导入前端，就谈不上系统化验证/编译）；
> - 新同学快速定位：Torch 模型如何进入 BoundFlow 的 Primal IR、哪些算子会被映射/保留、如何做最小 normalize；
> - 复现回归：给出最小测试入口。
>
> 主要依据：
> - `docs/change_log.md`（Phase 2 总账条目）

---

## 1. Phase 2 解决什么问题（目标与边界）

Phase 2 的目标是把仓库从“只有 IR 草图”推进到“能实际导入一个 Torch 模型并得到可校验的 Primal IR”：

1. **TorchFrontend 最小闭环**：`torch.export.export()` → FX graph → Primal IR（Node/Value/shape/dtype/params）。
2. **最小 normalize**：把一部分 FX/aten 形态规整成后续阶段更容易处理的 primitive（例如把部分 `call_method::*` 归一到 `reshape/transpose` 等）。
3. **测试钉子**：用一个小 MLP 的导入单测，证明“导入结果自洽且可 validate”，并能作为 Phase 3 的 IBP、Phase 4 的 Task pipeline 的输入来源。

Phase 2 的边界（刻意不做）：

- 不做 IBP/CROWN/BaB 等边界传播（Phase 3/6）；
- 不做 TVM/Planner/Task（Phase 4/5）；
- 不做 ONNX 前端（Phase 4D）。

---

## 2. Phase 2 的完成定义（Done Definition）

Phase 2 的“完成”以可导入、可校验、可复现为准：

1. 给定一个小 Torch MLP，能导入为 `BFPrimalProgram/BFPrimalGraph`；
2. 导入结果包含必要 meta：inputs/outputs、Value 的 shape/dtype、params 映射；
3. `graph.validate()` 可通过（结构自洽）；
4. 单测可复现跑通：`tests/test_torch_frontend_import.py`。

---

## 3. 核心实现落点（从哪看起）

### 3.1 Torch 前端：torch.export → Primal IR

- `boundflow/frontends/pytorch/frontend.py`
  - `export_mode="export"`：使用 `torch.export.export()` 获取 FX Graph
  - aten → primitive 映射：常见 `aten.*` 映射到 `linear/relu/add/...` 等 v0.1 primitive 名称
  - 未知 op：保留原名（方便 debug 与后续逐步补齐）
  - params 映射：将 placeholder 名映射到 `ExportedProgram.state_dict`，填充 `BFPrimalProgram.params`

### 3.2 最小 normalize

- `boundflow/frontends/normalize.py`
  - 对导入图做最小规范化（例如部分 `call_method::*` → `reshape/transpose`）
  - normalize 入口会调用 `graph.validate()`，把结构错误尽早暴露

---

## 4. 回归钉子与如何验证

### 4.1 Torch 前端单测（推荐入口）

- `tests/test_torch_frontend_import.py`
  - 验证：小 MLP 的 torch.export 导入、primitive 映射、输入/参数 kind、以及图校验

建议运行（conda env：`boundflow`）：

```bash
python -m pytest -q tests/test_torch_frontend_import.py
```

---

## 5. 已知限制与后续阶段关系

Phase 2 的限制是刻意选择的：

- primitive 覆盖有限：未知 op 先保留原名，不在 Phase 2 强行“完全覆盖”；
- normalize 只做最小规则：避免过早引入复杂 pass pipeline（Phase 5 再系统化）；
- 仅 Torch 前端：ONNX 前端与“前端统一”在 Phase 4D 落地。

Phase 2 对后续阶段的意义：

- **对 Phase 3**：提供可导入的 Primal IR 输入，使 IBP reference 能对齐真实 Torch 模型；
- **对 Phase 4**：为 Task/Planner/Executor 的抽象提供稳定语义输入（避免“无模型可导入”的空转）；
- **对 Phase 5**：为 bench/产线提供可重复生成的 workload 来源；
- **对 Phase 6**：为方法族（CROWN/αβ/BaB）提供一致的模型语义入口（Primal IR 不被扰动/spec 污染）。

