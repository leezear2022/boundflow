# Phase 1 总结：工程止血 + Primal IR 加固（Node/Value + validate）

> 注：仓库最早的“启动止血”在总账中以 `Phase 0/1` 合并记录（见 `docs/change_log.md` 首条）。本文按“Phase 1”口径总结这部分工作，便于与 Phase 2/3/4/5/6 的总结文档风格保持一致。
>
> 主要依据：
> - `docs/change_log.md`：`2025-12-17：Phase 0/1 首次止血与 IR 加固`

---

## 1. Phase 1 解决什么问题（目标与边界）

Phase 1 的目标是让仓库“从一团乱麻变成可持续开发的工程基线”：

1. **工程止血**：清理重复包结构、建立可编辑安装（editable install）路径，保证后续开发/测试不被路径问题卡住。
2. **Primal IR 加固**：把 IR 从早期草图升级为可校验的结构（Node/Value 双层 + type/meta），并提供 `validate()` 防止后续 planner/runtime 在不一致 IR 上返工。
3. **最小测试门槛**：提供环境 smoke 与 IR 单测，让“导入/校验”成为持续回归的硬门槛。

Phase 1 的边界（刻意不做）：

- 不实现任何 bound 算法（IBP/CROWN/…）；
- 不做前端完整导入（TorchFrontend 在 Phase 2）；
- 不做 Task/Planner/TVM（Phase 4/5）。

---

## 2. Phase 1 的完成定义（Done Definition）

Phase 1 的“完成”以工程可用性与 IR 可审计性为准：

1. 仓库支持 `python -m pip install -e .` 的标准开发安装；
2. `boundflow` 包路径干净（无重复/空壳目录造成的歧义 import）；
3. Primal IR 具备 Node/Value 结构与一致性校验（`BFPrimalGraph.validate()`）；
4. 至少有一条 CI/本地可跑的硬钉子：`tests/test_ir_primal_validate.py` 与 `tests/test_env.py`。

---

## 3. 核心改动落点（从哪看起）

### 3.1 工程化与包结构

- `pyproject.toml`：支持 editable install
- 删除重复目录：移除 `boundflow/boundflow/`（避免 `boundflow.boundflow.*` 路径混乱）

### 3.2 Primal IR：Node/Value 双层结构

- `boundflow/ir/primal.py`
  - IR 升级为 `Node`/`Value`/`TensorType`
  - 提供 `BFPrimalGraph.validate()`（结构一致性校验）

### 3.3 前端壳子对齐（为 Phase 2/4 清障）

- `boundflow/frontends/pytorch/frontend.py`、`boundflow/frontends/onnx/frontend.py`
  - 适配新的 `BFPrimalGraph()` 构造方式（当时为“壳子对齐”，真正可用导入在 Phase 2/4D）
- `boundflow/frontends/normalize.py`
  - normalize 入口调用 `graph.validate()`，让不一致 IR 尽早暴露

### 3.4 测试与环境 smoke

- `tests/test_ir_primal_validate.py`：IR validate 的硬钉子
- `tests/test_env.py`：boundflow import smoke + 打印 `CONDA_DEFAULT_ENV`

---

## 4. 回归与验证（建议命令）

```bash
python -m pytest -q tests/test_ir_primal_validate.py
python tests/test_env.py
```

---

## 5. 已知限制与后续阶段意义

Phase 1 的意义在于：把“工程与 IR 的地基”钉住，避免后续阶段返工：

- Phase 2 能在稳定 IR 上实现 TorchFrontend；
- Phase 3 的 IBP reference 能依赖 `validate()` 确保导入图结构不漂；
- Phase 4/5 的 Task/Planner/TVM 体系能把 Primal IR 当成可靠输入，而不是一边写系统一边修 IR。

