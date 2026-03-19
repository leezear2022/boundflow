# Phase 0 总结：仓库工程止血（可编辑安装 + 包结构清理 + 最小 smoke）

> 注：仓库总账里最早一条记录以 `Phase 0/1` 合并描述（见 `docs/change_log.md` 首条）。本文按“Phase 0”口径，仅聚焦其中偏工程/仓库卫生的部分；IR 结构化与 `validate()` 的部分在 `gemini_doc/phase1_summary.md` 展开。
>
> 主要依据：
> - `docs/change_log.md`：`2025-12-17：Phase 0/1 首次止血与 IR 加固`

---

## 1. Phase 0 解决什么问题（目标与边界）

Phase 0 的目标是把仓库从“不可稳定开发/不可稳定 import”的状态止血到一个可持续迭代的工程基线：

1. **可编辑安装（editable install）**：让开发者能用标准方式安装与导入（不靠临时 PYTHONPATH/重复目录）。
2. **包结构清理**：消除重复/空壳包目录导致的 import 歧义。
3. **最小 smoke**：提供最小环境检查路径，避免后续阶段在“环境/路径问题”上浪费时间。

Phase 0 的边界（刻意不做）：

- 不做 IR 设计（Phase 1）；
- 不做模型导入（TorchFrontend 在 Phase 2；ONNX 在 Phase 4D）；
- 不做任何边界传播算法（Phase 3/6）。

---

## 2. Phase 0 的完成定义（Done Definition）

1. 仓库可通过 `python -m pip install -e .` 正常安装；
2. `import boundflow` 不再因重复包目录产生歧义；
3. `.gitignore` 覆盖常见开发产物（避免污染仓库）；
4. 最小 smoke 能提示当前环境是否为预期的 conda env（`boundflow`）。

---

## 3. 核心改动落点（从哪看起）

- `pyproject.toml`
  - 支持 editable install（`python -m pip install -e .`）
- 删除重复/空壳目录：移除 `boundflow/boundflow/`
  - 避免出现 `boundflow.boundflow.*` 的迷惑路径
- `.gitignore`
  - 忽略 `__pycache__/`、`*.egg-info/`、`.pytest_cache/` 等
- `tests/test_env.py`
  - 最小环境 smoke：能导入 boundflow，并打印 `CONDA_DEFAULT_ENV`（非 `boundflow` 环境给出提示）

---

## 4. 回归与验证（建议命令）

```bash
python -m pip install -e .
python tests/test_env.py
```

---

## 5. 与 Phase 1 的关系

Phase 0 解决的是“工程基线”问题；Phase 1 解决的是“Primal IR 可校验”问题（Node/Value + `BFPrimalGraph.validate()`）。

两者合起来构成后续 Phase 2/3/4/5/6 的地基：先能稳定安装/导入，再在稳定 IR 上推进前端/算法/系统与 AE 工件链。

