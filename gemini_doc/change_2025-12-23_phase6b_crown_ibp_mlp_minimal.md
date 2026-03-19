# 变更记录：Phase 6B 起步——最小 CROWN-IBP（MLP: Linear+ReLU）

## 动机

Phase 6 的下一步需要把 “IBP forward + CROWN backward” 的最小闭环跑通（先从 MLP 的 `Linear + ReLU` 子集开始），以便：

- 验证三轴解耦/Stage pipeline 的可落地性；
- 为后续 multi-spec batching、α/β 优化、以及 BaB driver 打基础；
- 为 `PerturbationSet.concretize(A, x0)`（未来可能演进到 `LinearOperator`）建立最小工作路径。

## 主要改动

- 更新：`boundflow/runtime/perturbation.py`
  - 新增 `PerturbationSet.concretize_affine(center, A, b)`：将一般线性形式 `A @ x + b` 在输入扰动集合上数值化（当前实现覆盖 `LpBallPerturbation(p∈{∞,2,1})` 的张量版 `A`）。

- 新增：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)`：最小 CROWN-IBP 执行器（single-task，op 子集仅 `linear/relu`）：
    - forward：interval IBP 得到每个 ReLU 的 pre-activation bounds（用于固定松弛线性界）
    - backward：按 CROWN 规则反向传播线性界（upper/lower 分开，并按系数符号选择 upper/lower relaxation）
    - 输入处：调用 `PerturbationSet.concretize_affine(...)` 得到最终输出的上下界
  - `get_crown_ibp_mlp_stats(...)`：返回是否支持/不支持原因（用于未来 bench gate）

- 新增测试：`tests/test_phase6b_crown_ibp_mlp.py`
  - `L∞`：采样 soundness + upper bound 不劣于 IBP（更紧或相等）
  - `L2`：采样 soundness

## 如何验证

```bash
python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py
```

## 已知限制

- 仅支持 MLP 风格（`Linear + ReLU`）的单任务图；不支持 conv/残差/分支结构。
- `A` 目前以显式张量表示；后续按 `gemini_doc/bound_methods_and_solvers_design.md` 的 §7.1 可演进为 `LinearOperator`，以避免 CNN 上显式 `A` 的内存爆炸。

