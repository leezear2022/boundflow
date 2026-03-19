# 变更记录：补充 bound_methods 设计中的 `concretize` 实现模式

## 动机

在讨论多方法族（IBP/CROWN/αβ-CROWN）与多扰动集合时，`concretize` 往往是容易混淆的概念：它到底属于 domain、还是属于 perturbation、还是属于 solver？

为避免后续实现时“把 concretize 塞进各种 Domain 子类导致重复/爆炸”，本次把 `concretize` 的职责边界补充进三轴解耦设计文档。

## 主要改动

- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 在 §6 新增 `concretize` 的实现模式说明：
    - Interval（IBP）：状态已是数值 `(lb,ub)`，等价于直接返回。
    - Linear（CROWN/DeepPoly/αβ）：状态是线性形式 `A x + b`，在输入处调用 `PerturbationSet.concretize(A, x0)` 数值化。
  - 强调 `concretize` 的核心逻辑属于扰动集合（对偶范数/top-k/support function），并与 `gemini_doc/perturbation_support_design.md` 对齐。

## 验证

- 文档变更（无额外运行时验证）。

