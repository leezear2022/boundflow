# 变更记录：新增多范数输入扰动（L∞/L2/L1/L0）设计文档

## 动机

论文与系统设计需要回答：对于 `L∞/L2/L1/L0` 等不同扰动集合，线性算子（尤其第一层 affine/conv）对应的最紧上/下界公式不同，BoundFlow 应如何设计才能支持并保持可扩展性。

## 主要改动

- 新增：`gemini_doc/perturbation_support_design.md`
  - 提出 `PerturbationSet` 抽象与 support function 统一公式（`h_S(w)`），用于将不同扰动集合统一到线性算子边界计算。
  - 给出 `L∞/L2/L1` 的对偶范数公式，以及 `L0(k, eps)`（需明确采用 `||δ||_0≤k 且 ||δ||_∞≤eps`）对应的 `topk` 公式。
  - 说明与现有 interval IBP 的最小侵入式落点：泛化 `LinfInputSpec → InputSpec`，并将“输入扰动→第一层 affine”显式化；conv2d 第一版可先 sound 降级到 box，再迭代 tighten。
  - 补充测试/评测建议与论文一句话表述。

## 验证

- 文档变更（无额外运行时验证）。

