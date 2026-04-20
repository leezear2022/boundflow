# Phase 7A PR-10 之后的下一步计划

**更新时间**: 2026-03-29
**当前状态**: PR-10（structured ReLU backward）和 shared CROWN layout-only support（`reshape` / batch-preserving `permute` / `transpose`）已经合入本地并通过 `pytest tests/`

---

## 目标

把当前“正确性已闭合”的 shared CROWN 路径推进到“性能收益可量化、剩余 dense 点继续减少”的下一阶段。

## 建议优先级

### 1. 性能记录与收益归因

- 针对 ReLU-heavy case、residual/concat case、`permute -> reshape -> linear` case 记录：
  - 端到端耗时
  - 显存峰值或关键中间张量物化情况
  - 与 PR-9 / PR-10 前的对比
- 产出一份短文档，明确“structured ReLU backward”和“layout-only shared CROWN”分别带来的收益。

### 2. 继续消除 sign-split 的剩余 dense 点

- 重点检查仍走 `_split_pos_neg_dense(...)` 的复合 operator。
- 首要对象：
  - `RightMatmulLinearOperator`
  - 其他会在 ReLU 前主路径上高频出现的复合 operator
- 目标不是一次性设计更大的 operator algebra，而是优先消除热路径上最值钱的 dense materialization。

### 3. 补性能/结构测试

- 新增或扩展测试，显式锁定：
  - 某些 shared CROWN 热路径不再回退成普通 `DenseLinearOperator`
  - `split_pos_neg()` 的结构化实现不会破坏 `relu_alpha` / `beta` 梯度

## 非目标

- 不在下一步里扩展更广的 ONNX `reshape` 语义。
- 不在下一步里把 CROWN/BaB lowering 直接下沉到 TVM。
- 不顺手处理 first-layer infeasible detector 的独立 dense 点。

## 验收标准

1. 至少有一组 benchmark 能量化 PR-10 和 layout-only support 的收益。
2. 至少消掉一个 shared CROWN 热路径上的 `split_pos_neg()` dense 点。
3. `conda run -n boundflow python -m pytest -q tests/` 持续全绿。
