# Phase 7A PR-14 之后的下一步计划

**更新时间**: 2026-04-16
**当前状态**: PR-11 到 PR-14 已把 shared CROWN 的 benchmark、热点观测、ReLU pullback 接口与 `RightMatmul` 专用实现补齐。当前 ReLU workload 上的 `split_pos_neg_dense` hotspot 已清零；`permute_reshape_linear` 对 dense layout barrier 约有 `1.11x` 收益；但 ReLU 路径相对 dense ReLU barrier 仍约为 `0.53x~0.68x`。

---

## 目标

把 shared CROWN 主线从“旧 split hotspot 已清掉、性能瓶颈已经可见”推进到“ReLU path 的剩余 dense materialization 成本被量化并开始下降”的下一阶段。

## 建议优先级

### 1. 压缩 `relu_relax_pullback()` 内部的 dense materialization 成本

- 重点检查：
  - `RightMatmulLinearOperator.relu_relax_pullback()`
  - `SliceInputLinearOperator.relu_relax_pullback()`
- 先回答两个问题：
  - 哪些 `to_dense()` / broadcast / add 包装仍是主要开销来源；
  - 这些开销里哪些是 exact contract 真正要求的，哪些只是当前实现上的重复物化。
- 目标不是放松 `split_pos_neg()` 的 exact 语义，而是在 ReLU 专用 pullback 路径内减少重复 dense materialize、重复广播和无价值包装。

### 2. 继续补 observability，让性能解释能落到 operator 级别

- 在现有 `scripts/bench_phase7a_shared_crown_path_attribution.py` 基础上继续补最小必要观测：
  - `relu_relax_pullback()` 命中次数
  - 可能的 `to_dense()` / materialize 次数或尺寸统计
  - 如有必要，再补更细粒度 timing breakdown
- 保持现有 4 个 workload 不变，继续用同一口径复跑 CPU smoke 与 CUDA bench，避免换 workload 导致结论漂移。

### 3. 补回归测试，锁定 PR-14 之后的 contract

- 持续锁定三个 ReLU workload 的：
  - `split_pos_neg_dense_total == 0`
  - `split_pos_neg_dense_by_op == {}`
- 若后续对 `relu_relax_pullback()` 继续做 operator-specific 优化，增加针对 `RightMatmul` / `SliceInput` 的 exactness 回归测试。
- 如果 bench stdout JSON 再扩字段，同步补 schema/contract test，避免 observability 口径悄悄漂移。

### 4. 中期方向：继续做 ReLU 专用路径优化，而不是回头改 `split_pos_neg()` contract

- 如果 `RightMatmul` 仍是主要成本源，优先考虑：
  - 更局部的 operator-specific pullback 表示；
  - 可证明 sound 的中间表示；
  - 减少一次 backward 内重复 materialize 的缓存/复用策略。
- 不建议回退到“看起来像是 exact 的四项 sign split”，因为 PR-12 已经确认那条路会破坏逐元素 exact contract。

## 非目标

- 不放宽或重写 `split_pos_neg()` 的 exact contract。
- 不在下一步里把 CROWN/BaB lowering 直接下沉到 TVM。
- 不顺手扩更广的 ONNX `reshape` 语义。
- 不把 first-layer infeasible detector 等无关 dense 点混入这条主线。

## 验收标准

1. 能用同一套 benchmark/观测口径解释清楚：PR-14 后 ReLU path 为什么仍慢于 dense barrier。
2. 至少一项 `relu_relax_pullback()` 内部成本被实质削减，或 benchmark 结果显示 ReLU workload 继续回升。
3. `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py tests/test_phase7a_pr9_dag_linear_operator.py tests/test_phase7a_pr11_shared_crown_bench.py` 持续全绿。
