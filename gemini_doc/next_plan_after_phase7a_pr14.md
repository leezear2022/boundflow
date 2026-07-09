# Phase 7A PR-14 之后的下一步计划

**更新时间**: 2026-07-01
**当前状态**: PR-11 到 PR-14 已把 shared CROWN 的 benchmark、热点观测、ReLU pullback 接口与 `RightMatmul` 专用实现补齐。PR-15 补了 opt-in operator attribution，并把 `unknown_materialization` 清零。PR-16 增加 run-local dense cache；`concat_relu_mlp` 上能减少一部分 `RightMatmul` 重复物化，但 `relu_heavy_mlp` 仍以 cache miss 为主。PR-17 增加 final concretization policy，使 `structured` 与 `dense_barrier` 能在同一 benchmark 口径下比较。PR-18 增加 hybrid planner / capability table，并把 benchmark 的 `auto` policy 固化为 `relu_barrier -> structured`、`layout_only -> dense_barrier`。PR-19 新增 Phase 7B benchmark matrix / crossover study 脚本。PR-20 新增 cost model v1 后处理，把 matrix 数据转成带置信度和 guardrails 的离线规则证据。PR-21 完成正式 CPU matrix。PR-22 将 high-confidence CPU 规则推进 planner v2，目前只提升 `permute_reshape_linear small/bench -> structured`。PR-23 增加 Mac MPS benchmark support、aggressive/nightly env 定义与 MPS op coverage report；当前稳定环境 torch 2.5.1 的 MPS smoke 已跑通。

---

## 目标

把 shared CROWN 主线从“旧 split hotspot 已清掉、性能瓶颈已经可见”推进到“用 attribution + cache + policy + planner 建立 dense / structured / future lowering 的选择边界”。

## 建议优先级

### 1. Phase 7A 收口证据

- PR-15 已证明 attribution 可 side-effect free 地记录 op / shape / bytes / phase / reason。
- PR-16 已证明 run-local identity cache 语义安全，但只在部分路径上产生 hit。
- PR-17 已证明 final concretization 能在 structured / dense barrier 间显式切换且保持 exact bounds。
- PR-18 已把 capability table 与 auto planner 接入 benchmark。
- 继续保留 `split_pos_neg_dense_total == 0` 与 `unknown_materialization == 0` 作为回归约束。

### 2. 继续补 bench 证据，而不是继续手改 sign split

- 使用 `scripts/bench_phase7b_crossover_matrix.py` 保持现有 4 个 workload 不变，复跑：
  - `--final-concretization-policy structured`
  - `--final-concretization-policy dense_barrier`
  - `--final-concretization-policy auto`
- 或在 PR-19 matrix 口径下跑：
  - `--scales smoke,small,bench`
  - `--policies structured,dense_barrier,auto`
- 对比 `planner_decision`、materialization bytes、cache hit/miss 与 latency。
- 若 ReLU workload 仍由 `RightMatmul` exact sign split 主导，不继续伪 structured；将其作为 planner/cached dense 的证据。
- 若 layout-only workload 在 dense barrier 下稳定更快，可作为 selective lowering 的候选 hot path。

### 3. 补回归测试，锁定 PR-14 之后的 contract

- 持续锁定三个 ReLU workload 的：
  - `split_pos_neg_dense_total == 0`
  - `split_pos_neg_dense_by_op == {}`
- 若后续对 `relu_relax_pullback()` 继续做 operator-specific 优化，增加针对 `RightMatmul` / `SliceInput` 的 exactness 回归测试。
- 如果 bench stdout JSON 再扩字段，同步补 schema/contract test，避免 observability / planner 口径悄悄漂移。

### 4. Phase 7B / 7C 方向

- CPU 侧 PR-21/PR-22 已完成。
- CUDA 当前机器不可用；需要在有 CUDA 的机器上重复 PR-21 matrix。
- Mac MPS smoke 已可用；下一步应创建 `boundflow-mps-aggressive`，用 torch 2.12.1 跑 formal MPS matrix。
- 低置信度规则继续留在 evidence report，不推进 runtime planner。
- Phase 7B：继续扩大 workload scale sweep，寻找 structured / dense barrier 的 crossover。
- Phase 7C：只对已经稳定的 hot path 做 selective lowering，例如 layout-only pullback、final dense barrier 或 signed reduction。
- 如果 wrapper / Python dispatch 仍主导，转向 planner-level fusion 或 selective TVM lowering。
- 不建议回退到“看起来像是 exact 的四项 sign split”，因为 PR-12 已经确认那条路会破坏逐元素 exact contract。

## 非目标

- 不放宽或重写 `split_pos_neg()` 的 exact contract。
- 不在下一步里把 CROWN/BaB lowering 直接下沉到 TVM。
- 不顺手扩更广的 ONNX `reshape` 语义。
- 不把 first-layer infeasible detector 等无关 dense 点混入这条主线。

## 验收标准

1. 能用同一套 benchmark/观测口径解释清楚：PR-14 后 ReLU path 为什么仍慢于 dense barrier。
2. 能在 JSON 中看到 `operator_attribution`、`cache`、`final_concretization_policy`、`planner_decision` 与 `capability_table`。
3. `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr10_relu_barrier_structured.py tests/test_phase7a_pr9_dag_linear_operator.py tests/test_phase7a_pr11_shared_crown_bench.py` 持续全绿。
4. `--final-concretization-policy auto` 能稳定产出 `relu_barrier -> structured`、`layout_only -> dense_barrier` 的 planner 决策。
