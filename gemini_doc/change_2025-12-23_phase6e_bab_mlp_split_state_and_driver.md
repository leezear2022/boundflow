# 变更记录：Phase 6E（BaB MLP）起步——split state + α-CROWN bound oracle + priority-queue driver

## 动机

Phase 6D 已经把 α 的梯度链路与优化回归钉死，可以把 α-CROWN 作为一个“可复用的 bound oracle”。Phase 6E 进入 complete verification 的最小闭环：把 per-neuron split 约束变成可传递的运行时状态，并将 bound oracle 放进 BaB（Branch-and-Bound）控制流。

本次只追求 **最小可验证闭环（MLP 链式子集）**，为后续 β/αβ-CROWN 的 split 约束编码与 batching/caching 打地基。

## 本次改动

- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_split_state=...)`：forward IBP 记录 ReLU pre-activation bounds 时，按 split state 对 pre bounds 进行 best-effort 的区间收缩：
    - active（`x>=0`）：`l = max(l,0)`
    - inactive（`x<=0`）：`u = min(u,0)`
  - forward IBP 抽成 `_forward_ibp_trace_mlp(...)`，供 BaB 分支选择复用。

- 更新：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(..., relu_split_state=...)`：透传 split state 到 bound oracle，便于 BaB 节点评估与 warm-start 继承。

- 新增：`boundflow/runtime/bab.py`
  - `ReluSplitState`：`relu_input_name -> int{-1,0,+1}[H]` 的 split 约束表示（per-neuron）。
  - `solve_bab_mlp(...)`：最小 BaB driver（priority queue）：
    - 节点评估：`run_alpha_crown_mlp(..., warm_start=parent_best_alpha, relu_split_state=...)`
    - 分支选择：按 “最宽 ambiguous pre-activation interval” 选 neuron（`l<0<u` 且 `u-l` 最大）
    - prune：当 `lower >= threshold`（含容差）认为该节点已证明
    - leaf 判定：当无 ambiguous ReLU 时，若 `lower < threshold` 则返回 `unsafe`（存在违反的子域）
  - 1D Linf toy 的“完整性补丁”：
    - `_restrict_input_spec_linf_1d_for_first_layer_splits(...)`：当 split 约束对应 first-layer `w*x+b >=0/<=0` 时，将输入域从 `[x0-eps,x0+eps]` 交并得到新 interval，并用 1D Linf spec 表示（用于 toy complete 测试）。
    - 备注：这是为了 Phase 6E 的最小 complete proof 演示；更一般的 split 约束编码会在后续 β/αβ-CROWN 阶段替换为更系统的约束传播方式。

- 新增测试：`tests/test_phase6e_bab_mlp.py`
  - `test_phase6e_split_constraints_tighten_bounds`：在 1D toy 上用“显式缩小输入区间”验证 split 后 bounds 不变宽。
  - `test_phase6e_bab_proves_nonneg_with_suboptimal_alpha_init`：设置 `alpha_steps=0, alpha_init=0.5`（根节点 bound 偏松），BaB 通过 split 仍能证明 `ReLU(x) >= 0`（toy complete 闭环）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6e_bab_mlp.py
```

## 已知限制（TODO）

### 关于 “complete” 的口径（必须明确）

本 PR 的实现目标是把 **split state + α-CROWN bound oracle + BaB 控制流** 的工程分层跑通，并提供一个 **toy complete 演示**。
但需要强调：

- 对一般 ReLU 网络与一般 split 约束，仅靠 “区间收缩 +（α-）CROWN/ LiRPA bound propagation” 是 **sound** 的（不会把不满足的点误判为满足），但并 **不保证 complete**（可能无法在有限分支内证明/反证）。
- 当前实现中的 `_restrict_input_spec_linf_1d_for_first_layer_splits(...)` 是为了 1D Linf 的 toy case 提供“输入域收缩补丁”，用于演示 complete 闭环；它不是通用 split 约束编码。
- 通用 complete verifier 需要把 split 约束更系统地编码进 bound propagation（例如 β/αβ-CROWN 风格的 split constraint encoding），或退回到 LP/MIP 等更强但更慢的约束求解。
- 对照锚点：β-CROWN 的补充材料中给出了 “β-CROWN + BaB（针对 ReLU splitting）soundness & completeness” 的定理证明（Theorem 3.3）：https://proceedings.neurips.cc/paper_files/paper/2021/file/fac7fead96dafceaf80c1daffeae82a4-Supplemental.pdf

### 其它限制

- 当前 BaB 仍仅支持链式 MLP（Linear+ReLU）子集。
- split 约束的“输入域收缩”仅覆盖 1D Linf + first-layer 线性约束的 toy 情形；更一般的 split 约束编码需要 Phase 6F（β/αβ-CROWN）落地。
