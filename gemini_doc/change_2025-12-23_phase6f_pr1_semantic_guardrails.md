# 变更记录：Phase 6F PR-1 语义护栏补强——infeasible 三态化口径 + β 只放松 + 证书/测试可复核

## 动机

Phase 6F PR-1 的目标是“接口/可微/可剪枝优先”。但其中最容易破坏 soundness 的点是：

- 把 `feasible=False` 当成“猜测不可行”而误剪枝；
- β 的占位符逻辑若意外 tighten bounds，会引入 unsound；
- infeasible witness 若不可复核，回归时难定位。

本次补强将这些风险用 **语义护栏 + 回归断言**钉死，为 PR-2 的真实 β encoding 留出稳定接口。

## 本次改动

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `AlphaBetaCrownStats.feasibility: {"unknown","infeasible"}`：
    - `"infeasible"` 仅在 **可证明空域** 时返回；
    - `"unknown"` 表示未发现矛盾（不代表可行，不能用于 prune）。
  - β 占位符明确只放松 bounds（`lb -= penalty`, `ub += penalty`），并在 `reason` 里标注 PR-1 占位语义。
  - infeasible 情形增加 `infeasible_certificate`（type/opposite_pair + 证据字段），便于复核与调试。
  - 防止误剪枝：对 split shape 异常等输入错误改为抛异常，不再返回 `"infeasible"`。

- 更新：`boundflow/runtime/bab.py`
  - prune 条件改为 `stats.feasibility == "infeasible"`。

- 更新：`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - DoD-1 增加“只放松不 tighten”的断言：β 非零时 `lb<=alpha_only_lb` 且 `ub>=alpha_only_ub`。
  - DoD-2 断言 `infeasible_certificate` 非空（证书可复核）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py
```

