# 变更记录：Phase 6D（α-CROWN MLP）补强——修复 α 梯度链路 + objective 权重 + 梯度回归测试

## 动机

Phase 6D 引入了 `relu_alpha` 与 `run_alpha_crown_mlp` 的 K-step 优化循环。为确保 “α 真的能被 autograd 优化”，需要避免 ReLU backward 中的索引赋值断梯度，同时补齐一个专门的“梯度存在性”回归测试，防止后续重构引入 silent bug。

此外，`objective="both"` 与 `objective="gap"` 在默认权重下等价；为避免语义歧义，补充 `lb_weight/ub_weight` 以支持加权目标（不改变默认行为）。

## 本次改动

- 更新：`boundflow/runtime/crown_ibp.py`
  - `relu_alpha` 注入从 `alpha_l[amb] = alpha_broadcast[amb]` 改为 `alpha_l = torch.where(amb, alpha_broadcast, alpha_l)`，保持对 `alpha_broadcast` 的可微依赖，确保梯度能回传到 α。

- 更新：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp` 新增 `lb_weight/ub_weight`（默认 1.0/1.0），用于 `objective="both"` 的加权形式：`lb_weight*lb - ub_weight*ub`。

- 更新：`tests/test_phase6d_alpha_crown_mlp.py`
  - 新增 `test_phase6d_alpha_crown_relu_alpha_has_gradient`：直接对 `relu_alpha` 做 `loss=-lb.mean()` 反传，断言 `alpha.grad` 非空且非零。
  - 额外断言 `alpha.grad` 为 finite（防止 NaN/Inf 梯度的 silent 回归）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py
```
