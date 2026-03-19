# 变更记录：Phase 6G PR-1（αβ oracle）——multi-spec 真 batch 回归 + `spec_reduce` 口径固化

## 动机

Phase 6F PR-2 已把 split 约束从 patch/假设变成了可运行的 β 编码，并用 BaB 回归钉死了 “complete 的支点回到 β”。
Phase 6G 的目标转向“**不动语义，只做系统化收益**”，其中最先做、最不容易返工的一步是：

- 把 αβ oracle 路径在 `C:[B,S,O]` 下的 **multi-spec 真 batch** 语义用测试钉牢（对齐 Phase 6C 的收益叙事）。
- 固化 “multi-spec 的优化目标到底是平均还是最坏” 的口径：引入 `spec_reduce`，默认保持现状（mean），并为后续 verification 场景预留 `min/softmin`。

## 本次改动

### 1) 固化优化目标口径：`spec_reduce`

- 更新：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(..., spec_reduce={"mean","min","softmin"}, soft_tau=...)`
  - 说明：
    - `spec_reduce="mean"`：保持历史默认（对所有 `[B,S]` 元素做 mean）
    - `spec_reduce="min"`：lower 用 `min`（worst spec），upper/gap 用 `max`
    - `spec_reduce="softmin"`：lower 用 softmin，upper/gap 用 softmax（`soft_tau` 控制平滑）

- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp(..., spec_reduce=..., soft_tau=...)` 同步支持。
  - 对 first-layer infeasible 检测增加 `m==1` 的快速路径（避免无谓迭代）。

### 2) αβ oracle 的 multi-spec 真 batch 回归

- 新增：`tests/test_phase6g_alpha_beta_multispec_batch.py`
  - `multi-spec batch vs serial` 一致性（`steps=0` 口径，确保 batch 化不会引入语义差异）。
  - `forward 复用计数`：S=1 vs S=32 的 forward transformer 调用次数一致（证明 forward IBP 不随 spec 增长）。
  - `multi-spec 梯度链路`：在 `run_crown_ibp_mlp` 上对 `[B,S]` 的 loss 反传，断言 α/β 梯度 non-empty/finite/non-zero（防 batch 化时断梯度）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py
python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py
```

## 备注

- 本 PR 只钉 “oracle 在 spec 维度的真 batch 语义与目标口径”，不引入 BaB 的 node-batch（6G PR-2），也不引入缓存（6G PR-3）。
- `spec_reduce` 不影响 bound 的 soundness，只影响优化 tightness 与“多 spec 聚合目标”的口径；默认 `mean` 保持 backward-compatible。

