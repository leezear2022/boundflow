# 变更记录：Phase 6B（CROWN-IBP MLP）补强——L1 测试 + brute-force + multi-spec 入口

## 动机

在 Phase 6B 的最小 CROWN-IBP（MLP: Linear+ReLU）闭环跑通后，需要把“reviewer-proof”的关键点用测试钉牢，避免后续扩展 multi-spec / α 优化时引入 silent bug：

- `LpBall` 的对偶范数实现需要覆盖 `p=1`（此前测试只覆盖 `p=∞/2`）。
- backward ReLU relaxation 的“按系数符号选择 upper/lower”非常容易写反，最好用小维度 brute-force 直接兜底。
- 下一步要做 multi-spec batching，需要先在 CROWN-IBP 执行器层提供一个最小入口（`linear_spec_C`），让 `A` 的 spec 维度自然扩展。

## 主要改动

- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec.l1(...)` 便捷构造（`LpBallPerturbation(p=1)`）。

- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., linear_spec_C=...)`：新增多目标入口（`C` 形状 `[B,S,O]`，对应 `A` 的 spec batch 维度），为后续 multi-spec batching 做准备。
  - 数值稳定性：ReLU upper secant 的 `denom clamp` 从常量 `1e-12` 改为 `torch.finfo(dtype).eps`（更适合混精度/不同 dtype）。
  - dtype 保护：在使用 `torch.finfo` 前显式检查 bounds 为浮点 dtype，避免非预期整型输入导致报错。
  - 形状易用性：`linear_spec_C` 额外支持 `[S,O]`（自动 broadcast 成 `[B,S,O]`）。
  - 结构约束显式化：对 MLP 的“严格链式拓扑”做 gate（避免跳连/分支导致 silent wrong），并要求 `output_value` 必须是最后一个 op 的输出。
  - 清理未使用变量：删除 `_relu_relax` 中未使用的 `zeros/ones`。

- 更新：`tests/test_phase6b_crown_ibp_mlp.py`
  - 新增 `L1` 采样 soundness 测试（覆盖 `p=1` 的 row-wise dual norm 路径）。
  - 新增小维度 `L∞` brute-force 网格测试，用于捕获 backward ReLU 符号选择写反等错误。
  - 新增 multi-spec “串行对拍”测试：一次性 `C:[B,S,O]` 与循环 `C[:,s:s+1,:]` 的结果必须一致。
  - 新增 `C:[S,O]` broadcast 测试：与显式 `C:[B,S,O]` 结果一致。
  - 新增“非链式拓扑”拒绝测试：`get_crown_ibp_mlp_stats(...).supported=False` 且 `run_crown_ibp_mlp` 抛出 `NotImplementedError`。

## 如何验证

```bash
python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py
```

## 备注

- `linear_spec_C` 目前仅提供入口与形状约束；下一步 multi-spec “真 batch”会在该接口基础上，把 forward IBP 的 pre-activation bounds 复用到所有 spec，并将 `A` 的 spec 维度贯穿 backward。
- 当前实现仍是 DeepPoly baseline（不稳定 ReLU 的下界取 0）；后续要做 α/β-CROWN 时可在 `_relu_relax` 上引入可优化的 `alpha` 参数以收紧下界。
