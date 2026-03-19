# 变更记录：Phase 6H PR-5（不动语义）——扩展 E2E bench workload suite（小型非 toy MLP）

## 动机

Phase 6H 的主线已经具备 AE/论文交付形态（JSONL/CSV/fig + 一键 runner + schema/meta 固化）。下一步最划算的增强是：在不改 runtime 语义的前提下，把 `--workload` 从 `1d_relu` 扩展到 1–2 个 **小型但非 toy** 的 MLP case，使 time-to-verify 的图表更像论文主结果。

## 本次改动

- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 新增 `--workload` 选项：
    - `mlp2d_2x16`：输入 2D，隐藏层 `[16,16]`，输出 4D
    - `mlp3d_3x32`：输入 3D，隐藏层 `[32,32,32]`，输出 6D
  - 新增 `_make_chain_mlp(...)`：生成链式 MLP（Linear+ReLU）`BFTaskModule`，权重由 `seed` 控制，确保可复现。

- 新增：`tests/test_phase6h_workload_suite_smoke.py`
  - 对 `1d_relu/3dir_l2/mlp2d_2x16/mlp3d_3x32` 做最小 smoke（单组合开关），验证 JSON schema 能生成并可运行。

## 如何验证

```bash
python -m pytest -q tests/test_phase6h_workload_suite_smoke.py

# 示例：跑一个非 toy workload
python scripts/bench_phase6h_bab_e2e_time_to_verify.py \
  --device cpu --dtype float32 --workload mlp2d_2x16 \
  --oracle alpha_beta --steps 0 --max-nodes 256 --node-batch-size 32 \
  --warmup 1 --iters 3
```

## 备注

- 本 PR 只扩展 bench workload，不改 BaB/oracle 的求界语义与 solver 行为。
- `comparable/note` 机制仍然适用：若 batch/serial verdict 不一致，speedup 仅作为参考值。

