# 2026-07-01：Phase 7B PR-28 MPS custom Metal kernel feasibility

## 背景

PR-26 将 MPS 性能问题指向 materialization / dispatch / small-kernel overhead。PR-28 先做最小 custom Metal feasibility gate，验证 PyTorch 2.12 的 `torch.mps.compile_shader()` 是否能在本项目候选 hot path 上提供单 kernel 优势。

## 主要改动

- 新增：`scripts/report_mps_metal_kernel_feasibility.py`
  - 输出 schema：`mps_metal_kernel_feasibility.v1`。
  - 使用 `torch.mps.compile_shader()` 编译两个最小 Metal kernels：
    - `axpy`: `out = a * x + b`
    - `signed_weighted`: `out = x >= 0 ? x * pos_w : x * neg_w`
  - 对比 PyTorch MPS eager expression 与 custom Metal kernel。
  - 记录 p50 latency、speedup、`max_abs_diff`、allclose。
- 新增：`tests/test_mps_metal_kernel_feasibility.py`

## Evidence

命令：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_metal_kernel_feasibility.py \
  --sizes 4096,65536,1048576 \
  --warmup 5 \
  --iters 20 \
  > out/phase7b/phase7b_pr28_mps_metal_kernel_feasibility.json
```

结果：

| kernel | size | PyTorch MPS ms | custom Metal ms | speedup | max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| axpy | 4096 | 0.3082 | 0.2583 | 1.19x | 4.77e-7 |
| signed_weighted | 4096 | 0.3003 | 0.2827 | 1.06x | 0 |
| axpy | 65536 | 0.2522 | 0.2560 | 0.99x | 4.77e-7 |
| signed_weighted | 65536 | 0.2534 | 0.2468 | 1.03x | 0 |
| axpy | 1048576 | 0.2673 | 0.2614 | 1.02x | 4.77e-7 |
| signed_weighted | 1048576 | 0.3037 | 0.2575 | 1.18x | 0 |

Summary：

- allclose：true
- geomean speedup：`1.075x`
- best speedup：`1.194x`
- worst speedup：`0.985x`

## 结论

Custom Metal kernel 在候选 elementwise/signed-weighted hot path 上有温和收益，但收益不稳定，不能直接重写 runtime。下一步若继续 PR-28B，应选一个真实 BoundFlow hot path，例如 signed bias contribution 或 final concretization，做 end-to-end guarded lowering prototype。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_mps_metal_kernel_feasibility.py \
  tests/test_mps_dispatch_profile_report.py
```

结果：`3 passed`
