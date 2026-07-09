# 2026-07-01：Phase 7B PR-26 MPS dispatch/profile report

## 背景

PR-25B 显示 MPS env vars 不能提供足够强的 larger-scale 收益。下一步需要解释 structured path 在 MPS 上为什么仍慢于 dense baseline，重点看 materialization、cache、fallback、wrapper/dispatch 相关信号。

## 主要改动

- 新增：`scripts/report_mps_dispatch_profile.py`
  - 输出 schema：`mps_dispatch_profile.v1`。
  - 复用 Phase 7B `_collect_row()`，保留 timing、planner decision 和 operator attribution。
  - 额外整理 dispatch payload：
    - materialization total calls / bytes
    - by op / by reason / by phase
    - ReLU pullback by op
    - fallback by reason
    - dense cache hits/misses
  - 支持 `--with-mps-signposts`，用 `torch.mps.profiler.profile()` 包裹运行，便于 Xcode Instruments 采样。
  - 支持 `--with-torch-profiler`，输出 CPU dispatch/top-op 文本表。
- 新增：`tests/test_mps_dispatch_profile_report.py`

## Evidence

命令：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_dispatch_profile.py \
  --device mps \
  --workloads all \
  --scales smoke \
  --policy auto \
  --warmup 1 \
  --iters 3 \
  > out/phase7b/phase7b_pr26_mps_dispatch_profile_smoke.json
```

结果摘要：

- rows：4
- `unknown_materialization_total=0`
- `materialization_bytes_total=591872`
- `cache_hits_total=4`
- `cache_misses_total=48`

逐 workload：

| workload | structured ms | baseline ms | cache hits | cache misses | materialized bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| relu_heavy_mlp | 22.364 | 15.025 | 0 | 14 | 212992 |
| residual_relu_mlp | 15.290 | 10.666 | 2 | 12 | 147456 |
| concat_relu_mlp | 24.261 | 14.700 | 2 | 20 | 198656 |
| permute_reshape_linear | 1.704 | 0.877 | 0 | 2 | 32768 |

## 结论

MPS structured path 的主要问题不是 unknown fallback，而是仍有较多 materialization/cache miss 与 dispatch 级别开销。PR-27/PR-28 应优先考虑 final concretization / signed elementwise reduction / selective Metal kernel feasibility，而不是继续调整 MPS env vars。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_mps_dispatch_profile_report.py \
  tests/test_mps_aggressive_env_health_report.py
```

结果：`3 passed`
