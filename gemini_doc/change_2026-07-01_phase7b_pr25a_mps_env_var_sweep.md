# 2026-07-01：Phase 7B PR-25A MPS env-var sweep

## 背景

PR-23 建立了 Mac MPS aggressive lane，并在 `torch==2.12.1` 上验证了四个 Phase 7B workload 的 MPS coverage。初步 smoke 显示新 PyTorch MPS 带来明显绝对耗时下降，但 structured path 相对 dense baseline 仍未翻正。

本轮 PR-25A 继续做 Mac 探索，不改变 CPU planner，也不把任何 MPS 规则直接提升为默认策略。目标是系统性比较 PyTorch MPS 环境变量：

- default
- `PYTORCH_MPS_PREFER_METAL=1`
- `PYTORCH_MPS_FAST_MATH=1`
- both enabled

## 主要改动

- 新增：`scripts/report_mps_env_var_sweep.py`
  - 输出 schema：`mps_env_var_sweep.v1`。
  - 用子进程调用 `scripts/bench_phase7b_crossover_matrix.py`，确保 MPS env vars 在 `import torch` 之前生效。
  - 默认清除 `PYTORCH_ENABLE_MPS_FALLBACK`，防止 silent CPU fallback 污染 MPS 性能结论。
  - 支持 `--set-kmp-duplicate-lib-ok` 和 `--omp-num-threads`，用于 aggressive env 中规避当前 OpenMP duplicate runtime abort。
  - 支持 `--dry-run`，用于无 MPS 依赖的 schema / env case 测试。
  - 汇总每个 case 相对 default 的 structured / dense baseline 绝对耗时 geomean gain。
- 更新：`scripts/bench_phase7a_shared_crown_path_attribution.py`
  - `device_meta` 新增：
    - `mps_prefer_metal_env`
    - `mps_fast_math_env`
- 新增：`tests/test_mps_env_var_sweep_report.py`
  - 验证四个 env case 的 dry-run schema。
  - 验证 fallback 默认被清除。
  - 验证 KMP / OMP workaround 能进入子进程 env。

## 本机 smoke evidence

运行环境：

- conda env：`boundflow-mps-aggressive`
- torch：`2.12.1`
- device：`mps`
- dtype：`float32`
- workload：all
- scale：`smoke`
- policies：`structured,dense_barrier,auto`
- warmup / iters：`3 / 5`
- fallback：disabled

命令：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_env_var_sweep.py \
  --cases all \
  --workloads all \
  --scales smoke \
  --policies structured,dense_barrier,auto \
  --warmup 3 \
  --iters 5 \
  > out/phase7b/phase7b_pr25a_mps_env_var_sweep_torch212_smoke_w3i5.json
```

结果摘要：

| case | rows | structured abs gain vs default | baseline abs gain vs default | structured/dense geomean |
| --- | ---: | ---: | ---: | ---: |
| default | 12 | 1.000x | 1.000x | 0.604x |
| prefer_metal | 12 | 1.021x | 1.020x | 0.605x |
| fast_math | 12 | 1.011x | 1.012x | 0.603x |
| both | 12 | 1.026x | 1.029x | 0.602x |

观察：

- `PYTORCH_MPS_PREFER_METAL=1` 是本轮较稳的小幅正收益，structured 约 `+2.1%`，dense baseline 约 `+2.0%`。
- `PYTORCH_MPS_FAST_MATH=1` 单独使用收益更小；考虑到 fast math 可能影响数值细节，不应直接进入默认 lane。
- 所有 case 的 `mps_fallback_enabled=false`，没有 silent CPU fallback。
- MPS env-var sweep 没有改变 structured 相对 dense baseline 的大方向：structured path 仍慢于 dense baseline。

## 结论

PR-25A 不提升 planner 规则。`prefer_metal` 可以作为 Mac aggressive lane 的候选配置继续跑 `small/bench`，但还需要：

1. aggressive env 的 OpenMP duplicate runtime 冲突已在 PR-25C 通过 pip-only 数值栈解决。
2. 加入 correctness side check，尤其是 `FAST_MATH` 的 bound allclose / certified decision match。
3. 在 larger scale 上确认是否存在 crossover；若仍无 crossover，PR-26/PR-28 应转向 MPS profiler、dense final concretization policy 或 selective Metal kernel feasibility。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_mps_env_var_sweep_report.py \
  tests/test_phase7b_crossover_matrix.py
```

结果：`2 passed in 0.86s`

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_env_var_sweep.py \
  --cases all \
  --workloads all \
  --scales smoke \
  --policies structured,dense_barrier,auto \
  --warmup 3 \
  --iters 5
```

结果：`ok=4`、`fail=0`、`dry_run=0`。
