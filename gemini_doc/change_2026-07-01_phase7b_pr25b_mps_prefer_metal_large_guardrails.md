# 2026-07-01：Phase 7B PR-25B MPS prefer-metal larger-scale sweep + guardrails

## 背景

PR-25A 在 `smoke` scale 上发现 `PYTORCH_MPS_PREFER_METAL=1` 有小幅正收益，但该结果不足以进入 Mac planner 或默认 aggressive lane。PR-25B 将测试扩大到 `small,bench`，并补充 default vs prefer-metal 的 correctness guardrails。

## 主要改动

- 新增：`scripts/report_mps_prefer_metal_guardrails.py`
  - 父进程不 import torch，通过子进程分别运行 default 与 `PYTORCH_MPS_PREFER_METAL=1`。
  - 子进程计算 structured bounds、operator attribution、planner decision、certified-decision summary。
  - 父进程比较：
    - bounds `allclose`
    - certified decision match
    - fallback disabled
    - `unknown_materialization == 0`
  - 输出 schema：`mps_prefer_metal_guardrails.v1`。
- 新增：`tests/test_mps_prefer_metal_guardrails.py`

## Evidence

### Larger-scale sweep

命令：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_env_var_sweep.py \
  --cases default,prefer_metal \
  --workloads all \
  --scales small,bench \
  --policies structured,dense_barrier,auto \
  --warmup 5 \
  --iters 20 \
  > out/phase7b/phase7b_pr25b_mps_prefer_metal_small_bench.json
```

结果：

| case | rows | structured abs gain vs default | baseline abs gain vs default | structured/dense geomean |
| --- | ---: | ---: | ---: | ---: |
| default | 24 | 1.000x | 1.000x | 0.642x |
| prefer_metal | 24 | 0.996x | 1.003x | 0.637x |

结论：`prefer_metal` 在 `small,bench` 上没有稳定收益，structured geomean 约 `-0.4%`，baseline 仅约 `+0.3%`，低于 promotion 门槛，不进入 planner / 默认配置。

### Correctness guardrails

命令：

```bash
conda run --no-capture-output -n boundflow-mps-aggressive \
  python scripts/report_mps_prefer_metal_guardrails.py \
  --workloads all \
  --scales small,bench \
  --policies structured,dense_barrier,auto \
  > out/phase7b/phase7b_pr25b_mps_prefer_metal_guardrails.json
```

结果：

- `ok=24`
- `fail=0`
- fallback disabled
- `unknown_materialization == 0`
- certified decision match 全部通过
- `max_abs_diff=1536.0`，但在大数值 bound 下仍满足 `rtol=1e-4, atol=1e-4`

## 结论

`PYTORCH_MPS_PREFER_METAL=1` 可以保留为 Mac aggressive evidence-only 开关，但当前不推荐作为默认配置。下一步性能方向不应继续从 env vars 榨收益，而应转向 dispatch attribution / MPS profiler / selective Metal kernel feasibility。

## 验证

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_mps_prefer_metal_guardrails.py \
  tests/test_mps_env_var_sweep_report.py
```

结果：`2 passed`
