# PR-13 Reduced Artifact Appendix（2026-07-14）

## Scope

该工件复现 PR-13A–D 的 query contract、动态 batching、same-solver correctness 与 RTX 4060
reduced fixed/E2E 结果。它不复现 VNN-COMP full evaluation，也不证明 αβ/split 使用 PR-12
compiled Planner。

## 环境

- `conda activate boundflow`（activation hook 自动加载 `env.sh`）；
- Python 3.12.13、PyTorch 2.12.1+cu132；
- CUDA GPU；权威 D 工件使用 RTX 4060 Laptop GPU；
- 运行前建议：`pytest -q tests/test_env.py`。

## Correctness workflow

```bash
pytest -q \
  tests/test_phase7a_pr13a_query_contract.py \
  tests/test_phase7a_pr13a_fixed_replay.py \
  tests/test_phase7a_pr13b_dynamic_batch_manager.py \
  tests/test_phase7a_pr13c_same_solver_adapter.py
```

无 CUDA 时 custom-stream case 合法 skip；GPU 机器上单独执行：

```bash
pytest -q \
  tests/test_phase7a_pr13c_same_solver_adapter.py::test_pr13c_runtime_obeys_non_default_torch_stream
```

Expected：CPU/default sandbox 为 `14 passed, 1 skipped`；CUDA stream case 为 `1 passed`。

## A–C 生成命令

```bash
python scripts/run_phase7a_pr13a_fixed_replay.py \
  --out-dir artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714
python scripts/run_phase7a_pr13b_dynamic_batch_replay.py \
  --out-dir artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714
python scripts/run_phase7a_pr13c_same_solver.py \
  --out-dir artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714
```

Expected：A 8/8 replay；B dynamic/fault 均 8/8、0 loss；C 7/7 same-solver query/state、
solver counters 一致。

## D GPU reduced 命令

```bash
python scripts/benchmark_phase7a_pr13d_bab_runtime.py \
  --out-dir artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714 \
  --device cuda --max-nodes 16 --alpha-steps 1 --batch-size 8 \
  --memory-budget-bytes 1073741824 --warmup 1 --repeats 5
```

Expected files：`raw.jsonl`、`summary.json`、`manifest.json`。Expected semantic fields：

```text
summary.status = ok
summary.correctness_failures = 0
summary.unstable_e2e_rows = 0
fixed runtime_speedup_vs_per_node ≈ 96.52
hard E2E runtime_speedup_vs_per_node ≈ 9.93
hard E2E runtime_speedup_vs_batched_original ≈ 0.980
research_gate.non_toy_workload = false
closure_recommendation = VALIDATED-REDUCED
```

性能数字受 GPU/driver/负载影响；复现门禁优先检查 correctness、status/node count、query loss
和趋势，不要求跨机器逐位相同。

## Evidence chain

```text
command/config
  → raw JSONL
  → summary comparison/gates
  → manifest SHA/environment
  → PR-13 closure audit / Claims Map
```

失败、负收益和不适用项不应删除：easy root runtime 负收益、`compiled_plan_cache_applicable=false`
与 `pr12_planner_dispatches=0` 都是 closure 的必要证据。
