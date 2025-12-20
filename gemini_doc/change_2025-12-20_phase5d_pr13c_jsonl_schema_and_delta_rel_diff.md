# 变更记录：Phase 5D PR#13C（JSONL schema 文档 + schema_version/time_utc + cache delta + rel diff）

## 动机

PR#13A/13B 已经把 ablation matrix bench 的“统一入口 + JSONL 输出 + 计时公平性/可解释性”拉起来了。

为了让后续扩展（更多 workload / 更多旋钮 / 论文画图）不反复返工，本 PR 再补两类“硬钉子”：

1) **schema 固化**：提供一份可读的 JSONL 字段说明，明确计时口径与字段语义；  
2) **去歧义字段**：引入 schema 版本与 UTC 时间，并补齐 `cache delta` 与 `max_rel_diff`，避免累计统计/量级差异导致的误读。

## 本次改动

### 1) bench JSONL：增加 schema_version/time_utc + cache delta + max_rel_diff

- 修改：`scripts/bench_ablation_matrix.py`
  - 顶层增加 `schema_version`（当前 `0.1`）
  - `meta` 增加 `time_utc`（ISO-8601 UTC）
  - `tvm` 增加 `compile_cache_stats_delta_compile_first_run`：围绕首次运行（触发编译）阶段的 cache hit/miss/fail 增量，避免累计值歧义
  - `correctness` 增加 `*_max_rel_diff_*`：相对误差（分母 `abs(ref).clamp_min(1e-12)`），便于跨量级对照

### 2) 新增：bench JSONL schema 文档（字段→口径→用途）

- 新增：`docs/bench_jsonl_schema.md`
  - 固定 `bench_ablation_matrix.py` 的 schema，明确：
    - stdout 纯 JSONL payload（提示走 stderr）
    - `compile_first_run_ms` vs steady-state `run_ms_*` 的计时口径
    - compile cache 统计与 delta 的语义
    - correctness 的 abs/rel diff 含义

### 3) 文档索引注册

- 修改：`AGENTS.md`
  - 在“关键文档索引”中新增 `docs/bench_jsonl_schema.md`

## 如何验证

```bash
# smoke test 仍应通过（schema 顶层字段变更不会破坏 JSONL）
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py

# bench 产出应为纯 JSONL
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py \
  --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check \
  --output /tmp/boundflow_ablation.jsonl
python -c 'import json, pathlib; p=pathlib.Path(\"/tmp/boundflow_ablation.jsonl\"); json.loads(p.read_text().splitlines()[0]); print(\"ok\")'
```

