# 变更记录：PR#14C（Phase 5 对照）auto_LiRPA baseline 缓存 + init/cold/hot 计时 + correctness gate

## 动机

Phase 5 的矩阵消融需要第三方 baseline（auto_LiRPA）来支撑“正确性对齐 + 性能对比”的论文证据链。但 baseline 与 partition/reuse/static_plan/fusion 等旋钮无关，如果按矩阵点重复跑会：

- 将 16 点矩阵的 baseline 计算放大为 16 倍，浪费算力且拖慢 AE；
- 让 baseline 的计时口径与 BoundFlow 的 compile/cold/hot 不对齐，难以写进论文。

本次把 auto_LiRPA baseline 固化成：

- 进程内按 workload/shape/eps/method/warmup/iters 缓存（矩阵点复用）；
- 输出 init/cold/hot 计时字段（类比 TVM 的 compile vs run）；
- 输出 correctness gate（BoundFlow Python reference vs auto_LiRPA 的 bounds allclose + max diff）。

## 本次改动

- 更新：`scripts/bench_ablation_matrix.py`
  - 新增 auto_LiRPA baseline 的进程内缓存 `_AUTO_LIRPA_BASELINE_CACHE`：
    - key：`(workload,input_shape,eps,method,warmup,iters,device,dtype)`
    - value：`(jsonable_payload, lb, ub)`（lb/ub 仅用于 gate，不写入 JSONL）
  - baseline 字段升级为 init/cold/hot 口径：
    - `baseline.auto_lirpa.init_ms`
    - `baseline.auto_lirpa.run_ms_cold`
    - `baseline.auto_lirpa.run_ms_p50/p95`（warmup 后稳态）
    - `baseline.auto_lirpa.cache_hit`（是否复用缓存）
    - `available/reason/version/method/device`
  - 保留兼容字段映射（best-effort）：
    - `setup_ms ~= init_ms`
    - `compute_bounds_ms_* ~= run_ms_*`

- 更新：`scripts/postprocess_ablation_jsonl.py`
  - `ablation.csv` 扁平化新增：
    - `auto_lirpa_init_ms/auto_lirpa_run_ms_cold/auto_lirpa_run_ms_p50/auto_lirpa_run_ms_p95`
    - `auto_lirpa_cache_hit/auto_lirpa_reason/auto_lirpa_version/auto_lirpa_method`

- 更新：`docs/bench_jsonl_schema.md`
  - 增加 `baseline.auto_lirpa` 的字段说明与“baseline 缓存”约定。

- 更新：`scripts/run_phase5d_artifact.py`
  - 合并 JSONL 时强制补齐 `workload.model`（硬化：避免未来某条 workload 路径漏写导致 postprocess 混组）。

- 更新：`gemini_doc/artifact_claims_phase5d.md`
  - runner 示例命令推荐使用 `--workload all`（MLP+CNN 同口径产线）。

## 如何验证

- 无 TVM 环境（可跑）：
  - `python -m pytest -q tests/test_artifact_phase5d_smoke.py::test_phase5d_artifact_runner_allow_no_tvm_smoke`
  - `python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`

