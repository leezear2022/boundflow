# 变更记录：PR#13A 增强 compile cache 公平性字段（keyset digest）+ 主表补列

## 动机

矩阵消融中，`compile_first_run_ms`/compile 开销容易被质疑“是否因为 cache 命中导致不可比”。虽然已经记录了 `compile_cache_stats` 与 `compile_cache_stats_delta_compile_first_run`，但仍缺少“本次 run 里到底编译了哪些 task”的可比性指纹。

本次补充 `compile_keyset_digest/size`，用于在不输出大体量 compile_stats 明细的前提下，给每个配置点一个稳定的 task 集合摘要，便于后处理/审计与论文解释。

## 本次改动

- 更新：`scripts/bench_ablation_matrix.py`
  - 在 `tvm` 字段新增：
    - `compile_cache_tag`：镜像 `TVMExecutorOptions.compile_cache_tag`（便于后处理直接引用）。
    - `compile_keyset_size`：本次 run 编译过的 unique task key 数量。
    - `compile_keyset_digest`：对排序后的 task key 集合做 sha256 的短摘要（16 hex）。
- 更新：`scripts/postprocess_ablation_jsonl.py`
  - `ablation.csv` 扁平化新增：`compile_cache_tag/compile_keyset_size/compile_keyset_digest`。
  - `tables/table_main.csv` 增加三列均值：`compile_cache_*_delta_first_run_mean`、`compile_keyset_size_mean`，避免主表缺少“cache 可比性”证据链。
- 更新：`docs/bench_jsonl_schema.md`
  - 增加 `tvm.compile_keyset_*` 与 `tvm.compile_cache_tag` 的 schema 说明。
- 更新：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 在 TVM 可用时，contract test 额外断言 `compile_cache_tag/compile_keyset_size/compile_keyset_digest` 存在且类型正确。

## 如何验证

- 无 TVM 环境（本机可跑）：`python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py::test_pr13d_bench_jsonl_schema_contract_no_tvm_mode_still_writes_rows`
- 后处理回归：`python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_postprocess_enum_normalization.py`

