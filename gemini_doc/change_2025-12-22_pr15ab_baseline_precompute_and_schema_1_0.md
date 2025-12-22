# 变更记录：PR#15A/15B（baseline 外提预计算 + schema_version 冻结为 1.0）

## 动机

1) auto_LiRPA baseline 不依赖矩阵旋钮（partition/reuse/static_plan/fusion）。若在每个矩阵点内触发 baseline，容易带来：

- 不必要的重复开销（即使有 cache，也会重复走点内路径与条件分支）；
- 更复杂的边界条件（cache_hit/miss 行为与矩阵点顺序耦合）。

因此将 baseline 计算外提：在进入矩阵循环前预计算一次，再在每行 JSONL 中直接附加 baseline 字段。

2) Phase 5D 的 JSONL 字段与计时/分组口径已经稳定，后处理/contract tests 也已完善。为避免后续 Phase 6 扩展时“口径被冲掉”，将 `schema_version` 从 `0.1` 冻结升级为 `1.0`。

## 本次改动

### 1) baseline 外提预计算（矩阵循环外）

- 修改：`scripts/bench_ablation_matrix.py`
  - 在进入矩阵循环前，按 `(workload,input_shape,eps,method,warmup,iters,device,dtype,spec)` 预计算一次 auto_LiRPA baseline（best-effort）。
  - 矩阵点内不再触发 baseline 计算；每行 JSONL 直接附加 baseline payload（并将 `cache_hit=True` 标记为“复用预计算结果”）。

### 2) schema_version 冻结为 1.0

- 修改：`scripts/bench_ablation_matrix.py`
  - `schema_version`：`0.1` → `1.0`
- 修改：`docs/bench_jsonl_schema.md`
  - 更新当前版本号为 `1.0`
  - 在“版本演进”中注明 `1.0` 为 Phase 5D 冻结口径
- 修改：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 适配 `schema_version=1.0`
- 修改：`tests/test_phase5d_pr13e_postprocess_jsonl.py`、`tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`
  - 合成 JSONL 样例中的 `schema_version` 同步为 `1.0`（不改变测试语义，仅避免口径歧义）。

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_phase5d_pr14d_postprocess_baseline_dedup.py
```

## 备注

- baseline 预计算是 best-effort：若 auto_LiRPA 不可用（import/compute 失败），会保持原有“baseline 不可用”的输出语义，不影响矩阵点执行。
- `schema_version=1.0` 表示 Phase 5D 的字段集合与口径冻结；后续新增字段应 bump 版本并保持 postprocess 向后兼容。

