# 变更记录：Phase 5D PR#13F（bench 支持 eps/batch 覆盖，用于分组验证与消融扩展）

## 动机

PR#13E.1 的 postprocess 已将 `eps/input_shape/domain/spec` 纳入 group key。为了能用**真实 bench 输出**验证“不同 eps/input_shape 不会被混组”，以及为后续扩展实验矩阵预留入口，本 PR 给 `bench_ablation_matrix.py` 增加两个最小旋钮：

- `--eps`：覆盖 `workload.eps`
- `--batch`：覆盖输入 batch size（影响 `workload.input_shape`）

## 本次改动

- 修改：`scripts/bench_ablation_matrix.py`
  - 新增 CLI 参数：
    - `--eps <float>`：覆盖 Linf eps
    - `--batch <int>`：覆盖输入 batch size（当前仅支持 `workload=mlp`）
  - `_bench_one()` 新增参数 `eps_override/batch` 并应用到输入与 eps

- 更新：`docs/bench_jsonl_schema.md`
  - 增加“Workload 参数化（用于分组验证）”小节，说明 `--eps/--batch` 的用途与限制。

## 如何验证

```bash
# schema contract test 应保持通过
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py

# 生成两条真实记录（eps/input_shape 不同）并确认 postprocess 不混组
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --eps 0.1 --batch 4 --output /tmp/c1.jsonl
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --eps 0.2 --batch 2 --output /tmp/c2.jsonl
cat /tmp/c1.jsonl /tmp/c2.jsonl > /tmp/c.jsonl
conda run -n boundflow python scripts/postprocess_ablation_jsonl.py /tmp/c.jsonl --out-dir /tmp/c_out --no-plots
head -n 6 /tmp/c_out/tables/ablation_summary.csv
```

