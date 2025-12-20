# 变更记录：Phase 5D PR#13E.2（MANIFEST 换行修复 + --no-check 结构稳定）

## 动机

在真实验收中发现两点会影响“artifact 可读性/契约稳定性”：

1) `MANIFEST.txt` 使用了字面量 `\\n` 而不是换行符，导致人类阅读体验差；  
2) bench 使用 `--no-check` 时，JSONL 的 `correctness` diff 字段不出现（结构不稳定），会增加下游处理分支与口径歧义风险。

本 PR 做两个小修复：保证 MANIFEST 可读、保证 `correctness` 字段结构在 `--no-check` 下仍稳定（值为 null）。

## 本次改动

### 1) 修复 MANIFEST.txt 换行符

- 修改：`scripts/postprocess_ablation_jsonl.py`
  - `MANIFEST.txt` 写入从 `"\\\\n".join(...)` 改为 `\"\\n\".join(...)`，并确保文件末尾以换行结束。

### 2) `--no-check` 下 correctness 结构稳定（diff keys 输出 null）

- 修改：`scripts/bench_ablation_matrix.py`
  - `correctness` 字典初始化时包含所有 diff keys（`python_vs_{tvm,auto_lirpa}_max_{abs,rel}_diff_{lb,ub}`），默认值为 null
  - 当开启 correctness 检查时再填充值

### 3) 回归测试

- 修改：`tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 断言 `MANIFEST.txt` 不包含字面量 `\\n`
- 修改：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 新增：`--no-check` 场景下 diff keys 仍存在且为 null

### 4) 文档补充

- 更新：`docs/bench_jsonl_schema.md`
  - 说明 `--no-check` 时 diff 字段为 null（结构稳定），后处理仍按 missing 计数处理。

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py
```

