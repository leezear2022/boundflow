# 变更记录：Phase 5D PR#13E.1（postprocess hardening：缺失值/分组/流式读取/enum 修复）

## 动机

在跑真实 bench（尤其是 `--no-check`）与后续扩大矩阵时，后处理脚本需要避免 4 类“静默口径错误”：

1) **correctness 缺失不能当 0**：否则 summary 会误读为“误差完美=0”；  
2) **分组 key 不能漏掉 eps/input_shape/domain/spec**：否则不同实验会被混组平均；  
3) **JSONL 必须流式读取**：大矩阵（大量行）不能一次性读入内存；  
4) **enum 解析正则必须正确**：避免归一化失败导致 memory_plan_mode 等字段悄悄变形。

## 本次改动

### 1) 流式读取 JSONL

- 修改：`scripts/postprocess_ablation_jsonl.py`
  - `_read_jsonl()` 改为按行迭代文件，避免 `read_text().splitlines()` 的内存峰值。

### 2) 修正 enum 解析正则

- 修改：`scripts/postprocess_ablation_jsonl.py`
  - `_ENUM_VALUE_RE` 从 `:\\\\s*` 修正为 `:\\s*`，确保能解析 `"<MemoryPlanMode.DEFAULT: 'default'>"` 形态的 repr。

### 3) 分组 key 补齐工作负载维度

- 修改：`scripts/postprocess_ablation_jsonl.py`
  - 扁平字段新增 `domain/spec`（来自 `workload`）
  - `_group_key()` 追加 `input_shape/eps/domain/spec`，避免混组。

### 4) correctness 缺失显式化（不再当 0）

- 修改：`scripts/postprocess_ablation_jsonl.py`
  - summary 计算 `python_vs_tvm_max_rel_diff_max` 时：
    - 缺失（None）不参与 `max()`；若全部缺失则输出为空
    - 新增 `python_vs_tvm_rel_diff_missing` 计数

### 5) 回归测试补齐

- 修改：`tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 增加：
    - 缺失 correctness 时 summary 不应输出 0，并记录 missing=1
    - group key 纳入 eps/input_shape 后应分成两组
- 新增：`tests/test_postprocess_enum_normalization.py`
  - 覆盖 enum repr 解析（value 级）

### 6) 文档补充

- 更新：`docs/bench_jsonl_schema.md`
  - 增加“缺失值约定”说明：`--no-check` 时 summary 不把缺失当 0，并输出 missing 计数。

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py
conda run -n boundflow python -m pytest -q tests/test_postprocess_enum_normalization.py
```

