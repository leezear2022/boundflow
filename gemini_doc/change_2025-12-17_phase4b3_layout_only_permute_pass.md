# 变更记录：Phase 4B.3（layout-only `permute` 简化 pass）

## 动机

`permute` 是典型的 layout-only 算子：它不改变数值，只改变张量维度解释/内存视图。

为了避免后续（尤其 Phase 5）在 planner/后端里遇到大量“无意义的重排”：

- 需要在 planner 层先把 **确定可消去** 的模式消掉
- 并把 `permute` 明确成可优化对象（而不是“只是 executor 能跑”）

## 本次改动

### 1) 新增 planner pass

- 文件：`boundflow/planner/passes/layout_only.py`
- pass：`simplify_layout_only_ops()`

规则（v0）：

1. **连续 permute 合并**
   - `permute(p1) -> permute(p2)` 合成为 `permute(compose(p1,p2))`
2. **identity permute 消除**
   - `dims == [0,1,2,...]` 直接消去
   - 通过 value alias 重写后续 op 的输入与 task 输出，保证 SSA 环境一致
3. **向后兼容**
   - 将 `transpose` 视为 `permute` 处理（统一语义，便于后续优化）

### 2) 默认启用

- `boundflow/planner/interval_v0.py` 在生成 TaskOps 后默认调用该 pass。

### 3) 新增测试

- 文件：`tests/test_phase4b3_layout_permutes.py`
- 覆盖：
  - inverse permute 相邻出现可被消除
  - identity permute 可被消除
  - 非 identity permute 会保留，并带 `attrs["layout_only"]=True`

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase4b3_layout_permutes.py
conda run -n boundflow python -m pytest -q
```

