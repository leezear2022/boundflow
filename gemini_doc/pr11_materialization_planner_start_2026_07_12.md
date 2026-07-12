# PR-11 Materialization Planner 启动记录

> 日期：2026-07-12
> 基线：`263ea81` + 当前未提交的 ASPLOS 文档收敛修改
> 状态：PR-11 第一实现切片完成；尚未达到 PR-11 总体验收门禁。

## 1. 本切片目标

建立 method/autograd/memory-aware Planner 的最小可执行闭环，而不是继续在 PR-10 operator
内部硬编码策略：

```text
real CROWN query
  → shape-derived MaterializationContext
  → capability filter
  → safe-memory feasibility
  → latency selection
  → explicit plan/reason dump
  → CROWN / α / αβ runtime execution guard
```

## 2. 代码修改

### `boundflow/planner/materialization.py`

新增：

- `BoundMethod` 与独立的 `OptimizationStage`；
- `MaterializationAction`：dense、structured、reduce-batch；
- `TargetProfile` capability contract；
- `OperatorTreeSummary`、`MaterializationContext` 与 options；
- Always Dense、Always Structured、Method-Only、Memory-Threshold、Global policy；
- feasibility-first、latency-second 的确定性选择；
- capability/memory rejection reason 与 JSON-serializable plan dump；
- per-case measured Oracle；
- 无可行 action 时的 deterministic reduce-batch recommendation。

### Runtime 接入

- plain CROWN 可接受显式 `MaterializationPlan`；
- `plan_crown_materialization` 从实际 forward trace、spec/domain shape 和 dtype 生成 context；
- α-CROWN 与 αβ-CROWN 可接受 dense/reduce-batch plan；
- 当前 backend capability 未通过前，optimized-bound structured 在执行前确定性拒绝；
- reduce-batch 通过 `MaterializationReplanRequired` 返回 host runtime，不在 kernel 内静默缩 batch。

## 3. 测试证据

- PR-11 第一切片专项：15 passed；第二切片后专项增至 21 passed；
- 第二切片后全量：200 passed、1 skipped；
- Black：新增 Planner 与测试文件通过；
- Pylint：`boundflow/planner/materialization.py` 为 10.00/10；
- Mypy：`boundflow/planner/materialization.py` 无问题；
- `git diff --check` 在收尾检查中执行。

专项覆盖：

- dense/structured 的预算 regime；
- measured latency 的第二级选择；
- α/αβ capability filter；
- reduce-batch 与 host re-plan signal；
- per-case Oracle 的最快可行 action；
- plan dump 可序列化；
- 显式 plan 覆盖 legacy mode 且 bounds 对齐；
- 真实 query 的 shape-derived cost features；
- optimized-bound runtime 不可绕过 structured capability guard。

## 4. 当前证据边界

本切片证明 Planner API、合法性约束、执行 guard 和 Oracle 基础已落地，但尚未证明 C2 的系统
收益。以下仍是 PR-11 blocker：

- 当前 cost summary 是解释性 shape model，尚未用 workload-family held-out 校准；
- 尚无 multi-barrier Local Greedy 与全图 Global 的公平对照；
- 尚未建立 360 行 profile → calibration/held-out → JSONL evaluation runner；
- 尚未证明某个真实预算下 Always Dense OOM 而自动 Planner 成功；
- 尚未报告 held-out median/p90 Oracle regret；
- reduce-batch 目前返回 host re-plan signal，scheduler 自动拆批尚未接入。

因此 claims map 中 C2 只能从 `planned` 更新为 `partial`，不能标记 `validated`。

## 5. 下一切片

前五项已在第二切片完成，见 `gemini_doc/pr11_heldout_eval_2026_07_12.md`。结果同时证明单一
query 决策会使 Global 退化为 Memory-Threshold，因此下一步改为 multi-barrier/region 联合
placement；之后再决定是否将 reduce-batch 接入 BaB/domain scheduler。
