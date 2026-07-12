# 变更记录：PR-10 materialization instrumentation 起点

## 目标

遵循 ASPLOS 执行计划的 instrumentation-first 约束，在修改 ReLU operator 语义前先量化当前
dense barrier。此批不实现 Planner，不改变 ReLU relaxation，也不声称性能收益。

## 修改

- 新增 opt-in `MaterializationTrace` context；未开启时不保存事件。
- 每个事件记录 `reason`、`site`、operator type、logical shape、dense bytes、dtype、device 和
  lifetime，并提供 JSON-friendly summary。
- 将 `_backprop_relu_step` 的 upper/lower `to_dense()` 标记为 `relu_sign_split` barrier。
- 增加事件字段、字节统计与 context 隔离测试。

## 下一步

1. 将其他公共 fallback 接入相同 trace，生成 chain CNN 与 residual DAG 的真实 profile。
2. 以 dense reference 锁定数值与 α gradient。
3. 再设计 ReLU row-scaling operator；不在 instrumentation 批中改数学路径。

## 验证结果

- PR-10/ReLU/DAG 定向回归：16 passed；
- 全量：164 passed、1 个预期 skip；
- `pylint boundflow/runtime/materialization.py`：10.00/10；
- `git diff --check`：通过；`crown_ibp.py` 无全文件格式化噪声。
