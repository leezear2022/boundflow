# 变更记录：Phase 5A PR#2 追补（TaskIO buffer contract + branch/merge 回归）

## 背景

收到 review 指出 PR#2 的关键风险点：

- task 边界如果只靠 op list，TaskIO 合约不清晰，后续 5B（reuse）与 5E（lowering 签名）会返工
- “按 op 数二分”必须确保 cross-segment use/def 被显式连边
- 测试需要覆盖 branch+merge 的依赖模式

## 本次追补

- `BoundTask` 增加 TaskIO buffer 字段：
  - `input_buffers` / `output_buffers`（对齐 `StoragePlan.value_to_buffer`）
- `BFTaskModule.validate()` 在 `task_graph` 存在时强制要求任务填充 buffer IO，并校验与 StoragePlan 一致
- `interval_v0` / `interval_v2` 在生成 task 时同步填充 TaskIO buffer 列表
- PR#2 等价测试新增手工 branch+merge primal graph 用例

## 目的

提前钉住 TaskIO（buffer 级）与 cross-segment 依赖构图规则，确保 5B/5E 不返工。

