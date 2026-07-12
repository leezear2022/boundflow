# 变更记录：新增 BoundFlow ASPLOS 总体计划候选稿

## 动机

将当前 Phase 7A PR-9 的真实工程状态、后续 ReLU/operator/materialization/runtime 路线、
ASPLOS rapid-review 约束、实验与 artifact 证据链统一为一个顶层计划，供多模型和人工评审后
定稿。避免继续以“TVM 加速若干 bound 算子”作为论文主线。

## 修改

- 新增 `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`。
- 明确三项候选贡献：Operator-Preserving Bound IR、Materialization/Memory Planner、
  Repeated-Query Runtime。
- 明确 auto_LiRPA、α,β-CROWN、tensor-based certifier compiler 与 Luna 的边界。
- 以当前 PR-9 为基线，定义 Gate 0 与 PR-10 至 PR-15 的验收条件。
- 增加实验矩阵、两页 rapid-review 结构、2027/2028 Go/No-Go、artifact 和风险止损规则。

## 验证

- 对照当前 Git 基线、Phase 7A PR-9 后续计划与现有 artifact 流程进行一致性检查。
- 使用 ASPLOS 2027 官方 CFP/AE 页面及相关工作原始仓库或论文复核时效性信息。

## 边界

本次只写候选计划文档，没有启动 PR-10，没有修改系统实现，也没有将该计划标记为最终执行版。
