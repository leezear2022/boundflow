# 修改记录：MR1 Same-Solver 静态可替换性审计

> 日期：2026-08-26  
> 状态：正式执行完成，VALIDATED-NO-GO

## 修改

- 新增 MR1 只读静态 admission 审计，替代已被 MR0 关闭的 event-per-op share 路线；
- 冻结 RVIR v2、RVIR-v3 inventory、B3 formal 与 CIBC formal 输入 hash；
- 冻结10条 CIBC full-graph 逐调用 admission 规则；
- 冻结完整 coverage/ledger/replay/tamper 与零 eligible 的机械 NO-GO；
- 明确 NO-GO 只关闭当前 CIBC 整图直接替换，不关闭 operator/subgraph 或重新设计的
  structured CROWN 路线。

## 验证

- 预注册阶段只运行文档一致性与 `git diff --check`；
- 实现、测试、正式工件和 verdict 必须在后续提交完成，禁止回改本文件门槛。

## Formal result

- source=`a6b6d05`，394条activation call无损计数，ResNet2B ledger=`51`；
- 当前CIBC full-graph eligibility=`0/51`，机械verdict=
  `VALIDATED-NO-GO-MR1-CIBC-FULL-GRAPH-SAME-SOLVER`；
- 51/51均为activation-BaB、非IBP、split state present、provider-owned exact call；
- self-contained replay PASS，13/13 fully re-signed tamper rejected；
- targeted=`10 passed`，typing/lint/format通过；full regression=`1687 passed,3 skipped`。
