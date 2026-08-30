# BoundFlow ASPLOS'27 S4-1B0 round-aware activation gate 修改记录

status: validated
date: 2026-08-31
scope: S4-1A exchange round tracking only
performance-claimed: false

## 1. 问题

S4-1A round 1 外审要求修正后，DocOps 合法进入 round 2 的 `ready_for_audit`。既有 S4-1B0
activation checker 把开放 exchange 的 `next` 写死为
`wait-for-external-audit-s4-1a-round1`，因此对合法 round 2 状态返回 ERROR，而不是 WAIT。

## 2. 修改

- `_validate_docops_state` 显式接收 exchange round；
- 开放 exchange 的期望 next 动态绑定为
  `wait-for-external-audit-s4-1a-round{exchange_round}`；
- blocker ID 继续保持稳定的 `external-audit-s4-1a-pending`；
- 新增 round 1、round 2、stale-next 拒绝、closed-next 四个 self-test。

## 3. 边界

本修改只修复状态机验证，不改变 S4-1A 审计内容、不开放 S4-1B0，也不产生 correctness、timing
或 performance claim。round 2 未批准并由 executor close 之前，真实门禁仍必须返回 WAIT。

## 4. 验证结果

- activation gate self-test：`16/16 PASS`，含 4 个新增 DocOps round-state case；
- 当前 round-2 真实状态：`WAIT / external-audit-s4-1a-pending`，implementation/formal/timing/
  performance authority 全为 false；
- mypy：在 `scripts/` sibling-import 口径下两个 checker clean；
- Pylint：`10.00/10`；
- Black check、`git diff --check`：PASS；
- DocOps exchange validate、lint：PASS。
