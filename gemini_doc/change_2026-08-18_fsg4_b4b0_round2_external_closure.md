# FSG4/B4-B0 Round 2 外审关闭记录

日期：2026-08-18
状态：`VALIDATED-B4-B0-EXTERNALLY-APPROVED`

## 流程结论

DocOps exchange `fsg4-b4b0-five-fresh-20260818` 已在 Round 2 获独立外审批准，并由 executor
执行 `dol exchange close`。最终 finding 为 blocker 0 / major 0 / minor 0 / info 0；Round 1 F1
正式关闭。

## 独立复核证据

- code 常量、protocol 完整 frozen identity 与 manifest identity hash 三层一致；
- 从 source/model 重建 state、primal、split、topology、schedule 全匹配；
- 从 source PT 独立核对 14 个 lineage source tensor hashes，重建 4 个 receipt hashes，全匹配；
- 审计方自行实施 coordinated all-run topology/lineage 改写，并同步重签 capture、lineage、
  protocol、summary、replay、manifest 全链；两案均被 root replay 拒绝；
- v1 read-only replay 与 v2 replay 均通过；v2 正式完整性负例 11/11 rejected；
- raw 独立重算为 5 runs / 10 captures、108 tensors / 664,744 elements、max diff=
  `1.1920928955078125e-07`、sign exact；
- 定向=`24 passed`；全量=`1376 passed, 3 skipped, 6 warnings`；
- Black、scoped Mypy、scoped Pylint 10.00/10、diff、DocOps validate/lint 均通过。

## Claim 边界与下一步

B4-B0 只以 production evaluation-0 capture correctness/ownership 关闭。下一阶段只开放“另行
预注册 B4-B1 typed pure-PyTorch reference”；B4-B2、CUDA/TIR lowering、performance、memory 与
ASPLOS-ready claim 继续关闭。

完整外审报告：
`.docops/exchange/fsg4-b4b0-five-fresh-20260818/r002/audit_report_full.md`。
