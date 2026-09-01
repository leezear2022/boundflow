# S4-1B0 activation preflight 修改记录

date: 2026-08-30
stage: s04
performance-claimed: false

## 改动

- 新增 S4-1B0 待审期间 activation preflight；
- 记录当前 legacy backend blob、proposed 路径与 symbol collision 检查；
- 独立重算 construction model hash；
- 在仓库外重跑 TVM/CUDA ternary pack/select 位级原型；
- 记录首次 shape-mismatch 失败及修正，不隐去负结果；
- 冻结 S4-1A 外审批准后的最小提交顺序。

## 边界

- 没有 production、test、artifact 或第三方代码改动；
- 没有激活 S4-1B0 implementation/formal；
- 没有 correctness、memory、timing、same-solver、10x 或 ASPLOS-ready claim；
- S4-1A exchange 仍为 `ready_for_audit`，外审仍是唯一上游门禁。

## 验证

- current TVM/CUDA environment import：PASS；
- proposed files/schema/symbol collision scan：PASS；
- construction model SHA256：PASS；
- disposable TVM/CUDA IEEE-bit prototype：PASS；
- `git diff --check` 与 DocOps lint：交接前执行。
