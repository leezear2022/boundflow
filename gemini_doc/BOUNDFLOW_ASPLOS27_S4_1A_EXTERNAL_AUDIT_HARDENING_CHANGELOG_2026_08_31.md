# BoundFlow ASPLOS'27 S4-1A 外审修正记录

status: validated-pending-round2-external-audit
date: 2026-08-31
scope: S4-1A ordered buffer replay/static hardening
performance-claimed: false

## 1. 起因

S4-1A round-1 外审给出 `approve-with-minor-correction`，其中 F2/F3 为 exchange close
前强制修正，F1 为 replay 未绑定 `detail_code` 对应 `verification_reason` 的语义缺口。本次修正
不改写已经交付的 r001 artifact，也不修改其冻结 raw、sidecar、manifest 或 headline 数字。

## 2. 修改

1. `scripts/replay_asplos27_s4_1a_buffer_stdlib.py`
   - 把七个 fault 的冻结合同从 `(fault, detail_code)` 扩为
     `(fault, detail_code, verification_reason)`；
   - replay 同时校验 detail 与 reason，拒绝外审构造的 coherent-resign reason forgery。
2. `tests/test_asplos27_s4_1a_buffer_artifact.py`
   - 新增 output fault reason 被替换并重签后的专用负向测试。
3. `scripts/run_asplos27_s4_1a_buffer_worker.py`
   - 用 `del` 表达临时 Tensor 生命周期结束；
   - 用 `setattr` 安装/恢复 observer，消除三处 mypy 类型错误，不改变运行时调用顺序。
4. `boundflow/runtime/asplos27_s4_ordered_buffer_abi.py`
   - 按既有 S2 静态检查口径，对运行时惰性 TVM import 显式禁用 Pylint `import-error`。

## 3. Claim 边界

- 本次只关闭 replay 与静态检查口径缺口；
- 不新增 timing、speedup、same-solver、complete-query 或 ASPLOS-ready claim；
- S4-1B0 在 DocOps exchange 完成正式响应、复核和关闭之前仍不得执行 production 实现。

## 4. 验证结果

- S4-1A ordered-buffer + artifact 定向：`85 passed`；
- 全量：`2051 passed, 3 skipped`；新增 1 个通过用例即 reason coherent-resign 拒绝测试；
- stdlib replay：PASS（12 workers / 40 binary exact / 7 faults）；
- mypy `--explicit-package-bases`：7 个交付源文件 clean；
- Pylint：7 个交付源文件逐文件均为 `10.00/10`；合并运行会报告既有跨文件
  `duplicate-code`，因此不把合并口径写成 10.00；
- Black check：PASS；
- `git diff --check`：PASS；
- activation-gate self-test：`12/12 PASS`；在 exchange `changes_requested` 时真实门禁按预期返回
  `WAIT`，S4-1B0 未误开放；
- DocOps exchange validate/lint 在正式响应与 round-2 delivery 后复核。
