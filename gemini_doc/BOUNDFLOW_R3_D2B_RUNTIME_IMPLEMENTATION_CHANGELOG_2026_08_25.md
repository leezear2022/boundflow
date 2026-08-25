---
status: implemented-targeted-passed-formal-pending
updated: 2026-08-25T23:50:00+08:00
type: changelog
topic: boundflow
slug: r3-d2b-runtime-implementation
stage: s01
---

# R3-D2-B Runtime 修改记录

## 改动

- 新增 D1-C 子类，只拦截 backward `_coefficient_sign_pass` 内 residual11/residual6；
- 复用 D1-C 已编译 four-symbol module 与两个 caller-owned arena tail；
- forward/backward staged launch 和 bias alias 计数完全分离；
- raw B1 backward launch 从 15 降为 13，新增 4 个 staged launch；
- receipt 明确禁止 dense A、autograd history、workspace、fallback、eager/native shadow、timing 与
  performance claim；
- 新增完整 10/9 D1-C 对照、receipt 六类漂移和两个 ABI 拒绝测试。

本提交只做 correctness candidate，不含 timing runner、默认路由或 production adapter 切换。

## Targeted validation

- black：通过；mypy：clean；pylint：`10.00/10`；
- `tests/test_r3_d2b_staged_backward.py`：`3 passed`；
- 完整 10/9 terminal lower/α/sign 对 D1-C 通过；
- 正式 five-fresh 逐步轨迹与 tamper 尚未生成，因此 timing 仍关闭。
