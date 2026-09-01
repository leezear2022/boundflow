---
status: passed
updated: 2026-08-25T15:30:00+08:00
type: validation
topic: boundflow
slug: r3-d0-full-regression
stage: s01
---

# R3-D0 全量回归验证记录

- 命令：`pytest -q tests`
- 结果：`1615 passed, 3 skipped, 6 warnings in 666.58s`
- skip：1项TVM已可用时跳过重复allow-no-tvm编译；2项冻结VNN-COMP checkout不可用；
- warnings：既有TorchScript deprecation、profiler提示与pytree future warning；
- 无失败，无新增skip，不改变D0 claim边界。
