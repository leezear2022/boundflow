---
status: formal-closed-validated-opportunity
updated: 2026-08-26T01:05:00+08:00
type: changelog
topic: boundflow
slug: mr7r-unprofiled-host-recovery
stage: s01
performance_claimed: false
---

# MR7-R Unprofiled Host Recovery 修改记录

## 1. 目标

实现已预注册的 MR7-R：以五个独立 pair 比较已冻结的 MR6 diagnostic worker 与 MR7 unprofiled control
ledger，先证明 ledger 低扰动，再决定 host boundary 是否能开放 FCR-1/GC-0 correctness。

## 2. 当前修改

- 新增 `boundflow/runtime/mr7r_unprofiled_host_recovery.py`；
- 固定 `BL/LB/BL/LB/BL`、5 pair/10 fresh 的 mechanical derive；
- 复用 MR6/MR7 worker validator，不新增或改变测量 worker；
- 固定 semantic、30/27、cache/module/stream/fallback、host closure、host ratio、clock direction 门禁；
- 固定 15%/15 ms、4/5 run 与 required-region-speedup `<=10x` opportunity 门禁；
- 新增 CPU replay 与 fully re-signed host-call drift 单测。
- source derive 已冻结于 `0a1e79553e216ed5c34604a235d537288fcf8e19`；
- 新增 10-fresh formal runner，按 role 分别调用已冻结的 MR6 diagnostic 与 MR7 control worker；
- protocol 绑定 source、generator、代码 blob、MR6/MR7 artifact identity、外部仓库和全部门禁；
- replay 从 raw 重算 pair metric、perturbation/opportunity gate 与 status；
- 新增 12 类 fully re-signed raw tamper 和 repository artifact replay 测试。

## 3. Claim 边界

- 当前只是 source implementation，尚未生成 10-fresh artifact；
- 尚未判定 opportunity/no-go/invalid；
- `compiled_region_correctness_open` 由 formal raw 决定；
- `timing_open=false`、`performance_claimed=false`；
- 不开放图编译 implementation、same-solver/query/queue 或 ASPLOS claim。

## 4. 后续

1. 通过 formal tooling 专项测试、black/mypy/pylint；
2. 提交 generator freeze；
3. 运行 5 pair GPU formal；
4. 运行 replay、12 类 tamper 与全量回归；
5. 按冻结门禁形成 closure 并更新权威文档。

## 5. Formal 结果

- artifact：`artifacts/measurement-recovery/mr7r-unprofiled-host-recovery-v1/`；
- status=`VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY`；
- 10 fresh/5 pair，host ratio median=`1.00685785`，per-run min/max=
  `0.94277477/1.02633524`；
- boundary median=`20.333052% / 24.683788 ms`，qualifying=`5/5`；
- required parity region speedup=`1.91213674x`；
- semantic max diff=`2.7418137e-6`、sign exact；
- replay通过，12/12 fully re-signed tamper拒绝；
- 相关=`13 passed`，全量=`1821 passed,3 skipped`；
- `timing_open=false/performance_claimed=false`。

下一步只开放 GC-0/FCR-1 verification graph ABI 与 correctness 预注册。
