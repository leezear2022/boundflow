# 2026-08-13 — FSG3 Timing Schema 与 Replay

## 改动

- 新增B0/B1/B2、control/profile、36-worker冻结顺序的typed raw contract；
- 新增cold/query/core/GPU/compile/post-validation与peak memory独立metric；
- 新增semantic/queue、B1 exactly-once、B2 zero-provider/fallback和环境排他合同；
- lower-only upper使用finite payload + positive-infinity mask，避免JSON非有限值并保持哨兵exact；
- replay从36条raw run重算配对speedup、median/range/MAD/geomean、profile扰动、break-even和状态；
- 新增顺序、删run、counter、scope、semantic、environment、closure及perturbation负向测试。

## 边界

本切片不含真实GPU worker或正式数字；所有测试均为合成raw contract。`performance_claimed=false`。

## 验证

- 初始schema定向=`13 passed`、full=`1213 passed, 3 skipped`；pre-run upper-sentinel amendment后
  定向=`14 passed`，post-amendment full延后到real-worker切片统一执行；Black/mypy clean、
  Pylint=`10.00/10`；
- DocOps validate/lint在提交前记录。
