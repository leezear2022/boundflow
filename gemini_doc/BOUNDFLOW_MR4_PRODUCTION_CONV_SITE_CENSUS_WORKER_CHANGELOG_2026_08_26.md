---
status: implemented-pre-formal
updated: 2026-08-26T17:50:00+08:00
type: changelog
topic: boundflow
slug: mr4-production-conv-site-census-worker
stage: s01
---

# MR4 Production Conv Site Census Worker 修改记录

## 1. 实现

- 新增`scripts/run_mr4_production_conv_site_census_worker.py`；
- 在真实beta-split optimized outer call中扫描全部graph topology并hook三条冻结ReLU→Conv edge；
- 每site记录30字段左右的shape/dtype/device/α/β/handoff/output/O(1)账本，不执行candidate或计时；
- 记录完整outer/final α/module semantic state，供five-fresh验证observer不改变provider结果；
- 机械计算每site forward MAC units与candidate最低materialization bytes；
- model/property、三个外部仓库commit、device/stream与10/9 call lifecycle均fail closed。

## 2. 非正式单run预检（不进入formal raw）

- topology恰为C0 `/input-4←/input`、C1 `/input-12←/input-8`、C2 `/input-24←/input-20`；
- 三site各10 rows，grad-enabled=`9`，β均=`[6,0]`、numel=`0`；
- handoff content=`10/10`，pointer=`0/10`（provider ReLU生成新A，属于真实基线行为）；
- MAC units：C0=`1,327,104`、C1=`1,769,472`、C2=`884,736`，合计/P=`4.5x`；
- candidate最低materialization bytes/evaluation：C0=`172,056`、C1=`98,328`、C2=`73,752`；
- solver=`verified`、visited domains=`[6]`。

这些数字只证明worker绑定正确和正式实验值得运行，不是performance/share claim；预检目录不提交。

## 3. 验证

- Black：通过；
- mypy：clean；
- pylint：`10.00/10`（移除未使用import后）；
- RTX 4060 Laptop GPU真实worker：成功闭合。

## 4. 下一步

从clean worker commit实现MR4 formal validator/runner/tamper，再运行5 fresh。正式通过前MR5仍关闭。
