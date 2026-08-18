---
status: validated-b4b1-v2-reference-pending-regression-and-audit
updated: 2026-08-18T16:01:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 v2 deterministic reference 候选

## 结果

从execution-policy修正提交`d9164b8`生成
`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2`。root replay在冻结策略下：

- 5 fresh / 10 captures；
- 60 metrics / 196,380 elements；
- maximum absolute difference=`6.109476089477539e-07`；
- allclose/sign exact全过；
- summary hash=`becd8ae5...d744`；
- 从入口threads=1/4均得到相同records且退出后恢复调用方线程数。

v1因缺失执行策略字段被新replay fail-closed拒绝，保留为历史失效证据。integrity report进一步增加
source git HEAD与完整reference code revision绑定，避免probe只绑定自身源码却漏绑被测runner。

## 下一步

提交v2 artifact与probe绑定修正后，生成正式v2 integrity report并重跑related/full/static/DocOps；
外审批准前B4-B2/TIR/performance继续关闭。
