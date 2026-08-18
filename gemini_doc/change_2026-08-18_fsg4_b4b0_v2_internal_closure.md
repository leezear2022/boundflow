# FSG4/B4-B0 v2 内部关闭记录

日期：2026-08-18
状态：`VALIDATED-B4-B0-V2-PENDING-ROUND2-EXTERNAL-AUDIT`

## 起因

Round 1 外审发现原 v1 replay 只证明 fresh run 间相对一致，未把 topology 与 lineage 的绝对
身份绑定到冻结 source。全 5 run / 10 capture 协调一致改写并重签时，v1 verifier 会错误接受。

## 本轮变更

- source=`422a3ee96fe86d09bcb0f042b3757447ed94ae6a`；
- v2 protocol 在代码、protocol 和 manifest 三层绑定 source/model/state/schedule/primal/split/
  topology，以及逐锚点 anchor、lineage source tensors 与 round-trip receipt hashes；
- verifier 保留 v1 只读 replay 兼容，但 v2 必须匹配冻结绝对身份；
- 新增 coordinated-all-runs topology 与 lineage 两类完整性负向用例；
- 正式 artifact：`artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v2`；
- 正式完整性报告：
  `artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v2-tamper-report.json`。

## 冻结结果

- 5 个 fresh CUDA subprocess、10 个 production evaluation-0 captures；
- 108 组 tensors、664,744 elements；
- maximum absolute difference=`1.1920928955078125e-07`，sign exact；
- manifest hash=`27391e66acb6fc1146a6fc3f0d726a1b97d24af3df6b24f7294b362e4025be6b`；
- protocol hash=`2514fc21ca34a3647bed5df3352ccebe6dfe07d30147bb9ff2781a003b57ea4b`；
- summary hash=`db7f498c780ef722c182fa1db6d6e1d2baae29b96da73a8d0d644826f8e4413e`；
- frozen identity hash=`05b926ac8fc70f03ce6bd08a34b61ef6bf81cb27e02b019c0cb42c2c590c3e9d`；
- 11/11 outer-resigned 完整性负向用例拒绝，其中含 Round 1 的两类 coordinated rewrite；
- 定向回归=`24 passed`；全量=`1376 passed, 3 skipped, 6 warnings`；
- Black clean、scoped Mypy clean、scoped Pylint=`10.00/10`；
- `performance_claimed=false`，`tir_admitted=false`。

## 边界与下一步

本轮只恢复 B4-B0 capture correctness/ownership 的内部证据闭环，不能升级为 TIR correctness、
速度、显存或 ASPLOS-ready claim。必须回复 Round 1 F1 并通过 Round 2 独立外审；在批准前，
B4-B1 typed differentiable IR、B4-B2 TIR 和所有性能计时继续关闭。
