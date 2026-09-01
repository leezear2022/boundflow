# FSG4/B4-B0 Evaluation-0 Live Observer 候选记录

日期：2026-08-18

状态：`IMPLEMENTED-B4-B0-LIVE-OBSERVER-PENDING-FIVE-FRESH`

## 实现

- `B4BRegionLiveObserverV1`只在显式传入terminal optimizer时激活；
- observer仅在evaluation 0对`31/Gemm_14`和`25/Conv_8`拥有观测权；
- 诊断路径将锚点structured lower-A物化成参与后续ReLU+affine计算的同一tensor，
  并在真实backward中retain gradient；
- value/gradient在evaluation 0 backward后、第一次`optimizer.step()`前冻结，后续9次
  mutation不能改写capture；
- production compressed α/β在CUDA smoke中与native dense observation汇合，两个immutable
  `ProductionDifferentiableRegionCaptureV1`均通过validation。

## 新冻结的production事实

| 锚点 | incoming-A | beta | weight/attrs |
|---|---|---|---|
| `31/Gemm_14` | `requires_grad=false`，gradient absent | active pre-add与β gradient存在 | `(100,1024)` |
| `25/Conv_8` | `requires_grad=true`，gradient存在 | compressed beta empty，pre-add/β gradient absent | `(16,16,3,3)`，stride/padding/dilation=`1/1/1`，groups=1 |

这些事实表明empty-beta path必须明确缺席，不得用全零pre-add或全零gradient伪造
“执行了beta”。incoming-A custom backward的micro门禁仍需要在B4-B1/B2中对S-anchor使用
显式requires-grad clone，但不改写production raw事实。

## 验证

- B4-B/B3 schedule/PR-12/B4-A/B3 related：`53 passed`；
- 真实CUDA finalization smoke：`1 passed`，两个capture全tensor source device=`cuda:0`；
- 全量：`1369 passed, 3 skipped, 6 warnings`；
- Mypy：B4-B capture + terminal schedule clean；
- Pylint：`10.00/10`；
- `git diff --check`：PASS。

## 边界与下一步

本次是单进程smoke，不是5 fresh artifact，没有root replay/outer-resigned tamper，不支持
TIR correctness或performance claim。下一唯一动作是实现独立进程、raw-first的5-fresh B4-B0
artifact并关闭replay/tamper门禁。
