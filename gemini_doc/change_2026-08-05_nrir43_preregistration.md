# NRIR-43 Cross-Axis Verification Batch Schedule v1 预注册记录

## 起因

NRIR-42 已消除 scorer validation 重复并恢复 31/31 production coverage，但 fixed ResNet2B property 0
的 whole query 仍为 57–58 秒。frozen trace 显示 floor 约 22 秒，随后两条 selected clause 仍按顺序
执行；每条 31-node queue 的 31 个 objective scorer 也按节点发射。下一大杠杆是联合 batch，而不是
继续削 Python validation 常数。

## 本次变更

- 用 DocOps standalone plan/changelog 预注册 NRIR-43；
- 冻结唯一变量为跨 clause/node/candidate 的 typed ragged ready-work Schedule；
- 冻结 Phase A sibling scorer pack 与 Phase B two-clause coordinator 的顺序；
- 冻结 exact semantic/ownership、launch reduction、three-repeat timing 与 fail-closed rollback 门禁；
- 同步 current status、执行备忘、总体计划、claims map、README 与总变更账本。

## 当前结论

NRIR-43 尚未实现、尚无 artifact、尚无性能结论。NRIR-42 的 `VALIDATED-REDUCED` 与窄 claim boundary
保持不变，ASPLOS-ready 仍为 NO。
