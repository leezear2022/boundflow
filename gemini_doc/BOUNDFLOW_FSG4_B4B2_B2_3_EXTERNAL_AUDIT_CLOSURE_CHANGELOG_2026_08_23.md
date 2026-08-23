---
status: externally-approved
updated: 2026-08-23T12:47:40Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-3-external-audit-closure
stage: s01
---

# FSG4/B4-B2 B2-3 外部审计关闭

## Verdict

- DocOps exchange Round 1=`APPROVE`，0 blocker/major/minor，2 info；
- 最终状态=
  `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS`；
- exchange 已由`approved`正式执行`close`，resolution=`approved`；
- 只开放 B2-4 P-anchor sparse-source schedule；timing、B2-5、B4-B3保持关闭。

## Independent Evidence

- 无 autograd float64 闭合公式重算5份raw四路输出，最差diff=`1.8300339528209975e-06`；
- 现场GPU复现5 raw/20 metrics/92,190元素、runner max diff=
  `2.384185791015625e-06`、sign exact；
- template/schedule/module receipt hash逐位复现；DLPack=`19/19`，launch=`1/1`，
  fallback/eager=`0/0`；
- scheduled TIR结构遍历只得到`adjoint_conv[6,1,16,8,8]`与
  `output_bias_delta[6,1]`；8类篡改全部fail closed；
- targeted/related/full=`43/97/1457 passed`，3 skipped均为既有环境边界；静态与DocOps全过。

## Findings Disposition

1. module receipt的TIR/device hash在本地validate中不重编译：接受为info，冻结为B2-5 replay
   必须独立重编译并比对的明确门禁；B2-4不伪装已解决；
2. dense Conv缺独立shape-mismatch用例：随B2-4立即补齐，不改变B2-3已批准结论。

## Next Boundary

B2-4连续实现P-anchor compressed-alpha/empty-beta Conv correctness与最多12个预登记schedule
candidate ledger。本阶段不生成formal timing raw、不选择性能winner、不开放B2-5或B4-B3。
