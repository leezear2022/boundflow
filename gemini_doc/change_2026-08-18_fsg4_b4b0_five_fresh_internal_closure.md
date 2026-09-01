# FSG4/B4-B0 Five-Fresh 内部关闭记录

日期：2026-08-18

状态：`VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`

## 冻结身份

- source：`1dbb2de4bc29eb92457e2d24c3e627d638b6607a`；
- artifact：`artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v1`；
- manifest hash：`79059be7bee161b3774beda09b711484cd6aa3cefbb8cded93c8fc851c2601ab`；
- protocol hash：`d9e8a76e36806355f193d249574b38dfbbaf18c602fd2bcf40edb442d1350da1`；
- summary hash：`93b62ce30830ffb199f5fc8ddad6db61aff790b4925bccfc18afa238221f399c`；
- tamper report hash：`727ae2ad5bfacc0ad0fae4a60dffe59f58445d2f4ea7973451bfa186ec7b7f5a`。

## 门禁结果

- 5个独立CUDA subprocess，S/P两锚点各5份，共10份capture；
- root replay从raw tensor payload重建全部typed capture；
- 5次离散结构exact；108组tensor、664,744个元素逐项比较；
- 最大绝对差`1.1920928955078125e-07 <= 2e-4`，sign exact；
- S-anchor active-beta value/gradient存在，P-anchor empty-beta无伪造pre-add/gradient；
- source/device/layout/requires-grad、α-index/lookup、β-location/sign、round-trip、Conv attrs、
  default stream与no-alias全部绑定；
- state/start-node/topology/shape/alpha-index/beta-location/gradient/alias/stream九类
  outer-resigned攻击`9/9 rejected`；
- 定向`20 passed`；全量`1372 passed, 3 skipped, 6 warnings`；Black、Mypy、Pylint
  10.00/10与diff check通过。

## Claim边界与下一步

本阶段只关闭production evaluation-0 capture correctness/ownership，不是TIR correctness、region
speedup、whole-core/query speedup、memory或ASPLOS-ready。`performance_claimed=false`、
`tir_admitted=false`保持。下一唯一动作是外部审计B4-B0；批准后才开放B4-B1 typed
`DifferentiableLowerRegionIR`与pure-PyTorch reference，B4-B2 TIR仍关闭。
