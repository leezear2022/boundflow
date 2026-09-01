---
status: validated-r3-3-s-active-beta-correctness
updated: 2026-08-26T04:15:00+08:00
type: changelog
topic: boundflow
slug: r3-3-active-beta-implementation
stage: s01
---

# R3-3 Active-β Correctness 实现变更记录

## 1. 变更

- 新增五个独立 CUDA worker 的 R3-3 S-anchor active-β correctness 证据生成路径；
- 原始证据同时保存 candidate、B4-B1 PyTorch oracle、native α/β gradient、first-class
  template/schedule/instance 以及 module/projection/launch receipts；
- replay 从冻结 capture 重算 PyTorch oracle，不仅信任 raw 中的 reference tensor；
- 新增 12 类全外层重签 tamper 探针，覆盖 β tensor/location/sign、projection、
  launch、empty specialization、cache protocol、summary scope 与同改 oracle/candidate 攻击；
- 新增 formal replay 测试，只允许通过后开放 R3-3 isolated timing。

## 2. 协议勘误

原预注册把“五个独立进程”和“同一 cache miss→4 hit”写成同一组 receipt 门禁。
在任何 formal raw 生成前已拆成：

- 五个 correctness worker 各自 cold miss，保留进程级 fresh；
- 额外独立 cache-sequence probe 在同进程、同一 cache 下证明 `miss,hit,hit,hit,hit`。

该勘误不改变数值、ownership、workspace、tamper 或 scope 门禁。

## 3. 当前验证

- worker smoke：`beta_nonzero=6`，`performance_claimed=false`；
- black：PASS；
- mypy：4 个相关文件 clean；
- pylint：`10.00/10`；
- targeted：`11 passed, 3 skipped`，3 个 skip 均因 formal artifact/tamper 尚未生成。

## 4. 边界

本记录不声称 R3-3 已关闭，不记录性能，不开放 R3-4 或 same-solver。下一步是先提交
精确 source，再生成与该 source revision 绑定的 formal artifact。

## 5. 首轮 formal 回归失败及修正

首轮 source=`ee0e96d` 的专项结果为 5 worker、max diff=`8.642673492431641e-07`、
12/12 tamper、`14 passed`。但全量回归得到 `1650 passed,3 skipped,2 failed`：两个失败
均由同进程前序测试后 CPU oracle 逐字节 hash 不稳定触发。该 artifact 作废，不形成
closure。

修正复用 B4-B1 已测的 `_reference_execution_policy`，将 worker 内部 oracle 与 replay oracle
同时固定为 1 CPU thread、deterministic debug mode 2、highest matmul precision、MKLDNN off，
且离开 context 时恢复环境。必须从修正后 source 重生 artifact 并重跑全量回归。

## 6. 最终结果

修正后 source=`7350572` 从零重生 formal artifact；5 fresh/20 metrics max diff=
`8.642673492431641e-07`，active β gradient=30/30 nonzero，12/12 tamper，targeted=
`15 passed`，full=`1653 passed,3 skipped`。R3-3 correctness 已正式关闭，只开放另行预注册
的 isolated timing。
