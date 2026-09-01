---
status: validated-mr2-p-anchor-bridge-preregistration-open
updated: 2026-08-26T15:20:00+08:00
type: closure
topic: boundflow
slug: mr2-production-crown-owner-inventory-formal
stage: s01
---

# BoundFlow MR2 Production CROWN Subgraph/Owner Inventory 正式 Closure

## 1. Verdict

MR2正式route=
`OPEN-P-ANCHOR-PRODUCTION-EXACT-CALL-BRIDGE-CORRECTNESS-PREREGISTRATION`。

冻结候选仅P-anchor Conv与S-anchor Linear。按七层readiness ledger，P-anchor是唯一
`ready_for_bridge_correctness=true`的site；S-anchor未准入。该结论只开放新的P-anchor
production exact-call bridge **correctness预注册**，bridge尚未实现，timing/same-solver/R2均关闭。

## 2. Frozen identity

- source=`26233bf98bd26d8f34bd80a326893167e6ecaf59`；
- artifact=
  `artifacts/measurement-recovery/mr2-production-crown-subgraph-owner-inventory-v1/`；
- protocol hash=`846461e46c5ee8a563adb57e213d725d2a6c5524f7f30ccd5dcfb6acd873f2a5`；
- gap-matrix hash=`6353abd45c4911da652619100ad0a2a2e97872258a57580a5d0c776b0dbff428`；
- summary hash=`c761922bc0c621bf302e8ce9b5109cb16226d0fd369d2cbe9b9cbfc1ebe6e8ba`；
- final manifest hash=`37b384b3fb68073f2b4b6d86e6b60cdd54dffa83299888e030a536c0afb3a34e`；
- tamper hash=`b1c4f3cc4952be86ba2a5e770ccc6cf9b30023fc6c15bf6b2c75808464e235cd`。

工件自包含RVIR-v3 inventory、R3 P/S correctness、P trajectory、CIBC local evidence和MR1-S
closure snapshot；无本机路径，replay重算site ledger、matrix与route。

## 3. Readiness matrix

| gate | P `25/Conv_8` | S `31/Gemm_14` |
|---|---|---|
| production site identity | proven | missing |
| typed input/output ABI | proven | proven |
| state ownership | proven | proven |
| forward/backward correctness | proven | proven |
| optimizer trajectory correctness | proven | missing |
| multi-site consumer closure | bounded single-site | missing |
| production exact-call connection | missing | missing |

P的冻结ABI包含compressed α=`[2,1,6,86]`、absent β=`[6,0]`、bounds=`[6,16,8,8]`、
weight=`[16,16,3,3]`；structured owner无saved dense coefficient，已有10 evaluation/9 mutation
correctness。`production_connected=false`被原样保留，正是下一阶段唯一目标。

S已有active-β VJP correctness与30个非零β gradient，但artifact没有first-class start-node identity、
没有S-anchor 10/9 mutation，也没有adjacent consumer closure。因此不得借用P的轨迹或局部性能证据。

## 4. 为什么选择P

selection order在预注册中固定为P→S，且MR2不读取timing做admission。P被选中是因为前五层均
proven、multi-site被fail-closed限制为single-site、唯一missing gate是production connection；不是因为
历史CIBC局部数字更高。B4-B2 v1 physical NO-GO与后续local correctness/performance均只作为evidence
状态保留，不传播到下一阶段门槛。

## 5. Replay、tamper 与回归

- root replay：selected=`P:25/Conv_8`、route逐字节重算PASS；
- 12/12 fully re-signed tamper rejected：site、β shape、production-connected、trajectory、
  correctness、active β、MR1 eligibility、ledger、matrix和route；
- targeted=`10 passed`；
- mypy clean、pylint=`10.00/10`、Black/diff通过；
- full regression：`1697 passed, 3 skipped`（3项均为既有环境边界）。

## 6. 唯一下一动作

预注册MR3 P-anchor production exact-call bridge correctness：

1. 只接`25/Conv_8`一个site，保持β absent与single-site consumer边界；
2. provider baseline与candidate使用同一真实call pre-state；
3. bridge必须恰一次typed dispatch/emit/atomic commit，fallback/eager/native shadow=`0`；
4. five-fresh逐项比较lower、compressed dα、α/Adam mutation、termination-visible result；
5. split/history、其余start nodes和provider termination仍归αβ-CROWN；
6. 失败即关闭当前bridge，不扩site；通过也只开放单site bridge timing预注册。

## 7. Claim boundary

允许：现有冻结证据下，P-anchor是两个候选中唯一具备bridge correctness预注册资格的site。

禁止：bridge已实现、production coverage已提高、same-solver/query speedup、multi-site已解决、S-anchor
可借用P证据或ASPLOS-ready。
