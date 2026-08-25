---
status: validated-no-go-mr1-cibc-full-graph-same-solver
updated: 2026-08-26T14:45:00+08:00
type: closure
topic: boundflow
slug: mr1-static-same-solver-eligibility-formal-no-go
stage: s01
---

# BoundFlow MR1 Same-Solver 静态可替换性正式 NO-GO Closure

## 1. Verdict

`VALIDATED-NO-GO-MR1-CIBC-FULL-GRAPH-SAME-SOLVER`。

冻结 RVIR activation raw 共394条，其中 ResNet2B 51条。逐调用按预注册10条规则重算，当前
CIBC 17-op full-graph IBP executor的真实same-solver eligibility为`0/51`。因此不开放 direct
end-to-end B0/B3/candidate A/B，也不开放same-solver timing或R2实现。

这不是 CIBC 性能失败。它证明的是**当前替换边界错误**：真实调用是provider-owned
activation-BaB CROWN/αβ-CROWN call，不是独立IBP整图call。

## 2. Frozen identity

- source=`a6b6d05240d3d38a3c7f2e4565fc4b01b263b796`；
- artifact=
  `artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1/`；
- protocol hash=`7675661ab8c82db7bff8db2c391b648dd0c4aa21cbf0b4109acd704f3467e362`；
- coverage hash=`de5e7ad8174b65d4a3dbc454039b3a8540354905b9ad2e2c4a4e063d614d07c4`；
- summary hash=`7b575d33c9c144c1db442cb8c5f758b13781c4ce2c5fc374537a3faf8e8c6263`；
- final manifest hash=`c0a9cbcf6986a77a02b31dfe9cb44af0573ebdecf3165dc372d95650c09b774b`；
- tamper hash=`0faf191e69f0b615b3f90e49acb71af2022d2dd6714e4ef122004d7d02e45f68`。

工件自包含冻结的 activation raw、RVIR manifest、RVIR-v3 inventory、B3 manifest和CIBC
manifest；无`/home/`路径，root replay可从source重算ledger/coverage/summary/verdict。

## 3. Lossless coverage

- activation call=`394`：MLP=`343`、ResNet2B=`51`；
- method：CROWN=`386`、αβ-CROWN=`8`；
- phase：`activation_bab_bound=394/394`；
- split state present=`394/394`；
- requires-grad=`8`、no-grad=`386`。

没有因为模型、方法或状态不匹配而从分母删除调用。ResNet2B 51条均进入逐行ledger。

## 4. ResNet2B admission

| gate/reason | rejected calls |
|---|---:|
| phase不是initial/IBP graph evaluation | 51/51 |
| method不是IBP | 51/51 |
| split state存在或未解析 | 51/51 |
| requested output不是完整interval graph | 51/51 |
| semantics owner/backend为external exact provider | 51/51 |
| CIBC runtime contract未被该call证明 | 51/51 |
| dynamic state/lineage identity未闭合 | 51/51 |
| CIBC compile key/topology receipt缺失 | 51/51 |
| requires-grad owner存在 | 1/51 |

固定首因顺序下，51/51 primary reason都是
`solver_phase_not_initial_ibp`。完整reason set同时保留，避免首因掩盖多重所有权边界。

## 5. Replay、tamper 与回归

- root replay：`eligible=0/51`且verdict逐字节重算PASS；
- fully re-signed tamper：13/13 rejected，覆盖删除/重复call、model、phase、method、grad、split、
  owner、ledger eligibility/reason、coverage count、summary count/route；
- targeted：`10 passed`；
- mypy：3个实现文件clean；pylint=`10.00/10`；Black/diff通过；
- full regression：待最终记录。

## 6. Route propagation

1. 关闭“把当前 CIBC 17-op IBP整图直接替换现有ResNet activation-BaB call”的假设；
2. 不运行 B0/B3/candidate timing，因为候选在语义上尚未admit；
3. 不撤销 CIBC 独立IBP图`2.45631x` reduced claim，也不把它传播到query；
4. 不撤销B4-B2/R3已验证的CROWN局部TIR correctness，但其局部数字不自动形成真实call覆盖；
5. 下一只开放 **MR2 production CROWN subgraph/owner contract inventory**：在真实
   provider-owned call内部枚举可替换的Conv/Linear/coefficient-update边界、状态输入输出、
   backward/mutation owner和已有receipt覆盖；仍不计时、不实现candidate；
6. 只有MR2找到至少一个状态闭合的production subgraph，才允许另行预注册接入correctness。

## 7. Claim boundary

允许：现有冻结RVIR raw中，当前CIBC full-graph IBP executor对ResNet activation-BaB call的静态
eligibility为`0/51`，因此该直接接入假设NO-GO。

禁止：CIBC不能加速αβ-CROWN、CROWN子图没有机会、query会变慢、历史`2.45631x`无效、任何
same-solver/complete-query/queue speedup或ASPLOS-ready。

