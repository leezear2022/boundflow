---
status: validated-r3-0-contract
updated: 2026-08-25T04:25:00+08:00
type: closure
topic: boundflow
slug: r3-0-structured-owner-formal-closure
stage: s01
---

# BoundFlow R3-0 Structured Owner Formal Closure

> **历史证据警告（2026-08-25）**：本文件对应 v1 通用合同 fixture。后续复核发现其 alpha
> binding 是 dense native shape，因此本文件中的 `r3_1_open=true` 不再具有 admission 权威性。
> compressed-alpha v2 已修正并正式通过，当前权威 closure 为
> `BOUNDFLOW_R3_0_COMPRESSED_ALPHA_V2_FORMAL_CLOSURE_2026_08_25.md`。v1 的 validator、closure、
> liveness 与 tamper 证据继续保留，但不得用来证明 production-shaped R3-1 输入合同。

## 1. Verdict

R3-0 contract/static-validator 阶段正式关闭为 `VALIDATED-R3-0-CONTRACT`。

该状态证明 first-class lower-region DAG、Template/Instance、closure/fanout/bias ownership、两 scratch
liveness、dense escape、context reachability、saved-state ledger 和 claim receipt 可由 typed replay
fail closed。它不证明 custom backward、dα/dβ 数值、真实 saved-tensor hook、GPU memory 或性能。

## 2. Source 与 artifact

- source commit：`e9b11e3dae1ade98228f1c60d9bda1cffdd0eed2`；
- source clean：true；formal 唯一 dirty path=`.docops/ev.jsonl`；
- artifact：`artifacts/r3-structured-owner/r3-0-contract-v1`；
- protocol hash：`aa69145670826bd3066f67185ba21c789c1d7ebaac6170300eecc4a466719c67`；
- manifest SHA256：`2e60e35d1a742855900c6c1dbf533c20dd9a9907f252a4211ae6c5cc0c0cabdb`；
- replay stdout 与 summary 逐字节一致。

## 3. Contract receipt

| field | result |
|---|---:|
| node / edge | 8 / 8 |
| source op | 3 |
| scratch slot | 2 |
| saved logical bytes | 304,128 |
| saved unique storage bytes | 205,824 |
| saved coefficient bytes | 0 |
| dense escape / context tensor | 0 / 0 |
| production connected | false |
| timing recorded / performance claimed | false / false |

关键 hashes：template=`319dd908…dfbf`、instance=`7f0921fd…3e53`、bundle=`aebfe761…cb6a`、
summary=`83b2c8be…2e23`。

## 4. Negative evidence

40 个 targeted tests 通过。12/12 fully re-signed artifact mutations 被 semantic replay 拒绝，覆盖
start-node、topology edge、beta shape、split/history、consumer count、BiasSplit fraction、scratch
overlap、dense escape、context Tensor、production connected、performance claimed 和 summary gate。
tamper hash=`409a73435b64b6b6f26e7dbacf0f05c8ff3f096fe0b873dd783097fbdaa9de22`。
全量回归=`1568 passed, 3 skipped`；3个skip均为既有TVM重复编译或冻结VNN-COMP checkout环境边界。

## 5. Claim boundary

- 允许：R3-0 typed contract/static validation 已关闭；
- 不允许：把 contract-only synthetic pointer/ledger 当成真实 CUDA execution 或 memory measurement；
- 不允许：声称 custom VJP、terminal lower、dα/dβ、optimizer trajectory、speedup 或 ASPLOS-ready；
- R3-1 开放不代表 R3 路线 GO，只允许执行下一 correctness kill gate。

## 6. 下一唯一动作：R3-1

只接 `25/Conv_8`，一个 evaluation，optimizer mutation count=`0`。每个 candidate worker必须恰一次
forward和恰一次 custom backward；no-grad 只能 smoke。five fresh 必须与独立 native worker比较
terminal lower、sign、dα和 empty beta，并同时证明 saved dense A=`0`、scratch slot `<=2`、candidate
control无native shadow/fallback/eager、α/β version不变。本阶段不读取 latency，
`performance_claimed=false`。R3-1 未关闭前，R3-2A/2B 均保持关闭。

## 7. Links

- design：`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`
- implementation：`BOUNDFLOW_R3_0_STRUCTURED_OWNER_CONTRACT_IMPLEMENTATION_CHANGELOG_2026_08_25.md`
- R1 route closure：`BOUNDFLOW_CIBC_R1_A_FORMAL_NO_GO_CLOSURE_2026_08_25.md`
