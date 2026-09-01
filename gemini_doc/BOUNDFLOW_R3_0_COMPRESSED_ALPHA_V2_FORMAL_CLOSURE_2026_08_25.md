---
status: validated-r3-0-compressed-alpha-v2
updated: 2026-08-25T05:35:00+08:00
type: closure
topic: boundflow
slug: r3-0-compressed-alpha-v2-formal-closure
stage: s01
---

# BoundFlow R3-0 Compressed Alpha v2 Formal Closure

## 1. Verdict

R3-0 的 production-shaped 输入合同正式关闭为
`VALIDATED-R3-0-COMPRESSED-ALPHA-V2-CONTRACT`。v2 修正了 v1 fixture 把 P-anchor alpha 错绑为
dense native shape 的问题；v1 继续证明通用 validator 机制，但其 admission 字段不再权威。

本阶段仍是 contract-only：不证明 production 接入、真实 custom backward、数值正确性、GPU memory
或性能。

## 2. Source 与 artifact

- source commit：`8941e6665faf42b5d6d79650016a427493fbb612`；
- source clean：true；唯一允许 dirty path=`.docops/ev.jsonl`；
- artifact：`artifacts/r3-structured-owner/r3-0-contract-v2`；
- protocol hash：`ec38d1d4b9f971d928160d4d92b38cb4a7b8e6203a73860d3bf3af0bd9cdf732`；
- manifest hash：`682bdf6ed699a4e2355f7cd568dfbc2c4e8296c1a492d7021aa554f3d1aacbca`；
- manifest file SHA256：`a059599a1a589ea1c660191b276cfc6a0162d6c1460c87370d03b70a6fada92e`；
- replay stdout 与 summary 逐字节一致。

## 3. Corrected contract receipt

| field | result |
|---|---:|
| production alpha shape | `[2,1,6,86]` |
| beta shape | `[6,0]` |
| node / edge | 8 / 8 |
| scratch slot | 2 |
| saved logical bytes | 207,888 |
| saved unique storage bytes | 109,584 |
| saved coefficient bytes | 0 |
| dense escape / context tensor | 0 / 0 |
| production connected | false |
| timing recorded / performance claimed | false / false |

关键 hashes：template=`319dd908…dfbf`、instance=`a3e77a77…7279`、bundle=
`1c787901…5c63`、summary=`3fa9c479…dd8`。

## 4. Replay 与 negative evidence

- clean-source replay exit 0，stdout 与冻结 `replay_stdout.txt` 逐字节一致；
- 12/12 fully re-signed mutations 被 semantic replay 拒绝；
- tamper hash=`409a73435b64b6b6f26e7dbacf0f05c8ff3f096fe0b873dd783097fbdaa9de22`；
- 覆盖 start-node、topology、beta、split/history、fanout、BiasSplit、scratch overlap、dense
  escape、context tensor、production/performance claim 与 summary gate。

## 5. Claim boundary 与下一动作

允许的唯一升级是：R3-0 static contract 已绑定 production compressed-alpha shape。不得将 synthetic
pointer/ledger 写成真实运行结果，也不得形成 correctness、memory、speedup 或 ASPLOS-ready claim。

下一唯一动作是 R3-1：只接 `25/Conv_8`，一个 evaluation，optimizer mutation=`0`；candidate 必须
恰一次 forward 与恰一次 custom backward，并以五个独立 fresh native worker 比较 terminal lower、
sign、compressed dα 与 empty beta。同时证明 saved dense A=`0`、scratch slot `<=2`、无 native
shadow/fallback/eager。R3-1 不计时；通过前 R3-2A/2B 保持关闭。
