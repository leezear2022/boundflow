---
status: validated-r3-d2b-wrapper-research
updated: 2026-08-26T01:40:00+08:00
type: closure
topic: boundflow
slug: r3-d2b-wrapper-timing-formal
stage: s01
---

# R3-D2-B Wrapper-Inclusive Timing 正式关闭

## Verdict

R3-D2-B 以 `VALIDATED-R3-D2B-WRAPPER-RESEARCH` 正式关闭。5 个 fresh triplet 的
candidate/native geomean=`1.752001x`、worst=`1.724843x`，同时通过冻结 `1.20x` research gate；
coefficient-sign region worst=`53.9195x≥11.8762x`。只开放 R3-3 S-anchor active-beta correctness；
multi-site、R3-4+、same-solver、query/queue 与 ASPLOS-ready 继续关闭。

## 冻结证据

- source=`3ee5920a4d6b8d6fec1d08bd25ec9245152f6a24`；
- artifact=`artifacts/r3-structured-owner/r3-d2b-wrapper-timing-v1`；
- protocol hash=`9de4f248e0fdc0312910788bb08f94887645c3ba6788f06d4ae94eb287e94105`；
- summary hash=`3e8bf9103e632b8eb975ad6737b0b4d67cde62a15f3ed253b0dc8f95d5a027f3`；
- manifest hash=`d74ec8656573969dadf2da2aea31851552886c5ce89bf3b593e6f1f4bb5251fc`；
- 15 fresh workers × 30 host samples=`450`，每 worker 3 warmup；
- 12/12 fully re-signed tamper rejected；artifact tests=`2 passed`。

artifact summary 故意保持 `performance_claimed=false/r3_3_open=false/pending_tamper=true`，防止仅凭数值
gate 提前升级。正式 claim/open 由本 closure 在 tamper 完成后授予。

## 正式数字

| run | native ms | D1-C ms | D2-B ms | D2-B/native | D1-C recovery | region |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.154 | 395.721 | 56.368 | 1.7413x | 7.0203x | 55.7345x |
| 1 | 98.942 | 396.004 | 56.484 | 1.7517x | 7.0109x | 55.7779x |
| 2 | 100.359 | 397.510 | 58.184 | 1.7248x | 6.8319x | 53.9195x |
| 3 | 100.812 | 397.920 | 56.744 | 1.7766x | 7.0126x | 55.8272x |
| 4 | 100.792 | 397.826 | 57.072 | 1.7660x | 6.9706x | 55.2765x |

D1-C recovery geomean/worst=`6.968886x/6.831907x`。candidate 对 native terminal lower max diff=
`7.39098e-06`、α max diff=`2.38419e-07`、sign exact；candidate/D1-C peak allocated/reserved
五次均=`1.0x/1.0x`。

## 审计纪律

首轮 artifact 在 tamper 前提前写 claim/open，被拒绝；第二轮发现 protocol research threshold 可重签降级
而 replay 未拒绝，同样被拒绝。最终 replay 冻结 schema、source、code hashes、顺序、3/30、region
`11.8762x`、parity `1.0x`、research `1.2x`，再完整重跑。失败 artifact 均保留于 `/tmp`，不进入正式链。

本 claim 仅覆盖固定 P-anchor、单 production ResNet2B region、完整 10/9 local wrapper；不得外推为
same-solver、complete query、queue 或多模型收益。

