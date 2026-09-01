---
status: implemented-smoke-passed-formal-pending
updated: 2026-08-26T00:50:00+08:00
type: changelog
topic: boundflow
slug: r3-d2b-timing-worker
stage: s01
---

# R3-D2-B Timing Worker 修改记录

- 新增 native/D1-C/D2-B 三模式独立进程 worker；
- 每个 worker 固定 3 warmup + 30 个完整 10/9 host-wall sample；
- headline 后另跑一次 10-event coefficient-sign parent measurement，明确禁止进入 headline；
- 冻结 terminal、execution、D1-C/D2-B ownership、arena pointer 与 memory receipt；
- black、mypy、pylint `10.00/10` 通过。

单 triplet smoke：native/D1-C/D2-B median=`97.608/393.543/56.634 ms`，candidate/native=
`1.72349x`，D1-C recovery=`6.94886x`，raw/staged coefficient-sign=`55.7662x`；D2-B 与 D1-C peak
allocated/reserved相同。该结果只证明 worker 与量级可执行，`performance_claimed=false`，必须由 five-fresh
formal 重算。

## Formal replay 与状态机修正

首轮 formal 数值通过后发现 summary 在 tamper 前提前写入 claim/open；该 artifact 未提交并可恢复地保存在
`/tmp/r3-d2b-timing-premature-claim.I73iEd/`。修正后的 source=`5e4fed1` 完整重跑得到
candidate/native geomean/worst=`1.65905x/1.49073x`、D1-C recovery geomean/worst=
`6.79727x/5.97433x`、region worst=`47.9682x`。当前只标记 research gate passed pending tamper，
尚无正式性能 claim。

首轮 source-bound tamper 在 `protocol-research` 案例 fail：攻击者把 research threshold 从 `1.2` 改为
`1.0` 并重签 protocol/manifest 后，replay 仍接受。原因是 replay 只冻结 region gate 与顺序。该 artifact
不关闭；replay 改为逐项冻结 schema、3/30、region/parity/research 三门槛、顺序、source 与全部 code
hash，并要求新 revision 完整重跑。
