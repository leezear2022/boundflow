---
status: validated-r3-d2b-trajectory-correctness
updated: 2026-08-26T00:20:00+08:00
type: closure
topic: boundflow
slug: r3-d2b-correctness-formal
stage: s01
---

# R3-D2-B Staged Backward Correctness 正式关闭

## 1. Verdict

D2-B correctness 以 `VALIDATED-R3-D2B-TRAJECTORY-CORRECTNESS` 关闭。只开放 D2-B five-fresh
wrapper-inclusive timing；R3-3、same-solver、query/queue 与 ASPLOS performance claim 继续关闭。

本轮证明 staged residual6/residual11 可在 production coefficient-sign backward 内保持完整 10/9
optimizer 轨迹与结构化所有权。没有记录 latency，也没有更改默认 production route。

## 2. 冻结证据

- source revision：`f08054d3165976f1b5a1e97c22dbb014b47385de`；
- artifact：`artifacts/r3-structured-owner/r3-d2b-correctness-v1`；
- protocol hash：`8bcd373095bf1ed09e7c052d8880a6ef5c6f30e1ec370df650d054ecf1f94693`；
- summary hash：`c30f59f4d78d75fe092c524986a3a9ce2ed98bbdc62da3a05e91b56220702014`；
- manifest hash：`5c2abb782af9780503d1466f3199eb24b6fa8127e7d46ca004e2b11c6c9d9703`；
- 5 pair / 10 fresh process，顺序为 control-first、candidate-first 交错；
- 10 raw、protocol、summary、12-case tamper report 全部由 manifest SHA256 绑定。

## 3. 逐步语义

每个 process 独立构造 plan/trace/tensors/optimizer；每个 pair 对 10 个 evaluation 逐步比较：

| 字段 | tolerance | five-pair maximum diff |
|---|---:|---:|
| lower | `atol=rtol=2e-4` + sign exact | `0.0` |
| compressed dα | `atol=rtol=2e-4` + sign exact | `0.0` |
| α after update | `atol=rtol=2e-5` | `0.0` |
| Adam exp_avg | `atol=rtol=2e-5` | `0.0` |
| Adam exp_avg_sq | `atol=rtol=2e-5` | `0.0` |

alpha lineage、tensor SHA256、step projection hash、trajectory hash 均由 replay 从 raw 重算；固定 10
evaluations、9 optimizer mutations、9 scheduler mutations。最后一次无 update，Adam step 保持 9。

## 4. Ownership 与执行合同

每个 evaluation 同时验证：

- D1-C forward staged launch=`4`；
- D2-B backward staged launch=`4`；raw B1 backward launch=`13`；
- backward bias in-place alias=`2`；existing arena=`2`、scratch region=`2`；
- scratch 为 `s1[6144:12288]` 与 `s0[12288:18432]` 的 caller-owned tail；
- persistent dense A、saved autograd history、global workspace 均为零/false；
- fallback、eager candidate、native shadow 均为零；
- `timing_recorded=false`、`performance_claimed=false`。

12/12 fully re-signed tamper 拒绝，覆盖 lower、dα、alpha lineage、Adam moment、claim、scratch pointer、
control receipt、protocol tolerance/order 与 summary gate 漂移。

## 5. 验证

- black：通过；mypy：clean；pylint：`10.00/10`；
- targeted：`6 passed`；
- full regression：`1647 passed, 3 skipped, 6 warnings in 674.65s`；
- 三个 skip 均为既有 TVM duplicate-compile / frozen VNN-COMP checkout 环境边界。

## 6. 下一门禁

只开放 wrapper-inclusive timing。直接对照是 D1-C，并带回同 pair 冻结 native latency：

1. correctness worker 与 timing worker 分离，计时路径不得序列化 step tensor；
2. 先运行固定 warmup/readiness，再采集 ≥30 host-wall sample；
3. 同时采集 raw/staged coefficient-sign region，检查 worst region speedup 是否达到 D2-A 的
   `11.8762x`；
4. whole candidate/native geomean 与 worst 均 `≥1.20x` 才通过 research gate；
5. 若只恢复 native parity，则 reduced；若 parity 也失败，则 NO-GO；门槛不事后改写。

