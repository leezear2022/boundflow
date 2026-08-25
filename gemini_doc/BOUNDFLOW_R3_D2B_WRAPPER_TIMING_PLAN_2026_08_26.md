---
status: preregistered-formal-timing-open
updated: 2026-08-26T00:35:00+08:00
type: plan
topic: boundflow
slug: r3-d2b-wrapper-timing
stage: s01
---

# R3-D2-B Wrapper-Inclusive Timing 预注册

## 1. 比较对象

冻结 correctness closure source=`f08054d`。每个 run 用独立进程构造同一 production ResNet2B state，
交错执行三个 mode：

- `native`：原 αβ-CROWN/PyTorch exact solver region；
- `d1c`：forward staged、backward raw residual 的直接 control；
- `d2b`：forward 与 backward 都使用 staged residual 的 candidate。

五个 run 使用五个不同顺序；每个 worker 3 warmup + 30 个完整 10/9 wrapper host-wall sample。clock 为
`perf_counter_ns`，每个 sample 前重置 α，结束边界同步 non-default CUDA stream。不得用 CUDA-event sum
替代 wrapper headline。

## 2. Region 归因

30 个 headline sample 完成后，另跑一次 instrumented wrapper，只在 `_coefficient_sign_pass` parent 周围
记录 10 对 CUDA event。该 wrapper 不进入 latency headline；D1-C raw region 与 D2-B staged region 形成
同 run ratio。禁止把 symbol/kernel sum 当 region wall，禁止 overlap adjustment。

## 3. 冻结门禁

逐 run 先验证 terminal lower 对 native/D1-C `atol=rtol=2e-4` 且 sign exact，terminal α
`atol=rtol=2e-5`，execution/ownership receipt 全部成立。

- region physical gate：每个 run `raw_sign_ms/staged_sign_ms ≥ 11.8762x`；
- qualification：candidate/native geomean 与 worst 均 `≥1.00x`；
- research：candidate/native geomean 与 worst 均 `≥1.20x`；
- D1-C recovery 必须单列 `d1c/candidate`，但不是最终 claim；
- peak allocated/reserved 对 D1-C 不得上升；
- 若 research 通过则 `VALIDATED-R3-D2B-WRAPPER-RESEARCH`；只过 parity 则
  `VALIDATED-REDUCED-R3-D2B-PARITY`；parity 失败或 region gate 失败则 NO-GO。

不允许因结果修改 `11.8762x/1.00x/1.20x`。formal 通过前 `performance_claimed=false`；R3-3、
same-solver、query/queue 与 ASPLOS claim 继续关闭。

## 4. Artifact

artifact 必须包含 15 个 raw worker、protocol、summary、manifest 与 fully re-signed tamper report；replay
从 30 个 sample 重算 median、geomean、worst、region ratio、语义与最终 verdict。旧 D1-C artifact 只作
历史交叉核对，不替代同轮 control。

