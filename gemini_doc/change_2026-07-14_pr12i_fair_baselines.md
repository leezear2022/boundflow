# 2026-07-14：PR-12I 公平 baseline 与归因闭环

## 目标

在 PR-12H 冻结的 region-runtime 与 end-to-end final-bound 合同下，补齐 structured eager 和
TVM unfused 两个关键归因对照，并对 `torch.compile` 做不改写 workload 的条件探测。PR-12I
只回答 baseline 与机制归因，不提前做 compile amortization、profiler 或新 schedule。

## 实现

- 新增 `TVMUnfusedCrownExecutor` 和显式 `scaled_u/scaled_l` 输出的 Linear/Conv2d TIR；其
  PyTorch-owned workspace 会进入统一 allocator peak 口径；
- 统一比较 dense eager、structured eager、chunked-r512、TVM unfused、TVM fused TIR；
- region-runtime 包含 backend dispatch、输出/workspace 分配、DLPack/TVM-FFI 与 stream bridge；
- complete final-bound 每次计时均重建固定后端选择，并包含真实 plain-CROWN 与 concretization；
- default/custom stream 均使用被测 stream 的 CUDA Event，不在 timed path 使用全局同步；
- 所有失败、N/A 与 compile probe 原因均保留在 JSONL；
- 新增 JSONL→CSV→Pareto figure→summary/manifest 后处理与合同测试。

structured eager 保留 operator representation，无法公平映射到以 dense `A_u/A_l` 为边界的
region API，因此 region-runtime 明确记为 N/A，只在完整 final-bound 合同中比较。TVM unfused
则显式写回两份 scaled-A workspace，用于区分“TVM codegen”与“BoundFlow fusion 消除中间量”。

## 条件 `torch.compile` 结果

使用 `fullgraph=True, dynamic=False` 在三类未改写 complete final-bound workload、default/custom
stream 上实际探测。6/6 E2E probe 均在首次 capture/compile 时被 Dynamo 拒绝，直接原因是
`_relu_backward_mode` 的 `ContextVar.set` 无法被 fullgraph trace；region 合同因不代表完整
solver 而记为 N/A。没有拆掉 host-side graph/Planner 语义来迎合 TorchInductor，因此它不进入
数值/性能表，但结构化失败仍保留在 raw JSONL。

## 权威工件

```text
artifacts/phase7a-pr12/pr12i-baseline-v2-20260714/
artifacts/phase7a-pr12/pr12i-baseline-report-v2-20260714/
```

协议：3 个已消费 calibration workload；default/custom stream；warmup 5；5 个独立 group；
每组 10 次。共 72 行：54 `ok`、18 `not_applicable`、0 correctness failure。18 个 N/A 由
6 个 structured-region、6 个 torch.compile-region 和 6 个 torch.compile-E2E probe 组成。

关键 SHA256：

```text
raw.jsonl:   fe65faf578a6597aeac85c4435348ff1bd9fac85d06f353114406bf9573ebc31
baseline.csv: 38206a7bfa4078b95f3b9878c197ad9b5107a19571c7013203922d976f6b04f2
summary.json: 5d6ffe8a8f2d3ca14d98047ecf56c08ef46ddfd35108ba38a7f5655b88e264f6
```

## 结果与解释

默认 stream 的 complete final-bound 代表点：

| workload | eager latency / peak | structured | chunked | TVM unfused | TVM fused |
|---|---:|---:|---:|---:|---:|
| Linear memory-sensitive | 1.736 ms / 68.60 MiB | 3.324 / 44.51 | 2.112 / 46.95 | 6.883 / 44.51 | 8.644 / 28.46 |
| Conv unseen-width | 1.386 ms / 23.06 MiB | 2.128 / 14.63 | 1.345 / 38.15 | 2.975 / 17.44 | 1.768 / 11.82 |
| mini-ResNet | 7.234 ms / 18.83 MiB | 49.683 / 9.62 | 6.513 / 26.21 | 7.657 / 17.42 | 7.009 / 17.42 |

E2E geomean speedup 相对 eager：chunked 0.980×、structured 0.367×、TVM unfused 0.481×、
TVM fused 0.546×。fused 的 median peak ratio 为 0.512，并在 3/3 workload 处于 Pareto；
TVM unfused E2E 0/3 Pareto，说明仅换成 TVM codegen 不能解释 fused 的显存收益。另一方面，
fused 在 memory-sensitive Linear 明显更慢，不能作为统一 latency headline。

mini-ResNet 的绝对 bound 数值很大，故 TVM 路径出现 `max_abs=12288`，但相对误差约
`2.20e-7`，并通过统一 `rtol=atol=2e-4` 的 `torch.allclose`、finite 与 lower≤upper 门禁；原始
绝对值和相对值均保留，不能只展示较好看的相对误差。

## 判定

```text
PR-12I baseline implementation: PASS
PR-12I contract/correctness:     PASS
PR-12I performance headline:     NOT CLAIMED
PR-12 overall:                   IN PROGRESS
PR-13:                           BLOCKED
```

本阶段证明候选确有不同 latency-memory regime，也确认 TVM unfused 不是有竞争力的 E2E 候选。
下一阶段严格进入 PR-12J：真实拆分 compile/serialize/load/cache/restart，并按 query reuse 计算
break-even；不得根据本结果先调 TIR schedule。

## 收尾验证

```text
PR-12H/I focused:  9 passed
全量：              327 passed、1 skipped
mypy：              6 source files success
pylint：            6 core/script files 10.00/10
Black / diff check: 通过
```

9 条 warning 均为既有上游 deprecation/future warning；唯一 skip 是 TVM 已可用时避免重复编译
的 allow-no-tvm smoke。全量测试必须在已激活的 `boundflow` 环境中启动，确保其子进程继承正确
的 Python；直接用环境 Python 但不更新 `PATH` 会使旧 Phase-6 shell runner 的 `python3` 落到系统
解释器，这属于启动方式错误，不是测试失败。
