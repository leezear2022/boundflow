# 2026-07-14：PR-12J compile/load/cache 与重复查询摊销

## 目标

把 PR-12I 中“first-run 与 cold 的差值”升级为真实阶段拆分，并回答同 shape repeated-query 需要
多少次才能摊销 TVM fused CROWN 的编译或跨进程加载成本。固定查询数为：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

## 实现

- Linear/Conv2d schedule 新增接受已构造 PrimFunc 的入口，使 TIR generation 与 schedule 不再
  因 helper 内部重建而重复计时；
- 新增 `FusedCrownModuleCache`，cache key 覆盖 signature、target、schedule/code schema 与 TVM
  version；
- 每个 cache entry 为 `.so + JSON manifest`，加载前验证完整 canonical payload 与 library
  SHA256；
- 进程内冻结 signature 直接命中 packed function；跨进程通过 `tvm.runtime.load_module` 加载；
- 原子写入临时文件后 replace，损坏或不匹配 entry 会确定性重新编译；
- executor 可选接入该 cache，默认无 cache 时保持既有 lru 路径；
- runner 分离 Planner/region construction、TIR generation、schedule、`tvm.compile`、serialization、
  module load、first query、memory-hit cold/warm 和独立 Python 进程 disk hit；
- 后处理生成 compile phase CSV、Q-sweep CSV、三张 amortization 图、summary 与 manifest。

全量回归还修正两项集成合同：旧 tracking subclass 未调用父类 `__init__` 时，executor 以
`getattr(..., None)` 保持向后兼容；custom-stream 回归在 default-stream 生成输入后显式执行
consumer `wait_stream(producer)`，避免把调用方缺失的跨 stream 数据依赖误判为 FFI bridge 失败。
测试仍只同步 custom stream，不使用全局同步。

## 运行中发现并保留的失败

- `pr12j-amortization-v1-20260714`：Linear 跨进程命中，但 Conv signature 的 tuple 与 JSON list
  直接比较，导致 Conv/mini-ResNet 同 key 被误判失效；1 ok、2 fail。修复为 cache payload 在
  hash 与比较前统一 canonical JSON；
- `v2`：3/3 correctness/disk cache 通过，但 memory hit 每次重新读取并 SHA256 `.so`，人为污染
  warm path；
- `v3`：移除上述 warm-path 磁盘开销；
- `v4`：在 v3 基础上同时报告 module-load first-query 与完整 process-restart wall model，作为
  唯一权威版本。

这些目录均保留，不能删除 v1/v2 来隐藏测量实现问题。

## 权威工件

```text
artifacts/phase7a-pr12/pr12j-amortization-v4-20260714/
artifacts/phase7a-pr12/pr12j-amortization-report-v4-20260714/
```

协议：PR-12I 相同 3 个已消费 calibration workload；default stream；warmup 5；5 groups×10；
3/3 complete final-bound 与 process-restart disk-cache correctness 通过。每个 worker 的 cache
event 均为 `disk_hit`，没有跨进程隐式重编译。

关键 SHA256：

```text
raw.jsonl:        ab8794fbe592421d2310885e948207dc32d9788ad690a99d0c183385989c17a0
compile_phases:   c3907ddb25925c431eceb8b2754aeb7dcf1695a74bd51e01724380fa75b71eae
amortization.csv: bab089dd1139d9dc8d7f59985f7295977db6c78526e3f10c9e3c9e1f3d5bc8d8
summary.json:     8fdc1259d80524410760e15e4c06fc6ab9e7add4f88aa0ec138a7ff5d4081abd
```

## 结果

| workload | modules | TIR / schedule / compile / serialize | warm fused | eager / chunked | fresh / disk-first / process break-even vs eager |
|---|---:|---:|---:|---:|---:|
| Linear memory-sensitive | 1 | 1.07 / 1.36 / 480.00 / 71.69 ms | 8.557 ms | 1.736 / 2.112 | not amortizable |
| Conv unseen-width | 1 | 4.73 / 1.61 / 323.67 / 74.24 ms | 3.301 ms | 1.386 / 1.345 | not amortizable |
| mini-ResNet | 4 | 20.65 / 6.11 / 1299.12 / 295.05 ms | 6.847 ms | 7.234 / 6.513 | 4668 / 1062 / 4450 queries |

disk module load 本身仅为 0.17–0.60 ms，但新进程第一次 CUDA query 仍为约 350–419 ms，说明
共享库 load 不是 restart cold-start 的主要组成。完整 process wall 还包含 Python/import、query 与
结构化 correctness 输出，作为保守 process-restart 模型单独报告，不与纯 module-load 混写。

mini-ResNet fused warm 对 eager 有小幅收益，因此数学上可摊销，但三种 setup 都超过冻结的
1024-query 观察区间；对更快的 chunked baseline 则仍为 `not_amortizable`。Linear/Conv fused
warm 本身慢于两个 baseline，故不生成负 break-even 数字。

## 判定

```text
PR-12J phase decomposition:       PASS
PR-12J memory/disk/restart cache: PASS
PR-12J correctness:               PASS
amortizable within Q<=1024:       FAIL (0/3 vs eager)
PR-12 overall:                    IN PROGRESS
PR-13:                            BLOCKED
```

这项负结果提高了后续 Planner 的约束：compile-aware Planner 不能仅看到理论 disk-load 0.6 ms，
必须使用 first-query/process cold-start 与 expected query reuse；现有目标 workload 不应默认选择
fused 追求 latency。下一阶段严格进入 PR-12K profiler，不先优化 schedule。

## 收尾验证

```text
PR-12J focused/integration: 5 passed
全量：                       330 passed、1 skipped
mypy：                       6 source files success
pylint：                     6 core/script files 10.00/10
Black / git diff --check：   通过
```

9 条 warning 均为既有上游 deprecation/future warning；唯一 skip 是 TVM 已可用时避免重复编译
的 allow-no-tvm smoke。
