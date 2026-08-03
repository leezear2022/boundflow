---
status: completed
updated: 2026-08-04T04:21:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1
stage: s01
---

# Native Real-Network CUDA Memory Protocol v1 Plan

## Goal

- 在任何 CUDA 结果产生前，冻结 NRIR-2 retain-all/lifetime-reuse 的 device-level
  physical-memory 与 lower-only latency 测量协议。
- 实现 fresh-process runner、raw rows、summary、manifest 与 semantic replay/validation；环境
  不满足时必须 fail closed，不能生成 `status=ok` 或 performance claim。
- 用预注册门禁决定是否存在可重现的 real-network latency-memory Pareto；不以逻辑 arena
  byte ledger 替代 CUDA allocator counter。

## Scope

- 固定 workload 与 NRIR-2 相同：VNN-COMP 2021 ResNet2B prop0、相同 ONNX/VNNLIB/
  αβ-CROWN/intermediate-bound digest、相同 21-op native Bound graph。
- 两个被测 plan 仅为 `storage:native-retain-all-v1` 与
  `storage:native-lifetime-reuse-v1`；同一 Bound/PlanTemplate identity。
- 每个 plan 使用独立 fresh worker process，避免 allocator/cache state 跨 plan 污染。
- 正式配置冻结为 5 repeats；每 worker warmup 5、measured 20；每次 query 前后 CUDA
  synchronize。计时只覆盖 prepared native CROWN execution，不包含 ONNX import、forward
  bounds、compile、artifact load 或 correctness comparison。
- 每个 worker 在 measured loop 前记录 baseline allocated/reserved，并调用
  `torch.cuda.reset_peak_memory_stats()`；输出 peak allocated/reserved 与 baseline delta、每 query
  latency、device/driver/torch/CUDA identity、IR hash、result tensor hash。
- 每个 repeat 的启动顺序交替：偶数 retain→reuse，奇数 reuse→retain；两者仍是独立进程。
- 预注册门禁：10/10 worker rows `status=ok`；每 plan 5/5 correctness；结果 hash 相同；
  reuse median peak-allocated delta 至少降低 20%；reuse median lower-only latency 不超过
  retain 的 1.20×；所有 raw/summary digest 可 replay。
- `max_memory_reserved` 只报告不设通过阈值；无实际 OOM 不得声明 OOM rescue。

## Tasks

- [x] 冻结 workload、进程隔离、warmup/repeat/iteration、counter、计时边界与 Go/No-Go 阈值。
- [x] 审计本机 CUDA 环境：PyTorch `2.12.1+cu132`，但 CUDA unavailable、device count 0，
  `nvidia-smi` 无法连接驱动。
- [x] 实现 worker、orchestrator、schema validation、summary 与 manifest/replay。
- [x] 新增无 CUDA fail-closed、protocol config、aggregation/tamper tests。
- [x] 在本机生成 `environment_unavailable` probe evidence；未生成正式 benchmark artifact。
- [x] 完成静态/全量验证与 DocOps 收口；代码发布后由可用 CUDA 主机执行冻结命令。

## Validation

- 当前环境探针必须稳定返回非零 benchmark exit，并包含：torch/CUDA build、
  `cuda_available=false`、`device_count=0`、`nvidia-smi` failure；不得出现 measured rows。
- 单测必须覆盖 row count/identity/semantic hash、20% memory threshold、1.20× latency threshold、
  digest/tamper、缺 CUDA 与 worker 非零退出。
- 正式 CUDA artifact 只有在 10/10 rows 和所有预注册门禁通过后才允许写
  `performance_claimed=true`；否则为 `NO_GO` 或根本不生成 manifest。
- 本机 probe artifact 位于
  `artifacts/native-real-network-cuda-memory-protocol/environment-unavailable-20260804/`；
  `probe` 按预期 exit 2，`probe-replay` 复核 digest/status 后 exit 0，`generate` 在创建输出目录
  前 exit 2。
- 聚焦协议/runtime 回归 `17 passed`；全量 `484 passed, 37 skipped`；Mypy 4 files clean，
  Pylint 10.00/10，Black 与 `git diff --check` 通过。

## Rollback

- runner/protocol 不修改默认 runtime/backend 行为；删除新增 runner、tests、probe artifact 与
  本文即可撤回，不影响 NRIR-1/2 artifacts。

## Closure

- 本阶段关闭的是可执行、可重放的 CUDA 测量协议及 unavailable-environment 证据，不是
  CUDA 性能实验。当前没有 raw worker row、allocator reduction、latency、OOM rescue 或 speedup
  结果。
- 下一工程阶段为 representation semantic binding bridge：Plan representation decision 必须
  驱动真实 Bound rewrite/backend conversion 和 Schedule materialization，而不是只改变 metadata。

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_CHANGELOG_2026_08_03.md`
- prior: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
- publication: PR `#14`, head `f67fbfe`
