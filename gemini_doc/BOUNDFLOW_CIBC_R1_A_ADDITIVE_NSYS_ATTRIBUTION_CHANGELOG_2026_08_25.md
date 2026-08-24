---
status: implemented-pending-formal-run
updated: 2026-08-25T01:42:00+08:00
type: changelog
topic: boundflow
slug: cibc-r1-a-additive-nsys-attribution
stage: s01
---

# BoundFlow CIBC R1-A Additive/Nsight Attribution 修改记录

## Summary

- 在不改变默认 CIBC 执行行为的前提下，实现 opt-in source-op marker、fresh control/profile worker、
  CUPTI/NVTX clock anchor、Nsight SQLite graph-node owner 重放、artifact replay 与全重签篡改探针。
- `performance_claimed=false`；正式 6-pair 结果尚未生成，本提交只开放在 clean commit 上执行 R1-A
  formal，不开放 R1-B/R1-C/R2。

## Changes

- `boundflow/runtime/cibc_ibp_graph.py`：新增可选 `op_context_factory`；默认 `None` 时执行和输出保持原样，
  R1 runner 可在 graph warmup/capture 时为 17 个 source op 注入稳定 NVTX range。
- `scripts/run_cibc_r1_attribution_worker.py`：
  - 冻结 17-op production topology 与 6/2/6/2/1 bucket inventory；
  - control=`10 warmup + 20×50 replay`，profile=`10 warmup + 20×5 replay`；
  - lower/upper copy 均包含在 CUDA-event wall；
  - 支持 `torch` smoke 和外层 `nsys` 两种 profile backend；
  - Nsight backend 记录 20 个 group range 与 3 个双向 CUPTI/NVTX anchor。
- `boundflow/runtime/cibc_r1_nsys.py`：从只读 SQLite raw 重建：
  - 3-anchor affine error 与 formal clock receipt；
  - capture marker containment → original graph-node owner → cloned replay graph-node；
  - kernel/memcpy/runtime inventory、single stream、unowned/temporal fallback；
  - owner ledger 与四口径 timing ledger。
- `scripts/run_cibc_r1_attribution_artifact.py`：固定 model/source/topology/semantic identity、CP/PC 顺序、
  source code blob，支持 atomic raw-first generate/replay；formal 前置要求 clean tracked tree 与 `nsys`。
- `scripts/probe_cibc_r1_attribution_tamper.py`：9 类 payload 在同步重签 protocol/worker/summary/manifest 后
  仍必须由语义重算拒绝。
- 系统环境安装 Arch 官方 `extra/nsight-systems 2026.1.3.425-1`；`nsys status --environment` 显示
  timestamp counter、process-tree profiling 可用。system-wide CPU sampling fail 不影响本轮 CUDA/NVTX trace。

## Validation

- 新增/相关专项与 CIBC/FSG3/B3 artifact 组合：`49 passed`。
- Black：pass；mypy：clean；Pylint：`10.00/10`；`git diff --check`：pass。
- torch smoke artifact：generate/replay 逐字节一致；CUPTI admitted；profile perturbation
  `1.1702x/1.1761x`，按 `[0.95,1.05]` 正确拒绝；9/9 fully re-signed tamper rejected。
- 真实 Nsight 单 profile 探针（RTX 4060 Laptop）：
  - clock p95/max/residual=`1513/2845/496 ns`，slope/anchor drift=`0.974 ppm/13 ns`；
  - anchor error=`221/445/224 ns`，formal clock admitted；
  - capture graph node=`42`，clone map=`138`，20 group/100 replay；
  - kernel/memcpy/runtime/graph launch=`4200/200/520/100`；
  - owner events=`4400`，unowned=`0`，temporal fallback=`0`，stream=`[7]`；
  - profile median=`0.11558 ms`。该单探针不是正式 6-pair performance result。

## Decisions

- 不使用 torch-profiler 形成 formal share；其约 17% 扰动只保留为 smoke rejection evidence。
- Nsight owner mapping 只接受 capture marker containment、`originalGraphNodeId` clone edge 与 kernel
  `graphNodeId`，不提供 temporal fallback。
- flatten/view 无 CUDA graph node是合法的零 device-wall bucket；未归属设备事件仍必须为零。
- 单 stream 强制 `exclusive_wall == critical_path == overlap_adjusted_wall`；runtime/sync bucket只接收
  group wall 减去已归属 kernel/memcpy 的剩余，不以 overlap 修饰 headline。

## Follow-Ups

- 在本提交成为 clean source 后生成 6-pair formal artifact，并由 replay 重算 perturbation verdict。
- 若任一 pair 超出 `[0.95,1.05]`，R1-A formal NO-GO，R1-B 与后续优化保持关闭；不得调宽门槛。
- formal closure 后更新 claims map、execution memo/current status 与外审交接。

## Links

- plan: `gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- R1-0: `gemini_doc/BOUNDFLOW_CIBC_R1_0_CONTRACT_CLOCK_TOPOLOGY_CHANGELOG_2026_08_25.md`
