---
status: validated-no-go-r1a-attribution
updated: 2026-08-25T03:20:00+08:00
type: changelog
topic: boundflow
slug: cibc-r1-a-additive-nsys-attribution
stage: s01
---

# BoundFlow CIBC R1-A Additive/Nsight Attribution 修改记录

## Summary

- 在不改变默认 CIBC 执行行为的前提下，实现 opt-in source-op marker、fresh control/profile worker、
  CUPTI/NVTX clock anchor、Nsight SQLite graph-node owner 重放、artifact replay 与全重签篡改探针。
- `performance_claimed=false`；clean source 上的正式 6-pair 已完成，结果为
  `VALIDATED-NO-GO-R1A-ATTRIBUTION`。R1-B/R1-C/R1-D/R2 全部保持关闭。

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
- 正式 source=`fe80c754fabaa13ac917ba556d156f13b02a42ae`，顺序固定为
  `CP/PC/CP/PC/CP/PC`；六组 control median=`0.097669/0.097853/0.097690/0.097689/
  0.097690/0.097883 ms`，profile median=`0.115830/0.115955/0.115674/0.115645/
  0.115824/0.116074 ms`。
- 六组 profiler perturbation=`1.185947/1.184988/1.184093/1.183806/1.185633/1.185838x`，
  `0/6` 落入冻结的 `[0.95,1.05]`；CUPTI clock receipt 为 `3/6` admitted。因此
  `formal_attribution_admitted=false`，不能从这些 profile wall 形成 op share。
- 每组结构证据一致：`42` graph nodes、`138` clone mappings、`20` groups/`100` replays、
  `4200` kernels、`200` memcpy、`520` runtime APIs、`100` graph launches、`4400` owner events；
  unowned/temporal fallback=`0/0`，single stream=`[7]`。
- root replay 从六份 SQLite 和 raw 重算后逐字节复现 summary；summary hash=
  `16e46384830993bf5850dfd1ca84823795580ce31234f9e04634c6acb01ca583`。
- 9 类 scope/source/order/clock/semantic/timing/topology/verdict 全重签篡改均被拒绝；tamper hash=
  `112e23e0b333ca5fda18c34dca990e6602e1123964fb68337481b0d8b40fde5a`。
- 正式 artifact=`artifacts/cibc-r1-optimized-graph-attribution/resnet2b-prop0-v1`，含原始
  `.nsys-rep`/SQLite、owner ledgers、replay stdout 与 tamper report；manifest SHA256=
  `62a15384676b00dc006699928380258a36cf3113595e34d84db1f4a01d9d9e13`。

## Decisions

- 不使用 torch-profiler 形成 formal share；其约 17% 扰动只保留为 smoke rejection evidence。
- Nsight owner mapping 只接受 capture marker containment、`originalGraphNodeId` clone edge 与 kernel
  `graphNodeId`，不提供 temporal fallback。
- flatten/view 无 CUDA graph node是合法的零 device-wall bucket；未归属设备事件仍必须为零。
- 单 stream 强制 `exclusive_wall == critical_path == overlap_adjusted_wall`；runtime/sync bucket只接收
  group wall 减去已归属 kernel/memcpy 的剩余，不以 overlap 修饰 headline。
- 正式结果按预注册门禁关闭为 NO-GO：不放宽 `1.05`、不挑选 3 个 clock-pass pair、不把完整但
  不可准入的 owner ledger 用于 headline share。
- 一次早期正式尝试在 pair 2 遇到 clock rejection 后暴露 runner 会丢弃负证据；随后只修复
  “保存/序列化 NO-GO”流程，没有改变门槛或实验参数。可审计失败尝试保留在本机 ignored
  `.failed` 目录，不进入正式 artifact 与 claim。

## Follow-Ups

- R1-B same-solver share、R1-C query-local replay、R1-D feasibility 与条件 R2 不再开放；不能以
  “换 profiler/调宽扰动/重跑挑选”复活本协议。
- 依据既有 reprioritization 条款，下一独立工程阶段转为 R3-0：只实现 structured-owner/custom-VJP
  的 IR/Template/Instance、closure/liveness、receipt 和负向验证器；不接 production、不计时。
- R3-0 必须另立修改记录和 artifact/replay 合同；R3-1 仍由 R3-0 正式关闭结果门控。

## Links

- plan: `gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- R1-0: `gemini_doc/BOUNDFLOW_CIBC_R1_0_CONTRACT_CLOCK_TOPOLOGY_CHANGELOG_2026_08_25.md`
