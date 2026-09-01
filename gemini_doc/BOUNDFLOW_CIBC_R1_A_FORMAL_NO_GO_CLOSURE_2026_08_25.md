---
status: validated-no-go-r1a-attribution
updated: 2026-08-25T03:20:00+08:00
type: closure
topic: boundflow
slug: cibc-r1-a-formal-no-go-closure
stage: s01
---

# BoundFlow CIBC R1-A Formal NO-GO Closure

## 1. 结论

R1-A 已按冻结协议在 RTX 4060 Laptop 上完成六组 fresh control/profile 正式执行、raw-first
artifact、SQLite semantic replay 和全重签篡改探针。最终状态为：

`VALIDATED-NO-GO-R1A-ATTRIBUTION`

该结论关闭的是“用当前 Nsight/CUPTI additive profile 形成 optimized CIBC graph 的可传播
op-type wall share”这一测量路径。它不撤销既有 CIBC graph `2.45631x` reduced claim，也不说明
Linear/Conv/runtime 优化物理上无效；但在没有合格 share 的前提下，R1-B、R1-C、R1-D 和条件 R2
均不得继续。

## 2. 冻结身份与协议

- source commit：`fe80c754fabaa13ac917ba556d156f13b02a42ae`；tracked source clean，唯一允许的
  hook 追加为 `.docops/ev.jsonl`；
- formal artifact：`artifacts/cibc-r1-optimized-graph-attribution/resnet2b-prop0-v1`；
- pair order：`CP/PC/CP/PC/CP/PC`；
- control：`10 warmup + 20 groups × 50 replay`；
- profile：`10 warmup + 20 groups × 5 replay`；
- perturbation admission：每组必须位于 `[0.95,1.05]`；
- source topology：17 op，bucket inventory=`Conv 6 / Linear 2 / ReLU 6 / Add 2 / Flatten 1`；
- semantic receipt：lower/upper 共 `235,992` 元素，max diff=`0.000244140625`，sign exact。

## 3. 六组正式结果

| pair | order | control ms | profile ms | perturbation | clock | pair admitted |
|---:|:---:|---:|---:|---:|:---:|:---:|
| 0 | CP | 0.097669 | 0.115830 | 1.185947x | PASS | FAIL |
| 1 | PC | 0.097853 | 0.115955 | 1.184988x | FAIL | FAIL |
| 2 | CP | 0.097690 | 0.115674 | 1.184093x | PASS | FAIL |
| 3 | PC | 0.097689 | 0.115645 | 1.183806x | PASS | FAIL |
| 4 | CP | 0.097690 | 0.115824 | 1.185633x | FAIL | FAIL |
| 5 | PC | 0.097883 | 0.116074 | 1.185838x | FAIL | FAIL |

所以 perturbation admission=`0/6`，clock admission=`3/6`，formal admission=`0/6`。即便只看
clock 通过的 0/2/3，扰动仍为 `1.1838—1.1859x`，NO-GO 不依赖挑选时钟坏点。

## 4. 归属完整性与为什么仍不能形成 share

六份 profile 均独立重建出同一结构：

- graph node/clone mapping=`42/138`；
- profile group/replay=`20/100`；
- kernel/memcpy/runtime API/graph launch=`4200/200/520/100`；
- owner event=`4400`，unowned=`0`，temporal fallback=`0`，single stream=`[7]`。

这证明 owner reconstruction 机制完整，却不能越过 measurement admission。冻结协议要求计时扰动
和时钟同时通过；结构完整不是将受扰 profile wall 升级为真实 production share 的许可证。因此本轮
不报告 Conv/Linear/elementwise/runtime headline share，也不计算 query-local Amdahl route。

## 5. Replay、tamper 与失败证据纪律

- root replay 重新解析六份 SQLite、重建 clock/owner/timing ledger，并逐字节复现 summary；
- summary hash：`16e46384830993bf5850dfd1ca84823795580ce31234f9e04634c6acb01ca583`；
- 9/9 fully re-signed tamper rejected，覆盖 scope target、source digest、pair order、clock fit/raw、
  semantic receipt、timing median、production topology 和 summary verdict；
- tamper hash：`112e23e0b333ca5fda18c34dca990e6602e1123964fb68337481b0d8b40fde5a`；
- manifest SHA256：`62a15384676b00dc006699928380258a36cf3113595e34d84db1f4a01d9d9e13`。

早期正式尝试曾暴露“遇到 clock rejection 即异常退出、负证据未完整序列化”的 runner 缺陷。修复只
让 NO-GO raw 被保留和汇总，没有更改阈值、顺序、样本数或 workload。最终 artifact 从 clean source
重新运行；不采信早期结果，也不把多次运行当作挑选成功 pair。

## 6. Claim 与门禁传播

- 允许：当前 profile 路径在冻结扰动/时钟门禁下 `VALIDATED-NO-GO-R1A-ATTRIBUTION`；
- 保留：CIBC Conv horizontal graph 既有 `VALIDATED-REDUCED` 及其外审批准；
- 禁止：op-type production share、same-solver share、query/queue speedup、memory、跨模型、
  auto_LiRPA 或 ASPLOS-ready claim；
- 关闭：R1-B、R1-C、R1-D、基于其 admission 的 R2 和三方 formal；
- 不允许：放宽 1.05、换 profiler 后沿用当前预注册、只取 3/6 clock PASS 或跨 scope 代入历史
  `G=2.45631x`。

## 7. 下一阶段

按 R3 设计文档既有“R2 关闭或留下显式 reprioritization 记录”条款，本 closure 构成显式
reprioritization：下一阶段只开放 **R3-0 合同和静态验证器**。

R3-0 只实现 first-class lower-region DAG、Template/Instance、closure/liveness、saved-state/scratch
receipt 与负向篡改验证；不接 production、不计时、`performance_claimed=false`。R3-1 的 P-anchor
custom backward correctness 仍关闭，必须等 R3-0 formal artifact/replay 完成。

## 8. 证据入口

- protocol：`gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- implementation：`gemini_doc/BOUNDFLOW_CIBC_R1_A_ADDITIVE_NSYS_ATTRIBUTION_CHANGELOG_2026_08_25.md`
- artifact：`artifacts/cibc-r1-optimized-graph-attribution/resnet2b-prop0-v1`
- R3 design：`gemini_doc/BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`
