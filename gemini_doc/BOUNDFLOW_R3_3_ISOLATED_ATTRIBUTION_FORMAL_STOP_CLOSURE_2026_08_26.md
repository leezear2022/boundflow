---
status: validated-stop-attribution-quality
updated: 2026-08-26T12:45:00+08:00
type: closure
topic: boundflow
slug: r3-3-isolated-attribution-formal-stop
stage: s01
---

# R3-3 Isolated Attribution 正式 STOP Closure

## 1. Verdict

`VALIDATED-R3-3-ISOLATED-ATTRIBUTION-STOP-QUALITY`。

5 个独立 GPU process 均完成 raw、事件重建、互斥账本与 replay，但 `0/5` 通过 attribution
quality gate。按预注册规则 route 必须是 `STOP`；R3-4、same-solver 和任何由本轮 share 推导的
优化路线继续关闭。

这不撤销此前的 active-β correctness，也不改变 isolated timing 的
`VALIDATED-NO-GO-R3-3-S-ISOLATED-PHYSICS`。

## 2. Frozen source 与 artifact

- source：`1c15624f5b823e44d5c941dfa9425907000240c3`；
- artifact：`artifacts/r3-structured-owner/r3-3-isolated-attribution-v1/`；
- protocol hash：`44c9f1ab5459252bda86d681ee38a934886f20d4716f9ce644b2fadfc19d840f`；
- summary hash：`8b2d8db41b4aebb4251c325a6226a8a8870137cec4cd859f33a08a6334a5d899`；
- final manifest hash：`cb64f752fa3ea9f8fbb81f4e7f664810bfd4a6aaa4ce5c803782619df6391af2`；
- tamper report hash：`7720da70bc39552578a1dfd8c65722a325aebfd3f6168c441fb5d9efa384da71`。

每个 process 使用 capture ordinal `0..4`、10 warmup、30 个无 profiler CUDA-event sample，
再执行一个带显式 marker/CUPTI correlation 的 diagnostic capture。无 profiler median 为
`[1.401856, 1.341952, 1.374640, 1.408512, 1.405360] ms`。

## 3. Attribution quality 失败

五组 failure 完全一致：

- `profiler-perturbation`：profiled/unprofiled=`[2.761596, 2.805299, 2.671920,
  2.406102, 2.547823]x`，全部大于冻结上限 `1.20x`；
- `calibration-residual`：`[114079, 109856, 118976, 112544, 113599] ns`，全部超过
  `max(5 us, 2%)`；
- conservation error 均满足门禁；
- unexplained share=`1.868%–2.806%`，满足 `<=5%`；
- CUDA activity 全部有 correlation parent，containment fallback/unattributed 均为 0，且为单 stream。

因此失败不是“没有捕获事件”，而是 profiler 显著改变了被测 1.4 ms 小 wrapper 的成本结构，
同时校准收据不足以把该投影提升为可采信 share。

## 4. 非准入诊断投影

为便于复核，raw 保留以下 five-run 范围，但所有字段都标记
`admissible_for_route=false`：

- forward kernel union：`1.828%–2.089%`；
- backward kernel union：`2.949%–3.362%`；
- bridge/launch idle：`35.895%–39.548%`；
- autograd/allocation：`26.237%–27.715%`。

这些数字不能用于选择 KERNEL/BRIDGE/AUTOGRAD/CUMULATIVE，不是 speedup claim，也不能代入
query/queue Amdahl 公式。特别地，不能因为 bridge 的非准入投影接近 `36.35%` 就开放 ABI 优化。

## 5. Replay、tamper 与回归

- semantic replay：PASS，从 5 raw 逐 event 重建 ledger 与 route；
- fully re-signed tamper：`12/12 rejected`，覆盖 latency、calibration、event、ledger、capture、
  protocol gate、summary route 和 scope-open；
- targeted：`8 passed`；
- Black：PASS；mypy：clean；pylint：`10.00/10`；
- full regression：`1667 passed, 3 skipped, 6 warnings in 729.29s`；3 个 skip 均为既有
  环境/重复编译边界。

第一次 formal 启动在第 4 个 worker 非零退出，原子目录未提交且临时日志按 runner 当时行为清理；
同一 capture 随后独立复现通过，第二次从 ordinal 0 全量重跑并形成上述唯一正式 artifact。该首次
失败不参与任何统计。

## 6. 路线决定

1. 停止当前 fixed S-anchor R3-3 physical 分支；不实现 R3-4；
2. 保留 structured owner、active-β correctness、D2-B local wrapper 等机制证据；
3. 不放宽 profiler/clock 门槛，不从非准入 share 反向选择优化；
4. 后续若重新测量小 wrapper，只能新预注册低扰动的显式 CUDA-event/受控 A/B ablation，且它是
   新测量假设，不是本轮 route 的延续；
5. 系统主线回到尚未完成的 same-solver/query/queue 证据缺口；任何新方案仍须按 op type 冻结
   真实 query-local share 与 `G_query,k`，不得沿用独立 graph 或本轮 wrapper 投影。

## 7. Claim 边界

允许 claim：five-fresh profiler attribution 因质量门禁 fail closed，route=`STOP`。

禁止 claim：本轮证明 bridge/autograd 是真实 dominant、已找到可行优化、R3-4/same-solver 开放、
query/queue speedup、跨模型收益或 ASPLOS-ready。
