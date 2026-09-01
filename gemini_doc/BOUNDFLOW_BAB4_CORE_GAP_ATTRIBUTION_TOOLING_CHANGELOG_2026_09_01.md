# BAB4 core-gap 分段归因工具修改记录

status: live-profile-complete-optimization-in-progress
date: 2026-09-01
performance-claimed: false
external-audit-requested: false

## 1. 动机

BAB4 五组正式结果显示 complete-query geomean 为 `1.30018x`，但 core geomean 只有
`1.17718x`。已有 exact-call receipt 只能把 live core 拆为 rebind、完整 optimizer 和 handoff，不能
解释 optimizer 内四段 TIR、重算 backward 与 Adam 各占多少。

同时，raw 的 query phase 表明候选使用了 prepared verification request，而原 B4-A control 没有使用
同等 prepared request。查询级 `1.30018x` 因而混合了两类收益：四段 compiled executor，以及静态请求
准备移出 warm query。后续必须新增 prepared-control 对照，不能直接把全部 query 收益归因到 BAB4。

## 2. 新工具

新增 `scripts/run_bab4_core_gap_profile.py`：

- 在真实 same-solver BAB4 worker 中只 profile 第二次、即 production live optimizer；第一次 dummy
  warmup 不进入 profile；
- 用显式 marker 区分 terminal/residual/projection/input 四段 forward/backward、full-region forward、
  outer/recompute autograd、Adam、zero-grad 和 scheduler；
- 用 CUPTI/torch profiler 捕获 CUDA activity，并复用既有 R3-D0 event/ledger schema；
- unprofiled 分母来自已冻结五组 artifact 的五个 optimizer receipt；
- profiler 数字只用于 attribution，`profile_timing_claimed=false`、`performance_claimed=false`；
- 输出同时绑定正式 manifest、live worker、compiled plan/assets 和 10/9、76/36、fallback=0 合同。

新增 `tests/test_bab4_core_gap_profile.py`，确保 unprofiled 分母恰来自五个已准入 BAB4 live workers。

## 3. 下一步

1. 运行一次 fresh live profile，按 kernel union 与 host residual 找出最大桶；
2. 增加 `B4-A-PREP` 对照，使 baseline/candidate 都使用相同 prepared verification request；
3. 只对实测最大桶实施第一项优化，再跑 paired same-solver timing。

本工具不改变 production 执行，也不升级任何性能 claim。

## 4. 首轮 live profile 结果与第一项优化

首轮 fresh profile 捕获 1,586 个 CUDA activity。由于 profiler setup/teardown 误计入 host wall，且
`134/1586` kernel 使用 marker containment，首轮 `calibration_admitted=false`，因此数字只用于路由：

- residual forward：8 个 kernel/调用，compiled kernel sum 约 `9.35 ms`；
- projection forward：9 个 kernel/调用，compiled kernel sum 约 `8.95 ms`；
- input forward/backward：约 `7.02 ms`；
- Adam：约 `0.35 ms`，不是主瓶颈。

逐 kernel 检查发现 residual/projection 各自最慢的两个 kernel 正是 `entry_bias_delta` 与
`inner_bias_delta`。旧 schedule 只把 `(spec, domain)` 绑定到 GPU，`channel×height×width=1024`
的 reduction 在单线程内串行执行，每个 kernel 中位约 `220--242 us`。

因此第一项优化不是泛化调参，而是 verification-specific reduction schedule：

1. 将 `C×H×W` reduction fuse；
2. 按 `thread_extent=128` split；
3. 用 `rfactor` 生成 `(spec, domain, lane)` partial；
4. partial lane 绑定 `threadIdx.x`，最终 128-lane reduction 单独收口；
5. residual 与 projection 共四个 bias reduction 使用同一合法 schedule primitive。

该变换增加少量 reduction-final kernel，但把 1024 项串行 reduction 改为每 lane 约 8 项，目标是
消除 profile 中约 `17 ms` 的主要 GPU 串行区。正确性与真实性能仍需后续 fresh 复测。

首个非正式 live diagnostic 已观察到：

- optimizer：旧五组中位 `47.34 ms` → 新单次 `37.10 ms`；
- core：新单次 `203.91 ms`；
- query：新单次 `1103.26 ms`；
- lower/sign 与冻结范围一致，environment admitted。

该单次只说明优化值得进入 paired 测量，不构成 headline。

## 5. 公平对照修正

新增 `B4-A-PREP`：它保持原 B4-A executor，只把与 BAB4 完全相同的 prepared verification request
移到 query 外。新增 `run_bab4_rfactor_prepared_five_fresh.py`，五组交替比较
`B4-A-PREP ↔ BAB4`。这将分开回答：

1. prepared request 本身节省多少；
2. rfactor 后的四段 TIR executor 在相同准备边界下节省多少。

旧 artifact 测试也改为按其冻结 source commit 读取 git blob，而不是要求当前工作树永远等于旧代码；旧
raw/replay 仍保持不变。

## 6. Warm-state 二次校正

`B4-A-PREP ↔ BAB4` 五组诊断得到 query geomean `1.163729x`、core geomean `1.160650x`，但 raw
phase ledger 暴露出新的不匹配：BAB4 在 query 外执行了四段 optimizer warmup，B4-A-PREP 没有执行
原生 optimizer warmup。候选的 root incomplete 中位因此约为 `641.45 ms`，而 control 为
`800.41 ms`；这部分约 `159 ms` 差异发生在 BAB4 exact-call core 之前，不能归因给四段 TIR。

因此该五组只保留为诊断，不升级性能结论。新增两个配置：

- `B4-A-WARM`：prepared request + 原生 root optimizer warmup + 原 B4-A executor；
- `BAB4-WARM`：相同 prepared request + 相同原生 root optimizer warmup + BAB4 四段自身 warmup/executor。

新增 `run_bab4_rfactor_warm_five_fresh.py`，以五组交替 fresh 进程比较上述配置。只有该完全匹配的
warm-state 协议才能形成当前 BAB4 的 query/core 性能结论。

## 7. 完全匹配五组正式结果

`B4-A-WARM ↔ BAB4-WARM` 在 RTX 4060 Laptop GPU 上完成 5 对交替、10 个 fresh 进程：

| 指标 | 结果 |
|---|---:|
| complete-query geomean | `1.034617x` |
| complete-query worst pair | `1.023630x` |
| exact-call core geomean | `1.182383x` |
| exact-call core worst pair | `1.165240x` |
| lower max abs diff | `2.5629997e-6` |
| 候选额外 static prepare 均值 | `9.266469 s` |
| query 平均节省 | `24.595941 ms` |
| cold break-even | `376.748` queries |

5/5 environment admitted、discrete semantics exact、lower sign exact。raw replay PASS。结论降精度为：

- BAB4 rfactor 在完全匹配 warm state 下达到 query parity，且 5/5 都快；
- core 有约 18.2% 收益，但没有达到冻结的 `1.20x` core research gate；
- complete query 只有约 3.46% 收益，没有达到 `1.15x` research gate；
- `performance_claimed=false` 保持，旧 `1.30018x` 与中间 `1.16373x` 不再作为公平 headline。

中位时间账说明损失发生在集成传播而非正确性：

| scope | B4-A-WARM | BAB4-WARM | 观察 |
|---|---:|---:|---|
| query | `730.752 ms` | `711.619 ms` | 候选快约 19 ms |
| core | `250.999 ms` | `211.643 ms` | 候选快约 39 ms |
| pre-core | `479.498 ms` | `497.113 ms` | 候选反而慢约 17.6 ms |
| root incomplete | `239.447 ms` | `238.300 ms` | 基本相同，warm state 已匹配 |
| four-segment optimizer | N/A | `42.414 ms` | 仍是候选主热区之一 |
| rebind + handoff | N/A | `18.509 ms` | typed runtime 集成成本 |
| peak allocated | `318.401 MiB` | `320.018 MiB` | 候选多约 1.62 MiB |
| peak reserved | `390 MiB` | `394 MiB` | 候选多 4 MiB |

rfactor 本身相对旧 artifact 的 optimizer geomean 从 `46.851 ms` 降到 `42.468 ms`，约 `1.1032x`；
此前单次 `1.276x` 诊断高估了稳定收益。下一步应先归因并消除约 17.6 ms pre-core/集成损失，再继续减少
residual/projection 的多 kernel launch 和中间 materialization，而不是宣称当前已经达到论文性能门槛。

冻结 artifact：`artifacts/bab4-rfactor-warm-five-fresh/resnet2b-prop0-v1`。stdlib replay PASS，
`summary_hash=4725f1d5db74393884aed33b3ebb2329966df07f05d4b2c76a916086b67a716d`。专项 50 passed；
全量回归 `2218 passed, 3 skipped`，三个 skip 均为既有环境边界。
