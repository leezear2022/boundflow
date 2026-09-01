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
