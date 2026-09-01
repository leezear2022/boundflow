---
status: preregistered-r3-2b-local-timing
updated: 2026-08-25T05:48:00+08:00
type: plan
topic: boundflow
slug: r3-2b-wrapper-inclusive-timing
stage: s01
---

# R3-2B P-anchor 10/9 Wrapper-Inclusive Local Timing 预注册

## 1. 前置与范围

R3-2A 已以 `VALIDATED-R3-2A-P-TRAJECTORY` 关闭，逐步正确性、所有权和memory门禁成立。本阶段只回答：
在同一冻结P-anchor、同一10 evaluation/9 mutation轨迹下，compiled structured owner/custom VJP是否在
完整本地wrapper上达到物理收益。

这不是whole-core/query/same-solver测试。不得扩site、改变shape、调schedule或用局部kernel latency代替。

## 2. 两侧公平wrapper

共同包含在每个timed sample内：

- 新建单一P-α parameter group的PyTorch Adam；
- 新建ExponentialLR(gamma=`0.98`)；
- evaluation `0..9`；
- 每次lower与`d(-sum(lower))/dα`；
- ordinal `0..8`的Adam step、`clamp_(0,1)`和scheduler step；
- ordinal 9 terminal lower；
- Python loop、autograd/custom-Function dispatch与optimizer wrapper成本。

共同放在timed sample外：

- ONNX import、BF plan/trace、source-state binding；
- TVM compile、module cache和candidate persistent scratch/DLPack preparation；
- immutable tensor SHA/静态topology admission；
- warmup、initial α恢复、correctness/raw capture和memory capture。

native使用独立eager/autograd full-region oracle；candidate使用R3-1b compiled full-lower + custom compiled
P-α VJP。candidate不得调用native shadow/fallback/eager region。

## 3. 热路径receipt分层

R3-2A逐步content hash/replay是correctness capture，不属于每次production hot call必须重复执行的工作。
R3-2B在setup阶段完成一次static/immutable SHA admission；timed loop只允许O(1)检查：

- tensor tuple/shape/dtype/device/data_ptr identity；
- P-α version单调变化和evaluation ordinal；
- schedule/lr/update-after；
- custom forward/backward与mutation计数；
- current device/stream不漂移。

不得在timed path计算GPU tensor SHA、复制raw到CPU、创建formal receipt、运行profiler或reset memory stats。

## 4. 计时协议

- hardware/environment与R3-2A formal一致，CUDA device 0；
- native和candidate都在各自non-default torch CUDA stream执行；
- sample边界在同一device执行`torch.cuda.synchronize()`；
- headline用`time.perf_counter_ns()` host wall time，覆盖CPU dispatch、同步等待和GPU执行；
- 每个fresh worker：3个完整warmup + 30个完整measured samples；
- 每个sample前在计时外恢复同一initial P-α，sample后同步；
- 每worker统计30样本median；
- 5对fresh subprocess，顺序`NC/CN/NC/CN/NC`；
- pair speedup=`native_median/candidate_median`；headline为5个pair speedup geomean和worst pair；
- 不删除outlier，不续跑partial raw，不根据结果改样本数。

## 5. Untimed correctness与memory复核

每个worker在计时结束后另跑一次不计时capture，冻结terminal lower、terminal α、mutation count、环境和
execution counters。pair必须满足：

- terminal lower allclose(`2e-4/2e-4`)且sign exact；
- terminal α allclose(`2e-5/2e-5`)；
- mutation/scheduler=`9/9`，candidate forward/backward=`10/10`；
- candidate fallback/eager/native-shadow=`0/0/0`。

memory用独立untimed one-shot wrapper，在persistent candidate owner已经建立后reset peak；报告absolute
allocated/reserved peak。candidate两项均须`<=1.0x`同pair native。

## 6. GO / NO-GO

GO同时要求：

- 5 pair geomean speedup `>=1.20x`；
- worst pair `>=0.98x`；
- 全部correctness/counter/memory门禁通过；
- replay与fully re-signed tamper通过；
- 无计时口径或claim漂移。

GO仅允许claim `VALIDATED-R3-2B-P-LOCAL-WRAPPER`，并开放R3-3 active-beta correctness；不能直接开放
multi-site或same-solver。

任一失败则当前single-site实现正式kill为`VALIDATED-NO-GO-R3-2B-*`；不得靠扩site、删wrapper成本、
换GPU event-only latency或降低门槛挽救。

