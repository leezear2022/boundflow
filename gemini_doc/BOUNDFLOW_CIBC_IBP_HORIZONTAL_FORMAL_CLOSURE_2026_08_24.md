---
status: validated-reduced-cibc-ibp-conv-horizontal
updated: 2026-08-25T00:50:00+08:00
type: changelog
topic: boundflow
slug: cibc-ibp-horizontal-formal-closure
stage: s01
---

# CIBC IBP Horizontal Fusion Formal Closure

## Verdict

`VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`。

在RTX 4060 Laptop/sm_89、VNN-COMP 2021 ResNet2B property 0的IBP forward-bound图上，一个TIR
kernel同时计算center/deviation和lower/upper，正式替换BoundFlow原有weight正负拆分的四次PyTorch
Conv。6个真实Conv的算子层speedup geomean=`12.7951x`、worst=`9.1423x`；完整IBP CUDA Graph
speedup geomean=`2.45631x`、95% bootstrap lower=`2.45386x`、worst=`2.45091x`。预注册四项性能门禁
全部通过。

这是单模型、单property、单GPU、IBP Conv水平融合的reduced claim；不是auto_LiRPA对比、
alpha-CROWN/BaB/query speedup、显存收益或ASPLOS-ready结论。

## Frozen Identity

- source=`a52b1775e3b8185410e5600da0bde55b3b24102b`；
- artifact=`artifacts/cibc-ibp-horizontal-formal/resnet2b-prop0-v1`；
- protocol hash=`f083cd6bc16b007d17ba1f45850c0d89a8aead9139dd619288f3a11ae7924f5c`；
- summary hash=`baea55cb74a791275cad950bd93d266ccc5499f911e141e5757a2238266846e7`；
- manifest hash=`78df967521b9a57a6d5fcfdc55d39af85f5794b62b0f1ab16aeca44464d3b4e5`；
- tamper report hash=`19d429ec21a2eb69660f7e5a9584014347392977df76a4fa31ffd1c4fc0bfb9d`；
- source capture/model SHA256=`f42229dd…126dc`/`791aa24d…4a6d`。

## Operator Schedule Selection

候选集在运行前固定为64/128/256 threads，每个候选由一个独立进程对真实6 Conv各测30组×500次。

| threads | 6-Conv speedup geomean |
|---:|---:|
| 64 | 11.672159x |
| 128 | **12.795108x** |
| 256 | 12.725013x |

128-thread winner的逐算子结果如下；baseline是BoundFlow既有四Conv IBP公式，candidate是plan-owned
单launch TIR。

| graph ordinal | baseline median ms | TIR median ms | speedup |
|---:|---:|---:|---:|
| 0 | 0.048628 | 0.005319 | 9.1423x |
| 2 | 0.047706 | 0.003264 | 14.6166x |
| 4 | 0.049463 | 0.004267 | 11.5923x |
| 5 | 0.048391 | 0.002134 | 22.6715x |
| 8 | 0.047818 | 0.004265 | 11.2109x |
| 10 | 0.047543 | 0.004266 | 11.1448x |

## Whole-model Result

6个fresh进程按`BC/CB`交替，每个进程30组×100 replay。baseline和candidate都使用CUDA Graph，
每次计时都包含lower/upper输入copy，避免把Graph replay或漏计输入搬运冒充TIR收益。

| worker | order | baseline median ms | CIBC median ms | speedup |
|---:|:---:|---:|---:|---:|
| 0 | BC | 0.175928 | 0.071569 | 2.45815x |
| 1 | CB | 0.175088 | 0.071252 | 2.45731x |
| 2 | BC | 0.176010 | 0.071814 | 2.45091x |
| 3 | CB | 0.176809 | 0.071858 | 2.46054x |
| 4 | BC | 0.176808 | 0.072015 | 2.45517x |
| 5 | CB | 0.176560 | 0.071895 | 2.45580x |

## Measurement and Tolerance Disclosure

- operator 与 whole-model headline 都是 **steady-state warm execution**：TIR compile、schedule
  selection/autotune、Plan construction、CUDA Graph capture 和 warmup 不在 timed interval；
- whole-model 两侧 timed interval 都包含 lower/upper input copy 与 CUDA Graph replay，因此
  `2.45631x` 不是省略 candidate input 搬运得到的数字；
- cold compile、plan construction、graph capture 和 break-even 尚未由本 artifact 测量，不能从本
  closure 推导 JIT/cold-start claim；这些字段已转入 R1 预注册；
- `semantic_atol=semantic_rtol=3e-4` 已在 source=`a52b177` 的 protocol、正式 worker 运行前冻结，
  不是看到结果后的放宽；
- 实测最大差 `0.000244140625 = 2^-12`。外部审计核对该差值对应受影响数值指数档的一次 binary32
  spacing，并确认 sign exact；因此本轮 `3e-4` 是覆盖已观察单-ULP量级的预注册容差，不是提高
  算法误差预算。该解释不允许后续阶段自动沿用同一容差，新的 scope 仍须独立预注册。

## Semantics and Integrity

- 6/6 production Conv由TIR接管；其余ReLU/residual add/flatten/2 Linear保持同图执行；
- 全部中间interval与最终logit maximum absolute difference=`0.000244140625 < 3e-4`；
- 全部中间interval sign exact=`true`；
- protocol固定目标GPU、capture/model digest、code blob、schedule集合、顺序、repeat数和门槛；
- root replay从9份raw重算median、schedule winner、geomean、bootstrap、semantic和coverage，退出0；
- 10/10 fully outer-resigned tamper rejected，覆盖protocol claim、timing派生、semantic、coverage、
  worker identity、operator inventory、summary、schedule、hardware与input-copy receipt；
- targeted=`7 passed`；全量=`1492 passed, 3 skipped, 6 warnings`；Black/Mypy/Pylint=`10.00/10`；
  DocOps lint在交接前执行。

## 与原B4计划的关系

原B4纵向路线试图优化alpha-CROWN ReLU→Conv forward/backward region，先后完成typed correctness、
稀疏source、Conv/Linear TIR和production exact-call，但累计接管在B4-C1仅约`0.948x`，扩大到6个真实
materialization site后因dense autograd retention降至`0.337—0.349x`且显存增至`1.34x`，因此按
kill gate关闭。这里的CIBC结果不是给B4局部数字换名字，而是另起IBP forward-bound水平融合路线；
它直接解释了用户早期BoundConv原型为何能达到几十倍局部收益。

## Remaining Boundary

当前production default仍保持PyTorch，CIBC通过显式context/plan opt-in。若继续扩大claim，必须另行
预注册并验证：两层Linear水平融合、非CUDA-Graph eager amortization、第二model family、auto_LiRPA
same-workload对照、真实solver/query传播与独立进程显存测量。任何一项都不能由本artifact自动升级。
