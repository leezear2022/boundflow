---
status: preregistered-r3-d0-read-only-attribution
updated: 2026-08-25T06:20:00+08:00
type: plan
topic: boundflow
slug: r3-d0-microphysics-attribution
stage: s01
---

# R3-D0 Compiled Recurrence Microphysics Attribution 预注册

## 1. 问题与非目标

R3-2B 已证明当前candidate完整10/9 wrapper约`739–794 ms`，native约`96–104 ms`，geomean=
`0.133989x`。R3-2A同时证明语义和显存所有权成立。因此D0只回答：约`638–690 ms`额外成本来自
host dispatch、单kernel实现、跨kernel memory traffic还是组合。

D0只读测量，不修改schedule/kernel/production，不形成speedup claim，不开放R3-3。禁止在看到profile
后事后改region、marker、样本subset或候选路线门槛。

## 2. 冻结对象与样本

- source基线：R3-2B closure `d4c5b4c`及随后纯文档提交；
- workload：同一ResNet2B property-0、P-anchor、P-α 10/9 wrapper；
- native/candidate均复用R3-2B capture-free runtime；
- 5 fresh pair，顺序`NC/CN/NC/CN/NC`；
- 每worker 3次warmup后profile 1个完整wrapper；另测30次unprofiled host-wall sanity，必须与R3-2B
  median在`±15%`内，否则profile run不得用于headline attribution；
- 不裁剪kernel/event，不按结果选择pair。

## 3. 时钟与归因

每个worker冻结host `perf_counter_ns`、CUDA/CUPTI activity和显式NVTX/record-function边界。必须有
host↔CUDA calibration receipt；残差超预注册阈值则该worker不得产生share。

candidate marker层级：

```text
wrapper
  evaluation.00 ... evaluation.09
    forward.spec/linear/relu/residual/conv/concretize
    backward.sign/effective/recompute/compressed-gradient
    optimizer.adam / optimizer.clamp / optimizer.scheduler
```

每个compiled symbol报告launch count、kernel sum、p50/p95、host submit span和前后gap。相关性优先使用
CUPTI correlation id；只有缺失时允许explicit marker containment fallback，并单独计数。

native按PyTorch operator/aten/cuDNN kernel family报告相同字段，但不得用candidate symbol命名强行一一对应。

## 4. 必须输出的账本

每pair及汇总至少包含：

- wrapper host wall、CUDA kernel union、kernel sum、overlap、host-only residual；
- launch count和每launch平均host submit/gap；
- candidate各symbol family的count/sum/share/p50/p95；
- native各operator/kernel family的count/sum/share；
- forward、backward、optimizer三段host/GPU share；
- bytes可恢复时报告读写量；不可恢复时明确unknown，不用shape猜headline；
- candidate/native GPU-kernel-union ratio和host-residual ratio。

单stream且无overlap时不得使用overlap-adjustment制造headline。

## 5. 数学路由门禁

对每个pair用native median `N`、candidate median `C`冻结R3-2B目标budget：

```text
target_candidate = N / 1.20
required_saving = C - target_candidate
```

以formal中心值约算，target约`80–86 ms`，required saving约`653–708 ms`；正式值必须从各pair raw重算。

### Graph/dispatch支路

设candidate host-only residual=`H`。即使host residual无限消除后的乐观下界为`C-H`。只有5-pair
worst-case `C-H <= N/1.20`，且可capture launch占host residual>=`80%`，才允许预注册CUDA Graph支路。

### Kernel schedule/fusion支路

对任一kernel family share `s`，用：

```text
r_required = s / (target_candidate/C - (1-s))
```

分母必须为正。只有至少一个语义闭合family/相邻region的5-pair worst-case required speedup `<=10x`，且
该region无dense-A持久化、可在<=2 scratch内融合，才允许预注册schedule/fusion支路。

### 整体NO-GO

如果5-pair中任一pair的CUDA kernel union本身已大于`N/1.20`，且不存在required<=`10x`的闭合region；
或host无限消除与所有admitted region上限组合后仍达不到target，则R3-SO-CVJP performance路线整体NO-GO。

## 6. Artifact与篡改

formal raw绑定source/code blob、R3-2B artifact、environment、calibration、NVTX/CUPTI events和canonical
ledger。replay从events重建marker/kernel归属、share、Amdahl路由与verdict。

至少拒绝：kernel duration、correlation id、marker ordinal、fallback标记、host calibration、launch count、
region family、target budget、required speedup、summary route和performance claim的fully re-signed tamper。

## 7. 当前边界

本文只完成预注册。下一工程动作是实现diagnostic-only profiler worker与event replay；不得改R3-2B runtime
math或现有artifact。D0通过前CUDA Graph、kernel fusion/schedule tuning和R3-3全部关闭。

