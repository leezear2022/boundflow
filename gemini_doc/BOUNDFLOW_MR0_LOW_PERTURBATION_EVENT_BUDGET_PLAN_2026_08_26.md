---
status: preregistered-implementation-open
updated: 2026-08-26T13:30:00+08:00
type: plan
topic: boundflow
slug: mr0-low-perturbation-event-budget
stage: s01
---

# BoundFlow MR0 低扰动显式事件预算预注册

## 1. 背景与唯一目标

两条 profiler 归因路线已经分别 fail closed：

- CIBC R1-A：Nsight profile/control=`1.1838–1.1859x`，`0/6` perturbation admitted；
- R3-3 microphysics：torch profiler/control=`2.4061–2.8053x`，`0/5` attribution admitted。

MR0 不重跑上述 share，也不选择 Linear/Conv/bridge/autograd 优化。它只回答：在 CIBC production
17-op CUDA Graph 上，使用**无 profiler、无 NVTX、预分配 `torch.cuda.Event`**模拟 op-level 边界，
其测量扰动是否足够小，值得进入新的 MR1 internal-boundary prototype。

## 2. Frozen workload 与 identity

- VNN-COMP 2021 ResNet2B property 0；
- source capture SHA256=`f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc`；
- model SHA256=`791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`；
- CIBC candidate：128 threads、6 Conv horizontal lower/upper fusion、input copy included；
- production topology：17 op=`6 Conv + 2 Linear + 6 ReLU + 2 Add + 1 Flatten`；
- single stream、cache warm、CUDA Graph 已构造，compile/capture 不进入 timing。

## 3. Five-fresh paired protocol

- 5 个独立 GPU process，capture ordinal=`0..4`；
- worker order=`CI/IC/CI/IC/CI`，C=control，I=instrumented；
- 每 worker：20 control/instrumented warmup replay；
- event-pair budget=`1,4,8,17`，每个 budget 20 paired group，每侧每 group 100 replay；
- group 内 control 和 instrumented 使用同一外层 CUDA-event timing pair；
- instrumented 每次 replay 前记录 `budget` 个预分配 start event，graph replay 后记录对应
  `budget` 个 end event；event construction 和 `elapsed_time()` 读取在 timed region 外；
- control 除不记录内部预算 event 外，与 instrumented 相同；双方都包含 input copy、graph launch 和
  外层 timing event；
- 不调用 torch profiler、Nsight、NVTX、CUPTI activity collection 或 host callback。

该布局只校准 event-record budget，不声称已获得真实 node duration；17 对 event 是 production
op-level attribution 的最低完整预算。MR1 必须另行证明 event 与内部 op owner 的绑定。

## 4. 语义与环境门禁

- control/instrumented 必须使用同一个 immutable graph plan；
- final 与全部 intermediate lower/upper 在 instrumentation 前后 exact alias/shape/dtype/device 不变；
- instrumentation 不能增加 graph compile/capture、fallback、eager shadow 或 plan construction；
- GPU/driver/compute capability、温度/功耗/时钟在每 worker 前后冻结；
- event object 数固定为 `2 + 2*17`，worker 内预分配一次；
- 任何 nonfinite latency、event record error、stream drift 或 source digest mismatch 均 fail closed。

## 5. Admission gate

每 worker/budget 从 20 paired group 分别重算 control median、instrumented median 和：

```text
overhead_ratio = instrumented_median / control_median
```

MR0 只用最大预算 `17` 决策，其他预算用于敏感性披露。冻结 GO 条件全部成立：

1. 5 worker overhead ratio geomean `<=1.05x`；
2. paired bootstrap 95% upper `<=1.05x`，seed=`20260826`，samples=`10000`；
3. worst worker `<=1.08x`；
4. 每 worker semantic/stream/event-count receipt 通过；
5. 5 worker 均完成，不删除 outlier，不替换 budget。

若 GO，只开放 `MR1-internal-boundary-correctness`：在 graph capture 内绑定17个owner边界，仍不形成
share或性能 claim。若任一 gate 失败，结论=`VALIDATED-NO-GO-MR0-EXPLICIT-EVENT-BUDGET`，关闭
explicit-event op-level attribution；不得降低到只测1/4/8对后宣称完整 attribution 可用。

## 6. Artifact、replay 与 tamper

正式路径：

`artifacts/measurement-recovery/mr0-explicit-event-budget-resnet2b-v1/`

包含 protocol、5 raw、summary、manifest、portable logs 和 tamper report。root replay 必须从 raw
重算每个 median/ratio、bootstrap upper、gate 与 verdict。至少12类 fully re-signed tamper：source/
model digest、order、budget、sample、median、ratio、bootstrap、semantic、event count、scope open、
performance claim 和 summary verdict。

## 7. Claim boundary

MR0 最多允许 claim “显式 CUDA-event 的17-op记录预算在该 graph 上通过/未通过低扰动门禁”。
不得 claim op share、kernel duration、CIBC query speedup、same-solver admission、R2开放或
ASPLOS-ready。历史 CIBC `2.45631x` graph reduced claim与R3-3 STOP均不改变。

## 8. 执行顺序

1. 实现纯数据 derivation、worker、artifact/replay、negative tests；
2. targeted/typing/lint 通过后冻结 clean source commit；
3. 从 worker 0 开始生成 five-fresh formal；
4. 运行 12 类 tamper 与 full regression；
5. 根据机械 verdict 只开放 MR1 或关闭 explicit-event 路线。
