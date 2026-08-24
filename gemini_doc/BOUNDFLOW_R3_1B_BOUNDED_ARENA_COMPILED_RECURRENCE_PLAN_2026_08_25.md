---
status: implemented-r3-1b2-pending-clean-source-formal
updated: 2026-08-25T04:45:00+08:00
type: plan
topic: boundflow
slug: r3-1b-bounded-arena-compiled-recurrence
stage: s01
---

# BoundFlow R3-1b Bounded-Arena Compiled Recurrence 预注册计划

> **2026-08-25 R3-1b0 closure**：clean source static artifact/replay与6/6 fully re-signed tamper
> 通过；12-step、2 residual、2 scratch、每slot 73,728 B正式冻结。当前只开放b1 compiled
> full-lower forward；b2 custom VJP、b3 five-fresh与timing继续关闭。见
> `BOUNDFLOW_R3_1B0_TRACE_LIVENESS_FORMAL_CLOSURE_2026_08_25.md`。
>
> **2026-08-25 R3-1b1 implementation note**：compiled full-lower CUDA TIR 与 two-scratch
> launcher 已实现，单次真实 GPU smoke 的 lower max diff=`3.814697265625e-06`、warm dynamic
> allocated bytes=`0`。当前仍等待 raw-first artifact/replay/tamper，因此不提前关闭 b1；见
> `BOUNDFLOW_R3_1B1_COMPILED_FULL_LOWER_IMPLEMENTATION_2026_08_25.md`。
>
> **2026-08-25 R3-1b1 closure**：fresh-process artifact/replay 与10/10 fully re-signed tamper
> 通过；lower max diff=`3.814697265625e-06`，15 launches、2 scratch、70/70 DLPack、warm
> allocation=0。当前只开放b2 compiled P-alpha VJP；见
> `BOUNDFLOW_R3_1B1_COMPILED_FULL_LOWER_FORMAL_CLOSURE_2026_08_25.md`。
>
> **2026-08-25 R3-1b2 math gate**：P-alpha closed-form VJP 与 native autograd 的 max diff=
> `4.470348358154297e-08`、sign exact、nonzero=`281/281`。这只证明可在不跨 forward/backward
> 保存 dense A 的条件下归约；compiled TIR/custom backward仍未实现。下一只实现checkpoint/sign
> TIR，见`BOUNDFLOW_R3_1B2_P_ALPHA_VJP_MATH_REDUCTION_2026_08_25.md`。
>
> **2026-08-25 R3-1b2 implementation note**：10-symbol compiled VJP/custom Function单worker
> 已通过；lower/dα max diff=`3.93391e-6/6.14673e-8`、sign exact、2 scratch、saved dense A=0、
> warm allocation=0。当前等待clean-source artifact/replay/tamper，不提前关闭b2；见
> `BOUNDFLOW_R3_1B2_COMPILED_P_ALPHA_VJP_IMPLEMENTATION_2026_08_25.md`。

## 1. Why this branch exists

R3-1 M0 Python rematerialization 已证明 final lower 与 P-anchor compressed dα 语义可实现，但正式
five-fresh 同时证伪了两件事：

1. nested PyTorch autograd 的 peak allocated 固定为 native 的 `1.1181178686x`；
2. tensor-free ctx 与 zero saved dense A 不会自动产生 compiled bounded-arena region。

因此 R3-1b 不是给当前 prototype 做微调，也不是提前进入 R3-2A。它只替换 M0 内部执行机制：把
full lower coefficient recurrence 与 P-anchor VJP lower 到 first-class compiled schedule，并让所有
dense coefficient 只存在于两个 PyTorch-owned scratch 中。输入、oracle、tolerance 和 memory gate
沿用 R3-1，不得修改。

## 2. Frozen scope

只支持冻结 workload：

- model/property：VNN-COMP 2021 CIFAR10 ResNet2B property 0；
- batch/spec：`6/1`；start node=`25/Conv_8`；
- one evaluation，optimizer mutation=`0`；
- P alpha source=`alpha/%2Finput-24/%2F49 [2,1,6,86]`；
- P beta source=`beta/%2Finput-20/0/value [6,0]`，必须 absent，不得构造 dense zero beta；
- final output=`lower [6,1]`；唯一返回 gradient=`compressed dα [2,1,6,86]`；
- RTX 4060 Laptop / sm_89 / float32 / current non-default stream exact。

不支持 S-anchor active beta、10/9 optimizer、multi-start-node、timing、CUDA Graph、跨模型或
production default。

## 3. Exact recurrence that must be compiled

不是把 B4-B2 的局部 Conv kernel接到 frozen output adjoint。R3-1b 必须从 objective seed 沿真实
ResNet2B reverse topology执行完整 lower recurrence：

```text
33 → Gemm16 → ReLU31 → Gemm14 → Flatten29 → ReLU28
   → Add11 fanout {Conv10 → ReLU25 → Conv8, residual24} → merge24
   → ReLU23 → Add6 fanout {Conv4, Conv5} → merge18
   → ReLU19 → Conv2 → ReLU17 → Conv0 → input concretize
```

每个 ReLU 从相应 production compressed α、layout indices、fixed bounds 与 split/history重建 lower
slope；active beta 只在非 P nodes 依当前 frozen state进入 recurrence。所有 bias contribution 进入
独立 `[6,1]` scalar accumulator。任何 Python `LinearOperator.to_dense()`、native observer output
adjoint、candidate 内 native shadow 或 per-layer autograd graph 都禁止。

## 4. Two-scratch physical schedule

### 4.1 Arena ownership

- admission wrapper 预分配 `scratch_0`、`scratch_1` 两个 contiguous float32 Tensor；
- 每个 scratch capacity 冻结为本 trace 的 max live coefficient numel，至少覆盖
  `A[input]=[6,1,3,32,32]`；
- compiled module 通过 DLPack 只借用这两个 storage，pointer/size/alignment/device/stream 写入 receipt；
- module cache 只能保存 compiled code 与 tensor-free schedule，不能保存 Instance Tensor；
- warm launch 前完成所有 module build、DLPack view 和 scratch allocation。

### 4.2 Fanout rule

Add11/Add6 使用 depth-first accumulator schedule：一个 slot 保留 first-branch contribution，另一个
slot计算 second branch；merge 必须原位写回 accumulator。若 exact liveness simulation 证明任一
ordinal 需要第三个 coefficient buffer，R3-1b0 直接 NO-GO，不得静默临时分配。

### 4.3 Physical evidence

每次 launch receipt 必须报告：scratch pointer、capacity/high-water、owner interval、reuse ordinal、
dynamic allocation count、module/schedule hash。`scratch_count<=2` 由指针和 memory snapshot 双重证明，
不能只读静态 IR 字段。

## 5. Compiled forward and custom VJP

### 5.1 Forward

一个 region call 返回 final lower `[6,1]`。中间 coefficient 不进入 Python output，也不进入 ctx。
Function forward 保存 compact α/β、bounds、weights、input/spec 引用；ctx 只保存 plan/execution key、
schema 和 ordinals。

### 5.2 Backward

backward 必须调用独立 compiled VJP schedule，不得调用 `_evaluate_full_region`、
`run_crown_ibp_mlp_from_forward_trace` 或 `torch.autograd.grad`。M0 VJP 可按 segment 重放 forward
recurrence，但每一 segment 仍只能借用相同两 scratch。输出只含 P compressed dα；empty beta返回
`None`。

P dα 必须从 compiled recurrence 自身的 adjoint产生，禁止读取 B4-B1 capture 的
`output_lower_a_gradient` 或任何 native shadow。

## 6. Frozen gates

### Semantic

- 5 fresh native/candidate pairs，顺序仍=`NC/CN/NC/CN/NC`；
- final lower、compressed dα：`atol=rtol=2e-4`，finite、sign exact；
- alpha/beta/split/history/version exact，mutation=`0`；
- forward/custom VJP exactly=`1/1`。

### Ownership and physical memory

- saved dense A=`0`；Python-visible intermediate coefficient=`0`；
- scratch pointer count=`2`，第三 coefficient allocation=`0`；
- warm dynamic CUDA allocation count=`0`；
- fallback/eager/native shadow/implicit `to_dense()`=`0`；
- candidate/native peak allocated **and** reserved each `<=1.0x` in all 5 pairs；
- module/schedule/plan/source/topology/lineage hash exact。

### Claim

- `timing_recorded=false`、`performance_claimed=false`；
- 本阶段即使 memory 大幅下降也只能关闭 correctness/ownership，不得形成 speedup claim；
- 任一硬门禁失败即 `VALIDATED-NO-GO-R3-1B-*`，R3-2A继续关闭。

## 7. Implementation DAG

### R3-1b0：exact trace + liveness compiler

从 BFTaskModule 与 frozen production state 生成 typed recurrence nodes、fanout/post-dominator、shape、
bias owner 和 scratch interval；与当前 native evaluator 的真实 op order/shape逐项比较。只做 IR、
validator、negative tests，不运行 compiled kernel。

通过：topology/closure exact；liveness `<=2`；所有 source op、alpha/beta path、fanout consumer均绑定；
shape/topology/scratch tamper fail closed。

### R3-1b1：compiled full-lower forward

实现 TVM TIR module与 two-scratch launcher，只跑 no-grad final lower smoke。该 smoke 不关闭 R3-1，
但必须证明 zero-copy/current-stream、warm allocation=0、无 Python intermediate coefficient。

### R3-1b2：compiled P-alpha VJP

实现 mandatory custom backward 和 compiled dα schedule；单 worker 与独立 native oracle比较 lower/dα，
并记录真实 saved-tensor hook、memory snapshot与scratch high-water。

### R3-1b3：five-fresh formal

复用 M0 artifact 协议和 5-pair顺序，增加 module/schedule/scratch/dynamic-allocation receipts及 fully
re-signed tamper。只有 b3 全过才将 `r3_1_admitted=true` 并开放 R3-2A。

## 8. Kill / reroute rules

- b0 liveness需要 `>2` scratch：停止实现，回到 contract revision，不得加隐式第三 buffer；
- b1 无法做到 warm allocation=0：停止 b2，先修 arena ownership；
- b2 语义过但 allocated/reserved任一 `>1.0x`：正式 NO-GO，不开放 timing；
- compiled VJP 实现复杂度需要保存 dense A：违反路线定义，直接 NO-GO；
- 只有 M0 compiled correctness/memory通过后，才可讨论原设计中的 M1 bitpacked certificate。

## 9. Next executable action

提交 R3-1b2 clean source并生成单worker raw-first artifact、semantic replay与fully re-signed tamper；
通过前不启动five-fresh、不跑timing、不改 optimizer。
