---
status: validated-no-go-mr0-explicit-event-budget
updated: 2026-08-26T14:10:00+08:00
type: closure
topic: boundflow
slug: mr0-explicit-event-budget-formal-no-go
stage: s01
---

# BoundFlow MR0 Explicit Event Budget 正式 NO-GO Closure

## 1. Verdict

`VALIDATED-NO-GO-MR0-EXPLICIT-EVENT-BUDGET`。

在 CIBC production 17-op CUDA Graph 上，每 replay 记录17对预分配 CUDA event 会使整图时间增加到
control 的约 `2.14x`。five-fresh 的 geomean、bootstrap 95% upper、worst worker 三项均远高于
冻结 `1.05/1.05/1.08x` 门槛。因此 explicit-event op-level attribution 正式关闭，MR1 internal-
boundary correctness不开放。

## 2. Frozen identity

- source=`651e432068d44ba667f657b4c359065244a16792`；
- artifact=`artifacts/measurement-recovery/mr0-explicit-event-budget-resnet2b-v1/`；
- source capture/model SHA256=`f42229dd…126dc`/`791aa24d…4a6d`；
- protocol hash=`42c38a70bec402ce31692ac8b5c858933cc1104f61dd089a8b071ab70854e5f0`；
- summary hash=`0385afa44afe42ad55165162c5921bf8b152fbda479e10b2a98b22e157211193`；
- final manifest hash=`78a966c44132d7747e6ec98846a776ab09dad1d1eb7a295d6d098248bb6fd839`；
- tamper hash=`e6bd116ad8052f717e75f3d49b0f178df6848d3479aaa6d777a57f642e83e5cd`。

5 个独立 GPU process 的 order=`CI/IC/CI/IC/CI`；每 budget 20 paired group、每侧100 replay；
compile/capture/event construction不计时，input copy与CUDA Graph launch计入。

## 3. Budget sensitivity

| event pairs / replay | geomean | bootstrap 95% upper | worst worker | decision |
|---:|---:|---:|---:|:---:|
| 1 | `1.021632x` | `1.058469x` | `1.064672x` | disclosure only |
| 4 | `1.158736x` | `1.159466x` | `1.159738x` | disclosure only |
| 8 | `1.317715x` | `1.319083x` | `1.319695x` | disclosure only |
| 17 | `2.137191x` | `2.153191x` | `2.163574x` | **NO-GO** |

17对的5 worker ratio=`[2.138366,2.163574,2.115665,2.150273,2.118469]x`。1对虽然
geomean低于1.05，但bootstrap upper与worst均未过；更重要的是1对不能覆盖17-op owner，因此禁止
降级预算后宣称op-level attribution可用。

## 4. Correctness 与 scope

- 每 worker 36 个 lower/upper tensor、235992元素 exact；max diff=`0.0`；
- output pointer、shape/dtype/device contract、current stream 全部稳定；
- candidate Conv coverage=6，fallback/eager shadow=`0/0`；
- 36个event object只在timing前预分配；
- portable log无本机路径；
- `performance_claimed=false`、same-solver/R2均关闭。

## 5. Replay、tamper 与回归

- root replay从5 raw重算所有median/ratio/bootstrap/gate/verdict：PASS；
- 12/12 fully re-signed tamper rejected；
- targeted：`10 passed`；
- full regression：`1677 passed, 3 skipped`（3项均为既有环境边界）。

首次source=`3080af6` formal已得到同方向NO-GO，但最初的第1类tamper只改一个非中位样本，未改变
派生summary。该artifact保存在`/tmp/mr0-first-formal-3080af6-tamper-probe-incomplete`，不参与正式
结论。修正攻击为整组20个sample全重签改写后，以source=`651e432`从worker 0完整重跑。

## 6. Route propagation

1. 关闭 explicit-event per-op attribution，不开放 MR1；
2. 不回退 profiler/Nsight 路线，不放宽扰动/时钟门槛；
3. 不选择 Linear/Conv/runtime 优化，因为仍没有合格的 same-solver op share；
4. 下一步只开放**无计时的 same-solver static eligibility audit**：从现有 B3/RVIR raw逐调用判断
   CIBC full-graph executor是否在 topology/state/consumer 语义上可替换；
5. 只有发现真实 eligible 调用，才允许另行预注册 direct end-to-end B0/B3/candidate A/B；若0 eligible，
   直接关闭当前 CIBC→same-solver 接入假设。

## 7. Claim boundary

允许：17-op explicit CUDA-event record预算在冻结graph上以2.14x量级扰动而NO-GO。

禁止：event结果是op share、CIBC无法加速query、kernel/bridge/autograd哪个是dominant、same-solver/R2
开放、跨模型或ASPLOS-ready。
