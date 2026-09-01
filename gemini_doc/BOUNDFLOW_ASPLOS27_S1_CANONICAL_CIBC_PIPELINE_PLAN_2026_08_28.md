---
status: execution-active
date: 2026-08-28
type: plan
topic: boundflow
slug: asplos27-s1-canonical-cibc-pipeline
external-audit: deferred-by-user
performance-claimed: false
---

# ASPLOS’27 S1 canonical CIBC pipeline 执行计划

## 1. 目标

把已经验证的 CIBC IBP Conv winner 接入唯一的 production compiler path：

```text
ONNX/Primal interval task
  → exact-batch storage specialization
  → one standard Relax dataflow function
  → 6 CIBC paired lower/upper Conv PrimFunc + 2 cuBLAS Linear partitions
  → TVM VM executable
  → prepared DLPack bindings
  → static-address CUDA Graph replay
```

本阶段只证明 standalone IBP 图的 compiler-plumbing qualification，不外推 αβ-CROWN、same-solver query、
complete-query 或 ASPLOS headline。

## 2. 设计冻结

- 不新增 Plan/Task/Schedule/solver execution IR；现有 `BoundTask` 只是 frontend compatibility source；
- Relax 是唯一图级 production execution IR，TIR 是唯一 kernel IR；
- 6 个 Conv 复用既有 CIBC center/deviation 与 paired lower/upper 算法；
- Linear 不重复发明 GEMM，必须由 TVM 官方 cuBLAS partition 接管；
- ReLU/Add/Flatten 保持标准 Relax dataflow；
- warm run 不允许 per-op Python、DLPack view construction、eager/native shadow 或 fallback；
- CUDA Graph 只能捕获已经正确的 mixed VM；graph-stable VM output 在 prepare 阶段绑定，不能在 replay
  内增加错误或多余的 output copy；
- dynamic input 先做一次 finite/lower≤upper admission，warm replay 只做 pointer + tensor version 的 O(1)
  identity guard；参数同样由 pointer + version fail closed。

## 3. Identity 链

每个 prepared program 必须绑定：

```text
source task + parameter contents hash
  → exact-batch specialized storage hash
  → backend/schedule plan hash
  → source Relax IR hash
  → cuBLAS-partitioned/lowered Relax IR hash
  → device source hashes + target
  → compile receipt hash
  → invocation receipt
```

receipt 必须披露 17-op、6 CIBC call_tir、2 cuBLAS partitions、prepare/warm DLPack 数、input copy、
CUDA Graph replay、output materialization、fallback/eager-shadow 与 claim flags。

## 4. Correctness 门禁

- PyTorch interval graph、旧 direct-CIBC graph、canonical pipeline 三方比较；
- final lower/upper `atol=rtol=3e-4`，sign exact；
- 独立 debug module 输出全部 17 个中间值，定位 residual/fanout/Linear 累积误差；
- 6/6 Conv 与 2/2 Linear backend coverage exact；
- mutated input、schedule inventory、claim flip、fallback、DLPack、coverage 与 cuBLAS tamper 均 fail closed；
- production formal path 只返回 final pair，intermediate escape=0。

`3e-4` 在 formal 运行前写入 protocol；它与旧 CIBC artifact 数值相同，但本阶段仍独立执行三方比较，
不能仅引用旧结果。

## 5. 性能 protocol

- hardware：RTX 4060 Laptop / sm_89；
- 6 个 fresh process，顺序覆盖 `BDP/BPD/DBP/DPB/PBD/PDB`；
- B=PyTorch CUDA Graph，D=旧 direct-CIBC CUDA Graph，P=canonical pipeline CUDA Graph；
- 每进程 30 group，每对象每 group 200 replay；
- 三侧均计入 lower/upper input copy；compile/prepare/capture 不进入 warm headline，单独披露；
- 每 run 先取各对象 group median，再跨 run 计算 geomean 与 worst。

资格线：

1. `pipeline/PyTorch geomean >=2.20x`；
2. `pipeline/PyTorch worst >=2.00x`；
3. `pipeline/direct-CIBC geomean >=0.90x`；
4. correctness、coverage、fallback、identity 和 tamper 全通过。

资格通过也只设置 `s1_performance_admitted=true`，`performance_claimed=false` 与
`same_solver_claimed=false` 保持。

## 6. Kill gate 与后继

- mixed VM 未捕获若变慢，先归因 VM/launch，不改变 CIBC kernel；
- Linear 通用 DLight kernel若明显落后，优先 cuBLAS，不手写未经调优 GEMM；
- CUDA Graph 若改变语义或依赖未持有内存，立即关闭 Graph candidate；
- 任一门禁不过，S1 以 NO-GO/diagnostic closure 结束，不开放 S2；
- 全部通过后，唯一后继是 S2 coarse CROWN/custom VJP 进入同一 prepared path。
