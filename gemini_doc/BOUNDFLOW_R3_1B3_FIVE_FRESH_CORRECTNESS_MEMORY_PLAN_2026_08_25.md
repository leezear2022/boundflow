---
status: preregistered-r3-1b3-five-fresh
updated: 2026-08-25T05:32:00+08:00
type: plan
topic: boundflow
slug: r3-1b3-five-fresh-correctness-memory
stage: s01
---

# R3-1b3 Five-Fresh Correctness / Memory 预注册

## 1. 唯一问题

R3-1b2 已证明一个 fresh worker 的 compiled lower+dα correctness/ownership，但 R3-1 的最终 kill gate
要求五对独立进程中 candidate 的 PyTorch-visible physical peak allocated 和 peak reserved 都不高于
native。R3-1b3 只回答这个问题，不测 latency，不接 optimizer。

## 2. Frozen protocol

- pair count=`5`，worker count=`10`；
- spawn order=`NC/CN/NC/CN/NC`；每个 mode 是全新 subprocess；
- source/capture/model/plan/trace/module/device hash 全冻结；
- workload=`25/Conv_8`，one evaluation，mutation=`0`，P beta absent；
- exact interpreter、torch/CUDA、RTX 4060 Laptop / sm_89；
- native=production eager autograd oracle；candidate=b1 compiled forward + b2 compiled custom VJP；
- lower/dα=`atol=rtol=2e-4`、finite、sign exact；
- final α/β version、split/history、launch/ownership receipt exact；
- `timing_recorded=false`、`performance_claimed=false`。

## 3. Memory measurement

每个 worker 先完成共同的 model/state binding。candidate 还必须在测量前完成 module compilation、DLPack
views和所有 PlanInstance storage（两个 coefficient arena、4 sign bitmap、pre25 value、gradient/output）；
这些 live storage 已进入 reset 时的 absolute baseline，因此不会被排除在 absolute peak 之外。

随后执行：

```text
gc.collect -> torch.cuda.empty_cache -> synchronize
reset_peak_memory_stats
record allocated_before/reserved_before
execute exactly one native or candidate lower+dα
synchronize
record max_memory_allocated/max_memory_reserved
```

headline 只比较 absolute peak：

```text
allocated_ratio = candidate.peak_allocated / native.peak_allocated
reserved_ratio  = candidate.peak_reserved  / native.peak_reserved
```

increment 只披露，不参与门禁；不得在 reset 后调用 empty_cache，不得扣除 arena/sign/value workspace，
不得用 allocator rounding 事后改阈值。

## 4. Gates

所有五对同时满足：

- lower/dα allclose + sign exact；
- candidate compiled/custom VJP=`true/true`；forward/backward=`1/1`；
- b1 launches=`15/15`、b2 launches=`10`；
- scratch=`2`、saved dense A=`0`、warm dynamic allocated=`0`；
- fallback/eager/native shadow/mutation=`0`；
- `allocated_ratio<=1.0` 且 `reserved_ratio<=1.0`。

全部通过：`VALIDATED-R3-1B3`，`r3_1_admitted=true`，只开放R3-2A optimizer trajectory
correctness。任一失败：按证据命名 `VALIDATED-NO-GO-R3-1B3-*`，R3-2A继续关闭。无论结果如何，
本轮都不形成 speedup claim。

## 5. Artifact/replay

raw 使用每 worker 一个 `.pt`，保留完整 lower/dα tensor、physical memory receipt、execution receipt、
版本与环境；stdout不得泄露本机路径。manifest绑定所有raw/stdout、protocol、summary、source和code
revision。replay逐 worker重算tensor digest、五对语义、ratio、status与summary hash；fully re-signed
tamper至少覆盖数值、memory、compiled/custom VJP、scratch/saved state、claim和pair order。

## 6. 执行顺序

1. 实现worker/artifact/replay/tamper及synthetic negative tests；
2. 提交clean source；
3. 运行10个fresh subprocess；
4. replay + tamper + targeted/full regression；
5. 按预注册公式关闭为GO或NO-GO，不挑样重跑。

## 7. Formal run-0 环境纠正

首次 clean-source formal 在 run 0 native 完成、candidate import 阶段 fail closed：subprocess 环境把
`PYTHONPATH` 重置为仓库根，误删 Conda activation 提供的 TVM/TVM-FFI 路径，candidate 因
`ModuleNotFoundError: tvm` 退出。原子临时目录自动清理，未产生可续跑的部分 artifact，也没有
数值/memory 结果进入结论。修正仅把仓库根前置并保留冻结父进程 `PYTHONPATH`；必须提交新 clean
source 后从 run 0 重跑全部10 worker。
