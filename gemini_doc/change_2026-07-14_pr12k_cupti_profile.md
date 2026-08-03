# 2026-07-14：PR-12K CUPTI activity profile 与 TIR 止损判定

## 目标

在不修改 schedule 的前提下，对 PR-12I/J 暴露的 fused latency 问题做可复核归因，并从
PR-12L 的冻结分支中只选择一个后续动作。测量遵循 complete final-bound 合同，覆盖 Linear、
stride-1/2 Conv、两 block residual 和未见 width mini-ResNet；比较 eager、structured、chunked、
TVM-unfused 与 TVM-fused 五个 backend。

## Profiler 能力审计

本机 Conda 环境包含 Nsight Compute 2026.1.1 与 CUPTI：

```text
ncu:       $CONDA_PREFIX/bin/ncu
libcupti:  /opt/cuda/targets/x86_64-linux/lib/libcupti.so
driver:    RmProfilingAdminOnly=1
ncu probe: ERR_NVGPUCTRPERM
```

因此不能合法采集硬件 performance counter。本阶段降级为 `torch.profiler` 的 CUPTI activity
trace，只报告真实 kernel 名称、launch 次数、CUDA device activity time 与 launch API CPU time。
下列结论明确禁止：SpeedOfLight utilization、带宽/cache counter、achieved occupancy、scheduler
stall reason。没有使用 `sudo`，也没有修改驱动、内核或系统权限。

## 实现与工件迭代

- 新增 PR-12K profiler runner，输出每个 workload/backend 的 raw JSONL 与压缩 Chrome trace；
- 新增后处理，生成 activity/top-kernel/fused-comparison CSV、图、summary 与 SHA manifest；
- profiler 使用一个标记 range 包住 complete final-bound，但统计时排除该 inclusive range，避免
  把 CPU annotation 的聚合 CUDA time 当作额外 kernel；
- 所有 workload 都先和 eager final bound 做 finite、lower≤upper、allclose 门禁；
- v1 暴露 inclusive range 被重复计数；v2 修正计数但 top-kernel raw `count` 被序列化为 float；
  v3 raw 修复为整数 count；v4 只修复报告中旧的硬编码 `5/6` 文字，是权威 report。

权威工件：

```text
artifacts/phase7a-pr12/pr12k-cupti-v3-20260714/
artifacts/phase7a-pr12/pr12k-cupti-report-v4-20260714/
```

关键 SHA256：

```text
raw.jsonl:             7b9879397a4f9783a45961e526e5890ec2e2a8a9c304a7758718cb37ddab1cad
profiler_audit.json:   deda6829731efcee288a3fab7e1a6b737f0948f7546def9367895297b7917b3c
report summary.json:   8e5558b3c87f41b523b6211c799ff376aef591f7d95215a5af5e72ccd84e11da
```

## 结果

30/30 profile rows 状态为 ok，30/30 correctness 通过。与 TVM-unfused 相比：

| workload | fused/unfused CUPTI device time | 判定（5% 阈值） |
|---|---:|---|
| small Linear calibration | 0.986× | neutral |
| memory-sensitive Linear | 1.252× | regress |
| stride-1 Conv calibration | 1.685× | regress |
| stride-2 Conv | 1.026× | neutral |
| two-block residual | 1.238× | regress |
| unseen-width mini-ResNet | 0.637× | improve |

fusion 相对 TVM-unfused 每个 eligible region 只减少 2 次 launch，六个 workload 的最大整体 launch
降幅仅 1.96%。按 5% 阈值为 3/6 退化、1/6 改善、2/6 中性。该结果说明继续凭经验打磨单个
TIR schedule 缺少足够证据；同时 mini-ResNet 的改善证明 fused backend 仍应作为受约束候选保留，
不能删除。

## 判定

```text
PR-12K correctness/profile chain: PASS
hardware-counter attribution:     UNAVAILABLE (permission blocked)
PR-12L selected branch:           E_STOP_OPTIMIZING_TIR
PR-12 overall:                    IN PROGRESS
PR-13:                            BLOCKED
```

PR-12L 不再新增 schedule 参数或 TIR candidate，只冻结这个止损决定。下一项工程工作是 PR-12M：
把 compile/load/first-query/reuse 纳入 Planner，在全新 split 和多显存预算上决定 eager、chunked、
TVM-unfused 或 fused；不得回写 PR-12G final held-out。

## 验证

```text
PR-12K focused：2 passed
全量：            332 passed、1 skipped
mypy：            2 scripts success
pylint：          2 scripts 10.00/10
Black/diff check：通过
```
