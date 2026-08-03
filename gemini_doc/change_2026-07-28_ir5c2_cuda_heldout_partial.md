# 变更记录：IR-5C2 CUDA typed held-out（PARTIAL）

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> runner 基线：`1be9c19`
> 工件：`artifacts/ir5/heldout-adaptive-v2-20260728`
> 判定：IR-5 measured foundation 成立；IR-5 总门禁未关闭

## 1. fresh CUDA 执行范围

- 设备：NVIDIA GeForce RTX 4060 Laptop GPU；
- Python 3.12.12，PyTorch 2.12.1+cu132，CUDA 13.2；
- calibration 2 个、held-out 2 个确定性 plain-CROWN MLP；
- 每个 workload 实测 reference、PyTorch dense、PyTorch chunked、TVM fused TIR；
- 每个候选执行 1 次 cold + 9 次 warm；
- 每次执行均经过 BoundModule→PlanInstance→TaskIR→ScheduleIR→typed backend；
- 16/16 candidate measurement final bounds 与同 workload reference allclose；
  最大 lower/upper absolute diff 分别为 `0.0006103515625` / `0.00048828125`。

manifest 精确绑定 runner commit `1be9c19d1fe6ec7666252152e2fc82aaf0e296c9`。
目录内 6 个证据文件均有 SHA-256，integrity replay 与 4 个 reference workload 的
fresh semantic replay 均通过。

## 2. 公平 policy 结果

每个 held-out workload 使用 4 个预冻结 context（cold-single、cold-repeated、
warm-single、low-memory），共 8 contexts × 4 policies = 32 outcomes。

| Policy | feasible | Oracle regret p50 | p90 | max |
|---|---:|---:|---:|---:|
| fixed reference | 6/8 | 1.19811× | 1.19878× | 1.19878× |
| local greedy | 8/8 | 1.00000× | 1.00766× | 1.00766× |
| global | 8/8 | 1.00000× | 1.00766× | 1.00766× |
| oracle | 8/8 | 1.00000× | 1.00000× | 1.00000× |

高内存 64 MiB 下 Global 对两个 held-out workload 均选择 PyTorch dense。冻结低内存预算
（8,800,000 / 9,400,000 bytes）下均切换到 TVM fused：

- medium：dense peak `8,864,768`，TVM fused peak `8,742,912` bytes；
- large：dense peak `9,748,992`，TVM fused peak `9,309,184` bytes。

heldout TVM compile/setup event total 分别为 `370.685006` / `377.490001` ms。对应 warm
mean latency 为 `56.1440` / `54.7802` ms；dense 为 `54.4033` / `55.1999` ms。
因此 large warm-single 的 Oracle 是 TVM，而 Global 仍选 dense，regret `1.00766×`；
该负例未被隐藏。

每个 outcome 均保留 p50/p90/p99 latency、TTV、peak 与 regret。TTV 使用
`compile miss + mean(warm latency) × expected query count`；warm-cache context 明确把
compiled artifact key 标为 cached。

## 3. 防泄漏与作废 pilot

- calibration model 只读取 calibration JSONL；
- held-out measurements 只在选择完成后的 outcome/Oracle 计算中使用；
- workload split、query counts、cache context 和 resource budget 均写入 `split.json`；
- replay 会拒绝当前代码常量与 split/context artifact 的漂移。

开发期 v1 pilot 曾从 held-out 最小 peak 后验构造 low-memory budget，已判为方法学不合格。
最终 v2 在 fresh measurement 前冻结固定预算，v1 不进入提交和 claim。

## 4. 为什么仍是 PARTIAL

IR-5 原契约还有四个实质缺口：

1. calibration 与 held-out 仍是同一 MLP family 的不同 shape/seed，不满足
   workload-family held-out 或 leave-one-architecture-out；
2. evaluator 尚未接入 ordinary batching 与 fair batched-original；
3. 低内存切换由可行性约束驱动，不能表述成多个可行候选间 Global cost model 的优势；
4. 还没有 CNN/残差/non-toy workload，不能证明收益来自跨层规划/调度而非简单 batch packing。

因此：

- 可以声称 typed measured pipeline、预算可行性切换和 reduced MLP regret 已验证；
- 不得声称 IR-5 closure、ASPLOS-ready、Global 显著优于 Local；
- 不启动 IR-6 cached specialization。

## 5. 复核命令与下一步

```bash
conda run -n boundflow python scripts/run_ir5_heldout_adaptive_artifact.py \
  replay --artifact-dir artifacts/ir5/heldout-adaptive-v2-20260728 --semantic
```

下一切片为 **IR-5C3**：新增独立 CNN/残差 workload family，冻结
leave-one-architecture-out split，并把 ordinary batching / fair batched-original 映射到同一
typed observation 与 TTV/peak/regret evaluator。只有该门禁通过后才审计 IR-5 closure。
