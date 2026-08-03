# 2026-07-13：PR-12E runtime Pareto 与 PR-12F frozen held-out

## 目标与范围

在 `a7fe6c1` 的 PR-12D correctness closure 之后，建立不依赖全局 CUDA 同步的正式
runtime/network benchmark，并只用冻结 calibration 数据选择 `pytorch_eager` 或
`tvm_fused_tir`。本切片不修改 kernel schedule、不扩 α/β/grad/split capability，也不启动
PR-13。

## 实现

- 新增 `boundflow/planner/fused_crown_backend.py`：同 family、最近
  log(bytes-per-region) 的 calibration-only Planner；先做 graph/capability eligibility，再按
  显存预算和 calibration latency ratio 选择后端；每个决定记录 calibration case、预测 peak、
  ratio 与 reason。
- 新增 `scripts/benchmark_phase7a_pr12_runtime_pareto.py`：由冻结 split 构造 Linear、Conv 和
  三 block mini-ResNet；分离 compile-first、cold、warm host latency、CUDA-event latency、
  allocator peak；default/custom stream 分别测量；加入 fanout graph-ineligible fallback control。
- 新增 `scripts/postprocess_phase7a_pr12_runtime_pareto.py`：raw JSONL → candidate CSV → 两张
  Pareto 图 → planner summary → manifest/hash。
- 新增三组 contract tests，覆盖 split/region count、calibration/held-out 隔离、预算/eligibility
  决策、ineligible Oracle 过滤和 postprocess 证据链。

## 测量协议

正式 calibration 与 held-out 均使用：warmup 5、5 个独立测量组、每组 10 query；default 与
non-default custom stream 分开。GPU 时间使用同一被测 stream 上的 CUDA Events；host 时间按
每组一次 stream synchronize 后除以 query 数；timed region 不调用全局
`torch.cuda.synchronize()`。显存为 PyTorch allocator peak delta，同时记录 allocated、reserved、
output bytes 与 temporary-workspace upper bound。

工件：

- calibration：`artifacts/phase7a-pr12/pr12e-calibration-v1-20260713/`；12/12 rows OK；
- canonical held-out：`artifacts/phase7a-pr12/pr12f-final-heldout-v1-canonical-20260713/`；
  24/24 rows OK；
- CSV/figure/manifest：`artifacts/phase7a-pr12/pr12ef-report-v1-canonical-20260713/`。

`artifacts/` 按仓库规则不进入 Git；report manifest 固定三个 raw 输入和所有输出的 SHA-256。

## 结果

### Candidate Pareto（default stream）

| held-out case | eager / fused warm ms | eager / fused peak MiB | speedup | 结论 |
|---|---:|---:|---:|---|
| Linear unseen A | 1.065 / 0.979 | 0.062 / 0.033 | 1.088× | fused 更快且更省显存 |
| Linear unseen B | 1.063 / 0.986 | 1.455 / 0.663 | 1.079× | fused 更快且更省显存 |
| Linear memory-sensitive | 2.025 / 8.516 | 68.599 / 29.282 | 0.238× | 64 MiB 下只有 fused 可行，但慢 4.21× |
| Conv unseen width | 1.323 / 1.670 | 23.689 / 12.441 | 0.792× | 显存下降，latency 退化 26.2% |
| mini-ResNet 3 blocks | 5.941 / 6.139 | 18.451 / 17.047 | 0.968× | 显存小幅下降，latency 退化 3.3% |

所有 fused held-out 均通过 `rtol=atol=2e-4` 的最终 bound allclose、finite 与
`lower<=upper`。mini-ResNet 的最大绝对误差受 bound 量级放大为 12288，但最大相对误差仅
`2.20e-7`，因此不能只用绝对误差判断失败。

compile overhead 约 0.29–1.48 s。只有三个 warm-faster 点存在有限 break-even，约
2.2k–7.4k repeated queries；其它点在当前 schedule 下无 latency break-even。

### Planner held-out

```text
held-out cases:                         5
budget feasible:                        5/5
unsafe fusion:                          0
median latency regret:                  1.000×
p90/max latency regret:                 1.262× / 1.262×
profitable or budget-required choices:  3/5
fanout fallback controls:               1/1 eager, regret 1.000×
```

Planner 对两个普通 Linear 选择 fused 且命中 warm Oracle；对 memory-sensitive Linear 因 eager
超过 64 MiB 而选择唯一预算可行的 fused。calibration 中单一 Conv shape 没能预测 unseen Conv
与 mini-ResNet 的 latency reversal，二者误选 fused。这是合法、无数据泄漏的 held-out 负结果，
不得事后调阈值并回写同一 split。

## 阶段判定

```text
PR-12D correctness closure:              PASS
PR-12E runtime/Pareto evidence chain:     PASS
PR-12E performance target:               FAIL
PR-12F frozen held-out execution:         PASS
PR-12F planner quality:                   GUARDED / PARTIAL
PR-12 overall:                            IN PROGRESS
PR-13:                                    BLOCKED
```

原因：fused 明确扩展 memory Pareto，但没有达到“代表 workload 相对 structured eager 几何平均
>=2×”的内部目标，也缺 TVM-unfused/structured-eager 完整正式对照；Conv schedule 在 unseen
shape 上发生 latency reversal，Planner p90 超过 1.20×。当前证据足以否定“总是融合”并证明
memory-aware selection 必要，但不足以形成论文 headline。

## 下一步

1. 冻结本轮 raw/manifest，不再用 `pr12-final-heldout-v1` 调参；
2. profile Conv unseen 与 memory-sensitive Linear，定位 output-centric gather/reduction 的
   occupancy、memory throughput 和 launch 开销；
3. 只新增 calibration-v2/new split 后再训练 Planner，不污染 v1 held-out；
4. 补 structured eager 与 TVM-unfused/default 正式对照；
5. 只有新的 schedule/候选使 Conv/mini-ResNet Pareto 前沿前移后，才关闭 PR-12 或进入 PR-13。

## 收尾验证

```text
新增 PR-12E/F contract tests：8 passed
全量：307 passed、1 skipped
Mypy（Planner/两个脚本/三组测试）：success，6 files
Pylint（Planner/两个脚本）：10.00/10
Black / git diff --check：通过
```

全量第一次使用环境内 Python 绝对路径、但未激活 Conda PATH，Phase6h 子进程误调用系统
Python 并缺少 Torch；按仓库规定 `conda activate boundflow` 后重跑，全量通过。该失败不是
代码回归，也未被计入最终门禁。
