---
status: preregistered-not-run
updated: 2026-08-25T00:35:00+08:00
type: plan
topic: boundflow
slug: cibc-r1-scope-clock-query-local-attribution
stage: s01
---

# BoundFlow CIBC R1 Scope/Clock/Query-Local Attribution 预注册计划

## 0. 当前状态与一句话结论

本计划冻结 R1 的测量合同，**尚未实现 runner、尚未生成 artifact、尚未形成新性能 claim**。

R1 不再问“独立 ResNet2B IBP 图已经 `2.45631x`，所以 query 会快多少”，而是分两步回答：

1. 当前 CIBC candidate graph 的剩余时间究竟属于 Conv、Linear、ReLU、Add、copy 还是 runtime；
2. 真实 same-solver query 中，哪些 IBP 调用与该 candidate **结构同构且可合法接管**，其 B3-side
   exclusive/critical-path share `q_B3,k` 和 query-local effective speedup `G_query,k` 分别是多少。

独立 graph 的 `G_independent=2.45631` 只保留为历史先验和 tamper 对照，禁止代填任何
`G_query,k`。只要 query topology、shape、state owner、wrapper 或执行边界不同，query-local `G` 就必须
现场重测；测不到时按 `G_query,k=1` 保守处理，而不是沿用独立图数字。

## 1. Goal

- 冻结 graph/query/queue 三个 timing scope，消除跨 scope Amdahl 计算；
- 为约 `0.071–0.072 ms` 的 CIBC candidate graph 建立可重放的 op-type/ordinal/critical-path ledger；
- 用 CUPTI↔host/NVTX 校准 receipt 证明 event 属于同一时间轴；
- 在同一 αβ-CROWN solver 内测量 B0/B3 与 eligible-IBP 调用，形成 B3-side `q_B3,k`；
- 对 exact production shapes/state 分别测 `G_query,k`，重新计算 parity/research feasibility；
- 只开放数学上能恢复 B0 parity 的一个 R2 分支；不能到 parity 时以 NO-GO 关闭实现投入。

## 2. Non-goals

- 不修改 TIR schedule、thread candidate、数学公式或 production default；
- 不实现 Linear/Conv/epilogue 新融合，不启动 R3；
- 不宣称 auto_LiRPA/αβ-CROWN/query/queue/ASPLOS speedup；
- 不把 kernel sum、独立 graph share 或 profiler-only latency当作 complete-query 时间；
- 不用人为放大 batch 构造 memory headline；
- 不因结果不利而更换 workload、目标、threshold、worker subset 或时钟口径。

## 3. 冻结证据基线

- 当前预注册父基线：commit `2bbf182cc1411997489d777177640540dda925e7`；
- CIBC formal：`artifacts/cibc-ibp-horizontal-formal/resnet2b-prop0-v1/`；
- FSG3 B0/B2 formal：`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/`；
- FSG4 B3 formal：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`；
- NRIR49A selected-CROWN：
  `artifacts/nrir49a-g1-gpu-attribution/resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/`；
- solveability qualification：`artifacts/nrir49-g0-admission/ga403uv-pre-reboot-20260806-v7/`；
- R1 实现/正式运行必须在新的 clean source commit 上完成，manifest 绑定 plan blob、所有 touched code
  blob、模型/property、三个外部仓库 commit 和环境 identity；不得从本工作树直接形成 formal raw。

## 4. Scope 与目标冻结

| scope | share | target | 允许用途 |
|---|---|---:|---|
| whole IBP graph | `s_graph,k` | 实验单独冻结的 `T_graph` | graph 内 op 路由；不得外推 query |
| complete query | `q_B3,k` | B0-relative `1.00x/1.15x` | qualification/research |
| queue/BaB | `s_queue,k` | B0-relative `1.20x` | R1 不运行，只保留后续目标 |

固定系统目标：

```text
T_query_qualification = 1.00
T_query_research = 1.15
T_queue_research = 1.20
```

R1-A candidate-graph attribution 不设置新的 speedup 目标；它只要求保留已批准 CIBC graph 语义和
control latency envelope。R1-C 的实现准入以 complete-query qualification=`1.00x` 为硬门槛，
`1.15x` 只用于研究优先级。queue=`1.20x` 不得使用 query share 预判通过。

## 5. 条件式执行 DAG

```text
R1-0 schema/clock/topology tests
  -> R1-A candidate graph control/profile attribution
  -> R1-B same-solver B0/B3 eligible-IBP admission
  -> R1-C exact query-local region replay and G_query,k
  -> R1-D feasibility closure and single R2 route decision
```

- 任一级 artifact admission 失败，下一级关闭；
- R1-A 只读 candidate graph，不因为看到 share 而修改 bucket；
- R1-B 在读取 R1-A 结果前冻结 solver workload/config 与 eligible predicate；
- R1-C 只运行 R1-B 已冻结的 exact signatures，不新增“更好测”的 shape；
- R1-D 只读 raw 重算，不实现优化。

## 6. R1-A candidate graph attribution

### 6.1 冻结 bucket 与 ordinal

高层节点由 source topology 在运行前生成稳定 ordinal、op type、input/output shape、dtype、producer、
consumer 和 CUDA Graph node identity。bucket 恰为：

1. `input_copy_lower_upper`；
2. `cibc_conv`（6 个独立 ordinal）；
3. `linear`（2 个）；
4. `relu`（6 个）；
5. `residual_add`（2 个）；
6. `flatten_view`；
7. `graph_launch_runtime_sync`；
8. `unowned`（门禁要求为 0，不是可吸收杂项的桶）。

CUDA kernel 行缺 shape 时，只能从显式 correlation parent 和冻结 source node 恢复；receipt 必须同时
记录 `shape_source=correlation_parent` 与 parent digest。按时间邻近猜 shape、ordinal 或 owner 一律
禁止。NVTX range 名称固定为：

```text
boundflow.r1/<phase>/<ordinal>/<op_type>/<topology_hash_prefix>
```

### 6.2 control/profile fresh 顺序

- 6 个 control/profile pair，共 12 个 fresh process；pair 顺序固定为
  `CP, PC, CP, PC, CP, PC`；
- 每个 process 重新导入模型、构造 plan 和 CUDA Graph；不得 resume 部分结果；
- control：10 warmup，20 group × 50 replay；headline 只读 control CUDA-event wall；
- profile：10 warmup，20 group × 5 traced replay；收集 NVTX、CUPTI activity、CUDA Graph node、
  kernel、memcpy 和 runtime API；
- 两侧都包含 lower/upper input copy；compile、plan construction、graph capture、warmup 分别记录但
  不进入 steady-state share；
- `profile_median/control_median` 必须落在 `[0.95,1.05]`，否则该 pair attribution rejected；
- control 与 profile 的 topology/hash/launch/copy/semantic receipt 必须逐项相同。

### 6.3 四种时间口径

- `kernel_sum_ns`：只作设备工作量库存，不能作为 headline；
- `exclusive_wall_ns`：同一 owner 去除嵌套子 range 的 wall；
- `critical_path_ns`：按 stream dependency、event 和 graph edge 重建；
- `overlap_adjusted_wall_ns`：只在证明真实 overlap 后使用。

若所有 graph node 位于同一 stream，或 overlap interval count=`0`，headline 强制为
`exclusive_wall_ns == critical_path_ns`；overlap-adjusted 必须在
`max(1 us, 2% * graph_wall)` 内退化一致，否则 artifact rejected。

## 7. CUPTI↔host/NVTX 时钟校准

每个 profile process 在 trace 前后各采 64 组 native calibration triplet：

```text
host_before = CLOCK_MONOTONIC_RAW
gpu_stamp   = cuptiGetTimestamp()
host_after  = CLOCK_MONOTONIC_RAW
```

以 host bracket midpoint 拟合 `host_ns = a * gpu_ns + b`，receipt 保存全部原始 triplet、拟合参数和
残差，不只保存 summary。冻结门禁：

- 每端至少 64 个有效 triplet；
- p95 host bracket width `<=2 us`，max `<=10 us`；
- calibration fit max absolute residual `<=2 us`；
- trace 前后 slope drift `<=100 ppm`，offset drift `<=2 us`；
- Nsight Systems export 至少有 3 个已知 NVTX/CUPTI anchor，映射误差各 `<=2 us`；
- timestamp 非单调、event 越出 owner containment、缺 calibration raw 或 export receipt 均 fail closed。

这些阈值若在正式运行前的 runner smoke 不可实现，只能另提交 protocol amendment 并解释硬件/API
事实；不得运行 formal 后放宽。

## 8. R1-B same-solver eligible-IBP admission

### 8.1 公平比较边界

primary mode 固定为同一 αβ-CROWN solver：

```text
B0: original executor
B3: RVIR adapter + current BoundFlow executor
```

模型/property、branching、termination、timeout、alpha/beta steps、seed、GPU、power/thermal admission
相同。B0/B3 各自只执行一个 owner；禁止 native+candidate shadow。control 重新测 B0/B3 query wall，
不能把旧 `0.910001x` 当本轮测量结果，但必须同时报告新结果与旧 formal 的差异。

### 8.2 eligible predicate

一次调用只有同时满足以下条件才进入 `eligible_replaceable_ibp`：

- source/topology 与 R1-A 支持的 IBP region 同构；
- op inventory、ordinal、shape、dtype、device、layout、bounds owner 与 policy 全部 admitted；
- 无 α/β-gradient owner、split/history mutation、dynamic branch consumer 或 unsupported op 逃逸；
- CIBC compile key 已存在，fallback/eager/native shadow=`0`；
- region start/end 能由同一时钟域的显式 marker 和 graph edge闭合。

带 split/α/β 状态本身不自动判 eligible；只有这些状态不改变待替换 IBP region 的语义/拓扑且 receipt
能证明时才允许。否则记录具体 reject reason，不能把调用从分母删除。

### 8.3 op-type share

对 B3 侧每个互斥 op bucket `k` 记录：

```text
q_B3,k = eligible_exclusive_or_critical_wall_B3,k / complete_query_wall_B3
```

同时记录 `q_B0,k` 供解释，但传播公式只使用待优化 B3 侧 `q_B3,k`。所有 bucket 必须 disjoint；
嵌套 region 只保留最外层 wall，再由内部 exclusive ledger拆分，禁止重复计数。

## 9. R1-C query-local `G_query,k`

### 9.1 为什么必须重测

`G_independent=2.45631` 来自独立 ResNet2B IBP CUDA Graph，默认假设 Conv 主导、固定输入和固定图。
same-solver query 可能改变调用次数、shape、wrapper、copy、receipt、cache 与可接管 coverage。因此每个
eligible exact signature 都要在独立 fresh replay worker 中，用 R1-B 捕获的相同输入/state identity
比较：

```text
G_query,k = native_B3_region_wrapper_wall / existing_CIBC_region_wrapper_wall
```

wrapper wall 包含 hot-path admission、adapter、copy、dispatch、launch 和必要 identity receipt；静态
compile/plan 构造另记 cold cost。candidate 不得调用 native reference，correctness 在独立 worker
比较。若无法构造语义相同的 exact replay，`G_query,k=1` 且 `admitted=false`。

### 9.2 op-type 记录

每条 exact signature 至少记录：

- query/node/evaluation ordinal、op type、shape、dtype、layout、device、stream；
- topology/state/alpha/beta/split/history identity 与 eligible/reject reason；
- baseline/candidate wrapper、kernel、copy、receipt、sync wall；
- cold compile、cache miss/hit、plan construction 和 expected reuse；
- semantic max abs/relative diff、sign、discrete state、fallback/shadow counters；
- `G_query,k` 及 bootstrap interval。

对于尚无 candidate 的 Linear/ReLU/Add/runtime bucket，`G_query,k` 保持 `1`；R1-D 可以据 share 和所需
`r_required` 排名未来 R2，但不能编造 candidate speedup。

## 10. Feasibility 方程与机械路由

对单 stream/disjoint exclusive buckets：

```text
delta_k = q_B3,k * (1 - 1 / G_query,k)
F_query = 1 / (1 - sum(delta_k))
R_projected = R_current * F_query
```

有真实 overlap 时，禁止求和 `delta_k`，必须从 event DAG 做 counterfactual critical-path replay。单区域
反解仍为：

```text
r_required = s / (1 / T - (1 - s))
```

分母 `<=0` 表示单区域不可达。统计决策使用 6 fresh worker 的 paired bootstrap：

- `R_current`、`q_B3,k`、`G_query,k` 均从本轮 raw 重算，并给 95% interval；
- `G_independent=2.45631` 只生成一行 `historical_optimistic_sensitivity`，不得进入正式
  `R_projected`；
- R2 **qualification GO**：使用 conservative paired bootstrap 组合后，
  `R_projected_95pct_lower >=1.00`，且每个使用的 `G_query,k` admitted；
- R2 **research-priority GO**：`R_projected_95pct_lower >=1.15`；
- `<1.00`：关闭基于当前 candidate/coverage 的 R2 实现；允许保存 op ledger，不调低门槛；
- 多个支路都可达时，只开放预计 `delta_k` 最大且 correctness/工程风险最低的一支，其他仍关闭。

## 11. Artifact 与 replay

建议 formal 路径：

```text
artifacts/cibc-r1-optimized-graph-attribution/resnet2b-prop0-v1/
artifacts/cibc-r1-same-solver-admission/resnet2b-prop0-v1/
```

每个 artifact 至少包含：

- `protocol.json`、`manifest.json`、`code_revision.json`；
- `calibration_raw.jsonl`、`calibration_receipt.json`；
- `control_runs.jsonl`、`profile_runs.jsonl`、`event_raw.jsonl`；
- `topology.json`、`owner_ledger.jsonl`、`op_type_ledger.jsonl`；
- `same_solver_queries.jsonl`、`eligible_calls.jsonl`、`query_local_pairs.jsonl`；
- `summary.json`、`replay_stdout.txt`、`tamper_report.json`。

raw 先原子落盘，再生成 normalized/summary；部分 worker 不得 resume。root replay 必须从 raw 重建
clock mapping、event ownership、四种 wall、`q_B3,k`、`G_query,k`、bootstrap 与 route verdict，并逐字节
重现 summary canonical hash。

## 12. Fail-closed 与 tamper matrix

至少覆盖以下全重签篡改：

1. 修改任一 scope target；
2. 把 `s_graph` 改名/代入 `q_B3`；
3. 用 `G_independent` 填 `G_query,k`；
4. 修改 calibration slope/offset/residual 或删除 triplet；
5. 改 NVTX ordinal/topology/shape parent；
6. 把 unowned/temporal-fallback event 指给合法 owner；
7. 重复计入嵌套 region；
8. 修改 single-stream/overlap 判定；
9. 修改 eligible predicate/reject reason；
10. 删除 adapter/copy/receipt wrapper wall；
11. 修改 B0/B3 solver config、branch、timeout 或 source identity；
12. 修改 query-local pair latency/semantic/counter；
13. 把 unsupported `G_query,k=1` 改成候选 speedup；
14. 删除 failed worker、改变 CP/PC 顺序或 worker subset；
15. 修改 profile perturbation或 bootstrap route verdict；
16. 更新外层 digest 后重签任一上述 payload。

全部必须由 replay 的语义重算拒绝，不能只依赖文件 SHA。

## 13. Workload 与前端 admission

- R1-A 固定已批准的 VNN-COMP 2021 ResNet2B property 0，仅作 candidate graph attribution；
- R1-B 固定 FSG3/B3 formal 的同一 ResNet2B property/config，不因 R1-A 结果换 workload；
- solveability qualification 复用已冻结 `mnistfc:2`：双方 CPU/30 s/8 threads 均 `verified`，模型与
  property SHA 由 NRIR49 G0 artifact 绑定；它不用于性能调参；
- held-out/front-end admission 至少审计 MNISTFC、CIFAR ResNet2B、OVAL CNN 三族的实际 op inventory；
- `AveragePool` 仍 fail closed。新增 op 实现不属于 R1；只产出 missing-op/shape/dtype 清单；
- 第二 family 和至少一个 non-unknown workload 的 hash/timeout 必须在任何 R2 candidate 结果前冻结。

## 14. Validation gate

实现提交至少通过：

- R1 schema、clock-fit、scope-mismatch、owner/topology、Amdahl 与 tamper tests；
- CIBC interval/CUDA Graph 现有专项回归；
- FSG3/B3 same-solver runner/replay专项回归；
- touched files Black、Mypy、Pylint；
- full `pytest -q tests`；
- artifact root replay、tamper matrix、`git diff --check`、DocOps lint/exchange validate。

正式 GPU raw 必须来自 clean source、12 个 fresh R1-A process 和后续独立 R1-B/C workers。runner
smoke、单 process 或 profiler截图不能关闭阶段。

## 15. Claim boundary

R1-0/R1-A 完成后最多可以说“当前 CIBC candidate graph 的 read-only op/critical-path attribution 在
冻结硬件和时钟门禁下成立”；不能说 query 会加速。

R1-B/C 完成后最多可以说“same-solver eligible share 与 exact query-local candidate opportunity 已
测得，某个 R2 分支在冻结方程下具备/不具备恢复 parity 的数学准入”。这仍不是实际 query speedup。

只有 R2 实现后跑 B0/B3/cumulative candidate 三方 formal，才可形成 query candidate claim；只有
complete-query/queue/held-out/solved 总门禁通过，才可能形成 ASPLOS system claim。

## 16. Tasks

1. **本提交**：完成 R0 hygiene，并冻结本文；不运行 profiler；
2. 新 clean commit 实现 clock/topology/schema 与 negative tests；
3. 先跑 smoke 验证 calibration/perturbation，失败则在 formal 前 amendment；
4. 生成 R1-A 12-process artifact、replay、tamper 与 closure；
5. 在 R1-A 关闭后冻结并执行 R1-B/C same-solver artifact；
6. R1-D 机械计算 route verdict；只开放一个 mathematically reachable R2。

## 17. Rollback

- 本提交仅修改 R0 类型/文档并新增预注册；不改变 production default；
- R1 instrumentation 必须 additive、opt-in；删除 runner/marker 即可回退；
- 不合格 raw 保留为失败证据或移出 formal 路径，不能覆盖后续 v2；
- 任一 protocol amendment 必须发生在对应 formal worker 0 之前并保留 git/history。

## Links

- changelog: `BOUNDFLOW_R0_HYGIENE_R1_PREREGISTRATION_CHANGELOG_2026_08_25.md`
- parent recovery plan: `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
- CIBC closure: `BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md`
- roadmap: `boundflow_asplos_master_plan_2026_07_12.md`
