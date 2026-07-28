# BoundFlow ASPLOS 执行备忘录 v1.0

> 生效日期：2026-07-12
> 当前冻结基线：`57a854b`；closure tag：`pr13-validated-reduced`
> 当前研发分支：`feat/pr14-real-verification`
> 唯一执行顺序：**Gate 0 → PR-10 → PR-11 → PR-12 → PR-13 → PR-14**。
> 禁止同时启动 Planner、fused kernel 与 BaB runtime 三条主线。

> **2026-07-20 路线修订**：PR-14 No-Go 后对代码进行 IR-first 复审，确认现有
> `runtime/linear_operator.py`、`PlanBundle` 和拓扑执行循环不能分别等同于完整 Bound IR、
> Plan IR 和 Schedule IR。第 10 节原定的纯 `docs/asplos-c1-c2-story-freeze` 不再是下一工程
> 主线；后续按第 11 节和
> `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md` 推进。

> **2026-07-28 进度**：IR-1 Bound IR reference closure 与 IR-2 Plan IR
> `VALIDATED-REDUCED` closure 已完成；下一实现门禁是 IR-3 Schedule IR v1 + reference
> executor。不得继续把第 11 节的“尚未实现”历史描述当作当前状态。

## 1. 锁定的论文命题

BoundFlow 是面向神经网络验证中相关边界查询的 query- and memory-aware compiler/runtime。
它不重新发明 CROWN/αβ-CROWN/BaB，而是暴露 eager tensor execution 隐藏的结构、物化、
显存和跨查询复用决策。

三项正式贡献为：

1. **Structured Bound-Operator IR with Explicit Materialization Semantics**：保留可组合结构，
   显式表示 barrier、reason、bytes 与 lifetime；dense path 是参考语义，不承诺永不物化。
2. **Method-, Autograd- and Memory-Aware Materialization Planner**：在 bound method、
   differentiation/optimization stage、query workload、硬件 capability 和显存预算下选择
   物化、partition、fusion、batch、cache、recompute 与 storage/schedule。
3. **BaB-Oriented Repeated-Query Runtime for Multi-Spec and Domain Batches**：只把 multi-spec
   和 BaB domain batch 作为首篇主线；certified training 是第二客户端，其余场景为未来工作。

## 2. C2 的正式问题边界

输入为 `(G, Q, H, B, R)`：operator DAG、query 集合/分布、硬件 profile、显存预算和参考
bound 配置。计划为 `P=(m, π, f, b, c, r, s)`：materialization、partition、fusion、batch
layout、cache、recompute、storage/scheduling。

优化目标包含 amortized compile、execute、queue、transfer 和 peak-memory cost，约束为
`M_peak(P) <= B`，并要求 planned path 在相同浮点语义下保持 dense reference computation。
实现采用 candidate generation → staged cost-aware heuristic，不承诺精确联合求解；评价包含
fixed、local greedy、global heuristic 与 small-graph exhaustive oracle。

## 3. 状态有效性规则

Runtime 中的缓存对象必须标记为以下之一：

- `EXACT_REUSE`
- `CONDITIONAL_REUSE`
- `WARM_START_ONLY`
- `INVALIDATE`

| 对象 | Multi-spec | BaB 父→子 | 参数更新后 |
|---|---|---|---|
| 图结构 | EXACT_REUSE | EXACT_REUSE | EXACT_REUSE |
| Planner 模板 | EXACT_REUSE | EXACT_REUSE | CONDITIONAL_REUSE |
| 编译 kernel | EXACT_REUSE | EXACT_REUSE | shape/dtype 不变时 CONDITIONAL_REUSE |
| 参数相关常量折叠 | EXACT_REUSE | EXACT_REUSE | INVALIDATE |
| intermediate bounds | CONDITIONAL_REUSE | WARM_START_ONLY 或 INVALIDATE | INVALIDATE |
| α 参数 | CONDITIONAL_REUSE | WARM_START_ONLY | 通常 INVALIDATE |
| β/split state | INVALIDATE | 子节点专属 | INVALIDATE |
| 输出 bounds | INVALIDATE | INVALIDATE | INVALIDATE |

禁止把父节点 intermediate bounds 直接描述成子节点的有效精确结果。

## 4. Correctness/Soundness 术语

1. **数学 soundness**：由 CROWN/IBP/αβ transformer 与 solver 保证。
2. **编译变换语义保持**：dense/operator/planned/fused path 在相同浮点语义下保持参考计算。
3. **实现验证**：dense reference、allclose、gradient comparison、auto_LiRPA comparison、
   sampled concrete sanity 和 deterministic replay。

没有 outward rounding、误差 envelope 或 proof checker 时，不宣称 GPU FP32 对实数语义具有
严格 numerical soundness。论文统一使用：

> preserving the reference bound computation under the same floating-point semantics

## 5. 立即执行的 Gate 0

- 将当前环境迁移、TVM/tvm-ffi ABI、Conda hooks 和 PyTorch 2.12 reshape 兼容整理为独立边界；
- 去除 `crown_ibp.py` 全文件 Black 噪声；
- 建立统一 build/run workflow；
- 运行激活/反激活、nvcc、TVM CUDA、TVM↔Triton、auto_LiRPA 与全量测试；
- 将 MLP/CNN baseline 从单次 quick 升级为多次 reduced evidence；
- 只在 Gate 0 干净后启动 PR-10 instrumentation。

## 6. PR-10 的成功标准

- dense reference 数值等价；
- α gradient 等价；
- CROWN、α、αβ、BaB、CNN、DAG 回归通过；
- 主 coefficient 不永久退化为 dense；
- fallback/materialization reason、count、bytes 可追踪；
- materialization count/bytes 下降；
- Python lazy path 不强制当场加速，端到端性能门槛属于 PR-12；
- 不接受无法解释的严重 runtime 或显存退化。

PR-10 的第一步必须是 instrumentation，再改 ReLU operator。

## 6.1 PR-10 最终判定

- 表示、正确性与研究机会门禁：PASS；
- structured 统一默认策略：被证据否定；
- 默认保持 dense，structured 仅作为 feature-gated memory escape capability；
- plain CROWN 代表点 peak 降约 29.8%，但慢约 9.17×；α/αβ structured 出现显存恶化与
  6 个 OOM；
- 不再打磨 Python structured 特例，唯一主线转为 PR-11。

## 6.2 PR-11 最小执行边界

- 显式输入 `bound_method`、`requires_grad`、`optimization_stage`、alpha/beta/split state、
  spec/domain batch、operator summary、memory budget/available memory、reuse 与 target；
- v1 action 仅为 dense、structured、reduce-batch；capability filter 禁止当前 α/αβ optimize
  选择 structured；
- 先满足安全显存预算，再在可行候选中最小化 latency；不可行时确定性缩 batch 并 re-plan；
- 基线为 Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy、
  Global Planner、Oracle；
- 按 workload family held-out，不随机拆分相邻 shape。

当前进度（2026-07-12）：context/capability/action/plan dump、真实 CROWN shape summary、
CROWN/α/αβ runtime guard、per-case Oracle、architecture-family cost-model split 与 final held-out
matrix 已落地；全量 200 passed、1 skipped。mini-ResNet held-out 上 Global 为 239/239 feasible、
0 unexpected、median/p90 regret 1.0，但与 Memory-Threshold 决策完全相同，p99/max 仍为
5.44×/9.17×。因此下一 blocker 是 multi-barrier global placement，不是继续调单一 query
threshold；scheduler 自动缩 batch 仍未完成，PR-11/C2 状态保持 partial。

第三切片已新增 multi-barrier placement：Local 独立选最快可能超预算，Global 可联合选择 mixed
dense/structured 组合并由 runtime 按 ReLU source value 执行；合成机制与两 ReLU 数值/trace
门禁通过，全量增至 207 passed、1 skipped。该机制尚缺真实 barrier-level cost profile 与
mini-ResNet held-out Oracle，不得用合成结果替代论文证据。

Barrier evaluator 与 Global Retry 已进一步落地：在一个 final mini-ResNet `spec=32/domain=8` 的
8-budget reduced matrix 上，Global Retry 为 7/7 feasible、0 unexpected、median/p90 regret
1.159×/1.562×；Always Structured median 为 5.486×，Memory Threshold 为 2.668×。host runtime
已有真实 CUDA OOM catch/blacklist/retry 状态机，但当前结果仍是 measured budget-rejection replay，
随后真实受控 OOM 也已完成：380 MiB process-local cap、mini-ResNet s128/d32，all-dense 真实 OOM
后 all-structured 成功，3/3 独立重复稳定。BaB 长生命周期 scheduler、timeout 与状态泄漏验证仍
未完成。

随后补齐了 `latency_rank_stratified_v1` 有界候选序列：两个最快候选、80%/90% latency-rank
候选和最低 predicted-peak fallback，总尝试数上限为 5。mini-ResNet s32/d8 与 s128/d8 两组
held-out 均为 7/7 feasible、0 unexpected，median regret 为 1.159×/1.171×，p90 为
1.722×/1.221×，最大尝试数为 3/5；真实 380 MiB OOM 实验也已改走同一 bounded runtime 入口。
当前下一门禁是独立 workload/query 与 BaB scheduler，而不是继续调这两个点的候选分位数。
本切片收尾为全量 216 passed、1 skipped，Mypy/Pylint/diff check 均通过。

独立并行 branched-ResNet held-out 随后给出 No-Go：128/128 combinations 正确，有界 retry
9/9 feasible、0 unexpected，但 median/p90 regret 为 1.976×/4.494×。审计还确认现 evaluator
读取 held-out candidate 的 trace logical bytes，故只能称 profile-guided replay。PR-11 下一唯一
主线改为从 IR shape/fanout/live interval 静态生成 topology/liveness-aware barrier cost；完成前不进入
PR-12。
独立 topology 切片收尾为全量 217 passed、1 skipped，profiler Mypy/Pylint/diff check 通过。

Static-v3 随后移除了 evaluator 对 candidate trace 的 feature 依赖：Task IR + forward shape 静态
生成 shape/FLOPs/bytes/reuse/batch axes 以及 fanout/live-span/depth/merge/path summary。所有 profile
执行 3 次独立 shuffled order 并按 pattern 聚合 median；6-family/36-budget LOO 联合冻结
ridge=.001、retry factor=1.30。三组 final held-out 共 23/23 feasible、0 unexpected，median regret
为 1.000×/1.194×/1.880×，p90 为 1.747×/1.194×/2.377×，最坏 max 3.160×。
StaticPlacementQuery→model load→candidate generator→plain-CROWN runtime 已连通并通过真实 OOM
3/3。PR-11 closure audit 判定为 validated-reduced；统一 QueryState/BaB wiring 按原计划保留到
PR-13，当前可在独立提交冻结后进入 PR-12，不把 reduced 证据扩大成论文级 C2 complete。

冻结前高-regret 归因进一步表明：9 个 `regret >= 1.5` case 全部首先属于 bounded candidate
set 未包含 measured oracle，而非已有候选的 cost-model misrank；7 个 backend-gap flag 仅是
PR-12 待验证假设。PR-12 因此收敛为无梯度 plain CROWN 的 ReLU+Linear/Conv fused TIR，
不得回写 PR-11 profile，也不得包装成 Planner 修复。

PR-12 kernel foundation 已进一步覆盖 Linear 与 Conv 1×1/3×3、stride 1/2。Conv 使用显式
DSCOHW/OIHW/DSCIHW layout、原始 input-shape/output-padding contract 与 output-centric gather；
CUDA matrix 四项输出对齐，三个代表 codegen 点为 0 stack/spill/local-memory 指令。calibration
sanity 中前三点快于 PyTorch dense eager，但 stride-2 medium 仍慢 1.717×。因此当前状态只能是
kernel-level correctness/mechanism PASS。

PR-12D 已将 dense-boundary fused region 接入真实 plain-CROWN backward：显式 execution step
消费 Affine→ReLU，后端无关 executor 可在 Torch dense reference 与 TVM fused TIR 间切换；
Linear chain、stride-1/2 chain CNN、residual 与 stride-2 downsample mini-ResNet-like block 的
最终 bounds 对齐，DLPack storage alias 成立。随后复审发现 fanout contribution 丢失与
TVM-FFI custom-stream race；修复后 v1 只 fuse single-consumer Affine→ReLU，fanout/stale plan
确定性 fallback，并以 `tvm_ffi.use_torch_stream` 桥接 stream。multi-block mini-ResNet、fanout
soundness 与 adversarial custom-stream 回归通过，全量为 299 passed、1 skipped。PR-12D
correctness closure 现为 PASS。随后 PR-12E/F 建立 calibration-only backend Planner 与
default/custom-stream runtime Pareto：calibration 12/12、held-out 24/24 candidate rows 正确，
5/5 held-out 预算可行、0 unsafe，median/p90 regret 为 1.000×/1.262×。fused 在所有 held-out
降低 peak，但 memory-sensitive Linear 慢 4.21×，unseen Conv/mini-ResNet 也发生 latency reversal；
仅 3/5 选择更快或为预算唯一可行。故证据链 PASS、性能门禁 FAIL、Planner quality 仅
guarded/partial，PR-12 overall 继续 IN PROGRESS，PR-13 继续阻塞。

PR-12G 随后没有回写 v1 held-out，而是先从 TIR source/schedule 归因 Linear 长 reduction，增加
`pytorch_chunked_r512` 预算型候选，再冻结全新 multibackend-v2 split。authoritative v2 证据为
calibration 48/48、held-out 36/36 candidate rows 正确；offline calibration-only Planner 在 5 个
held-out 上 5/5 预算可行、0 unsafe，exact Oracle 3/5，median/p90 regret 1.000×/1.054×，并分别
选择 eager/chunked/TIR 1/2/2 次。selected geomean 相对 eager 为 1.081×，memory-sensitive
Linear 同时满足 64 MiB 预算并比 eager 快 1.481×。这使 reduced 多后端 Planner quality 通过，
但不能替代 structured-eager/TVM-unfused baseline、真实 profiler 与 2× headline 门禁；PR-12
overall 和 PR-13 状态不变。

PR-12H 已切换到证据闭环阶段：`44f87ae` 以本地 tag `pr12g-validated-reduced` 冻结；kernel、
region-runtime、end-to-end final-bound 三层 benchmark contract 有机器可读 schema。审计确认旧
fused-sanity 的 allocation contract 不公平，旧 PR-12E/G candidate timing 又不包含 timed
region matching/Planner，故统一保留为 `compliant=false` historical evidence，不重写旧数值。
下一唯一工程阶段是 PR-12I structured eager/TVM-unfused 公平 baseline，仍禁止启动 PR-13。

PR-12I 已在新合同下补齐 structured eager 与显式 scaled-A workspace 的 TVM-unfused 对照：
正式 v2 为 72 rows（54 ok、18 N/A、0 correctness failure）。complete final-bound 中 TVM fused
geomean 仅为 eager 的 0.546×，但 median peak ratio 为 0.512 且 3/3 Pareto；TVM-unfused 为
0.481×、0/3 Pareto，说明 fusion 的主要已验证价值是消除中间物化而非普遍加速。条件
`torch.compile(fullgraph=True)` 在三类 workload、两种 stream 上均因 `ContextVar.set` 无法
capture，已保留结构化 N/A，未为迎合 baseline 改写 workload。下一唯一阶段为 PR-12J
compile/load/cache amortization；PR-12 overall 与 PR-13 状态不变。

PR-12J 已把 TIR generation、schedule、compile、serialization、module load、memory hit 与独立
进程 disk hit 分离。authoritative v4 为 3/3 correct、0 hidden recompile。Linear/Conv 因 fused
warm 本身较慢而不可摊销；mini-ResNet 对 eager 的 fresh/disk-first/process break-even 为
4668/1062/4450 queries，均超出 Q≤1024，且对 chunked 仍不可摊销。v1 的 Conv tuple/list cache
验证 bug 和 v2 的 warm-path SHA 污染均保留为失败证据。下一唯一阶段为 PR-12K profiler；不得
以 module load 仅 0.17–0.60 ms 掩盖 process first query 约 350–419 ms 的事实。

PR-12K 在不改 schedule 的前提下完成 6 workload×5 backend 的 complete final-bound CUPTI
activity profile，30/30 correctness 通过。Nsight Compute 2026.1.1 实测因
`RmProfilingAdminOnly=1` 返回 `ERR_NVGPUCTRPERM`，因此只报告 kernel/activity time 与 launch，
禁止 bandwidth/cache、occupancy、stall 等硬件 counter claim。fusion 对 TVM-unfused 最大整体
launch 降幅仅 1.96%；按 5% 阈值为 3/6 device-time 退化、1/6 改善、2/6 中性。PR-12L 唯一
选择分支 E：停止继续手工调孤立 TIR，保留 fused 作为 Planner 候选；下一工程阶段是全新 split、
多预算和 expected-reuse 驱动的 PR-12M compile-aware Planner。PR-13 继续阻塞。

PR-12L 已将该结论冻结为唯一分支 `E_STOP_OPTIMIZING_TIR`，且没有 TIR/schedule/runtime 代码
变化。Linear tiled reduction、CUDA Graph/dispatch、chunk-size family 与 Conv capability 扩展均
不进入本次 closure；它们若未来重启，必须使用新假设和新 split。PR-12M 只能推进
capability→budget→risk→amortized latency Planner，并一次性消费全新 final held-out。

PR-12M 已完成上述 Planner 与全新 v3 held-out。calibration/final candidate 均 25/25 correct，
fit 前 manifest 明确 final 未消费且 fit/replay model SHA 一致。16/32/64/128 MiB/unbounded ×
Q1/Q32/Q1024 共 75 decisions；72 个存在实测可行 candidate 的机会全部选到可行 backend，
0 unsafe，feasible median/p90/max regret 为 1.000×/1.000×/1.016×。计划随 reuse/budget 在
eager/chunked/structured/fused 间变化；3 个 16 MiB capacity failure 单列。下一唯一阶段为
PR-12N closure audit，仍禁止启动 PR-13。

PR-12N 最终判定为 `VALIDATED-REDUCED`，closure tag `pr12-validated-reduced`。它不满足 full
`VALIDATED`，因为 Q≤1024 compile 摊销为 0/3、硬件 counter 不可用、收益只在部分 regime，
且尚无真实 BaB/VNN-COMP；但 non-toy E2E Pareto、预算价值、自动选择与独立 held-out 足以避免
`MECHANISM-ONLY`。PR-13 硬门禁因此 GO/READY，但本 closure 不启动 PR-13；后续只允许推进
真实 multi-domain/BaB query runtime，不回到 PR-12 TIR 试参。

PR-13A 随后正式建立 state-versioned `BoundQuery`、完整 compatibility key、四级 state-validity
规则和 BaB recorder。现有 host solver 生成的 8-query two-ReLU smoke 固定流为 8/8 replay、
max abs diff 0、0 loss/duplicate。该结果只关闭 contract/replay foundation；PR-13B dynamic
BatchManager、same-solver multi-backend、non-toy TTV 与 tail latency 均未完成。

PR-13B 现已补齐 exact-key dynamic buckets、budget first-fit、fill/timeout/deadline、OOM 二分重试、
结果顺序恢复和 queue/fill/latency counters，并通过现有 αβ dense executor 做真实 physical pack/
unpack。8-query smoke 动态 3 batches 为 8/8、0 loss；OOM fault 8→4+4→四个 2 后仍 8/8。
当前仍只称 foundation；下一阶段是 PR-13C same-solver adapter。

PR-13C 已把 query runtime 作为 optional bound-call adapter 接回同一 `solve_bab_mlp`。αβ
steps=3/batch=4 smoke 中 original/runtime query ID、bounds、branch、αβ state 与 solver
status/node counters 全部一致（7/7、0 loss）；forged plain capability 在 executor 0 调用时拒绝。
单次 wall time 不具权威性。

PR-13D/E 随后在 RTX 4060 上完成 5-repeat fixed/E2E reduced 评估并以
`VALIDATED-REDUCED` 关闭：fixed runtime 相对 per-node 96.52×、相对 batched original 1.024×；
hard 16-node E2E 分别为 9.93×/0.980×，status/node count 一致。结果证明 runtime 能保留 batching
收益，但不证明超越普通 batching。αβ/split 对 PR-12 compiled Planner 不兼容，non-toy/VNN-COMP、
真实 OOM 和完整 TTV 未完成；ASPLOS-ready 仍为 NO。

## 7. 投稿门禁

- **7 月 26 日**：PR-10 与真实 materialization profile；
- **8 月 5 日**：第一次硬 Go/No-Go；必须已有非平凡 Planner、held-out 非 toy workload、
  首个 latency–memory Pareto、不同预算下不同计划、0 unexpected OOM，并报告相对 Oracle
  的 median/p90 latency regret；
- **8 月 14 日**：fused task 与 headline v0；
- **8 月 15 日**：BaB prototype 与前两页初稿；
- **8 月 20 日**：主实验冻结；
- **8 月 24 日**：最终投稿决定；
- **8 月 25 日后**：禁止新增技术功能。

8 月 5 日任一核心条件缺失，立即切换 ASPLOS 2028。

## 8. 公平 baseline

- PyTorch eager、`torch.compile`/TorchInductor、TVM default Relax/TIR；
- BoundFlow dense、always-lazy、fixed barrier、local planner、global planner；
- auto_LiRPA、α,β-CROWN、条件允许时 Luna；
- 最重要的端到端对照是：**相同 host solver，只替换 original executor 与 BoundFlow executor**。

每个后续 PR 必须回答：消除什么瓶颈、改善哪个北极星指标、为哪项论文贡献增加证据、如何
验证参考语义、原始 JSONL/表图/manifest 在哪里。

## 9. PR-13 后批准路线：PR-14 Verification-Aware Execution on Real Verification Workloads

PR-13 已以 `VALIDATED-REDUCED` 关闭。其 fixed/E2E 大幅逐节点 speedup 主要来自普通物理
batching；相对公平 batched original 没有稳定净加速。因此下一阶段不得回到 PR-10B.2、继续
孤立 TIR 调优或重新设计 BaB 算法。

PR-14 的唯一目标是量化并验证已有 `BoundQuery`、Planner、multi-backend execution 和 same-solver
adapter 在真实 complete-verification workload 中的 coverage 与作用：

1. PR-14A：真实 verifier/workload adapter、query distribution 与 backend eligibility coverage；
2. PR-14B：固定真实 query replay、backend eligibility 与公平 original-batched 对照；
3. PR-14C：只在 Go 后运行 CIFAR CNN、multi-block ResNet、VNN-COMP 代表实例的完整评估。

PR-14 不重新实现 query recorder；PR-13A 的 state-versioned contract、split lineage 和 fixed replay
是唯一基础。完整门禁见 `gemini_doc/pr14_execution_plan.md`。在真实 workload、0 query loss、
same-solver correctness 和相对 batched-original 的可归因证据成立前，ASPLOS-ready 继续为 NO，
C3 不得描述成“更快的 BaB runtime”。

## 10. PR-14A/B 最终判定：VALIDATED-NO-GO

PR-14A observer 在官方 MLP/CNN 与 VNN-COMP ResNet-2B 上记录 540 个真实 bound calls；
initial phase 有 143/146 个 query 含 capability-legal region，但 activation-BaB 为 0/394。
因此 PR-14B 只允许 replay initial plain-CROWN，不新增 α/β/split kernel。

PR-14B 使用真实 `x_L/x_U/C` 和 exact per-element box。simple MLP 的 external replay 与
BoundFlow eager/chunked/TVM lower 完全对齐，但 external 请求 lower-only，而当前 BoundFlow
总是 lower+upper，故不产生公平性能 claim。VNN-COMP ResNet-2B nominal forward 与 ONNX 对齐到
`1.67e-6`，但 whole-query lower 对 external max diff `796.765`，符号仅 3/9；same-solver
替换会改变 incomplete-verifier decision。

硬决策：

1. PR-14C 不启动，不用 full E2E 绕过 bound-equivalence gate；
2. 不继续调 TIR，不新增 α/β/split kernel，不重写 verifier 算法；
3. C3 降级为支撑 C1/C2 的 query/state/capability infrastructure；
4. 原判定的下一分支为 `docs/asplos-c1-c2-story-freeze`；该项已被第 11 节的 IR-first 复审
   取代。若未来研究 external-semantics-preserving region adapter，仍必须另立新假设。

最终证据见 `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`。ASPLOS-ready 继续为
NO，直到 C1+C2 paper-level story 独立通过评审门禁。

## 11. 2026-07-20 IR-first 路线纠正

PR-14 后复审发现，C1/C2 不能只靠整理已有 story 达到 paper level：

1. `boundflow/ir/bound.py` 仍是占位骨架，结构化系数语义主要存在于 runtime Python 对象中；
2. `PlanBundle` 及 PR-11/12 的局部计划对象尚未汇合为带统一引用、合法性和 replay 的 Plan IR；
3. 现有 scheduler 只是 TaskGraph 拓扑执行，项目不存在一等 Schedule IR；
4. PR-13/14 的 query/runtime 结果不能弥补上述编译器核心缺口。

因此下一工程分支修订为 `feat/compiler-ir-stack-v1`，顺序冻结为：

```text
Bound IR v1
  -> Plan IR v1
  -> Task IR + Schedule IR lowering
  -> reference/backend runtime migration
  -> adaptive PlanInstance evaluation
```

在三层 IR 的 typed schema、verifier、deterministic dump/hash 和最小端到端闭环完成前：

- C1 只能称 runtime mechanism foundation；
- C2 只能称局部 planner/backend mechanism validated-reduced；
- C3 保持降级，不以普通 batching 或计划中的 JIT 重新包装；
- 不新增 α/β/split kernel，不重写 BaB，不继续孤立 TIR 调优。

完整对象边界、迁移关系、JIT/状态有效性门禁和逐阶段 DoD 见新的架构契约文档。

2026-07-28 状态追加：

- IR-1 typed Bound IR、plain-CROWN lowering、dense/structured interpreter/rewrite 已通过；
- IR-2 typed PlanTemplate/PlanInstance、builder/selector、state-validity、legacy assembly 与
  deterministic artifact replay 已通过 reference closure；
- 当前 `artifacts/` 不含 PR-11/12 raw planner records，因此不声称历史逐记录迁移；
- 下一步从 IR-3 Schedule IR 实现继续，C1/C2 在 runtime/backend/E2E 前仍不得升级。
