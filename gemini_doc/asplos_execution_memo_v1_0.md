# BoundFlow ASPLOS 执行备忘录 v1.0

> 生效日期：2026-07-12
> 当前 integration base：`9d55b0a`；历史 closure tag：`pr13-validated-reduced`、
> `ir5-final-validated-nogo`
> 当前研发分支：`feat/native-real-network-memory-plans-v1`
> 唯一执行顺序：**Gate 0 → PR-10 → PR-11 → PR-12 → PR-13 → PR-14**。
> 禁止同时启动 Planner、fused kernel 与 BaB runtime 三条主线。

> **2026-07-20 路线修订**：PR-14 No-Go 后对代码进行 IR-first 复审，确认现有
> `runtime/linear_operator.py`、`PlanBundle` 和拓扑执行循环不能分别等同于完整 Bound IR、
> Plan IR 和 Schedule IR。第 10 节原定的纯 `docs/asplos-c1-c2-story-freeze` 不再是下一工程
> 主线；后续按第 11 节和
> `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md` 推进。

> **2026-08-03 最终状态**：IR-1—IR-4 narrow plain-CROWN compiler/runtime 已
> validated-reduced；IR-5D prepared execution remediation 已完成。fresh residual-v3
> final correctness/replay 全过，但 Global p90 1.26160×、gray 无 Pareto、无预算切换。
> IR-5 最终 VALIDATED-NO-GO；当前 ASPLOS system-performance 路线停止，IR-6 不启动。

> **2026-08-03 correctness 后续**：独立 RVIR 路线已以 CPU VALIDATED-REDUCED 关闭。
> ResNet external-semantics initial-CROWN 等价恢复；activation external exact call 已进入
> Bound/Plan/Task/Schedule typed stack。该结果不撤销 IR-5 No-Go，也不构成性能 claim；详见
> 第 12 节。

> **2026-08-04 P0 路线选择**：真实 production Schedule-memory 准入门禁为 `NO_GO`。
> Reduced residual path 有 arena/launch ownership，但没有 materialization、storage choice 或
> budget-driven decision switch；真实 ResNet 仍是单 external opaque call。不得直接重开 IR-5/
> IR-6，下一分支是 `feat/native-real-network-bound-ir-v1`，详见第 13 节。

> **2026-08-04 NRIR-1 结果**：固定 ResNet2B 的 main initial-CROWN backward 已从 external
> opaque wrapper 变为 21 个 native Bound/Task regions 和 21 次 Schedule launch；五层 hash
> 绑定 external-bound payload，CPU lower max diff `7.15256e-7`、sign 9/9。关闭等级只为
> correctness/compiler ownership VALIDATED-REDUCED；下一步是 NRIR-2 多计划/memory decision，
> 不是直接宣布性能结果。详见第 14 节。

> **2026-08-04 NRIR-2 结果**：同一 real ResNet Bound IR/PlanTemplate 已加入 retain-all 与
> lifetime-reuse 两个 storage plan；预算会切换 PlanInstance/Schedule，runtime 按 selected
> last-use 提前释放值。logical/observed peak 为 `1,860,912`/`442,656` bytes，两计划
> bitwise equal。该 closure 仍为 CPU mechanism/correctness，不是 CUDA memory/performance。
> 详见第 15 节。

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
- IR-3 typed Task/Schedule schema、逐 Task reference semantics、control trace 与 artifact v2
  已通过 synchronous reference closure；
- IR-3 关闭时冻结的下一步曾为 IR-4 production backend/runtime migration；该动作现已由
  下方 IR-4A—E 完成记录取代，C1/C2 在 IR-5 公平证据前仍不得升级为 paper-level complete。
- IR-4A 已新增跨 Bound/Plan/Task/backend 的 typed dispatch key 和 PyTorch reference
  prepared-task adapter；这只是迁移入口，chunked/structured/TVM/query runtime 仍为 pending。
- IR-4B 已把 PyTorch dense/structured/chunked 接入 typed registry；chunked fused Task 在 CUDA
  上真实调用原 executor。下一门禁是 TVM typed compile/cache，不是孤立 kernel 调优。
- IR-4C 已完成 TVM fused/unfused typed dispatch、dispatch-key cache v2、跨进程 disk replay
  与 Schedule semantic OOM fallback；
- IR-4D 已完成 capability-gated typed compiler query、Plan/Task cache、exact-version dense
  state payload、真实 load/store/task skip 与 fresh-process artifact；PR-13 α/β 请求保持
  external No-Go，不降级为 plain CROWN；
- IR-4E 已新增 `plain_crown_typed_ir` BoundQuery capability，并让 PR-13
  DynamicBatchManager 只通过正式 adapter 调用 typed compiler；旧 `SameSolverQueryRuntime`
  默认关闭，仅 PR-13 历史回归显式 opt-in，且错误/审计保留 PR-14 No-Go；
- IR-4 已以 validated-reduced 关闭。下一步进入 IR-5 adaptive PlanInstance；不得提前启动
  IR-6 cached specialization，也不得把 compiler closure 升级成 α/β external integration。
- IR-5A 已新增 query-time memory/deadline/cache/distribution context，并按 uncached
  compile/setup 在 expected query count 上摊销选择；cold/repeated/warm context 可产生不同
  typed PlanInstance；在 IR-5A 时点仍需 fixed/local/global/oracle 与 held-out 系统证据。
- IR-5B 已冻结四策略共享 observation/context 的公平 evaluator，输出 tail/TTV/peak/regret；
  当前 artifact 明确为 synthetic contract，不得写成 held-out 性能结果。
- IR-5C1/C2 已冻结 calibration-only CUDA runner 和资源 context，并在 fresh typed MLP
  artifact 上得到 Global 8/8 feasible、p50/p90 regret 1.000×/1.00766×；高内存选择
  PyTorch dense，低内存选择 TVM fused。该时点结果仅为 PARTIAL：同-family split、
  ordinary batching/fair batched-original 与 non-toy workload 尚缺，随后由 IR-5C3 补测。
- IR-5C3 已用 MLP calibration→chain-CNN held-out 和 fair batched-original 补齐关键口径；
  correctness/feasibility 全通过，但 Global p50/p90 regret 为 68.065×/70.263×，
  64/512 MiB 均选 chunked且无 Pareto。当前 IR-5 v1 VALIDATED-NO-GO，IR-6 blocked。
  profile 指向 query hot path 重复 validate/hash。
- IR-5D 已把静态 validate/hash/dispatch key 移入 prepared capsule，并分离 audit/production
  trace；在旧 CNN 上使用 from-forward-trace 公平计时的 calibration median 比值最快为
  0.880×/0.896×。该诊断不撤销 No-Go；其后按预注册门禁执行了一次 residual final。
- IR-5E 已新增 residual fanout/add typed workload，并冻结 chain-CNN calibration →
  residual-CNN final v2、from-forward-trace baseline、p90≤1.20 与 Pareto 判定字段。
  `7401/7402` 随后首次生成时因输入身份协议错误失效并永久退役。
- IR-5F 首次 v2 生成在 semantic gate 中止：同 seed 不保证不同 batch shape 的随机输入
  具有前缀关系。参数一致但 input max diff 为 3.735/2.167；无 summary/manifest，不能作
  性能结论。只允许显式 slice batched input、升级 schema 并旋转 fresh identities。
- IR-5G 已实现上述唯一修复：single 输入 exact clone batched query zero，并在 bound
  comparison 前做 tensor identity gate；v3 `7501/7502` 随后按协议运行一次并冻结。
- IR-5H v3 final correctness/integrity/semantic replay 全过，但 Global p90 regret
  `1.26160× > 1.20×`，gray compiler frontier 只有单点且无 multi-budget switch。
  按冻结止损规则，IR-5 保持 VALIDATED-NO-GO，禁止继续旋转 final 或启动 IR-6。

## 12. 真实 Verifier IR correctness 路线关闭

IR-5 No-Go 后另立的 `feat/real-verifier-ir-integration-v1` 不继续性能调参，只修复并审计
PR-14 暴露的两个 correctness 缺口：

1. ResNet initial-CROWN 通过显式 external intermediate bounds 与 adaptive ReLU slope，
   lower max diff 从历史 `796.765` 降为 `3.09944e-6`，sign 从 3/9 恢复为 9/9；
2. activation-BaB 作为 provider-owned external exact operation 进入 Bound/Plan/Task/Schedule
   stack。历史 394/394 query 可生成五层 IR hash；当前 CPU 真实运行 377/377 dispatch 完成，
   observer on/off 均访问 380 domains 且 final lower 一致。

范围必须按三条口径分开：

- fused BoundFlow kernel replacement 仍为历史 `0/394`；
- typed external-call admission 为 `394/394`，但历史 v1 identity 有明确 limitation；
- 当前 adapter v2 exact execution 为 `377/377`，external αβ-CROWN 继续拥有算法和 termination。

全量回归 `452 passed, 37 skipped`，artifact fresh-process replay 通过。因本机 CUDA 不可用且
external lower-only 公平性能合同未建立，关闭等级为 correctness/integration
VALIDATED-REDUCED，ASPLOS system-performance 总判定仍为 NO。

## 13. Production Schedule IR + Memory P0 门禁

RVIR closure 后没有凭对象名称直接宣布 Schedule IR 已成为论文主线，而是对当前 production
控制面做了独立、可重放的 P0 audit：

1. residual-final-v3 的 8 个 workload/backend case 均由 Schedule IR 覆盖完整 10-op Bound
   graph，并显式拥有 budget check、arena allocate/free、batch loop 与 launch；
2. 这些 case 没有 `MaterializeAction`，且每个 template 只有一个 batch/storage candidate；
3. 64/512 MiB 下 PlanInstance hash 会变化，但 region/representation/backend/batch/storage/state
   决策均不变化；冻结 artifact 同样没有 multi-budget switch，双 workload Pareto 失败；
4. VNN-COMP ResNet 的 51 个 activation call 五层 IR hash 可逐条复算，但每条主图仍只是一个
   provider-owned `EXTERNAL_VERIFIER_CALL` 和一个 launch；
5. 当前没有 production OOM-rescue artifact。

因此 `feat/production-schedule-memory-v1` 不准入。下一唯一工程问题改为：能否把一个冻结真实
residual network 的 main compute lower 为非 opaque、multi-region native Bound IR，并先通过
external-semantics correctness。只有此后存在至少两个合法 storage/batch plan、预算触发真实
决策切换，且出现 baseline OOM rescue 或可重现 memory Pareto，才允许重开 Schedule-memory
性能路线。P0 artifact 位于
`artifacts/schedule-p0/production-schedule-memory-p0-20260804/`。

## 14. Native Real-Network IR v1 与下一门禁

NRIR-1 固定 VNN-COMP 2021 `resnet_2b.onnx`、prop0 VNNLIB、αβ-CROWN commit 和 6 组逐 ReLU
external preactivation bounds。新的 portable payload 对 identity/tensor tamper fail closed，并让
aggregate digest 进入 ReLU state version 和 Plan provenance。

执行结果：17 个 Primal ops lower 为 21 个 native Bound ops；PlanInstance 选择 21 个 singleton
reference regions；Task IR 与 Schedule IR 分别拥有 21 units/launches；Bound/Task external-call
count 均为 0。五层 hash fresh replay 一致，final lower 对 external oracle max diff
`7.152557373046875e-07`、sign 9/9。

这只证明真实主 backward 已进入编译器 IR。external intermediate bounds 仍由 αβ-CROWN 提供，
NRIR-1 冻结时 Plan 只有一个 dense storage/full batch、没有 materialization alternative，也没有
GPU/timing。storage-axis 后续已由 NRIR-2 完成；历史获准顺序修订为：

```text
NRIR-2 real-graph storage alternatives + runtime last-use (completed)
  -> fresh CUDA physical-memory/OOM protocol, if device is available
  -> representation semantic binding + real materialization
  -> sliced batch execution
  -> only then reconsider Schedule-memory/performance claim
```

artifact 位于 `artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`；实现与复现命令
见 `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`。

## 15. Native Real-Network Memory Plans v1 与下一门禁

NRIR-2 没有改变 NRIR-1 的 Bound semantics。它从原 dense storage baseline 派生：

1. retain-all：独占对齐 byte ranges，所有 value 保留到 final op；
2. lifetime-reuse：使用 verified exact last-use，只让 lifetime 不重叠的值复用 arena range，
   Task runtime 在消费完成后删除对应 tensor/operator reference。

固定真实 ResNet 上，二者共享 Bound hash `16e27f31...80fb` 和 PlanTemplate hash
`359ee68f...43f3`。高预算选择 retain-all（`1,860,912` bytes）；预算降至 `442,656` 选择
lifetime-reuse；再减 1 byte 以 `memory_budget_exceeded` 拒绝。低内存计划有 386 对合法 alias、
85 个 final-task 前释放。两计划 final lower/upper bitwise equal，对 external lower max diff
`7.152557373046875e-07`、sign 9/9。

该结果关闭 real-graph storage decision mechanism，但不关闭 performance：

- `442,656` 是 Plan/Schedule arena 与 runtime live-value ledger，不是 CUDA allocator counter；
- runtime release 会删除引用，但当前没有 `torch.cuda.max_memory_allocated/reserved` 或 OOM rescue；
- `0.001 ms` policy cost 只用于稳定排序，标注 `policy_cost_not_benchmarked`；
- Plan representation decision 仍未绑定 Bound rewrite/backend semantics，Schedule
  `MaterializeAction` 仍只记账；full-query batch 也尚未被 slice execution 消费。

下一动作优先尝试 fresh CUDA physical-memory protocol。只有实际 device measurement 同时通过
correctness、重复运行和 baseline OOM/Pareto 门禁，才可进入性能主张；若 CUDA 不可用，则转向
representation semantic binding bridge，不等待或伪造设备结果。artifact 位于
`artifacts/native-real-network-memory-plans/vnncomp21-resnet2b-prop0-cpu-v1/`。

## 16. Native CUDA Physical-Memory Protocol v1 与环境边界

NRIR-3 已在任何正式 CUDA 结果产生前冻结并实现双 storage 的设备测量协议：

- 5 个 repeat，每个 plan/repeat 独立 fresh process；偶数 retain→reuse、奇数反向；
- 每 worker 5 warmup、20 measured，计时只覆盖 prepared lower-only native CROWN execution；
- 同步采集 baseline/peak allocated 与 reserved delta，保留全部 latency samples；
- 模型、intermediate-bound、环境、worker PID、Bound/PlanTemplate、result hash 和 raw→summary
  派生关系全部 fail closed/replay；
- 只有 reuse median allocated delta 至少降低 20%，且 median latency 不超过 retain 1.20×，
  才允许 `performance_claimed=true`；reserved 只报告，无实际 OOM 不声明 rescue。

当前主机的 PyTorch CUDA build 为 13.2，但 driver/device 不可用。`probe` 已以 exit 2 生成
digest-protected `environment_unavailable` artifact，`generate` 在输出目录和 measured row 产生前
exit 2。因此本阶段只关闭 protocol implementation，不产生 performance No-Go/Go。全量回归
`484 passed, 37 skipped`，Mypy/Pylint/Black/diff check 均通过。

冻结顺序继续为：

```text
CUDA protocol implemented; device run pending external availability
  -> representation semantic binding + executable MaterializeAction (completed)
  -> sliced batch execution
  -> only then reconsider Schedule-memory/performance claim
```

下一分支不得只新增 representation metadata/hash；至少一个 Plan decision 必须驱动真实 Bound
rewrite/backend conversion，并在固定 ResNet 上以 dense reference/external oracle 双重校验。

## 17. Native Representation Semantic Binding v1 与下一门禁

NRIR-4 已让 source Plan representation decision 真实决定 execution Bound program，而不是只改变
candidate ID：dense policy 执行原 21-op graph；structured-affine policy 由 binder 生成另一份
49-op graph，其中 14 cast + 14 materialize 全部与 selected transition、source Schedule action、
Task 和 Launch 一一对应。execution graph 使用独立 Plan/Task/Schedule identity。

固定 ResNet fresh replay 中，高预算选择 dense/retain-all，`442,656` bytes 选择
structured-affine/lifetime-reuse，再减 1 byte fail closed。dense/structured lower 最大差
`9.5367431640625e-07`，二者均对 external oracle allclose、sign 9/9。artifact 位于
`artifacts/native-real-network-representation-binding/vnncomp21-resnet2b-prop0-cpu-v1/`。

这只关闭 C1/C2 的 representation semantic binding mechanism：当前 structured operator 仍包装
dense tensor，storage 仍按 dense-equivalent bytes 记账；policy 与 storage 绑定也不能归因为
structured compression。`performance_claimed=false`，禁止 memory/latency/CUDA/OOM/Pareto/speedup
表述。

当前唯一代码顺序修订为：

```text
NRIR-4 representation binding (completed)
  -> real-network sliced batch execution
  -> execute frozen NRIR-3 CUDA protocol when a device becomes available
  -> only with physical evidence reconsider Schedule-memory/performance claim
```

下一分支至少要让一个 domain/spec/sample batch decision 改变实际 Task/Schedule slice 和 query
accounting；仅新增 batch metadata、hash 或 synthetic loop 不算完成。
