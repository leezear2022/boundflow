# BoundFlow ASPLOS 执行备忘录 v1.0

> 生效日期：2026-07-12  
> 当前提交基线：`263ea81`（PR-10 complete, feature-gated）
> 唯一执行顺序：**Gate 0 → PR-10 → PR-11 → PR-12 → PR-13**。  
> 禁止同时启动 Planner、fused kernel 与 BaB runtime 三条主线。

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
kernel-level correctness/mechanism PASS；end-to-end CROWN、正式 Pareto、final held-out 与
compile amortization 仍 pending，PR-13 继续阻塞。

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
