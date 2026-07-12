# BoundFlow ASPLOS 执行备忘录 v1.0

> 生效日期：2026-07-12  
> 当前提交基线：`ce36a51`（Phase 7A PR-9）  
> 唯一执行顺序：**Gate 0 → PR-10 → PR-11 → PR-12 → PR-13**。  
> 禁止同时启动 Planner、fused kernel 与 BaB runtime 三条主线。

## 1. 锁定的论文命题

BoundFlow 是面向神经网络验证中相关边界查询的 query- and memory-aware compiler/runtime。
它不重新发明 CROWN/αβ-CROWN/BaB，而是暴露 eager tensor execution 隐藏的结构、物化、
显存和跨查询复用决策。

三项正式贡献为：

1. **Structured Bound-Operator IR with Explicit Materialization Semantics**：保留可组合结构，
   显式表示 barrier、reason、bytes 与 lifetime；dense path 是参考语义，不承诺永不物化。
2. **Query- and Memory-Aware Materialization Planner**：在 query workload、硬件 profile 和
   显存预算下选择物化、partition、fusion、batch、cache、recompute 与 storage/schedule。
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

## 7. 投稿门禁

- **7 月 26 日**：PR-10 与真实 materialization profile；
- **8 月 5 日**：第一次硬 Go/No-Go；必须已有非平凡 Planner、非 toy workload、首个
  latency–memory Pareto、不同预算下不同计划；
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
