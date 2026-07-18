# BoundFlow 当前状态：PR-13 Closure 之后

> 状态日期：2026-07-18
> 冻结基线：`57a854b` / annotated tag `pr13-validated-reduced`
> 当前研发分支：`feat/pr14-real-verification`
> 总判定：PR-10/11/12/13 已关闭；ASPLOS 执行为 **CONDITIONAL GO**，ASPLOS-ready 仍为 **NO**。

## 1. 当前真实阶段

BoundFlow 已经完成从边界表示到 query runtime prototype 的主干：

| 层次 | 状态 | 已验证边界 |
|---|---|---|
| Structured bound representation / materialization trace | validated foundation | dense/structured 数值与梯度对齐；barrier、bytes、reason、lifetime 可观测 |
| Method/autograd/memory-aware Planner | validated-reduced | 1,416 次执行、472 个聚合 pattern、final held-out 23/23 feasible |
| Fused/multi-backend CROWN execution | validated-reduced | eager/chunked/structured/TVM fused 多预算选择；收益只在部分 regime |
| Query runtime | validated-reduced | `BoundQuery`、state validity、dynamic batching、same-solver adapter、reduced GPU E2E |
| 真实 complete verifier integration | PR-14A validated-partial | 已有官方 MLP/CNN 与 VNN-COMP ResNet-2B 共 540 个真实调用；尚无 external fixed replay |
| ASPLOS 最终系统主张 | 未冻结 | C3 必须根据真实 workload 证据决定保留、降级或改写 |

历史 `main@263ea81` 只到 PR-10 closure，不能再作为项目当前状态入口。跨会话恢复必须同时检查
research branch、annotated tag 与 closure 文档，不能只看 `main`。

## 2. 已经成立的证据

### C1：Structured Bound Representation

- ReLU 后主 coefficient 可以保持结构化 operator；
- materialization barrier 有稳定 trace schema；
- dense/operator/planned 路径在相同浮点语义下有 reference comparison；
- structured 不是统一默认策略：plain CROWN 的部分显存收益伴随明显 latency 代价，α/αβ
  structured 还会增加 autograd peak 并出现 OOM。

### C2：Query- and Memory-Aware Multi-Backend Planner

- PR-11 已完成静态 topology/liveness feature、global placement、bounded retry 和真实 OOM fallback；
- PR-12 已建立 eager、chunked、structured、TVM fused 候选及 compile-aware、多预算选择；
- final held-out 中可行机会 72/72 选到可行 backend，feasible p90 regret 为 1.000×；
- fused 的稳定价值主要是减少中间物化/peak memory，而不是普遍降低 latency。

### C3：Query Runtime Prototype

- PR-13A 已有 state-versioned `BoundQuery`、compatibility key、split lineage、fixed replay；
- PR-13B 已有 dynamic batching、deadline/budget、OOM bisection、顺序恢复和可观测 counters；
- PR-13C 已把 adapter 接回同一 host solver，只替换 bound-call execution；
- PR-13D reduced GPU 中，fixed/E2E 相对逐节点为 96.52×/9.93×，但 hard E2E 相对公平
  batched original 仅 0.980×。

因此 96×/9.93× 必须归因于物理 batching，不能描述成 runtime abstraction 的独立加速。

## 3. 当前真正缺口

1. **真实 fixed replay**：把 initial plain-CROWN 的 external tensor payload 冻结为可重放工件；
   当前只有 identity/profile，没有 parent lineage 与 payload。
2. **Backend phase 闭环**：真实 coverage 已证明 initial phase 143/146 eligible，但
   activation-BaB 为 0/394；PR-14B 只能窄化到前者，禁止为后者新增 kernel。
3. **公平端到端对照**：同 solver、property、branch/split、seed、timeout 下，比较 original
   batched executor 与 BoundFlow，而不是逐节点 baseline。
4. **C3 定位决策**：只有当 query-aware scheduling/cache/multi-backend 相对 batched original
   有可归因收益时，C3 才保留为核心贡献；否则降级为支撑 C1/C2 的执行基础设施。

## 4. 下一阶段唯一主线

下一阶段为 `PR-14: Verification-Aware Execution on Real Verification Workloads`，详细执行门禁见
`gemini_doc/pr14_execution_plan.md`。

PR-14 不重新实现 query recorder、BaB 算法、branch heuristic 或 split strategy。它复用 PR-13
contract，先量化真实 query coverage，再接 executor/replay，最后才做 full verification evaluation。

明确禁止：

- 回到 `bench/pr10b2-real-bab-fixed-domain-replay`；
- 继续无证据地调孤立 TIR/kernel；
- 新建 persistent GPU BaB queue；
- 把 reduced chain-CNN 结果写成 VNN-COMP/non-toy 结论；
- 把逐节点 speedup 当成相对成熟 batched verifier 的 headline。

## 5. 权威阅读顺序

1. 本文；
2. `gemini_doc/pr14_execution_plan.md`；
3. `gemini_doc/pr13_closure_audit_2026_07_14.md`；
4. `gemini_doc/pr13_execution_status.md`；
5. `gemini_doc/asplos_claims_map.md`；
6. `gemini_doc/asplos_execution_memo_v1_0.md`。
