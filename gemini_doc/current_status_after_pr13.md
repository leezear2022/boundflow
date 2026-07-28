# BoundFlow 当前状态：PR-13 Closure 之后

> 状态日期：2026-07-19
> 冻结基线：`57a854b` / annotated tag `pr13-validated-reduced`
> 当前研发分支：`feat/pr14-real-verification`
> 总判定：PR-14B 为 **VALIDATED-NO-GO**，PR-14C 不启动；ASPLOS-ready 仍为 **NO**。
> 2026-07-20 修订：本文保留 PR-13/14 历史证据，但第 4 节下一路线已由 IR-first 复审取代。
> 2026-07-28 进度：IR-1 Bound IR 与 IR-2 Plan IR 的最小 reference contract 已分别关闭；
> IR-3B Schedule control/executor/trace foundation 已完成；当前唯一下一阶段为 IR-3C typed
> Task IR + per-task reference execution。IR-3C typed Task schema/lowering/linkage 已完成，
> 当前具体缺口收敛为 IR-3D per-task semantic executor + closure audit。

## 1. 当前真实阶段

BoundFlow 已经完成从边界表示到 query runtime prototype 的主干：

| 层次 | 状态 | 已验证边界 |
|---|---|---|
| Structured Bound IR | IR-1 reference closure validated | typed schema/lowering/verifier、dense/structured rewrite/interpreter；生产 backend/runtime 待迁移 |
| Plan IR / Planner | IR-2 reference closure validated-reduced | typed builder/selector/verifier/state-validity/replay；Schedule IR 尚未实现 |
| Fused/multi-backend CROWN execution | validated-reduced | eager/chunked/structured/TVM fused 多预算选择；收益只在部分 regime |
| Query runtime | validated-reduced | `BoundQuery`、state validity、dynamic batching、same-solver adapter、reduced GPU E2E |
| 真实 complete verifier integration | PR-14B validated-no-go | 540-call coverage + MLP/ResNet fixed replay；activation 0/394，ResNet bound-equivalence fail |
| ASPLOS 最终系统主张 | C1/C2/C3 均不足 | C3 已降级；Bound/Plan reference closure 已有，下一步建立 Schedule/runtime/backend 闭环 |

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

## 3. PR-14 已关闭的问题

1. **真实 coverage**：540 calls 中 initial 143/146 region-level eligible；activation-BaB 0/394；
2. **真实 fixed replay**：MLP lower 等价，但 requested outputs 不同，性能 N/A；
3. **non-toy bound equivalence**：ResNet nominal forward 正确，whole-query lower max diff
   `796.765`、符号 3/9，不能接入 same-solver；
4. **C3 定位**：无公平 batched-original 净收益证据，已降级为支撑 C1/C2 的基础设施。

## 4. 下一阶段唯一主线

PR-14 implementation 到此停止。原定 `docs/asplos-c1-c2-story-freeze` 已被代码级复审否定：
仅整理 story 不能弥补 Bound IR 占位、Plan IR 分散和 Schedule IR 缺失。下一工程分支改为
`feat/compiler-ir-stack-v1`，按 Bound IR → Plan IR → Task/Schedule IR → runtime/backend
迁移推进。完整门禁见
`gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。仍不得用 PR-14C
E2E 绕过 bound-equivalence gate。

截至 2026-07-28，Bound IR 与 Plan IR 的 reference closure 已完成；当前从上述顺序的
**Task/Schedule IR** 开始继续，不回滚重复实现 IR-1/2。IR-2 closure 的 raw historical
artifact 缺失边界见 `gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`。

明确禁止：

- 回到 `bench/pr10b2-real-bab-fixed-domain-replay`；
- 继续无证据地调孤立 TIR/kernel；
- 新建 persistent GPU BaB queue；
- 把 reduced chain-CNN 结果写成 VNN-COMP/non-toy 结论；
- 把逐节点 speedup 当成相对成熟 batched verifier 的 headline。

## 5. 权威阅读顺序

1. `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`；
2. 本文（PR-13/14 历史状态）；
3. `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`；
4. `gemini_doc/pr14a_real_query_coverage_2026_07_19.md`；
5. `gemini_doc/asplos_claims_map.md`；
6. `gemini_doc/asplos_execution_memo_v1_0.md`。
