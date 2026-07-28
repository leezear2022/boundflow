# BoundFlow 当前状态：PR-13 Closure 之后

> 状态日期：2026-07-19
> 冻结基线：`57a854b` / annotated tag `pr13-validated-reduced`
> 当前研发分支：`feat/compiler-ir-stack-v1`
> 总判定：PR-14B 为 **VALIDATED-NO-GO**，PR-14C 不启动；ASPLOS-ready 仍为 **NO**。
> 2026-07-20 修订：本文保留 PR-13/14 历史证据，但第 4 节下一路线已由 IR-first 复审取代。
> 2026-07-28 进度：IR-1 Bound IR、IR-2 Plan IR、IR-3 Task/Schedule IR 的最小
> synchronous reference contract 已分别关闭；IR-4 production backend/runtime migration
> 已以 validated-reduced 关闭。IR-4A typed dispatch key + PyTorch reference
> adapter 已完成 foundation；IR-4B dense/structured/chunked typed registry 已通过，
> IR-4C TVM fused/unfused、dispatch-namespaced cache 与 semantic fallback 已通过；
> IR-4D typed plain-CROWN query→Plan/Task/Schedule、精确 state payload 与计算跳过已通过；
> IR-4E 已把 PR-13 manager 接入 typed compiler，并把 legacy α/β 改为默认关闭的
> historical opt-in。IR-5C3 independent workload-family + fair batching 已完成并给出
> VALIDATED-NO-GO；如继续，唯一补救是 IR-5D prepared execution capsule。
> IR-5A 已完成 query-time memory/deadline/cache/distribution context 与 amortized selector；
> 这只是 mechanism。
> IR-5B 已完成统一 observation 上的 fixed/local/global/oracle evaluator 与 synthetic
> contract artifact。IR-5C2 已产出 fresh CUDA typed MLP measured artifact：Global 8/8
> feasible，p50/p90 regret 1.000×/1.00766×，但同-family split、fair batching baseline
> 与 non-toy workload 仍缺。IR-5C3 随后用 MLP→CNN architecture-held-out 和 fair
> batched-original 补齐口径，Global p50/p90 regret 恶化为 68.065×/70.263×，且无多预算
> 切换/Pareto，因此当前 IR-5 v1 为 VALIDATED-NO-GO。
> IR-5D 已把静态 validate/hash/dispatch 移入 prepared execution capsule，并在已消费
> CNN 上以 from-forward-trace 公平边界得到 `0.880×`/`0.896×` 最快 median 诊断；
> 该结果仅为 calibration，不撤销 No-Go。新的 frozen residual-CNN final 尚未执行。
> IR-5E 已冻结 CUDA-only chain-CNN calibration→residual-CNN final v2 协议；正式
> v2 首次执行因 fixed-single 重新采样的 input 与 batch 第一 query 不同而
> PROTOCOL-INVALID；未生成 manifest，`7401/7402` 已退役。不得将此写成系统性能结果。

## 1. 当前真实阶段

BoundFlow 已经完成从边界表示到 query runtime prototype 的主干：

| 层次 | 状态 | 已验证边界 |
|---|---|---|
| Structured Bound IR | IR-1 reference + IR-4 backend closure validated-reduced | typed schema/lowering/verifier、dense/structured rewrite/interpreter、PyTorch/TVM typed execution |
| Plan/Task/Schedule IR | IR-2/3 reference + IR-4 runtime closure validated-reduced | typed builder/selector/task lowering/schedule verifier/per-task semantics/query/state/backend artifacts |
| Fused/multi-backend CROWN execution | validated-reduced | eager/chunked/structured/TVM fused 多预算选择；收益只在部分 regime |
| Query runtime | validated-reduced | `BoundQuery`、state validity、dynamic batching、same-solver adapter、reduced GPU E2E |
| 真实 complete verifier integration | PR-14B validated-no-go | 540-call coverage + MLP/ResNet fixed replay；activation 0/394，ResNet bound-equivalence fail |
| ASPLOS 最终系统主张 | C1/C2/C3 均不足 | C3 已降级；IR-1—4 narrow closure 已有，下一步补 IR-5 自适应与公平 held-out 证据 |

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

截至 2026-07-28，Bound IR、Plan IR、Task/Schedule IR 的 synchronous reference closure
与 IR-4 backend/runtime validated-reduced closure 均已完成；当前进入
**IR-5 adaptive PlanInstance**，不回滚重复实现 IR-1/2/3/4。IR-2
raw historical artifact 缺失边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`；IR-3 closure 证据见
`gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`。

IR-4D 已证明可验证 plain-CROWN 请求能够通过 typed query 入口完成
PlanInstance→TaskIR→ScheduleIR→backend，并已实现 exact-version dense state
load/store/task skip。PR-13 α/β 请求仍因 PR-14 whole-query mismatch 在 compiler 入口显式
No-Go。IR-4E 随后把 `plain_crown_typed_ir` 请求接入 PR-13 DynamicBatchManager，并把旧
`SameSolverQueryRuntime` 设为默认拒绝、仅 historical replay 显式 opt-in。IR-4 现以
validated-reduced 关闭；下一工程阶段为 IR-5，不得把此 closure 写成 α/β external
integration 或 ASPLOS 性能结论。

IR-5A 已让 cold/repeated/warm-cache 与 per-query memory/deadline 进入 PlanInstance
identity、provenance 和 runtime cache namespace。同一 template 可合法切换不同 plan。
IR-5B/C2 随后完成四策略 evaluator 与 fresh CUDA typed MLP artifact：Global 在 8/8
contexts 可行，p50/p90 Oracle regret 为 1.000×/1.00766×，高内存选择 dense、冻结低内存
选择 TVM fused。IR-5C3 随后冻结 MLP calibration→chain-CNN held-out，并加入 fixed-single、
ordinary typed batching 与 legacy fair batched-original。全部 correctness/feasibility gate
通过，但 batched-original 约 0.506–0.508 ms/query，Global 约 34.449–35.678 ms/query，
p50/p90 regret 68.065×/70.263×；64/512 MiB 都选择 chunked，无 memory Pareto。

profile 将主要问题定位到 query hot path 重复 Plan/Bound/Task validate、stable hash、
canonical JSON 与 dispatch-key 构造。当前 IR-5 v1 以 VALIDATED-NO-GO 关闭；
ASPLOS-ready 判定仍为 NO，IR-6 明确 blocked。如继续，唯一允许的补救是 IR-5D prepared
execution capsule，并必须在新 frozen CNN/residual split 上重新过 fair p90≤1.20× 门禁。

IR-5D remediation 现已实现：prepared Bound/Task program 冻结静态参数与 identity，
Plan cache 复用预计算 dispatch key，production trace 不在 timed path 生成中间 tensor
SHA；同时新增 from-forward-trace legacy baseline，使双方都只计 CROWN backward。
在旧 gray/color CNN 上的 20-sample CUDA calibration 中，最快 typed/legacy median 比值为
`0.880×`/`0.896×`。这些 workload 已被消费，故只能证明优化方向，不能升级 claim。
下一步是先冻结新的 residual-CNN final split，再一次性运行完整 fair evaluator 和 replay。

IR-5E 现已完成该 protocol freeze：新 workload 含真实 residual fanout/`add_backward`，
baseline 固定为 from-forward-trace，并显式输出 p90≤1.20、双 workload latency-memory
Pareto 与 multi-budget switch 字段。此时仍没有 final 数字；`7401/7402` 不得在 protocol
commit 前运行。

实际首次运行发现同 seed、不同 batch shape 的 `torch.randn` 不保证前缀一致，导致
fixed-single 与 batched-first 输入不同。v2 在 semantic gate fail closed，未形成 summary/
manifest，未进入正式性能判定。下一步只允许修复显式 input slicing、升级 protocol 并旋转
fresh identities；IR-6 继续 blocked。

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
