# BoundFlow 当前状态：PR-13 Closure 之后

> 状态日期：2026-08-04
> 当前 integration base：`d21bdee`（NRIR-2 merge）；PR-13 历史基线：`57a854b` / tag `pr13-validated-reduced`
> 当前研发分支：`feat/native-real-network-cuda-memory-protocol-v1`
> 总判定：IR-5 final **VALIDATED-NO-GO**；PR-14B 同为 No-Go、PR-14C/IR-6 不启动；
> ASPLOS-ready 为 **NO**。
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
> 该结果仅为 calibration，不撤销 No-Go；其后 residual-CNN v3 final 已完成并失败。
> IR-5E 已冻结 CUDA-only chain-CNN calibration→residual-CNN final v2 协议；正式
> v2 首次执行因 fixed-single 重新采样的 input 与 batch 第一 query 不同而
> PROTOCOL-INVALID；未生成 manifest，`7401/7402` 已退役。不得将此写成系统性能结果。
> IR-5G 已用 exact batched-input slice 修复方法学，并冻结 v3 `7501/7502`；
> backend/budget/shape/阈值均未按 v2 timing 调整，随后只运行一次。
> IR-5H v3 final 已完整生成并 replay：correctness 全过，但 Global p90 `1.26160×`
> 超过 `1.20×`，gray 无 compiler Pareto，且无多预算切换。IR-5 最终
> VALIDATED-NO-GO；停止当前 ASPLOS system-performance 路线，IR-6 不启动。
> 2026-08-03 RVIR 后续：真实 verifier correctness/integration 已 CPU
> VALIDATED-REDUCED；这不撤销 IR-5/ASPLOS performance No-Go。
> 2026-08-04 P0 后续：production Schedule-memory 准入审计为 `NO_GO`。Residual reduced
> 路径有完整 arena/launch ownership，但没有 materialize、storage 选择或预算决策切换；真实
> ResNet 的 51 个 activation call 仍各自是一个 external opaque launch。下一分支改为
> `feat/native-real-network-bound-ir-v1`。
> 2026-08-04 NRIR-1 后续：固定 ResNet2B initial-CROWN 已生成 21-op native Bound graph、
> 21 Tasks 与 21 launches，Bound/Task external-call count 为 0；五层 hash fresh replay 一致，
> lower max diff `7.15256e-7`、sign 9/9。该结果只关闭 CPU correctness/compiler ownership；
> external intermediate bounds、单 storage/batch、0 materialization 与无性能 claim 的边界保留。
> 2026-08-04 NRIR-2 后续：同一真实 ResNet Bound IR/PlanTemplate 已加入 retain-all 与
> lifetime-reuse 两个 storage plan。1,860,912/442,656 bytes 预算阈值会切换 PlanInstance 与
> Schedule arena；低内存路径在 Task 边界提前释放 85 个 runtime values，并有 386 对合法
> physical aliases。两计划 bitwise 相同、external max diff `7.15256e-7`、sign 9/9。该结果只
> 关闭 CPU storage-plan correctness/ownership；不是 CUDA allocator peak、OOM rescue 或性能证据。
> 2026-08-04 NRIR-3 后续：fresh-process CUDA protocol 已冻结并实现，包含 5 repeats ×
> 5 warmup × 20 measured、allocator allocated/reserved delta、交替进程顺序、prepared lower-only
> timing、20% memory 与 1.20× latency 门禁及 raw semantic replay。本机
> `cuda_available=false`，所以只生成 `environment_unavailable` probe artifact；正式 benchmark
> 在创建输出目录前 exit 2，`performance_claimed=false`。下一步转 representation semantic bridge。

## 1. 当前真实阶段

BoundFlow 已经完成从边界表示到 query runtime prototype 的主干：

| 层次 | 状态 | 已验证边界 |
|---|---|---|
| Structured Bound IR | IR-1 reference + IR-4 backend closure validated-reduced | typed schema/lowering/verifier、dense/structured rewrite/interpreter、PyTorch/TVM typed execution |
| Plan/Task/Schedule IR | IR-2/3 reference + IR-4 runtime closure validated-reduced | typed builder/selector/task lowering/schedule verifier/per-task semantics/query/state/backend artifacts |
| Fused/multi-backend CROWN execution | validated-reduced | eager/chunked/structured/TVM fused 多预算选择；收益只在部分 regime |
| Query runtime | validated-reduced | `BoundQuery`、state validity、dynamic batching、same-solver adapter、reduced GPU E2E |
| 真实 complete verifier integration | RVIR CPU correctness/integration validated-reduced | ResNet external-semantics max diff 3.10e-6、sign 9/9；typed external-call admission 394/394；真实在线 dispatch 377/377 |
| Production Schedule + Memory P0 | NO-GO | residual 8/8 完整 arena ownership，但 0 materialize、单 storage、0 budget decision switch；真实 ResNet 51/51 为单 external launch |
| Native real-network IR NRIR-1 | correctness/compiler ownership validated-reduced | ResNet2B 17 Primal ops → 21 native Bound/Task regions/launches；五层 hash 绑定 external-bound payload；max diff 7.15e-7、sign 9/9；仍无 memory choice/GPU/performance |
| Native real-network memory NRIR-2 | storage-plan correctness/ownership validated-reduced | 同一 real graph/template 的 retain-all 1,860,912 B 与 lifetime-reuse 442,656 B；预算决策切换、386 alias pairs、85 early releases、双计划 bitwise equal；无 CUDA allocator/performance claim |
| Native CUDA memory protocol NRIR-3 | protocol implemented / environment unavailable | fresh worker、5×5×20、allocator/timing/identity/replay 门禁已实现；本机 0 CUDA device，只保留 fail-closed probe，不产生 performance claim |
| ASPLOS 最终系统主张 | IR-5 final VALIDATED-NO-GO | IR-1—4 narrow closure 保留；Global p90/Pareto 失败，当前 system-performance 路线已关闭 |

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

## 4. IR-first 路线执行结果与关闭状态

PR-14 implementation 已停止。原定 `docs/asplos-c1-c2-story-freeze` 被代码级复审否定后，
历史工程主线切换到 `feat/compiler-ir-stack-v1`，并已按 Bound IR → Plan IR →
Task/Schedule IR → runtime/backend → adaptive evaluation 完整执行。契约见
`gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。仍不得用 PR-14C
E2E 绕过 bound-equivalence gate。

截至 2026-08-03，Bound IR、Plan IR、Task/Schedule IR 的 synchronous reference closure
与 IR-4 backend/runtime validated-reduced closure 均已完成；IR-5 adaptive PlanInstance
也已执行到 fresh residual final，并以 VALIDATED-NO-GO 关闭。不得回滚重复实现
IR-1/2/3/4，也不得继续旋转 IR-5 final。IR-2
raw historical artifact 缺失边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`；IR-3 closure 证据见
`gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`。

IR-4D 已证明可验证 plain-CROWN 请求能够通过 typed query 入口完成
PlanInstance→TaskIR→ScheduleIR→backend，并已实现 exact-version dense state
load/store/task skip。PR-13 α/β 请求仍因 PR-14 whole-query mismatch 在 compiler 入口显式
No-Go。IR-4E 随后把 `plain_crown_typed_ir` 请求接入 PR-13 DynamicBatchManager，并把旧
`SameSolverQueryRuntime` 设为默认拒绝、仅 historical replay 显式 opt-in。IR-4 现以
validated-reduced 关闭；其后 IR-5 已完成并失败。不得把 IR-4 closure 写成 α/β external
integration 或 ASPLOS 性能结论。

IR-5A 已让 cold/repeated/warm-cache 与 per-query memory/deadline 进入 PlanInstance
identity、provenance 和 runtime cache namespace。同一 template 可合法切换不同 plan。
IR-5B/C2 随后完成四策略 evaluator 与 fresh CUDA typed MLP artifact：Global 在 8/8
contexts 可行，p50/p90 Oracle regret 为 1.000×/1.00766×，高内存选择 dense、冻结低内存
选择 TVM fused。IR-5C3 随后冻结 MLP calibration→chain-CNN held-out，并加入 fixed-single、
ordinary typed batching 与 legacy fair batched-original。全部 correctness/feasibility gate
通过，但 batched-original 约 0.506–0.508 ms/query，Global 约 34.449–35.678 ms/query，
p50/p90 regret 68.065×/70.263×；64/512 MiB 都选择 chunked，无 memory Pareto。

profile 曾将主要问题定位到 query hot path 重复 Plan/Bound/Task validate、stable hash、
canonical JSON 与 dispatch-key 构造；IR-5D 已完成该补救。随后 fresh residual final 仍以
Global p90 `1.26160×` 和 gray Pareto 缺失失败。ASPLOS-ready 判定为 NO，IR-6 不启动，
IR-5 内部不存在仍被证据允许的后续旋转；独立 NRIR 路线按第 8 节推进。

IR-5D remediation 现已实现：prepared Bound/Task program 冻结静态参数与 identity，
Plan cache 复用预计算 dispatch key，production trace 不在 timed path 生成中间 tensor
SHA；同时新增 from-forward-trace legacy baseline，使双方都只计 CROWN backward。
在旧 gray/color CNN 上的 20-sample CUDA calibration 中，最快 typed/legacy median 比值为
`0.880×`/`0.896×`。这些 workload 已被消费，故只能证明优化方向，不能升级 claim。
该 calibration 当时只用于决定是否值得运行 final；其后 residual-CNN v3 final 已完成并失败。

IR-5E 完成了 protocol freeze：新 workload 含真实 residual fanout/`add_backward`，
baseline 固定为 from-forward-trace，并显式输出 p90≤1.20、双 workload latency-memory
Pareto 与 multi-budget switch 字段。v2 因输入身份协议错误失效，`7401/7402` 已退役。

实际首次运行发现同 seed、不同 batch shape 的 `torch.randn` 不保证前缀一致，导致
fixed-single 与 batched-first 输入不同。v2 在 semantic gate fail closed，未形成 summary/
manifest，未进入正式性能判定。当时唯一允许的处置是修复显式 input slicing、升级
protocol 并旋转 fresh identities；该处置已由 v3 完成，IR-6 始终未启动。

v3 runner 先对 fixed-single 与 batched query zero 做 `torch.equal`，再检查 final bounds；
split 记录 exact-clone contract。`7501/7502` 已按预注册协议运行一次并永久冻结。

v3 正式 artifact 已执行并绑定 `971a317`。Global 8/8 feasible，p50 regret
`1.00385×`，但 p90 `1.26160×`；失败来自 color warm-cache context 选择 TVM
（0.53146 ms/query）而 dense 为 0.42577 ms/query。color 有 latency-memory tradeoff，
gray 的 TVM 同时更快更省内存，只有单点 frontier，故双 workload Pareto 门禁失败。
IR-5/IR-6 路线按预注册止损规则关闭。

IR-5 当时冻结的最优先候选是与 `7501/7502` 独立的真实 Verifier IR correctness；该候选现
已由 RVIR 路线执行并按第 6 节关闭。其完成只解除 correctness/integration blocker，不授权
重新提出性能 claim。

明确禁止：

- 回到 `bench/pr10b2-real-bab-fixed-domain-replay`；
- 继续无证据地调孤立 TIR/kernel；
- 新建 persistent GPU BaB queue；
- 把 reduced chain-CNN 结果写成 VNN-COMP/non-toy 结论；
- 把逐节点 speedup 当成相对成熟 batched verifier 的 headline。

## 5. 权威阅读顺序

1. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_PLAN_2026_08_04.md`；
2. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_PLAN_2026_08_03.md`；
3. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`；
4. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`；
5. `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`；
6. `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`；
7. `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`；
8. `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`；
9. 本文（含 PR-13/14 历史状态与第 6—11 节当前修订）；
10. `gemini_doc/asplos_claims_map.md`；
11. `gemini_doc/asplos_execution_memo_v1_0.md`。

## 6. RVIR 关闭后的当前边界

PR-14B 的 `796.765` 与 `0/394` 仍是当时 local whole-query/fused replacement 路径的正确历史
结论；它们已被新的 correctness 路线分解，而不是被删除：

- external intermediate bounds + adaptive slope 的 ResNet initial-CROWN 已通过，max diff
  `3.09944e-6`、sign 9/9；
- fused replacement coverage 仍是 `0/394`；
- provider-owned typed external-call admission 是 `394/394`；
- adapter v2 当前 CPU exact-call execution 是 `377/377`，observer on/off 的 status、380
  domains 与 final lower 一致。

历史 394 行仍缺 split tensor values、requested polarity 与 parent lineage，artifact 已逐行标注；
当前 377 行补齐 lower-only 与 347 parent links。v2 artifact 进一步冻结这 377 条在线 query 与
377 条 typed execution record 原文；fresh replay 会逐条复核 query/record 顺序、parent
precedes child、完成状态和五层 IR hash，不再只信任生成端摘要。全量回归为
`452 passed, 37 skipped`（RVIR closure 基线）；在线 raw replay v2 合并前的最新回归为
`460 passed, 37 skipped`。
当前没有被证据授权的 CUDA/performance claim，下一性能研究必须另立公平 lower-only 合同与
fresh GPU protocol，不能直接复用本 correctness artifact。

## 7. Production Schedule IR + Memory P0 判定

`artifacts/schedule-p0/production-schedule-memory-p0-20260804/` 对 IR-5 residual-final-v3
和 RVIR v2 做了 digest-first、semantic-replay 审计：

- 2 workload × 4 backend 的 8 个 residual structural case 均由 Schedule IR 覆盖 10/10
  Bound ops，并显式执行 check-budget、arena allocate/free、batch loop 与 9/10 个 region launch；
- 但 8 个 template 均只有一个 batch 和一个 storage candidate，且没有任何
  `MaterializeAction`；64/512 MiB 虽生成不同 PlanInstance hash，实际 decision signature
  8/8 完全相同；
- 冻结 residual-final-v3 原有结论仍是 no multi-budget switch、双 workload Pareto 失败；
- VNN-COMP ResNet 51/51 activation call 的五层 IR hash 全部可重编译，但每条 Bound graph
  只有一个 `EXTERNAL_VERIFIER_CALL`，Schedule 也只有一个 external launch，主计算与数值
  语义仍由 αβ-CROWN provider 拥有；
- baseline OOM rescue 没有冻结证据，只能记为 not demonstrated。

因此不能直接启动 `feat/production-schedule-memory-v1`。当时批准的下一代码路线是
`feat/native-real-network-bound-ir-v1`：先把一个冻结真实 residual network 的主计算 lower
为 native multi-region Bound IR，并通过 external-semantics correctness oracle；之后才允许增加
多个 storage/batch 候选、重开 memory feasibility 与 GPU 性能门禁。

## 8. Native Real-Network IR v1 判定

NRIR-1 已在固定 VNN-COMP 2021 ResNet2B prop0 上完成 P0 要求的第一步：

- model/VNNLIB/αβ-CROWN commit 与 6 组 external intermediate bounds 均有 digest；portable
  payload 可由 `torch.load(weights_only=True)` 加载，ordinal/name/shape/dtype/tensor/aggregate
  identity 任一变化均拒绝；
- ONNX/Primal topology 为 17 ops（Conv 6、ReLU 6、Add 2、Flatten 1、Linear 2）；native
  plain-CROWN lowering 生成 21 个 Bound ops、21 个 Task units 与 21 次 Schedule launch；
- Bound IR 与 Task IR 的 `EXTERNAL_VERIFIER_CALL` 均为 0。external-bound aggregate hash 进入
  每个 ReLU relaxation state version，并继续进入 Plan provenance，所以五层 hash 对 oracle
  payload 内容敏感；
- fresh replay 的 native lower 对 αβ-CROWN final lower max diff
  `7.152557373046875e-07`，allclose 门限 `2e-4/2e-4`，sign 9/9；
- artifact 显式 `performance_claimed=false`，当前只有一个 dense storage、一个 full-query batch、
  0 materialization candidate，external verifier 仍负责 forward intermediate bounds。

结论为 CPU correctness/compiler ownership `VALIDATED-REDUCED`，不是完整 native αβ-CROWN 或
性能关闭。其 storage-axis 下一门禁已由 NRIR-2 按第 9 节完成；representation/materialization
与 sliced batch execution 仍未完成，不能因 storage switch 自动升级。

## 9. Native Real-Network Memory Plans v1 判定

NRIR-2 保持 NRIR-1 的 Bound graph、external semantic payload 与 reference backend 不变，只在同一
PlanTemplate 中加入两个可验证 storage plan：

- `native-retain-all-v1` 使用不相交对齐区间，并把所有 value lifetime 延长到 final op，
  Schedule arena 和 runtime observed residency 均为 `1,860,912` bytes；
- `native-lifetime-reuse-v1` 使用 compiler-derived exact last-use，确定性复用不重叠 lifetime
  的 byte ranges，Schedule arena 和 observed residency 均为 `442,656` bytes；
- 高预算选择 retain-all；预算为 `442,656` 时选择 lifetime-reuse；再减 1 byte 时 selector 以
  `memory_budget_exceeded` 拒绝；两者共享 Bound hash `16e27f31...80fb` 与 PlanTemplate hash
  `359ee68f...43f3`，但 PlanInstance/Task/Schedule identity 均不同；
- runtime 在 Task 前检查输入 resident，Task 后按 selected `live_to_op_id` 释放引用。真实图
  lifetime-reuse 有 386 对合法 physical aliases、85 个 final-task 前释放；
- 两计划 lower/upper bitwise 相同，对 external lower max diff
  `7.152557373046875e-07`、sign 9/9。parent NRIR-1 artifact 原五层 hash replay 不变。

结论为 storage-plan correctness/runtime ownership `VALIDATED-REDUCED`。`performance_claimed=false`
必须保留：当前 byte ledger 是 Plan/Schedule logical arena 与 runtime residency contract，不是
`torch.cuda.max_memory_allocated`、真实 allocator reuse、latency、OOM rescue 或 speedup。

representation 审计同时发现：当前 Plan 的 representation decision 不能自动改写 Bound IR；
structured 执行依赖另一份 rewritten module，而 Schedule reference executor 只记录
`MaterializeAction`。因此本轮没有加入假的 structured candidate。下一步应先尝试 fresh CUDA
physical-memory protocol；若 GPU 不可用，则冻结 runner/protocol 并推进 representation semantic
binding bridge，不得用 metadata/hash 代替执行证据。

## 10. Native CUDA Physical-Memory Protocol v1 判定

NRIR-3 已把 NRIR-2 双 storage 的设备测量方法冻结成可运行实现：每个 plan/repeat 使用独立
worker process，5 个 repeats 中交替启动 retain/reuse，每 worker 5 warmup、20 measured；计时
只覆盖 prepared native CROWN backward，并以同步后的 `max_memory_allocated/reserved` baseline
delta、result hash、Bound/PlanTemplate identity 与原始 latency samples 形成 replay-grade artifact。

本机 PyTorch 为 `2.12.1+cu132`，但 `torch.cuda.is_available=false`、device count 0，
`nvidia-smi` 无法连接 driver。因此：

- 冻结 probe artifact 为 `environment_unavailable` 且 `performance_claimed=false`；
- 正式 `generate` 在创建输出目录或 measured row 前 exit 2；
- 没有 CUDA allocator reduction、latency Pareto、OOM rescue 或 speedup 结论；
- 协议测试/全量回归为 `17 passed` / `484 passed, 37 skipped`，静态门禁通过。

协议实现已完成，设备实验待可用 CUDA 主机按原参数执行。当前无需停等硬件；下一代码路线为
representation semantic binding bridge，使 Plan representation 与 `MaterializeAction` 真正改变
Bound/backend execution，并先通过真实 ResNet 双路径语义一致性。

## 11. Native Representation Semantic Binding v1 判定

NRIR-4 已关闭 NRIR-2/3 明确指出的“表示选择只停留在 metadata”缺口：

- source PlanTemplate 对固定 21-op ResNet Bound graph 提供两个全局一致 policy；高预算选择
  `native-dense-v1` + retain-all，`442,656` bytes 选择
  `native-structured-affine-v1` + lifetime-reuse，`442,655` bytes fail closed；
- structured policy 的每个 selected transition 与 source Schedule `MaterializeAction`、rewritten
  execution Bound op 一一绑定。真实图插入 14 个 `REPRESENTATION_CAST` 与 14 个
  `MATERIALIZE`，execution graph 从 21 ops 变为 49 ops；49 个 op 均各自进入 Task 与 Launch；
- rewritten Bound graph 使用独立 execution PlanTemplate/PlanInstance/Task/Schedule hash；没有把
  source PlanTemplate 冒充成对另一 Bound hash 仍有效；
- dense/structured lower 最大差 `9.5367431640625e-07`；二者对 external lower 均 allclose，
  sign 9/9；artifact digest 与 fresh semantic replay 通过；
- selector 新增 storage-compatible prefix pruning，在不改变可行解集合的前提下避免真实图
  dense/structured 全排列的指数枚举。

结论为 representation binding/compiler ownership `VALIDATED-REDUCED`。当前 structured value
由 `DenseLinearOperator` 包装 dense tensor，execution storage 对每个 structured binding 仍保留
至少 dense logical bytes。因此不得声明 compression、memory reduction、latency、CUDA、OOM、
Pareto 或 speedup；source policy 与 NRIR-2 storage 的耦合仅用于确定性预算选择，物理内存收益仍
没有被 NRIR-4 证明。

下一代码门禁是 real-network sliced batch execution：Plan 的 domain/spec/sample batch decision
必须改变实际 Task/Schedule slicing 与 query accounting，并保持 dense/structured、single/batched
语义一致。CUDA NRIR-3 设备实验作为环境可用时的独立待办，不阻塞该代码路线。
