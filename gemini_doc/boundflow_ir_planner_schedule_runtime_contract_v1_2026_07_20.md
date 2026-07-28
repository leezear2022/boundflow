# BoundFlow IR—Planner—Schedule—Runtime 架构契约 v1.0

> 日期：2026-07-20
> 状态：架构重置后的实施基线；不构成任何 ASPLOS claim 升级
> 适用范围：PR-14 `VALIDATED-NO-GO` 之后的编译器主线
> 代码基线：`feat/pr14-real-verification@b143717`

## 0. 执行结论

BoundFlow 下一阶段不能继续以“补 BaB runtime”或“接更多 verifier”为主线，也不能把现有
`runtime/linear_operator.py`、`PlanBundle`、`TaskGraph.topo_sort()` 分别等同于完整的 Bound IR、
Plan IR 和 Schedule IR。

新的唯一工程主轴是建立并验证以下一等 IR 链：

```text
Primal IR + Verification Spec
              |
              v
          Bound IR
              |
              v
           Plan IR
              |
              +----------+
              |          |
              v          v
          Task IR    Schedule IR
              |          |
              +----+-----+
                   v
         Backend IR / compiled kernels
                   |
                   v
        Verification-aware runtime
```

其中：

- **Bound IR** 定义验证计算的语义；
- **Plan IR** 定义编译器选了什么以及为什么合法；
- **Task IR** 定义可以交给后端编译/执行的粗粒度单元；
- **Schedule IR** 定义这些任务及 host/device 动作如何执行；
- **Runtime** 执行 Schedule IR、维护动态状态，并在门禁内请求重新规划；
- **TVM/TIR** 是后端，不定义 BoundFlow 的核心抽象；
- **Query IR** 是 workload/context 协议，正交地向 Planner 和 Runtime 提供动态输入。

PR-10—PR-14 的已有工作保留为机制、负结果和迁移资产，但不得用这些局部对象代替上述 IR。

## 1. 为什么必须重置架构判断

### 1.1 当前代码的真实状态

| 层次 | 当前实现 | 审计结论 |
|---|---|---|
| Primal IR | `ir/primal.py` 有 value/node/type、图校验和前端来源 | 可作为起点，但 attrs/params 仍有弱类型 |
| Bound IR | `ir/bound.py` 的 `DomainState` 为空；`Spec.objectives`、transformer attrs 为 `Any`/dict | 占位骨架，不是 C1 所需的一等 IR |
| Task IR | `ir/task.py` 有 Task/Buffer/StoragePlan/TaskGraph | 只正式表示 `INTERVAL_IBP`，CROWN 与 α/β/split 未进入 TaskKind |
| Plan IR | `planner/core.py::PlanBundle` 聚合 task/storage/cache/layout/lowering/meta | 容器骨架；多个 `Any`/dict，不能表达统一联合决策 |
| 局部计划 | `MaterializationPlan`、`MaterializationPlacementPlan`、`ExecutionCandidate` | 各自可审计，但彼此平行，未汇合为 Plan IR |
| Schedule IR | 无 BoundFlow 一等 Schedule IR | `runtime/scheduler.py` 仅按 TaskGraph 拓扑序循环 |
| 局部 schedule | `FusedCrownExecutionStep.schedule_id`、placement retry ladder | 局部执行描述/过程式策略，不能表示全程序调度 |
| Query/runtime | `BoundQuery`、compatibility key、batcher、same-solver adapter | PR-13 的有效基础设施，但不是编译器语义层 |
| Backend | TVM TIR、fused CROWN、compile cache | 有真实机制与负面性能证据，但未由统一 IR 链驱动 |

### 1.2 旧叙事的三项错误

1. **把 runtime LinearOperator 叫作 Structured Bound-Operator IR。**
   它验证了结构化系数与物化机会，但缺少独立 module/value/op/type/verifier/pass/serialization
   契约，因此只能称 C1 的 runtime mechanism foundation。
2. **把 PlanBundle 和若干计划记录叫作全局 Plan IR。**
   当前决策对象没有统一引用关系、版本、输入 IR hash、跨决策合法性或一次性 verifier。
3. **把 TaskGraph 拓扑执行叫作 schedule。**
   依赖 DAG 不能表达 allocate/free、materialize、transfer、event、batch loop、retry、fallback、
   dynamic re-plan boundary，因而不是 Schedule IR。

### 1.3 两份外部设计材料中可保留和必须收紧的内容

可保留：

- Runtime 应执行 verification execution plan，而不是成为普通模型推理 runtime；
- BaB 搜索控制保留在 host，粗粒度 bound task 交给后端；
- offline planning 与 runtime adaptation 必须区分；
- spec/domain/training 是不同批处理轴，不能混为一个 tensor batch；
- 编译 kernel、计划模板和动态状态需要分级缓存及明确失效规则。

必须收紧：

- PR-13 已表明相对逐节点的巨大收益主要来自普通 batching；**BatchManager 目前不是独立创新**；
- PR-12J 已表明 fresh/process/disk compile 成本在当前工作负载常不能于 `Q<=1024` 摊销；
  **JIT 只能是稀有的、收益预测通过门禁的 fallback**；
- BaB 父子或兄弟不能默认精确复用 intermediate bounds、α、β 或 split state；所有对象必须遵守
  `EXACT_REUSE / CONDITIONAL_REUSE / WARM_START_ONLY / INVALIDATE`；
- 不能假设每个 bound task 都适合 Relax；当前已验证路径主要是 TIR/PyTorch executor，后端接口
  必须允许 TIR-only、library call 或其他实现；
- certified training 不进入首个 IR 闭环，避免同时扩展 autograd、动态搜索和编译调度三条主线。

## 2. 分层所有权

### 2.1 Primal IR 与 Spec IR

Primal IR 只表示原始网络计算，不承载验证算法状态。Verification Spec 必须成为独立、可 hash、
可验证的输入合同，至少包括：

- perturbation 集合及参数；
- objective/margin 张量语义；
- input/output value 绑定；
- spec batch 轴及静态/动态维度；
- requested lower/upper/intermediate outputs；
- numeric policy。

首个切片可以把 Spec 作为强类型对象而非完整图 IR，但禁止继续使用无约束的
`objectives: Any`。

### 2.2 Bound IR：语义 IR

Bound IR 必须独立于 executor，并至少拥有以下一等对象：

```text
BFBoundModule
  - primal_ref / primal_hash
  - spec
  - domain_config
  - values: BoundValue[]
  - ops: BoundOp[]
  - inputs / outputs

BoundValue
  - value_id
  - type: shape + dtype + layout + device constraints
  - role: interval | coefficient | bias | relaxation | split | objective
  - polarity: lower | upper | both
  - batch_axes: spec | domain | sample | none
  - state_version / provenance

BoundOp
  - typed op kind
  - typed inputs / outputs
  - primal origin
  - reference semantics id
  - effects and legality constraints
```

首个支持的 `BoundOp` 集合应围绕一个窄而完整的 plain-CROWN reference path：

- affine/conv backward propagation；
- ReLU relaxation；
- coefficient compose / bias accumulate；
- concretize/objective reduce；
- explicit `Materialize` 与 `RepresentationCast`；
- input/spec binding。

`Dense`、`Structured`、`Chunked` 是同一参考语义的表示/执行选择，不应通过隐藏的 Python 类型
分支改变数学语义。α、β、split、cuts 首先作为显式 capability/state schema 进入 IR，只有完成
reference semantics 和迁移测试后才进入优化路径。

Bound IR verifier 至少检查：

1. SSA/value 唯一性和 use-def 完整性；
2. lower/upper polarity、shape、dtype 和 batch axes 一致；
3. domain/spec/state version 合法；
4. materialization/cast 前后参考语义一致；
5. fanout、residual merge 和 objective reduction 的贡献不丢失；
6. α/β/split capability 不被 plain-CROWN backend 误消费；
7. 所有 fallback 均显式且可追踪。

### 2.3 Plan IR：决策 IR

Plan IR 是 Planner 的唯一正式输出，不再使用多个互不引用的 plan record 拼接事实。

```text
BFPlan
  - schema_version
  - bound_module_hash
  - planner_config_hash
  - hardware_profile_id
  - workload_profile_id
  - regions: RegionPlan[]
  - representations: RepresentationDecision[]
  - materializations: MaterializationDecision[]
  - partitions / fusions
  - batching: BatchDecision[]
  - cache / recompute: StateDecision[]
  - storage: StorageDecision[]
  - backends: BackendDecision[]
  - cost_summary
  - rejected_candidates + reasons
  - provenance
```

每项决策必须：

- 引用稳定的 Bound IR value/op/region id；
- 说明 capability、memory 和 state-validity 合法性；
- 给出预测 latency、peak memory、compile/setup cost 和置信/风险信息；
- 保留被拒候选及拒绝理由；
- 能被确定性序列化、hash 和 replay；
- 通过一次跨决策 verifier，而非只分别调用局部 `validate()`。

必须区分：

- `PlanTemplate`：由模型、Bound IR、硬件和 backend library 决定的静态候选空间；
- `PlanInstance`：结合当前 query bucket、可用显存、cache 状态和 deadline 后选择的执行实例。

现有对象的迁移关系：

| 现有对象 | v1 归属 |
|---|---|
| `MaterializationPlan` | `RepresentationDecision`/单 barrier candidate evidence |
| `MaterializationPlacementPlan` | 多 region `MaterializationDecision` |
| `ExecutionCandidate` | `BackendDecision` 候选 |
| `StoragePlan` | `StorageDecision` 的 lowering 结果或 Task IR 附件 |
| `FusedCrownExecutionStep` | region/backend 决策降低后的 Task/Schedule 片段 |
| `PlanBundle.meta` | 仅 debug/provenance；不得再承载语义性计划 |

### 2.4 Task IR：后端工作单元 IR

Task IR 位于 Plan IR 之后，表示后端可编译或可调用的粗粒度工作单元。它不决定全局次序，也不
拥有 BaB 搜索策略。

v1 必须补足：

- `TaskKind`：至少区分 interval propagation、plain-CROWN region、concretization、state update；
- typed TaskOp attrs，逐步消除关键路径 `Dict[str, Any]`；
- 明确 task inputs/outputs、parameter/state dependencies 和 memory effects；
- backend capability id、compiled artifact key 和 reference implementation id；
- task-level dynamic dimensions/shape constraints；
- 与 Plan IR region id、Schedule IR task id 的双向可审计引用。

α/β optimization、split-bound 和 solver constraint task 只在相应语义闭环完成后加入，不通过一个
泛化的 `CROWN` 枚举掩盖差异。

### 2.5 Schedule IR：执行 IR

Schedule IR 描述一次 PlanInstance 如何在 host/device 上执行。它与 TaskGraph 的关系是：

> TaskGraph 给出必须满足的依赖；Schedule IR 给出满足这些依赖的一种具体、可验证执行方式。

最小 action 集：

```text
Allocate(buffer, arena, bytes)
Free(buffer)
Materialize(src, dst, representation)
Transfer(src, dst, direction, stream)
Launch(task_id, backend_artifact, bindings, stream)
RecordEvent(event, stream)
WaitEvent(event, stream)
BatchLoop(axis, slices, body)
StateLoad / StateStore / StateInvalidate
CheckBudget(bytes)
Fallback(candidate_id, reason)
Retry(policy_id, bounded_attempts)
RequestReplan(reason, constrained_context)
EmitResult(query_ids, outputs)
```

v1 不需要立即实现多 GPU、异步编译或通用控制流，但 IR 必须预留 block/region 结构，避免未来把
所有动态动作重新藏回 Python。

Schedule verifier 至少检查：

1. TaskGraph dependency 和 use-before-def；
2. allocate/free 生命周期及 physical alias 冲突；
3. 预测/静态可证的 peak memory 不超过计划预算；
4. stream/event happens-before；
5. query id 在 batch split/merge 后无丢失、重复或错序；
6. retry 次数有界，fallback capability 合法；
7. state load/reuse 满足有效性等级；
8. runtime re-plan 只能改变允许的执行决策，不能改变 Bound IR 数学语义。

### 2.6 Query IR 与 Runtime State：正交输入

现有 `BoundQuery`/`QueryBatch` 不下沉为 Bound IR 节点。它们提供：

- 当前 method/stage、spec/domain batch 和 state lineage；
- model/weight/input/spec/split/alpha/beta 版本；
- deadline、requested outputs 和 numeric policy；
- compatibility key 和 plan specialization key。

Planner 用它实例化 `PlanTemplate`；Runtime 用它绑定 Schedule IR，并恢复 per-query 结果。

## 3. Offline Planner、Runtime Planner 与 JIT 的边界

### 3.1 Offline Planner

输入：Bound IR、硬件能力、backend capability、离线 profile。
输出：候选 region、PlanTemplate、Task IR templates、可用 compiled artifact keys。

它负责较昂贵但与单个动态 query 无关的分析：

- graph/region partition；
- representation/materialization candidate generation；
- fusion/backend candidate generation；
- shape/fanout/liveness/static bytes 特征；
- kernel family 和 compile cache key；
- small-graph oracle 或离线 cost-model calibration。

### 3.2 Runtime Planner

输入：PlanTemplate + query bucket + available memory + cache/state validity + deadline。
输出：PlanInstance + Schedule IR。

允许动态改变：

- 合法候选中的 backend/representation；
- spec/domain batch 切分；
- cache/recompute/eviction；
- bounded fallback/retry；
- 已预先声明的 specialization 参数。

禁止动态改变：

- bound method 或参考数学语义；
- objective/perturbation；
- α/β/split capability 边界；
- 未经 verifier 的 graph rewrite；
- 为追求完成率而静默丢 query、改 requested outputs 或更换 numeric policy。

### 3.3 JIT 门禁

JIT 不是默认 Level-2 路径。只有同时满足下列条件，Runtime 才能发出 `CompileRequest`：

1. 现有 compiled candidates 均不满足 capability/shape 或预计显著劣于新 specialization；
2. compile cost、预计复用次数和 warm latency 有显式预测；
3. 预计 amortized latency 优于不编译 fallback，并给出 break-even；
4. compile request 有确定 cache key、并发去重和失败 fallback；
5. 不阻塞 correctness-critical host search；
6. 结果经过 reference comparison 后才进入可复用 cache。

PR-12J 的现有结果是这项门禁的反证基线：在新 workload 证明可摊销前，论文不得把 adaptive JIT
列为已成立贡献。

## 4. Verification-aware Runtime 的职责

Runtime 是 Schedule IR executor，而不是第二个 Planner，也不是第二套 verifier。

建议的逻辑组件为：

```text
VerificationRuntime
  - PlanInstanceCache
  - ScheduleExecutor
  - BatchCoordinator
  - StateStore
  - CompileArtifactCache
  - ReplanController
  - SearchAdapter
```

职责边界：

- `BatchCoordinator` 只合并 compatibility key 完全相容的 query；
- `StateStore` 实施 state version 和 validity，不猜测父子状态可复用；
- `ReplanController` 只在 Schedule IR 显式 re-plan 点触发，并记录原因；
- `SearchAdapter` 接入现有 host solver，不重新实现 branch/prune/termination；
- `ScheduleExecutor` 执行 allocate/launch/event/retry/result restoration；
- backend executor 只执行 Task IR，不观察全局 BaB 队列后偷偷改策略。

三类 batching 必须分别记录：

| 轴 | 语义 | v1 状态 |
|---|---|---|
| spec batch | 同一 domain 下多个 objective | 首个 IR 闭环必须支持 |
| domain/BaB batch | 多个相容 domain query | 保留 PR-13 机制，迁移后再评估 |
| training sample batch | certified training/autograd | 首篇非目标 |

## 5. 端到端不变量

每个编译/执行结果必须能回答以下问题：

1. 使用的是哪个 Primal/Spec/Bound IR hash？
2. 哪个 Planner 版本生成了什么候选，为什么选择/拒绝？
3. Plan IR 的每个 region 决策降低成哪些 Task IR？
4. Schedule IR 以什么顺序、stream、buffer 和 fallback 执行？
5. 哪些状态被 exact reuse、warm start 或 invalidate？
6. backend artifact 的源码/ABI/cache key 是什么？
7. 输出是否保持相同浮点语义下的 dense reference computation？
8. query 是否有丢失、重复、错序或 requested-output 不一致？

统一 correctness oracle 分层：

```text
Bound IR interpreter
      ==
Plan IR reference lowering
      ==
Schedule IR reference executor
      ==
optimized backend execution
      ==（适用时）
external verifier same-query semantics
```

PR-14B ResNet 的 `max diff 796.765` 表明最后一层目前不成立；它不能通过内部三层一致性测试被
绕过。首个 IR 闭环可先在内部语义上完成，但 external adapter 只能在 exact semantics contract
明确后恢复。

## 6. 实施路线与硬门禁

### IR-0：契约冻结与历史 claim 纠偏

交付：本文档、Claims Map 降级、执行备忘录追加路线纠正、旧架构评审标记过时。
门禁：不得再把 runtime object、PlanBundle 或 topo loop 分别称为完整三层 IR。

### IR-1：Bound IR v1 最小语义闭环

范围：窄 plain-CROWN Linear/ReLU/Conv 子集 + residual/fanout + objective concretization。
交付：typed schema、builder/lowering、verifier、deterministic dump/hash、dense interpreter。
门禁：

- MLP、chain CNN、fanout/residual 的 reference final bounds；
- materialize/structured/dense rewrite 前后数值一致；
- malformed polarity/shape/state/fanout 必须 fail closed；
- 不通过 `runtime/crown_ibp.py` 的隐藏分支定义 IR 语义。

### IR-2：Plan IR v1 与现有决策迁移

范围：representation、barrier placement、backend、batch、storage/lifetime。
交付：PlanTemplate/PlanInstance、统一 verifier、旧计划适配器、确定性 replay。
门禁：

- 同一输入产生相同 plan hash；
- 每个旧 PR-11/12 决策能映射到新 IR 或明确标记 unsupported；
- 无语义性决策藏在 `meta`/`Any`；
- capability/memory/state-validity 交叉校验。

### IR-3：Schedule IR v1 与 reference executor

范围：单 GPU、单 host、同步 launch 为基础，显式 batch loop、memory、fallback/retry。
交付：schedule lowering、verifier、reference executor、trace/replay。
门禁：

- 与 TaskGraph dependency 一致；
- peak-memory 静态账本与 runtime trace 对齐；
- OOM retry 有界且不丢 query；
- custom stream/event happens-before 有回归；
- TaskGraph topo loop 仅作为对照，不再是正式 scheduler。

### IR-4：现有 backend/runtime 迁移

范围：PyTorch dense/chunked/structured、TVM fused/unfused、PR-13 query runtime。
门禁：

- backend 只消费 Task/Schedule IR；
- 现有 PR-10—13 关键 correctness tests 不退化；
- compile cache、capability rejection 和 state validity 均由新 id/hash 驱动；
- PR-14 的 external mismatch 保持显式 No-Go，不被 fallback 隐藏。

### IR-5：自适应 PlanInstance

范围：memory/cache/query-distribution 驱动的 runtime selection，默认不增加 JIT。
门禁：

- 相对 fixed、local greedy、ordinary batching、公平 batched original；
- 多预算下选择不同合法计划；
- held-out workload 上报告 Oracle regret、TTV、tail latency、peak memory；
- 证明收益来自跨层规划/调度，而不是单纯 pack batch。

### IR-6：条件式 cached specialization

只有 IR-5 显示 compiled candidate coverage 是主要瓶颈、且新 workload 的 break-even 可接受时启动。
否则保持 planned hypothesis，不实现、不写入 headline。

## 7. ASPLOS 贡献的重新定义

当前状态不是“runtime 差最后一截”，而是 C1/C2 的正式编译器抽象尚未实现。因此论文贡献只能在
IR-1—IR-5 有真实证据后升级：

1. **C1：Verification Bound IR with Explicit Representation and Materialization Semantics**
   必须是可验证、可降低、可序列化的一等 IR，不是 runtime operator collection。
2. **C2：Cross-Layer Plan and Schedule IR for Bound Execution**
   联合表示 representation、materialization、backend、batch、storage 和 dynamic fallback，并
   证明全局选择优于局部/固定策略。
3. **C3：Adaptive Execution of Evolving Verification Workloads**
   只有相对公平 ordinary batching 仍有可归因收益时才保留为独立贡献；否则 runtime 是 C1/C2
   的实现载体。

北极星问题改为：

> BoundFlow 能否通过显式 Bound、Plan 和 Schedule IR，对验证 workload 中表示、物化、批处理、
> 内存、后端和动态状态进行合法且可审计的跨层编译，并在真实 workload 上改善
> latency–memory–time-to-verify Pareto？

## 8. 明确非目标

在首个闭环完成前不做：

- 新 α/β/split GPU kernel；
- 新 BaB 搜索/branching 算法；
- certified training/autograd 完整支持；
- 每个 BaB node 的即时 TVM 编译；
- 多 GPU/分布式 runtime；
- 为 external ResNet replay 强行拟合一个语义不明 adapter；
- 继续调 PR-12 已冻结的孤立 TIR schedule；
- 只写 ASPLOS 前两页而不补编译器核心。

## 9. 下一分支与提交序列

PR-14 No-Go 证据应保留。完成本次文档纠偏后，从当前已包含 PR-14 证据的基线创建：

```text
feat/compiler-ir-stack-v1
```

建议按可独立审计的提交推进：

1. `docs: freeze Bound Plan Schedule IR contracts`
2. `feat(ir): add typed Bound IR v1 and verifier`
3. `feat(ir): add deterministic Bound IR dump and interpreter`
4. `feat(planner): unify decisions in Plan IR v1`
5. `feat(schedule): add Schedule IR verifier and reference executor`
6. `refactor(runtime): execute lowered schedule instances`
7. `bench: compare fixed local and global IR-driven plans`

不得在第 2—5 步之前以目录重组或空接口宣称 compiler architecture 已完成。

## 10. 完成定义

本文档的完成只意味着路线和接口责任冻结，不意味着实现完成。`compiler-ir-stack-v1` 只有在以下
证据全部存在时才可关闭：

- Bound/Plan/Schedule 三层均有 typed schema、verifier、deterministic dump/hash；
- 存在 `Primal+Spec -> Bound -> Plan -> Task+Schedule -> reference/backend` 的可运行闭环；
- 至少一个非 toy residual/fanout workload 通过逐层 reference comparison；
- 旧 PR-11/12/13 机制通过正式适配器消费新 IR，而不是保留第二条隐藏主路径；
- 动态 fallback/retry、state validity、query restoration 有 fail-closed 回归；
- 论文 Claims Map 只依据新工件升级，不沿用历史措辞自动继承 claim。
