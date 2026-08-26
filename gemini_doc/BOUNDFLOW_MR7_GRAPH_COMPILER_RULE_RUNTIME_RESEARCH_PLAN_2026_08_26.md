---
status: research-design-only
updated: 2026-08-26T00:45:00+08:00
type: research-plan
topic: boundflow
slug: mr7-graph-compiler-rule-runtime-research
stage: s01
execution_authority: false
performance_claimed: false
---

# 基于 MR7 的 BoundFlow 图编译、验证规则与物理运行时调研计划

> **2026-08-26执行状态更新**：MR7-R已正式通过，boundary median=`20.333%/24.684 ms`、5/5过
> 门禁，required region speedup=`1.91214x`。GC-0/FCR-1预注册已由外审批准并关闭；GC0-0通用
> schema与direct negative legality tests已实现并完成内部验证，待独立外审。capture、analysis、
> lowering、runtime与timing仍关闭，`performance_claimed=false`；GC0-0外审批准后才允许预注册
> GC0-1。见MR7-R formal closure、GC-0/FCR-1预注册与GC0-0 changelog。

> **2026-08-26后继状态**：GC0-0已由独立外审批准并关闭；只开放GC0-1 capture/analysis预注册，
> 不开放实现、lowering、runtime或timing。GC0-1必须把schema-level shallow rejection与具有完整
> graph witness的analysis rejection分开计数。

## 0. 文档地位与一句话结论

本文是基于 MR7 raw、当前 BoundFlow/TVM 代码和外部编译系统的一次**架构调研与预设计**，不是新的
性能 closure，也不改变当前权威执行顺序。

- MR7 的正式状态仍是 `INVALID_MR7_ATTRIBUTION`；
- 本文初稿时唯一开放实验为 MR7-R；该实验现已正式通过并关闭；
- 本文中的 MR7 share、required speedup 和 route 都标为 `[diagnostic-only]`；
- 本文不开放 MR7-A/B/C、FCR timing、same-solver、query、queue 或 ASPLOS claim；
- `performance_claimed=false`。

一句话结论：

> BoundFlow 下一代性能路径不应继续扩大“PyTorch 每个站点调用一个 TIR kernel”的 bridge，
> 也不应另接一套不理解验证语义的通用编译器；应建立
> `BoundFlow Verification IR → verification-aware Relax graph → fused TIR → physical arena/execution graph`
> 的单向 lowering，让规则合法性、kernel fusion、buffer lifetime 和 command submission 在同一编译边界内闭合。

这不是“把求解器整个静态编译”的承诺。第一阶段仍保留 host solver 的终止、verdict 和动态策略，只编译
稳定、重复、可类型化且可证明 closed 的 tensor region。

## 1. 为什么要在 MR7 后讨论图编译，而不是继续调一个 kernel

### 1.1 已有结果呈现出稳定的收益传播断层

| 层级/实验 | 冻结结果 | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| B4-B2 局部 TIR | `4.89834x` | 合法局部 kernel 可以快 | 不能推出 production wrapper/query 快 |
| CIBC Conv operator | `12.7951x` geomean | lower/upper 横向融合很有效 | 不能推出 CROWN/BaB/完整 query 快 |
| CIBC ResNet2B IBP graph | `2.45631x` | 多个真实 Conv 收益能传到一张小图 | 图中仍有 Python/PyTorch dispatch，且 scope 仅 IBP |
| R3 D2-B P-anchor wrapper | `1.752001x` | structured owner + bounded arena + staged VJP 可在一个签名上闭合 | 不能外推 active-β 或多 site |
| R3-3 active-β S-anchor | `0.668275x`，incremental allocation `10.9375x` | 同一 schedule/ABI 不可跨状态签名外推 | 不能据此否定所有 structured region |
| B4-C2 dense retention | `0.337–0.349x`，peak `1.3401x` | dense A 跨层保存是已证伪边界 | 不能复活同一路线继续堆站点 |
| MR5 三 site bridge | `0.83440665x` | 独立 kernel + framework crossing 不传播 | 不能否定全 region 编译 |
| MR6 guard reduction | 对 MR5 `1.03312564x`；对 provider `0.90300665x` | guard 不是唯一瓶颈 | 不能把其余 residual 自动归为 optimizer |

共同模式是：kernel 能快，但当前 production ownership 仍在 PyTorch/auto_LiRPA。layout、autograd、临时
buffer、α/β state、optimizer mutation、launch、同步和输出 materialization 没进入同一个编译单元，局部
收益在边界处被消耗。

### 1.2 MR7 的独立 raw 复核

从 `artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1/raw.json` 独立重算，三个
unprofiled control 的 host ledger 为：

| pair | outer | FFI/DLPack/stream | layout/materialize | post-output | 三项合计 |
|---:|---:|---:|---:|---:|---:|
| 0 | `127.786 ms` | `8.3612%` | `1.3482%` | `10.1553%` | `19.8646%` |
| 1 | `133.125 ms` | `8.4210%` | `1.3019%` | `9.7232%` | `19.4461%` |
| 2 | `130.642 ms` | `7.7699%` | `1.2403%` | `17.0298%` | `26.0399%` |

按正式实现使用的“分类 share 各取中位数后求和”口径，得到：

- `[diagnostic-only]` boundary share=`19.8183476%`；
- `[diagnostic-only]` boundary absolute=`25.8910325 ms`；
- `[diagnostic-only]` FFI/DLPack/stream=`8.3611584%`；
- `[diagnostic-only]` layout/materialization=`1.3019218%`；
- `[diagnostic-only]` post-output=`10.1552673%`；
- 57 个 FFI span，30 forward + 27 backward launch，117 个 layout span；
- host ledger 的 `optimizer_and_residual≈77.35%` 是 `outer-known` 的补项，**不是被直接测量的 optimizer
  时间**。

MR7 还有一个容易误用的数字：`device_kernel_share=8.6915%` 的分母是 profile worker 的 CUDA-event
outer，不是 host outer。因此不能将 `8.69%` 与 host `19.82%` 相加，也不能称它为“kernel 占 host
outer 8.69%”。

MR7 之所以正式失效，是三个 profile/control CUDA-event ratio 为：

```text
1.239399
1.039553
1.096733
```

第一项超过冻结上限 `1.10`。所以这些数据只能设计 instrumentation 和 architecture，不能开放优化资格。

### 1.3 MR7 仍然给出的结构信息

尽管 performance attribution 无效，以下**结构事实**不依赖 share headline：

1. production bridge 的 30/27 launch 与 57 次 FFI span 实际存在；
2. 当前每个 site 仍做 DLPack view/pointer 往返、stream/device 验证和输出转换；
3. `route_relu`/`route_conv` 仍在 Python 中做 α 重建、`permute/transpose/contiguous`、zero allocation、
   finite/status 检查；
4. 当前 lower path 的 kernel 与 runtime ownership 分离；
5. 因而“把多算子、内存和提交放进一个 closed region”是合理研究假设，但尚不是已经通过的机会结论。

## 2. 当前系统到底具备了什么，缺了什么

BoundFlow 不是没有 IR。问题是已有对象主要完成了**描述、验证和 replay**，尚未组成唯一的 production
physical lowering path。

### 2.1 已有资产

#### Verification/Bound IR

- `BoundTensorType` 已有 shape、dtype、layout、device 以及 sample/spec/domain 三轴；
- `BoundValue` 已有 role、polarity、representation、state version 和 primal lineage；
- `BoundAffineStateRef` 能表示上下界 coefficient/bias typed state；
- `VerificationSpec` 已类型化 perturbation、objective、polarity 和 numeric policy。

代码入口：`boundflow/ir/bound.py`。

#### Plan/Task/Schedule IR

- `RegionCandidate` 表达 fused op span 及其显式输入/输出；
- `PlanTemplate/PlanInstance` 已表达 representation、materialization、backend、batch、storage 和 state
  选择；
- `StorageBinding` 已有 arena id、offset、size、representation 和 live interval；
- `TaskIRUnit` 已有 typed boundary、state/external dependency、memory effect 和 backend binding；
- `ScheduleModule` 已有 launch、allocate、materialize、transfer、record/wait event、state commit/free 等
  action schema；
- `BackendDispatchKey` 已能绑定 Bound/Plan/Task/backend/artifact identity。

代码入口：`boundflow/ir/plan.py`、`boundflow/ir/task_v1.py`、`boundflow/ir/schedule.py`、
`boundflow/runtime/task_backend_dispatch.py`。

#### 已验证的专用合同

- `DifferentiableLowerRegionIRV1` 已冻结 α/β、sign selection、bias reduction、affine contraction 顺序；
- `StructuredCoefficientHandleV1.to_dense()` 在 production 直接拒绝 dense escape；
- `R31BBoundedArenaTraceV1` 已表达 12-step reverse recurrence、residual branches 和两个 ping-pong
  scratch slot；
- R3 custom backward 已证明“forward 不保存 dense A、backward 重算”的合同方向；
- CIBC TIR 已证明 lower/upper 可以在一个 reduction/kernel 中横向融合；
- device atomic commit、artifact replay、fully re-signed tamper 已提供 correctness 基础设施。

这些专用对象是可迁移的语义原型，但不少固定于 ResNet2B、lower-only、指定 start node 或特定 shape，
不能直接冒充通用 production IR。

### 2.2 实质缺口

| 层 | 已有 schema | production 缺口 |
|---|---|---|
| semantic effect | role/polarity/state version | α/β/split/history/optimizer/branch mutation 未统一为 effect token |
| graph capture | 手工 region/evidence | 无 production trace→semantic graph、最大合法 region、闭包分析 |
| rewrite | dense/structured 单类 rewrite | 无 rule registry、guard、冲突、定点、proof receipt |
| tensor program | Bound/Task + 专用 TIR | 无通用多算子 reduction/view/alias/VJP/saved-state IR |
| backend | 两 op fused CROWN、专用 CIBC | 仍多为硬编码签名和逐站点调用 |
| schedule | stream/event action schema | reference executor 多数 action 只记 trace，lower 默认 `stream_id="sync"` |
| arena | offset/lifetime 账本 | 没有一块真实 CUDA allocation + typed views + epoch/lease |
| parallelism | TaskGraph DAG | production scheduler 仍为串行 topo loop，IBP-only |
| cache/AOT | stable hashes 骨架 | 有路径仍含 Python object identity，execute 时可重复 lower |
| state/commit | typed store/atomic plan | clone/hash/细粒度 check 仍可能进入 hot path，未形成 device resident state graph |

当前代码实际是三套尚未完全合流的系统：

1. 旧 `BFTaskModule/PlannerPass` 管 IBP 和 task-level reuse；
2. Bound/Plan/Task/Schedule 管 typed correctness、artifact 和 replay；
3. RVIR/R3/B4/MR 专用对象管真实 αβ-CROWN state 与 TIR 实验。

下一版的首要工程不是造第四套对象，而是规定三者的**单向迁移接口**和唯一 production lowering。

## 3. CIBC 对图编译路线的启示与边界

### 3.1 当前 CIBC 并非“只做简单替换”

现实现已经做了关键的 verification-specific horizontal fusion：

- baseline 的正负 weight clamp 与四次 Conv 被一个 center/deviation reduction 取代；
- 同一个 TIR kernel 内累计 center 和 deviation，并一次写出 lower/upper；
- lower/upper 共用 combined output buffer；
- module、DLPack views、output buffer 可缓存；
- 完整 ResNet2B IBP 对照两侧都用 CUDA Graph，candidate input copy 计入计时。

这正是为什么单 Conv 可达 `12.7951x`，整图仍有 `2.45631x`。

### 3.2 当前 CIBC 尚不是通用图编译器

- 当前 TIR 是手写固定 signature，不是从 high-level Bound/BC graph 自动识别；
- schedule 只 sweep `threads_per_block∈{64,128,256}`；
- 每个 output thread 串行 reduction，没有 shared-memory tile、vectorization、tensor-core、warp
  specialization 或 1000-trial MetaSchedule；
- 仅 `groups=1`、float32、当前 CUDA shape family；
- Linear、ReLU、residual add、flatten 等仍由 Python/PyTorch 逐 op dispatch；
- 没有跨层 producer-consumer fusion、统一 arena、optimizer、branch/top-k 或 queue runtime。

所以准确表述是：

> BoundFlow 已实现 CIBC 最核心的 Conv-IBP horizontal tuple fusion，并做了小规模静态 schedule sweep；
> 尚未实现自动图识别、完整 fused-TE autotuning 和 verification-aware physical runtime。

用户提供的本地 `docs/CIBC_for_DAC.pdf` 提出的 high-level BC IR、同 shape/相似 reduction 横向融合、
fused tuple TE 与 schedule search，应该被吸收到新规则系统，而不是另开一条互不相通的 CIBC runner。

## 4. 外部图编译系统能借什么，不能借什么

### 4.1 TVM Relax/TIR：作为首选 lowering，不作为 verification semantic owner

当前 vendored TVM revision 为
`6248b5db43505fbcfb13cc289d11877d5d2649e8`。其 CUDA pipeline 已包含：

```text
LegalizeOps
→ AnnotateTIROpPattern
→ FoldConstant
→ FuseOps
→ FuseTIR
→ DLight default schedule
→ CallTIRRewrite
→ StaticPlanBlockMemory
→ RewriteCUDAGraph
→ LowerAllocTensor
→ KillAfterLastUse
→ LowerRuntimeBuiltin
```

因此不需要先造 fusion、buffer planning、CUDA Graph 的通用基础设施。真正需要补的是：

1. 将合法 verification region 表达成 Relax dataflow；
2. 在交给通用 pass 前，由 BoundFlow 验证 α/β/split/history、mutation、VJP、alias 和 dense-A
   lifetime；
3. 把 Relax/TIR/arena/schedule/module hash 重新绑定进 BoundFlow receipt。

不能直接让普通 `FuseOps` 决定全部融合。TVM 的 op pattern/post-dominator 知道数据流，却不知道 lower/upper
方向、start-node keyed α、active β、split/history 或 optimizer mutation。

### 4.2 MLIR：借鉴 legality/effect/bufferization，不作为第一后端

MLIR 最值得借鉴：

- `legal/dynamic/illegal` conversion target；
- analysis conversion：先报告哪些 op 可合法化，不实际改写；
- PatternRewriter 的 guarded DAG-to-DAG rewrite；
- destination-passing style；
- One-Shot Bufferize 的 whole-function SSA use-def/alias analysis；
- ownership-based deallocation。

对应到 BoundFlow：先做 analysis-only rule admission；只有所有外部 use、effect、alias、VJP 和 state
version 都闭合时才真正 rewrite。第一阶段不接完整 MLIR backend，因为 TVM 已有 Relax/TIR/CUDA 实现，
接 MLIR 会重复基础设施并切断现有 CIBC/R3 资产。

### 4.3 XLA：借鉴 fusion→schedule→buffer→command graph 分层

XLA 的有用架构是：

```text
HLO fusion
→ schedule
→ dataflow/alias/buffer assignment
→ thunk sequence
→ compatible thunk subsequence → command buffer
```

其中 schedule 先考虑 peak memory，latency-hiding 可能增加 memory，必要时再 rematerialize；buffer
assignment 将不重叠 live range 放入大 buffer slice；command buffer 最后才把兼容 kernel/copy 序列变成
CUDA/HIP Graph。

BoundFlow 应采用相同分层，但必须增加 verification effect 和 proof receipt。XLA 的通用 cost/alias 规则不能
判断 split/history、α/β mutation 或 verifier termination，故不接 XLA 依赖。

### 4.4 PyTorch 2 编译栈：作为 capture probe/oracle，不作为最终 owner

FX/AOTAutograd/Inductor 适合：

- 快速检测某 region 是否 `fullgraph` capture；
- 生成 forward/backward graph，验证 save-vs-recompute；
- 作为独立性能上界和 oracle。

但 state mutation、Python control 和 custom op 容易 graph break/specialize；opaque custom op 会阻断跨边界
fusion，最终 allocator/runtime 仍由 PyTorch 持有。因此可做短 timebox probe，不把它作为最终 production
runtime。

### 4.5 CUDA Graph 与 stream-ordered allocator：只解决提交/分配，不替代融合

CUDA Graph 适合 MR7 所示的固定 10-evaluation/9-mutation topology 和大量短 launch，但它只降低 command
submission，不能消除 kernel 间 HBM materialization。单 kernel/region fusion 仍由 Relax/FuseTIR 完成。

CUDA graph memory node 和 `cudaMallocAsync/cudaFreeAsync` 的 lifetime 必须受 graph/event/stream ordering
约束。固定热区优先 static/persistent arena；动态尾部才使用 stream-ordered pool。

## 5. 目标架构

```text
auto_LiRPA / provider state / model
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│ L1 BoundFlow Verification Semantic IR              │
│ role/polarity/start-node/α/β/split/history/effect  │
└───────────────────────┬────────────────────────────┘
                        │ typed admission + legal region
                        ▼
┌────────────────────────────────────────────────────┐
│ L2 Verification Graph IR / guarded rewrite         │
│ pure epochs, views, reductions, VJP, mutation token│
└───────────────────────┬────────────────────────────┘
                        │ lower legal tensor regions
                        ▼
┌────────────────────────────────────────────────────┐
│ L3 TVM Relax → fused TIR                           │
│ fusion, layout, schedule, certified in-place       │
└───────────────────────┬────────────────────────────┘
                        │ bufferize + schedule
                        ▼
┌────────────────────────────────────────────────────┐
│ L4 Physical Arena + Execution Graph                │
│ slices, leases, streams, events, command graphs    │
└───────────────────────┬────────────────────────────┘
                        │ coarse commit / status
                        ▼
              host solver / queue / verdict
```

关键所有权：

- L1 决定“语义上能不能改”；
- L2 决定“图上改成什么”；
- L3 决定“kernel 如何生成和调度”；
- L4 决定“buffer、stream、event 和 replay 如何实际运行”；
- PyTorch 在迁移期只做 frontend/oracle/fallback，不再是 candidate region 内的 tensor runtime。

## 6. 第一等公民 Graph IR

建议新增通用 `VerificationProgram/VerificationRegion`，但复用 `BoundValue`、`TaskIRUnit`、
`RegionCandidate`、`StorageBinding` 和 dispatch identity，不另造平行 identity 系统。

### 6.1 Value 类型

每个 SSA value 至少携带：

```text
tensor_type: shape/dtype/layout/device
axes: sample/spec/domain
role: parameter/interval/coefficient/bias/alpha/beta/split/history/optimizer/output
polarity: lower/upper/both/none
representation: dense/structured/compressed/chunked/scalar/bitpack
lineage: model/op/start-node/domain/property
state_version: immutable epoch or mutation ordinal
storage_class: persistent/scratch/saved/output/host-visible
alias_set: exact/may-alias/no-alias
```

### 6.2 Op 类型

至少需要：

- pure tensor op：view、layout、gather、sign-select、affine、Conv/Linear、reduction、norm、epilogue；
- verification op：relaxation、concretize、split injection、branch score、top-k；
- VJP op：custom VJP、rematerialize、gradient reduction；
- state op：load、evaluate、mutation、project/clamp、commit、rollback；
- runtime op：arena lease、launch、record/wait event、status write/read；
- host barrier：verdict、termination、unsupported dynamic decision。

### 6.3 Effect token

普通 tensor SSA 不足以表达 solver。建议每个 effectful op 显式消费并生成 token：

```text
AlphaState(site, version)
BetaState(split_history, version)
OptimizerState(evaluation, mutation, version)
DomainState(domain_id, lineage, version)
QueueState(epoch, fairness_order)
CommitState(transaction, epoch)
```

任何 rewrite 跨越 token version 都非法。这样 mutation barrier 不再依赖 Python 调用顺序的隐含约定。

### 6.4 Pure epoch 与 effect epoch

- pure epoch：相同 state version 内的 tensor DAG；可 canonicalize、fusion，必要时做受限 equality
  saturation；
- effect boundary：α/β update、optimizer mutation、split、queue commit、termination；默认不可跨越；
- effect fusion：只有专门规则证明完整 state transition 等价时才允许，如 `OptimizerStep`。

## 7. Verification-aware 规则系统

### 7.1 每条规则的最小 schema

```text
rule_id / rule_version
lhs_pattern / rhs_builder
semantic_preconditions
effect_and_version_preconditions
ownership_alias_liveness_preconditions
shape_dtype_layout_constraints
external_use_prohibition
numeric_policy / endpoint_policy
VJP_and_saved_state_contract
estimated_eliminated_launches
estimated_eliminated_materialized_bytes
fallback_boundary
proof_or_oracle_obligation
rule_receipt_hash
```

规则匹配成功不等于可提交。必须依次经过：

1. structural match；
2. dynamic legality；
3. effect/version closure；
4. external-use/post-dominator closure；
5. alias/liveness/dense-A escape；
6. VJP/saved-state contract；
7. profitability；
8. rewrite receipt。

### 7.2 规则分类

#### A. 无条件 canonicalization

- view/reshape/transpose/permute chain 合并；
- 不复制的 reshape 变为 view；
- zero bias 变成 typed neutral element；
- sample/spec/domain 轴规范化；
- compressed α/β logical lookup 规范化；
- mutation 显式化为 state-in/state-out token。

这类规则只改变表达，不进行成本驱动搜索。

#### B. 带验证 guard 的语义融合

| ID | 规则 | 主要融合内容 | 核心合法性 |
|---|---|---|---|
| `V-H1` | IntervalTwinReduction | lower/upper、center/deviation、bias epilogue | 同 weight/domain；方向不可交换；输出 `l≤u` |
| `V-R1` | ReluAffineLower | A sign、α slope、β/intercept、Conv/Linear、bias | bound direction、zero endpoint、start node exact |
| `V-R2` | CompressedAlphaGather | α gather/lookup 直接内联 consumer | path/index/shape/version exact；禁止 dense α |
| `V-R3` | BetaSplitInjection | β location/sign/split mask 与 coefficient update | active/empty 分 specialization；history exact |
| `V-D1` | ResidualFactorization | residual affine branches、skip、bias epilogue | DAG closure；共享节点一次；bias token 守恒 |
| `V-C1` | TerminalConcretize | 最后 coefficient + center/radius/norm + commit | 仅 typed terminal consumer；dense A 不 escape |
| `V-VJP1` | RegionCustomVJP | forward + minimal-save + rematerialized backward | grad owner仅α/β；拒绝 higher-order；endpoint冻结 |
| `V-O1` | OptimizerStep | evaluate+VJP+Adam+project/clamp | evaluation/mutation/moment/version 顺序 exact |
| `V-B1` | BranchScoreTopK | score+reduction+top-k+compact decision | tie-break/domain/history/termination exact |
| `V-M1` | ArenaReuse | scratch/output slot 与 layout/copy 复用 | live range不重叠；跨stream有happens-before |
| `V-S1` | DomainBatchParallel | 独立 domain/branch 合批或多 stream | signature相容、无alias、commit/fairness不变 |

#### C. 成本控制的等价选择

- fused vs unfused；
- save vs recompute vs bitpacked certificate；
- one-kernel vs command graph；
- per-signature schedule；
- single stream vs dependency-aware multi-stream；
- static arena vs async pool；
- AOT/cache vs JIT fallback。

### 7.3 必须冻结的特殊语义

1. CROWN lower/upper 不是普通双输出：A 正负号选择不同 relaxation；
2. `A==0` 的端点所有权必须与 native exact，不能随意把 `>=0` 改成 `>0`；
3. α 必须带 start-node key、path、feature index 和 version；
4. β 必须带 location/sign/split/history，empty β 不能伪造 dense zero；
5. residual 是 DAG，不是 tree，不能重复共享子图或重复 bias；
6. tensor 可按冻结 tolerance 比较，但 sign、split/history、mutation ordinal、branch/top-k、termination
   必须 exact；
7. 如果未来声明浮点 soundness，需另加 outward-rounding/error-envelope 合同，不能由 allclose 自动升级。

### 7.4 不在第一版上无限 e-graph

第一版采用 deterministic ordered rule registry，并要求每条规则单调降低至少一个冻结指标：

```text
(boundary count, materialization bytes, launch count, peak arena, estimated latency)
```

仅在无 mutation 的 pure epoch 内，后续才可 timebox equality saturation。effectful state op 仍为 barrier。

## 8. Pass pipeline

### P0 — Production capture → Semantic Graph

输入 RVIR-v4 topology/state、MR7 C2→C1→C0 10/9 关系和 Bound typed values，输出通用
`VerificationProgram`。schema 不编码 C0/C1/C2、ResNet2B 或固定 start node。

### P1 — Semantic canonicalization

规范 axis/layout/view、zero/bias、compressed α/β lookup、sign-select、finite/status、state token。每个
rewrite 产生 before/after hash 与 side-condition receipt。

### P2 — Analysis-only legal region discovery

模仿 MLIR analysis conversion：只报告可合法化的最大 closed regions，不实际改图。输出每个失败候选的
明确拒绝原因：external use、effect crossing、alias、dynamic shape、unsupported op、dense escape 等。

### P3 — Region formation

第一个通用 region 覆盖：

```text
typed state input
→ compressed α/β lookup
→ relaxation/sign selection
→ layout normalization
→ Conv/Linear bound propagation
→ bias/reduction/epilogue
→ minimal-saved-state VJP
→ persistent output/status
```

host verdict、termination 和 unsupported dynamic control 留在 region 外。

### P4 — Guarded rewrite/fusion

按稳定 rule registry 应用 `V-H1/V-R1/V-R2/V-R3/V-D1/V-C1/V-VJP1`；不先做贪心
latency search。每个 fused candidate 保留 unfused oracle variant。

### P5 — Layout propagation 与 boundary minimization

第一目标不是追求某个 kernel FLOPS，而是依次最小化：

1. PyTorch↔compiled boundary；
2. DLPack descriptor/pointer crossing；
3. materialized bytes；
4. launch；
5. 估计 critical-path latency。

### P6 — Destination-passing 与 certified in-place

先把输出 buffer 显式传入，再做 alias/liveness 分析。只有 BoundFlow ownership verifier 证明旧值死亡且
无 alias 时，才生成 TVM `call_tir_inplace`。不能全局直接打开，因为 TVM 明确警告该 op 在 type system
中被视为 pure，实际会 mutation。

### P7 — Op-level liveness 与 ArenaPlan

复用 `StorageBinding`，扩展为 physical contract：

- persistent/scratch/saved/output 四类 storage；
- byte offset、alignment、alias set、live range；
- stream/event happens-before；
- epoch/lease 和 rollback；
- recompute choice；
- dense-A escape verifier。

即使首版仅单 stream，schema 也必须携带 event fence，避免未来推翻 ABI。

### P8 — Lower to Relax/TIR

- region 输入/输出为显式 buffer；
- layout、epilogue 和 custom VJP 在 region 内；
- forward/backward 在同一 module family；
- module receipt 绑定 semantic IR、rule set、Relax、TIR、schedule、arena hashes；
- DLight 先给零搜索 baseline；MetaSchedule 只调正式热点。

### P9 — Physical ExecutionGraph lowering

图节点至少有 compiled region、host decision、state load/commit、event、arena lease、fallback、async
status。保留当前 Schedule executor 做 reference/replay，新增真正执行 stream/event/arena 的 driver。

### P10 — AOT/cache admission

持久 key 至少绑定：

```text
semantic graph hash
rule-set hash
tensor-program/Relax/TIR hash
arena-plan hash
shape/dtype/layout/state signature
numeric/endpoint policy
device capability
compiler/runtime revision
```

禁止 Python `id(...)` 进入跨进程 key。静态 structure/hash/alias proof 移到 compile/admission；warm path
只保留 O(1) identity、epoch、status 和 commit counter。

### P11 — CUDA Graph/command graph

先稳定 pointer/arena，再捕获固定 shape/topology 的 compatible command subsequence。TVM
`RewriteCUDAGraph` 默认关闭，只有满足内部 static allocation、constant shape、kernel-only、无 control flow
时启用。

### P12 — Query/queue integration

最后才把多个 domain/property/branch 放入 ready queue。多 stream 只用于真正独立节点；C2→C1→C0
依赖链不能伪造并行。termination/fairness/commit order 必须保持 exact。

## 9. 物理运行时设计

### 9.1 PlanInstance 成为资源 owner

每个 prepared plan instance 持有：

- compiled Relax/TIR module；
- persistent parameter/state views；
- arena allocation 与 typed slices；
- stream/event pool；
- CUDA Graph exec pool；
- status/error ledger；
- rule/module/schedule/arena/graph receipt。

warm call 不再构建 DLPack、不再创建 output tensor、不再逐 kernel 做 Python stream/device 检查。

### 9.2 Arena 不是 Python env 删除

当前 `StoragePlanRuntime` 的 last-use release 主要是从 Python session env 删除对象。新 runtime 要真正：

1. 为 static region 分配一块 aligned device allocation；
2. 按 byte offset 建立 typed views；
3. 用 alias/liveness/happens-before 证明 slice 可复用；
4. 用 epoch/lease 防止并发 execution 误复用；
5. mutation 失败时只提交 coarse rollback/status，不把半成品暴露给 host state store。

### 9.3 dense A 的硬约束

- Python/IR/autograd 边界只允许 structured coefficient handle；
- production `to_dense()` fail closed；
- dense A 仅存在于 tile-local/shared memory 或两个 ping-pong scratch；
- region forward 仅输出 `[domain,spec]` lower/upper 和 compact receipt；
- backward 从 compressed α/β、bounds、weights、split/history 重算；
- 若保存 sign certificate，优先 bitpack；不得保存 float dense A；
- optimizer mutation 后只 rebind compact leaf/version，不保留上一 evaluation scratch。

### 9.4 多 stream 的正确边界

可并行：

- 独立 domains/properties；
- residual 的真正独立 branches；
- 无 state/arena alias 的 sites；
- copy/compute 且 event 依赖完整。

不可假设并行：

- 同一 α/β mutation chain；
- C2→C1→C0 的因果链；
- 共享 scratch/optimizer state；
- 会改变 queue fairness、termination 或 branch order 的工作。

因此 liveness 必须基于 happens-before，不只是拓扑 ordinal。Conv 已经占满 GPU 时，多 stream 也可能更慢，
必须保留 single-stream candidate。

### 9.5 CUDA Graph exec 并发

固定 graph instance 不能被当成无限并发资源。并发 domain 需要 graph-exec/arena lease pool，或明确串行
复用。graph key 绑定 shape、site/state schema、arena layout 和 module hash；cold capture/instantiate 成本
单独核算 break-even。

## 10. Fusion 与 schedule 必须分开

推荐后端顺序：

```text
BoundFlow semantic canonicalization
→ verification-aware DPL/pattern rewrite
→ FuseOpsByPattern
→ generic FuseOps/FuseTIR
→ certified in-place
→ DLight baseline
→ hotspot-only MetaSchedule
→ static memory plan
→ CUDA Graph lowering
```

规则选择决定“哪些 op 在一起”；schedule 决定“这个 fused region 如何执行”。先调单算子、再改变
fusion boundary，旧 tuning 结论自动失效。

MetaSchedule workload key 必须绑定完整 fused IR、shape、dtype、GPU arch、active-β/empty-β、state
signature 和 semantic rule version。CIBC 的 `64/128/256` per-op winner 已说明不存在一个全局 block size；
R3 P-anchor `1.752x` 与 active-β `0.668x` 更说明 schedule/ABI 必须按 verification signature 选择。

## 11. Cost model 与 planner 边界

### 11.1 第一版不是训练一个泛化 global planner

IR-5 已经证明缺乏充分 plan differentiation 时，通用 global planner claim 不成立。第一版采用可解释的
lexicographic/constraint objective：

```text
hard constraints:
  semantic/effect/VJP/alias/memory budget legal

objective:
  minimize boundary_count
  then materialized_bytes
  then launch_count
  then peak_arena
  then predicted_critical_path
  plus compile_amortization
```

planner claim 只限 shape/state/device/cache-context keyed static specialization。

### 11.2 MR7 数据只能初始化 instrumentation

MR7 的 `[diagnostic-only]` 分类可用来决定下一轮要测哪些 counter，但不能直接成为 cost-model weight 或
正式 route。特别是 `optimizer_and_residual` 必须继续拆分，不能把 77.35% 全交给 optimizer fusion。

### 11.3 成本账

每个 candidate 至少估计/实测：

- launch/FFI/DLPack count；
- global read/write/materialized bytes；
- arena high-water；
- kernel/event critical path；
- save/recompute bytes 与 FLOPs；
- compile/tune/capture/instantiate；
- cache hit rate 与 break-even；
- host commit/status cost。

## 12. 首个 vertical slice

### 12.1 启动条件

MR7-R 已证明 unprofiled ledger 低扰动并关闭机会门禁；正式结果为 boundary median=
`20.333%/24.684 ms`、5/5 qualifying。该结果只开放下述 ABI/correctness，不开放 performance timing。

### 12.2 FCR-1 首个 region

MR7-R 通过后，第一刀不是薄 wrapper，而是：

```text
typed production state
→ α/β compressed lookup
→ relaxation/sign
→ input layout normalization
→ fused lower/upper Conv/Linear propagation
→ bias/reduction/epilogue
→ minimal-saved-state custom VJP
→ persistent arena output/status
→ coarse atomic commit
```

候选 region 内要求：

- 无 PyTorch tensor math；
- 一次 admission，device state 保持 resident；
- 无逐 op DLPack view/pointer round-trip；
- 无 dynamic output allocation；
- dense A 不跨 region；
- host 只看 compact status/commit。

### 12.3 四个实施切片

1. `GC-0 Graph/ABI correctness`：capture、typed region、effect、arena ABI、replay；无 timing。
2. `GC-1 Guarded fusion`：DPL/registry、external-use closure、FuseTIR、float64/VJP oracle；无 timing。
3. `GC-2 Physical arena/runtime`：certified in-place、static memory、真实 views/lease/status；结构计数。
4. `GC-3 Command graph/timing`：固定 topology 后启用 CUDA Graph，比较普通 compiled region 与 replay，
   再按冻结协议与 provider 比较。

不能将四刀合在一个事后优化提交中。

## 13. 后续路线与门禁

| 阶段 | 目标 | 通过后只开放 | kill/no-go |
|---|---|---|---|
| `GC-0` | 通用 semantic graph、effect token、analysis-only legality | GC-1 rewrite correctness | graph仍硬编码ResNet/site或effect不闭合 |
| `GC-1` | 首个 closed region + proof receipt + VJP | GC-2 physical arena | dense escape、外部use、trajectory不等价 |
| `GC-2` | 真 arena、persistent views、O(1) warm admission | GC-3 command graph | allocation/crossing未显著下降或peak>baseline |
| `GC-3` | unprofiled region timing、cold/warm/break-even | optimizer region | wrapper-inclusive未达预注册门槛 |
| `GC-4` | 10/9 optimizer evaluate/mutation region | branch/score | mutation/moment/version任何不等价 |
| `GC-5` | branch score/top-k/split materialization | multi-domain scheduler | tie-break/termination/fairness漂移 |
| `GC-6` | ready queue、domain batching、多stream | same-solver formal | query/queue收益不传播或显存失控 |

每阶段 correctness 关闭后才单独预注册 timing。任何局部结果不能跳级升级为 query/queue claim。

## 14. 验证与实验矩阵

### 14.1 正确性

- float64 独立 closed-form/reference，而不是 candidate 自比；
- lower/upper、sign、dα/dβ；
- final α/β、optimizer moments、evaluation/mutation ordinal；
- split/history/domain lineage；
- branch/top-k/tie-break/termination exact；
- residual bias-token conservation；
- fallback/eager escape/dense escape=0；
- fully re-signed semantic/effect/alias/arena/schedule/topology tamper 拒绝。

### 14.2 结构指标

- launch 数；
- FFI/DLPack/pointer crossing；
- dynamic allocation；
- materialized bytes/global bytes；
- arena high-water/fragmentation；
- saved state 与 dense-A count；
- sync/event/stream dependency；
- compiled region/graph cache hit。

### 14.3 性能层级

```text
kernel
→ fused region
→ custom VJP wrapper
→ optimizer step
→ exact-call site
→ same-solver query
→ queue
```

每层都报 candidate、baseline、absolute latency、geomean/worst/bootstrap、显存、冷/热和 break-even，画收益
传播瀑布。不能只报最快 kernel。

### 14.4 workload

- IBP Conv/Linear、Conv→ReLU→residual；
- CROWN P-anchor empty β、S-anchor active β；
- residual diamond、adjacent two-site、six-site lower region；
- 10 evaluation/9 mutation；
- branch score/top-k/split；
- MLP、plain CNN、ResNet，以及至少两个 held-out family；
- 至少一个在 timeout 内 baseline/candidate 都能给出非-unknown verdict 的公开 property。

### 14.5 消融

- horizontal fusion off；
- cross-op fusion off；
- verification guard off（只作拒绝/错误演示，不形成 candidate）；
- arena off；
- dense checkpoint vs recompute vs bitpacked sign；
- global schedule vs signature-keyed schedule；
- single vs dependency-aware multi-stream；
- eager launch vs CUDA Graph；
- fine hot-path guard vs admission + coarse receipt。

## 15. 主要风险与停止规则

### R1：图只是新 wrapper

若 candidate region 内仍逐 op 调 PyTorch、逐 op DLPack 或逐 op allocation，视为架构失败，不进入 timing。

### R2：通用 pass 破坏 verification state

任何 α/β key、split/history、mutation ordinal、branch exactness 漂移即 fail closed；不得用 allclose 掩盖
discrete state 差异。

### R3：dense A 重新跨层存活

saved dense coefficient count 必须为 0；peak memory `>1.0x` 时不得扩 site，除非事先冻结 memory-for-speed
例外且仍满足硬预算。

### R4：过早上多 stream/CUDA Graph

若 pointer/arena/state identity 尚不稳定，不得 capture；若并行不缩短 critical path 或增加 peak/抖动，
保留 single-stream/AOT variant。

### R5：调优吞噬研究周期

DLight 先建立覆盖 baseline；只有 formal attribution 证明 region 是热点，才给 MetaSchedule timebox。搜索预算、
workload hash 和数据库在实验前冻结。

### R6：性能无法向 query 传播

region 即使快，也必须回到 B0/B3/candidate same-solver 三方协议。若 wrapper/query/queue 未过各自门禁，
claim 只停留在对应层级。

## 16. 研究贡献定位

不建议把论文故事写成“把 CROWN 放到 GPU”或“又做了一次 horizontal fusion”。更有区分度的北极星是：

> 一套 proof-carrying verification graph rewrite 系统：在保留 bound direction、start-node keyed α/β、
> split/history、optimizer/branch effects、custom VJP 和 dense-coefficient lifetime 的前提下，联合选择
> fusion、rematerialization、arena 和 command schedule，并证明局部收益如何传播到 solver/query/queue。

与通用 graph compiler 的差异不在 pattern matching 本身，而在：

- verification effect/type；
- guarded legality；
- discrete trajectory exactness；
- dense-A lifetime prohibition；
- proof/receipt chain；
- 局部→region→wrapper→query→queue 的传播实验。

## 17. 与当前权威路线的关系

MR7-R已通过后的执行顺序为：

```text
MR7 INVALID → MR7-R opportunity validated（已完成）
        ↓（当前只开放）
GC-0/FCR-1 ABI + correctness预注册
        ↓
GC-1 guarded fusion correctness
        ↓
GC-2 physical arena/runtime
        ↓
GC-3 command graph + 独立预注册 timing
```

因此本文已成为 MR7-R 后的工程蓝图；MR7-R原预注册由formal closure关闭。本文不授权直接改 TIR、
打开 CUDA Graph 或跑性能搜索。历史预注册见
`BOUNDFLOW_MR7R_UNPROFILED_HOST_RECOVERY_PLAN_2026_08_26.md`。

## 18. 开工前应冻结的最小接口

MR7-R 关闭后，先写 `GC-0` 预注册，至少冻结：

1. `VerificationProgram/Region/EffectToken` schema；
2. analysis-only legality result 与拒绝原因；
3. rule schema、稳定 ID/version、receipt；
4. Relax lowering ABI；
5. physical arena slice/lease/epoch；
6. dense-A escape 与 saved-state ledger；
7. AOT/cache identity；
8. correctness、tamper、structural counter；
9. `performance_claimed=false`；
10. correctness closure 后才允许另写 timing plan。

## 19. 外部一手资料

### TVM

- [TVM Operator Fusion](https://tvm.apache.org/docs/arch/fusion.html)
- [Relax Dataflow Pattern Language](https://tvm.apache.org/docs/deep_dive/relax/dpl.html)
- [Relax transform API](https://tvm.apache.org/docs/reference/api/python/relax/transform.html)
- [Relax VM](https://tvm.apache.org/docs/arch/relax_vm.html)
- [DLight API](https://tvm.apache.org/docs/reference/api/python/s_tir/dlight.html)
- [MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
- [USMP tracking/RFC](https://github.com/apache/tvm/issues/8404)

### MLIR/OpenXLA/PyTorch/CUDA

- [MLIR PatternRewriter](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [OpenXLA GPU Architecture](https://openxla.org/xla/gpu_architecture)
- [OpenXLA HLO to Thunks](https://openxla.org/xla/hlo_to_thunks)
- [PyTorch 2 paper](https://pytorch.org/assets/pytorch2-2.pdf)
- [AOTAutograd](https://docs.pytorch.org/functorch/stable/aot_autograd.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA stream-ordered allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)

### Verification 与 graph optimization 边界

- [auto_LiRPA](https://arxiv.org/abs/2002.12920)
- [β-CROWN](https://arxiv.org/abs/2103.06624)
- [TASO](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)
- [egg](https://arxiv.org/abs/2004.03082)
- [Welder](https://www.usenix.org/system/files/osdi23-shi.pdf)
- [Checkmate](https://arxiv.org/abs/1910.02653)
- [IOS](https://proceedings.mlsys.org/paper_files/paper/2021/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf)

## 20. 本地证据入口

- `artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1/`
- `gemini_doc/BOUNDFLOW_MR7_LAUNCH_MATERIALIZATION_FORMAL_INVALID_CLOSURE_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_MR7R_UNPROFILED_HOST_RECOVERY_PLAN_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_FULLY_COMPILED_VERIFIER_RUNTIME_V1_ARCHITECTURE_2026_08_25.md`
- `gemini_doc/BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_R3_D2B_WRAPPER_TIMING_FORMAL_CLOSURE_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_R3_3_ACTIVE_BETA_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`
- `boundflow/ir/bound.py`
- `boundflow/ir/plan.py`
- `boundflow/ir/task_v1.py`
- `boundflow/ir/schedule.py`
- `boundflow/ir/structured_lower_region.py`
- `boundflow/ir/r3_bounded_arena.py`
- `boundflow/runtime/schedule_ir_executor.py`
- `boundflow/runtime/storage_plan_runtime.py`
- `boundflow/runtime/mr5_multi_conv_production_bridge.py`
- `boundflow/runtime/r3_structured_owner_custom_backward.py`
- `boundflow/backends/tvm/cibc_ibp_conv.py`
- `boundflow/3rdparty/tvm/python/tvm/relax/backend/cuda/pipeline.py`

## 21. 最终判断

MR7 没有正式证明 boundary share，但已经足够警告我们：继续在 PyTorch-owned runtime 中逐站点插入快
kernel，极可能重复 MR5/MR6 的收益传播失败。仓库同时已经拥有 strong typed semantics、专用融合原型、
bounded arena 原型、artifact/replay/tamper 纪律，以及 vendored TVM 的图融合、静态内存和 CUDA Graph
基础设施。

因此最小而正确的下一代编译路线是：

1. BoundFlow 保有 verification semantics 和 rewrite legality；
2. Relax/TIR 保有 graph lowering、fusion 和 kernel schedule；
3. physical runtime 保有 arena、stream、event、command graph 和 state commit；
4. 从一个 closed region 的 vertical slice 开始，以结构计数和 correctness 先关闭；
5. 在 MR7-R 及后续独立 timing 门禁前，不形成性能 claim。

这条路线既吸收 CIBC 的横向融合，也吸收 R3 的 structured owner/rematerialization，并且正面解决 MR7
暴露的 framework boundary；它不是继续堆算子，而是让编译器第一次真正拥有一段验证器热路径。
