# BoundFlow GC-0/FCR-1 Verification Graph ABI + Correctness 预注册

> **2026-08-26状态注**：本预注册已由exchange `gc0-fcr1-prereg-20260826` Round 1批准并由
> executor关闭；批准只开放GC0-0 generic schema/direct negative subset。GC0-1须另行预注册与外审，
> timing/performance全程关闭。正文门禁保留为冻结合同。

status: preregistered-approved-closed-gc0-0-only
date: 2026-08-26
predecessor: `9c5f3867c657078cb6ba980a613b686c5a08f2d2`
predecessor-state: `VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY`
timing-open: false
performance-claimed: false
implementation-open: GC0-0-only（外审已批准并由executor关闭；GC0-1及以后仍关闭）

## 0. 冻结声明

本文在 GC-0/FCR-1 实现、formal raw、性能计时和 winner 选择之前冻结。冻结后不得根据实现难度、
correctness 结果、kernel 数或性能直觉修改以下内容：

1. verification graph 的语义、effect、alias、arena 和 lowering ABI；
2. 首个 vertical slice 的范围、三类 production signature 和 10/9 轨迹；
3. analysis-only legality 与拒绝原因；
4. 独立 oracle、数值/离散容差、结构计数、replay 和 tamper；
5. GO/NO-GO、停止条件、后继开放范围；
6. `timing_recorded=false`、`timing_open=false`、`performance_claimed=false`。

本文的完成只表示协议冻结，不表示 graph compiler、Relax/TIR region、physical arena、custom VJP 或
production replacement 已实现。只有独立外审批准本文后，`GC0-0` 实现才可开始。

## 1. 起点、问题与本阶段目标

### 1.1 已有正式证据

MR7-R 已证明当前三站点 production bridge 的低扰动 host boundary 是稳定、可达的优化机会：

- boundary median=`20.333052%`；
- boundary absolute median=`24.683788 ms`；
- `5/5` run 同时通过 `>=15%` 与 `>=15 ms`；
- 旧 bridge 到 parity 所需 region speedup=`1.91213674x`；
- 当前结构有 `57` 次 forward/backward launch、约 `540` 次 crossing 和 `117` 个 layout span；
- MR5 三独立 site wrapper 只有 `0.83440665x`，说明局部 TIR 收益没有传播到 production outer。

`1.91213674x` 是后续性能候选的要求，不是当前速度。GC-0 不运行 timing，也不形成 speedup claim。

### 1.2 现有代码事实

可复用资产：

- `TaskIRUnit` 已有 typed input/output、external/state dependency、memory effect 与 backend binding；
- `PlanTemplate/PlanInstance`、`BackendDispatchKey` 已有跨层稳定 hash 与 fail-closed linkage；
- differentiable Linear/Conv TIR 已有 template/schedule/module/launch receipt；
- R3 已有 compressed α/β layout、minimal-saved-state custom backward、two-scratch liveness、bounded
  arena、10/9 optimizer trajectory 与 staged VJP 原型；
- CIBC 已证明 lower/upper horizontal Conv fusion 与 signature-keyed schedule 的局部价值。

尚不满足通用 graph compiler 的事实：

- `DifferentiableLowerRegionIRV1` 固定 `provider_start_node="/49"`、lower-only、dense、single-consumer；
- `R31FullRegionPlanV1` 固定 6 个 ReLU、domain=6、spec=1 和 P-anchor shape；
- 多个 TIR/runtime 类按 Linear/Conv/site 分裂，receipt 与 arena ownership 不能表达整个 closed region；
- 当前 production bridge 仍逐 site 进入 PyTorch/custom Function、逐 site materialize/layout/commit；
- 旧类型能证明局部实验，但不能充当 FCR-1 的通用 schema。

### 1.3 GC-0/FCR-1 唯一目标

建立并正式证明一套**与模型名、节点名和站点序号无关**的 verification graph ABI，使一个 closed
multi-op verification region 可以被：

```text
capture
→ typed semantic graph
→ analysis-only legality
→ guarded rewrite plan
→ Relax/TIR lowering ABI
→ physical arena plan
→ prepared runtime/custom VJP
→ coarse atomic commit
→ artifact replay/tamper
```

本文冻结 FCR-1 全部 ABI/correctness 路线，但严格分门禁执行：GC-0 只实现 graph/legality/lowering/arena
ABI 与 replay；GC-1 才实现 guarded rewrite、closed region 与 custom VJP correctness；GC-2 才实现真实
physical arena/prepared runtime 和结构计数。它们均不验证 latency。

## 2. 非目标与禁止捷径

GC-0 明确不做：

- 不跑 candidate/provider timing，不读取 clock 选 winner；
- 不启用 CUDA Graph、多 stream、MetaSchedule、DLight winner search；
- 不修改 auto_LiRPA solver 的 branch、termination、queue 或 verdict；
- 不把完整 solver 静态编译；
- 不扩 AveragePool/BatchNorm 等无关 frontend op coverage；
- 不用 lower-only/empty-β 数字外推 active-β；
- 不把 CIBC `12.7951x`、IBP graph `2.45631x` 或局部 TIR `4.89834x` 当成本阶段结果；
- 不允许 per-op PyTorch fallback、per-op CustomFunction、per-op DLPack、dynamic output allocation；
- 不允许在 graph schema 中出现 ResNet2B、`/49`、`25/Conv_8`、`31/Gemm_14`、C0/C1/C2；
- 不允许仅生成 JSON ledger 而没有真实 prepared module、arena view 和执行路径；
- 不允许 candidate/reference 同源 TIR 自比代替独立 oracle。

## 3. 分层所有权

### 3.1 BoundFlow semantic owner

BoundFlow 拥有：

- bound direction/polarity；
- start-node keyed α lookup、direction/spec index 与 state version；
- β active/empty、location、sign、split/history 与 version；
- ambiguous ReLU relaxation 和 `A==0` endpoint policy；
- primal/bound topology、residual bias token、lineage；
- optimizer evaluation/mutation ordinal、atomic commit 与 rollback；
- effect、alias、dense-A lifetime、VJP owner 和合法性证明；
- rule registry、receipt 和 fail-closed decision。

### 3.2 TVM owner

TVM Relax/TIR 只在 BoundFlow legality 通过后拥有：

- dataflow graph lowering；
- approved op/rule fusion；
- TIR PrimFunc 生成、schedule 与 target codegen；
- module import/export identity；
- static workspace 和 device function ABI。

TVM 不得推断或改变 α/β/split/history/optimizer/branch 语义。

### 3.3 Runtime owner

BoundFlow runtime 拥有：

- prepared module/cache；
- persistent arena、slice/view/lease/epoch；
- pointer、device、stream identity；
- execution/status/commit receipt；
- coarse atomic commit 与失败时完整 rollback；
- warm-path O(1) identity counters。

PyTorch 在 candidate region 外保留 frontend、oracle、host solver 和 fail-closed fallback owner；candidate
region 内不得执行 PyTorch tensor math。

## 4. Verification Graph ABI v1

所有对象必须 canonical JSON、`allow_nan=false`、稳定排序并可计算 SHA-256。稳定 ID 不得包含 Python
object id、绝对路径、进程号或随机临时名。

### 4.1 `VerificationProgramV1`

冻结字段：

```text
schema_version
program_id
semantic_owner
source_graph_hash
parameter_schema_hash
numeric_policy_id
target_contract_id
region_ids[]
entry_region_ids[]
external_value_ids[]
external_effect_ids[]
rule_registry_hash
program_hash
```

约束：

- `semantic_owner="boundflow.verification-graph/v1"`；
- program ID 来源于 semantic content，不来源于 model filename；
- region/value/op/effect ID 全局唯一；
- entry region 可多于一个；GC-0 formal 只对一个代表性 closed lower region 做 schema
  construction、admit/lower ABI 与 canonical replay，不执行 production region；
- source/parameter/numeric/target identity 任一变化均使 program hash 变化。

### 4.2 `VerificationRegionV1`

冻结字段：

```text
region_id
op_ids[]
input_value_ids[]
output_value_ids[]
parameter_value_ids[]
external_use_ids[]
effect_input_ids[]
effect_output_ids[]
saved_state_ids[]
gradient_owner_ids[]
entry_op_ids[]
exit_op_ids[]
postdominator_witness
closed_world
fallback_policy
```

`closed_world=true` 只有在所有中间 value 无 region 外 consumer、所有 effect 可排序、所有 output 都有
显式 owner 时成立。`fallback_policy="reject-before-launch"`；运行中不得回退。

### 4.3 `VerificationValueV1`

每个 tensor/scalar/state value 冻结：

```text
value_id, role, shape, dtype, device_kind, layout, strides
axis_roles, polarity, representation, requires_grad
state_version, lineage_id, storage_class, alias_set
producer_op_id, consumer_op_ids, external_use_count
present, finite_policy
```

枚举至少包括：

- `role`: coefficient、bias-token、lower-bound、upper-bound、alpha、beta、split、history、parameter、
  optimizer-state、status、commit-token、scratch；
- `axis_roles`: domain、spec、channel、height、width、feature、beta-slot、direction；
- `polarity`: lower、upper、both、none；
- `representation`: dense、compressed-indexed、sparse-location、scalar、token；
- `storage_class`: external-borrowed、parameter-resident、arena-persistent、arena-scratch、saved-minimal、
  host-status。

empty β 必须是 first-class `shape[...,0]` 或显式 absent value，不能伪造 dense zero tensor。

### 4.4 `VerificationOpV1`

冻结字段：

```text
op_id, op_kind, semantic_version
input_value_ids[], output_value_ids[]
parameter_value_ids[]
effect_read_ids[], effect_write_ids[]
attributes
bound_direction
numeric_policy_id
vjp_contract_id
source_op_ids[]
```

GC-0 vocabulary 至少能表达：

- spec seed / incoming coefficient；
- compressed α gather；
- sparse/empty β injection；
- ReLU lower/upper relaxation 与 sign select；
- Linear/Conv2d right propagation；
- reshape/view/layout normalization；
- residual diamond/join；
- bias reduction/token accumulation；
- input concretization；
- minimal-state VJP；
- compact status 与 coarse commit。

未列入 vocabulary 的 op 必须在 admission 前拒绝。

### 4.5 `VerificationEffectTokenV1`

冻结 effect kind：

```text
alpha-state
beta-state
split-history
optimizer-state
domain-lineage
queue-state
commit-state
runtime-arena
```

每个 token 包含 `effect_id/kind/resource_id/input_version/output_version/access/ordinal`。read 可并行；write
必须单 writer 且拓扑排序。GC-0 candidate region 不拥有 queue/termination mutation，但必须将其声明为
外部 effect boundary，不能隐式忽略。

### 4.6 `VerificationVJPContractV1`

冻结字段：

- primal input/output value；
- incoming adjoint；
- α/β gradient owner 与 compressed output layout；
- saved/recomputed value 集合；
- endpoint/subgradient policy；
- higher-order policy=`reject`；
- dense-A escape policy=`forbid`；
- mutation policy=`none-inside-vjp`。

允许保存的状态只包括 compressed α/β、index/location/sign、必要的小型 activation/sign mask、参数/arena
identity 和 effect version。不得保存跨层 dense A 或 PyTorch autograd history。

## 5. Rule 与 legality ABI

### 5.1 `VerificationRuleV1`

每条规则冻结：

```text
rule_id, rule_version, pattern_kind
input_op_kinds[], output_op_kinds[]
semantic_guards[], shape_guards[], effect_guards[]
alias_guards[], external_use_guards[], vjp_guards[]
replacement_builder_id
estimated_boundary_elimination
estimated_materialization_elimination
fallback_policy
```

GC-0 只冻结 registry/schema 和 reference rewrite plan；正式 rule execution 属于 GC-1。不得在 GC-0
宣称 fusion 已完成。

首批 rule ID 冻结为：

- `V-R1-relax-sign-affine-v1`；
- `V-R2-compressed-alpha-gather-v1`；
- `V-R3-sparse-beta-inject-v1`；
- `V-D1-residual-diamond-v1`；
- `V-C1-terminal-concretize-v1`；
- `V-VJP1-minimal-saved-state-v1`；
- `V-M1-certified-arena-reuse-v1`。

CIBC lower/upper tuple rule `V-H1` 进入 registry，但 GC-0 formal lower-only production instance不授予其
rewrite/performance claim。

### 5.2 `LegalityResultV1`

analysis-only pass 不编译、不执行、不计时，输出：

```text
admitted
region_id
ordered_op_ids[]
boundary_input_ids[]
boundary_output_ids[]
external_use_witnesses[]
effect_order_witnesses[]
alias_witnesses[]
dense_escape_witnesses[]
vjp_witnesses[]
rejection_reasons[]
analysis_hash
```

`admitted=true` 时 rejection 必须为空且所有 witness 完整；`false` 时至少一个稳定拒绝原因。

### 5.3 冻结拒绝原因

至少实现并专项测试：

1. `UNSUPPORTED_OP_KIND`；
2. `DYNAMIC_SHAPE_UNBOUND`；
3. `DTYPE_OR_DEVICE_MISMATCH`；
4. `LAYOUT_NOT_NORMALIZABLE`；
5. `REGION_EXTERNAL_USE`；
6. `REGION_NOT_POSTDOMINATED`；
7. `STATE_VERSION_MISMATCH`；
8. `EFFECT_ORDER_CONFLICT`；
9. `ALPHA_START_NODE_MISMATCH`；
10. `ALPHA_INDEX_OR_DIRECTION_MISMATCH`；
11. `BETA_ACTIVE_EMPTY_MISMATCH`；
12. `BETA_LOCATION_SIGN_HISTORY_MISMATCH`；
13. `BOUND_POLARITY_MISMATCH`；
14. `ENDPOINT_POLICY_MISMATCH`；
15. `RESIDUAL_BIAS_TOKEN_UNCLOSED`；
16. `UNSAFE_ALIAS_OR_LIFETIME`；
17. `DENSE_A_ESCAPE`；
18. `VJP_OWNER_OR_SAVED_STATE_MISMATCH`；
19. `HIGHER_ORDER_GRAD_UNSUPPORTED`；
20. `QUEUE_OR_TERMINATION_EFFECT_CROSSED`；
21. `RUNTIME_FALLBACK_REQUIRED`；
22. `RECEIPT_IDENTITY_MISMATCH`。

## 6. Relax/TIR lowering ABI

### 6.1 `VerificationLoweringRequestV1`

绑定：program/region/analysis/rule-registry/numeric-policy/target/GPU/TVM source hashes；输入必须是
admitted region，且不得携带 runtime tensor payload。

### 6.2 Relax ABI

Relax function 的逻辑参数顺序冻结为：

```text
external coefficients/status inputs
compressed alpha/beta/split inputs
resident parameters
arena handle + slice table
effect versions
```

逻辑返回为：

```text
arena-backed output views
compressed alpha/beta VJP views
bias/lower status
effect/commit receipt
```

实际 device tensor 不允许通过 Python list/dict 动态返回；返回结构在 admission 时绑定。

### 6.3 TIR ABI

每个 PrimFunc 必须声明：

- read/write region；
- alignment、layout、dtype、shape；
- alias/noalias 与 in-place 条件；
- stream/device contract；
- workspace slice；
- semantic/rule ID provenance；
- forward/backward/epilogue role。

GC-0 不冻结具体 thread/block winner，但 schedule policy ID、TIR hash 与 device source hash 必须进入 receipt。

### 6.4 `LoweringReceiptV1`

至少绑定：

```text
program_hash, region_hash, analysis_hash, rule_registry_hash
relax_module_hash, tir_module_hash, schedule_hash, device_source_hash
arena_plan_hash, vjp_contract_hash, module_receipt_hash
target_triple, gpu_arch, tvm_commit, tvm_ffi_commit
numeric_policy_id, source_revision
timing_recorded=false, performance_claimed=false
```

replay 侧必须重新构造 graph、重新做 legality、重新 lower/compile 并逐层比 hash；不能只校验 receipt 格式。

## 7. Physical arena 与 prepared runtime ABI

### 7.1 `PhysicalArenaPlanV1`

冻结字段：

```text
arena_id, device_kind, dtype_partitions
total_bytes, alignment, high_water_bytes
slices[]
live_intervals[]
alias_sets[]
happens_before_edges[]
lease_policy, epoch_policy
saved_state_ledger, dense_escape_count
```

每个 slice 包含 value、offset、size、alignment、storage class、first/last use、alias set、recompute policy、
read/write owner。offset/size 重叠只有在 live interval 不重叠且规则显式批准时合法。

### 7.2 `PreparedVerificationRegionV1`

prepared runtime 必须：

- admission/compile 时一次绑定 module、parameter、arena 和 views；
- 每次 outer evaluation 只更新 dynamic compressed state 和 version；
- warm path 不创建 per-op DLPack view，不分配 output tensor；
- device/current stream 双向 identity 验证；
- launch 前验证 O(1) program/module/arena/state version；
- launch 后只返回 compact status/commit token；
- 任一错误在 commit 前 fail closed，provider state 完整 rollback。

### 7.3 FCR-1 最终结构目标

这是 GC-2 关闭时的 correctness/architecture acceptance，不是 latency claim。GC-0 只冻结这些字段并
验证 symbolic plan/identity，不得声称真实 runtime 已满足：

- host→compiled-region submission：每次 evaluation 最多 1 次 forward，每次 mutation 最多 1 次 backward；
- 10/9 outer 总 submission=`10/9`，不再是三 site `30/27`；
- warm per-op DLPack/pointer crossing=`0`；
- warm dynamic output allocation=`0`；
- candidate region PyTorch tensor op=`0`；
- fallback/eager/native shadow=`0/0/0`；
- persistent dense-A count=`0`；
- saved dense-A count=`0`；
- arena/pointer/module/stream identity 全轨迹不漂移；
- internal TIR kernel 数单独披露，不把一次 region submission 伪写为一个 kernel。

若最终 lowering/runtime 不能达到上述结构目标，GC-2/FCR-1 correctness 不得关闭，也不得进入 timing。

## 8. 首个 vertical slice 与 workload

### 8.1 通用 scope

首个 region 必须从 typed production state 开始，以 compact status/coarse commit 结束，并包含：

```text
compressed alpha/beta lookup
→ ReLU relaxation/sign
→ layout normalization
→ Linear/Conv propagation
→ residual/bias/reduction/epilogue
→ input concretization
→ minimal-saved-state VJP
→ persistent arena output/status
```

host optimizer、termination 和 queue 仍在 region 外，但其 state version/effect/commit 必须显式建模。

### 8.2 三类必须通过的 signature

1. `P-empty-beta`：Conv P-anchor，真实 empty β `[D,0]`；验证 lower、compressed dα、10/9 optimizer trajectory；
2. `S-active-beta`：Linear S-anchor，active β、location/sign/split/history 与 nonzero dβ；不得借 P 性能或
   empty-β specialization；
3. `multi-conv-trajectory`：C2→C1→C0 closed region，三类 stride/shape，10 evaluation/9 mutation，
   residual/bias token 与 atomic rollback。

P/S/C0/C1/C2 只属于测试 instance；schema、rule、lowering 和 runtime 代码不得引用这些名字。

### 8.3 负向 workload

每个拒绝原因至少有一个最小 synthetic graph；此外必须覆盖：

- external consumer 出现在 region 外；
- residual diamond 缺一支或 bias token 丢失；
- α start-node、direction、spec index、feature index 或 version 被改；
- β active↔empty、location、sign、history 被改；
- effect write 重排或 mutation ordinal 重复/缺失；
- arena offset、alignment、lifetime、alias set 被改；
- saved dense A 或 PyTorch autograd tensor 被加入；
- source/module/TIR/schedule/stream/device identity 被改；
- forward/backward/commit 次数多一或少一；
- candidate/reference 同时篡改并重签外层 digest。

## 9. Correctness formal 协议

### 9.1 source freeze 与顺序

1. 独立外审批准本预注册；
2. 以批准的 plan commit 作为实现父节点；
3. `GC0-0` 至 `GC2-2` 按第11节分阶段DAG逐刀单独提交，阶段间必须先关闭和外审；
4. formal generator source 在 raw 生成前冻结；
5. raw-first，从 position 0 开始；任何部分结果不得 resume；
6. 五组 fresh process，每组同时执行 production oracle、independent closed-form oracle 和 candidate；
7. 固定 order=`PCM/CPM/PCM/CPM/PCM`，其中 P=production oracle、C=candidate、M=mathematical
   oracle；M不计时；
8. 本阶段不得记录或输出 latency 字段。

### 9.2 独立 oracle

需要两层独立 reference：

- production oracle：冻结 auto_LiRPA/PyTorch exact call；
- mathematical oracle：float64、无 autograd 的 closed expression，手写 Linear/Conv transpose、ReLU
  relaxation、β injection、bias reduction、residual 与 concretization；VJP 用闭式 adjoint 重算。

candidate 不得调用 production oracle 或 closed-form helper；replay 不得信任 raw 中已有 reference tensor，
必须从冻结 input/state 重算。

### 9.3 数值与离散门禁

所有 finite float32 输出与两个 oracle 比较：

- `atol=2e-4`；
- `rtol=1e-5`；
- sign exact；
- NaN/Inf 任一出现即失败；
- lower≤upper/shape/dtype/device/layout exact；
- compressed α/β gradient 与 owned index/location exact；
- unowned α/β gradient exact zero；
- split/history/domain lineage、evaluation/mutation ordinal、branch/termination-visible status exact；
- 10 次 lower、9 次 dα/dβ、α/β、Adam moments、clamp 与最终 commit 全轨迹逐步比较；
- evaluation 5 注入失败后，所有 mutable state、arena epoch 和 commit version exact rollback。

不得因 observed diff 较小而事后收紧/放宽容差；formal summary 同时披露最大绝对/相对差和 sign count。

### 9.4 分阶段结构门禁

#### 9.4.1 GC-0 ABI/analysis/lowering 门禁

- GC0-0：三 signature 均可由通用 schema 构造并 canonical round-trip；这不是 legality admission或执行；
- GC0-0：22 类 rejection enum/schema 完整，直接可触发子集全部 fail closed；
- GC0-1：三 signature 的 analysis-only legality 与全部 22 类 negative graph 才要求逐项执行；
- GC0-2：program/region/analysis/rule/lowering/TIR/schedule/symbolic-arena/module hash 稳定；
- GC-0 全阶段不得记录 timing，不得执行 production candidate region。

#### 9.4.2 GC-1 semantic/VJP 门禁

五组 fresh 均必须满足：

- 三 signature 全 admitted，所有 GC-1 semantic/VJP negative graph 全 rejected；
- 双 oracle、lower/sign、compressed dα/dβ、state/trajectory/rollback 满足 §9.3；
- production outer lifecycle=`10 evaluation/9 mutation/1 commit`；
- fallback/eager/native shadow=`0/0/0`；
- semantic/rule/lowering/module identity 全组稳定。

#### 9.4.3 GC-2 physical runtime 门禁

五组 fresh 均必须满足：

- region submission=`10 forward/9 backward`；
- warm per-op crossing/dynamic allocation/PyTorch tensor op=`0/0/0`；
- saved/persistent dense A=`0/0`；
- current device/stream、module、arena pointer、slice offset、lease、epoch按合同保持；
- internal kernel、global bytes、workspace 和 arena high-water 完整披露，但不形成性能判断。

## 10. Artifact、replay 与 tamper

### 10.1 artifact v1

目录至少包含：

```text
manifest.json
protocol.json
code_revision.json
program.json
regions.jsonl
legality.jsonl
rules.jsonl
lowering_receipts.jsonl
arena_plans.jsonl
raw.json
summary.json
replay_stdout.txt
tamper_report.json
```

manifest 绑定所有 payload SHA-256、source/generator commit、三个外部仓库 commit、Python/Torch/TVM/
CUDA/GPU 环境。payload 不得含 `/home/`、用户名、临时目录或未解析环境变量。

### 10.2 semantic replay

replay 必须从 raw frozen input/state：

1. 重建 `VerificationProgram/Region/Value/Op/Effect`；
2. 重跑 analysis-only legality；
3. 重建 rule registry/rewrite plan；
4. 重做 Relax/TIR lowering 与 module compile；
5. 重建 arena plan、prepared views 和 VJP contract；
6. 重算两个 oracle 与 candidate outputs；
7. 重算 trajectory、structure gates、summary 与所有 hash；
8. stdout 与冻结 `replay_stdout.txt` 逐字节一致。

只比较 JSON、只重算 outer digest 或只检查 module receipt 格式均不合格。

### 10.3 fully re-signed tamper

至少 22 类，每类修改内层 payload 后重算所有外层 digest：

1. op kind/semantic version；
2. bound polarity；
3. α start-node/direction/spec index；
4. α feature index/version；
5. β active/empty；
6. β location/sign/history；
7. external use/postdominator；
8. effect order/version；
9. residual branch/bias token；
10. VJP owner；
11. saved dense A；
12. higher-order policy；
13. arena offset/size/alignment；
14. arena lifetime/alias/lease/epoch；
15. Relax hash；
16. TIR/device source/schedule hash；
17. module/source revision；
18. stream/device/pointer identity；
19. launch/submission count；
20. fallback/eager/PyTorch-op count；
21. trajectory float/reference 与 candidate 同改；
22. `timing_recorded` 或 `performance_claimed` 改为 true。

必须 `22/22 rejected`；不得依赖未重签的 outer digest 轻易拒绝。

## 11. 分阶段提交 DAG

实现只能按以下顺序；每阶段关闭和外审后才允许写下一阶段的实现预注册：

1. `GC0-0 schema`：通用 graph/effect/rule/legality 类型、完整22类rejection enum/schema，以及无需
   analysis pass即可独立触发的constructor/identity/fallback/polarity/VJP负例；依赖拓扑、
   postdominator、effect-order或alias analysis的negative graph明确留到GC0-1；
2. `GC0-1 capture-analysis`：从现有 Bound/Task/R3 capture 构建通用 graph，analysis-only legality；
3. `GC0-2 lowering-arena-abi`：Relax/TIR request、receipt、symbolic arena/lease/epoch identity、
   semantic replay；不执行 production region；
4. `GC0-3 closure`：schema/legality/lowering-ABI artifact、tamper、独立外审与 formal closure；
5. `GC1-0 prereg`：另行冻结 admitted rules、closed region 与 custom VJP correctness 实验；
6. `GC1-1 rewrite-vjp`：P empty-β、S active-β、multi-site 10/9 的 guarded rewrite/module/VJP correctness；
7. `GC1-2 closure`：five-fresh 双 oracle、trajectory、replay/tamper、独立外审；
8. `GC2-0 prereg`：另行冻结真实 physical arena/prepared runtime 结构实验；
9. `GC2-1 arena-runtime`：真实 arena、persistent views、coarse commit 与结构计数；
10. `GC2-2 closure`：结构 artifact/replay/tamper、独立外审；完成后才允许另写 timing plan。

不能合并为一个事后优化提交。每刀均需 targeted/full regression、Black/Mypy/Pylint、DocOps change/
validation/lint；任何代码实现前必须确认其父提交包含已批准且未修改的本文。

## 12. Acceptance criteria

本文把 acceptance 分成不可跳级的三层。GC-0 只由 Plan-AC1—Plan-AC3 关闭；GC-1 只由
Plan-AC1—Plan-AC3 加 Plan-AC5/Plan-AC6 中相应 correctness 项关闭；GC-2/FCR-1 最终关闭才要求
Plan-AC1—Plan-AC7 全部满足。外部 exchange 中的验收项统一写作 Audit-AC，禁止与本节重名。

### Plan-AC1 — 预注册与范围

- plan 在实现前提交并外审批准；
- schema/代码不硬编码模型/site；
- P empty-β、S active-β、multi-site 10/9 三 signature 全覆盖；GC0-0/GC-0关闭时只要求通用schema
  construction与canonical round-trip能表达三者，legality admission和production execution分别属于
  GC0-1与GC-1；
- timing/performance 字段强制 false。

### Plan-AC2 — Graph/legality

- 通用 program/region/value/op/effect/VJP schema canonical/stable；
- closed-world、external use、postdominator、state/effect/alias/dense escape 都有 witness；
- GC0-0完整冻结22类稳定拒绝枚举并测试无需analysis的直接子集；GC0-1关闭时才要求22类negative
  graph全部由analysis-only legality逐项拒绝。

### Plan-AC3 — Lowering identity

- Relax/TIR/module/schedule/target/source 全链路 hash；
- replay 重新 lower/compile 后逐层一致；
- unsupported/different identity 在 launch 前拒绝。

### Plan-AC4 — Physical runtime

- arena 是真实运行 storage，不是只读 ledger；
- persistent view/pointer/lease/epoch identity 成立；
- warm per-op crossing、dynamic output allocation、PyTorch tensor op 均为 0；
- region submission 精确 `10/9`，fallback/eager/native shadow=0；
- saved/persistent dense A=0。

### Plan-AC5 — Semantics/VJP/trajectory

- 两个独立 oracle；
- 五组 fresh 三 signature 数值、sign、owned/unowned gradient、state/effect/trajectory/rollback 全过；
- max diff 在冻结容差内，离散值 exact。

### Plan-AC6 — Artifact/replay/tamper

- raw-first、manifest/source/environment 完整、零本机路径；
- semantic replay 重建 graph→compile→arena→oracle→summary；
- `22/22` fully re-signed tamper rejected。

### Plan-AC7 — 工程质量与 claim

- targeted 与 full test 通过；skip 逐项为冻结环境边界；
- Black/Mypy/Pylint/diff/DocOps lint 通过；
- authority docs 无 claim 漂移；
- `timing_open=false/performance_claimed=false/ASPLOS-ready=false`。

## 13. GO、NO-GO 与后继

### 13.1 GO

GC-0 的 Plan-AC1—Plan-AC3 通过后，状态只能是：

```text
VALIDATED-GC0-VERIFICATION-GRAPH-ABI
```

它只开放 GC-1 correctness 预注册，不开放 GC-1 实现、GC-2 或 timing。

GC-1 correctness 独立关闭后，状态只能是：

```text
VALIDATED-GC1-FCR1-GUARDED-REGION-CORRECTNESS
```

它只开放 GC-2 physical runtime 预注册。最终 Plan-AC1—Plan-AC7 全部通过后，状态只能是：

```text
VALIDATED-FCR1-VERIFICATION-GRAPH-ABI-CORRECTNESS
```

最终状态只开放：

- `GC-3` wrapper-inclusive timing 的独立预注册；
- 后续 optimizer region 仍须单独 correctness 预注册。

它不直接开放 GC-3 CUDA Graph、same-solver、query、queue 或性能 claim。

### 13.2 NO-GO/INVALID

以下任一成立即停止当前实现：

- schema 或执行路径仍硬编码 ResNet/site；
- graph 只是薄 wrapper，candidate 内仍逐 op PyTorch/DLPack/allocation；
- external use/effect/postdominator/alias 不能由 analysis-only pass 证明；
- dense A 跨 region 或进入 saved state；
- active-β、residual 或 10/9 trajectory 不等价；
- arena 只是账本、prepared view identity 不真实；
- replay 不能重 lower/compile，或 fully re-signed tamper 有一项未拒绝；
- 通过改变 workload、删步骤、放宽容差或借用局部数字才能过门禁。

语义/身份/证据失败为 `INVALID-GC0-FCR1`；架构范围完整但结构目标失败为
`VALIDATED-NO-GO-GC0-FCR1-ARCHITECTURE`。两者都不得进入 timing。

## 14. 外部审计问题

审计方必须独立回答：

1. schema 是否真正与模型/site 无关，还是把硬编码藏进 attrs/hash；
2. effect/alias/postdominator/external-use 是否足以证明 region closed；
3. active-β 与 empty-β 是否是两个真实 specialization；
4. minimal saved state 是否仍通过别名保留 dense A/autograd history；
5. Relax/TIR receipt 是否由重编译校验，而非格式检查；
6. physical arena 是否真实承载执行 storage；
7. 10/9 submission/crossing/allocation/PyTorch-op 结构计数是否从 raw 独立重算；
8. 两个 oracle 是否独立于 candidate；
9. replay 是否能拒绝 candidate/reference 同改且完全重签的攻击；
10. 是否存在 timing、speedup、query/queue 或 ASPLOS claim 漂移。

## 15. 证据入口

- `gemini_doc/BOUNDFLOW_MR7R_UNPROFILED_HOST_RECOVERY_FORMAL_CLOSURE_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_MR7_GRAPH_COMPILER_RULE_RUNTIME_RESEARCH_PLAN_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_FULLY_COMPILED_VERIFIER_RUNTIME_V1_ARCHITECTURE_2026_08_25.md`
- `gemini_doc/BOUNDFLOW_MR5_MULTI_SITE_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_R3_3_ACTIVE_BETA_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md`
- `boundflow/ir/task_v1.py`
- `boundflow/ir/differentiable_lower_region.py`
- `boundflow/ir/r3_bounded_arena.py`
- `boundflow/runtime/r3_structured_owner_custom_backward.py`
- `boundflow/runtime/task_backend_dispatch.py`

## 16. 当前唯一下一步

先对本文做独立外审。批准前不写 `GC0-0` 代码；批准后只从通用 schema、analysis-only legality 和拒绝
测试开始。任何人不得跳到 TIR 调优、CUDA Graph 或 timing。
