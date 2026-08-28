---
status: diagnostic-complete-code-closed
date: 2026-08-28
type: diagnostic
topic: boundflow
slug: asplos27-s4-evaluator-abi-terminal-handoff
stage: s04
execution-authority: false
code-change-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4：all-state evaluator ABI与terminal handoff实施蓝图

## 0. 结论

S4 candidate热路径应直接拥有production compressed lower-direction α与sparse β，不应继续以RVIR reference的
dense native α/β作为optimizer参数。dense native state只保留两种用途：

1. S4-1/S4-2独立oracle；
2. ordinal 9结束后一次性展开，供现有KFSB与atomic copy-out兼容边界消费。

每个evaluation的ABI固定为一个ordered state tuple、一个ordered gradient tuple和一个lower。ordinal 9额外返回
one-shot lower/lA handoff；ordinal 0—8必须没有handoff。不得在热路径构造按site Python dict。

terminal lA与六site coefficient-adjoint values的物理元素数相同，均为37,464个float32。推荐用同一组
phase-tagged slots：pass B结束时slot是`V_i=d lower/dT_i`；pass C在某site导出gradient后，ordinal 9才允许用
ReLU transform前的incoming lA覆盖已经消费完的V。handoff必须恢复`[D,S,*feature]`typed view。这样无需第三个
CROWN pass，也无需新增149,856 bytes峰值。

## 1. 三种state representation不能混为一谈

### 1.1 production compressed source

冻结snapshot中有六个α source tensor，总stored元素8,496：

```text
[2 directions, 1 start-spec, 6 domains, compressed width]
```

但当前lower-only copy-out只覆盖`source[0,0]`；`source[1,0]`从pre snapshot原样保留。因此：

- stored α：8,496 float32；
- optimizer-active lower α：`708 widths × 6 domains = 4,248` float32；
- preserved α direction：4,248 float32；
- optimizer-active sparse β：6 float32；
- empty β：五个合法零宽view。

P-anchor对应stored 1,032、optimizer-active 516。无论按stored还是active统计，P coverage均为
`86 / 708 = 12.1468926554%`。这些仍只是state coverage，不是runtime share。

### 1.2 RVIR native dense reference

`initialize_rvir_v4_native_pre_state`把compressed source scatter成六个dense tensor：

```text
6 × (2048 + 1024 + 1024 + 1024 + 1024 + 100) = 37,464 alpha
37,464 beta
37,464 int8 split
```

`execute_rvir_v4_native_optimizer_trace`在dense α/β上运行Adam，最终`_project_alpha/_project_beta`再投影回12条
production path。这条路线已经有provider parity，是极强oracle；但若作为S4 candidate热路径，会带来dense moments、
每evaluation compressed↔dense binding或修改现有TIR ABI，违背representation保持目标。

### 1.3 S2/S3 compiled owner

S2/S3直接绑定production compressed tensor和layout map。P-only S3已经证明host Adam可以直接修改compressed tensor，
compiled forward/backward无需dense optimizer state。S4应把这一模式推广到全部六site，而不是退回dense owner。

## 2. S4 candidate state owner

### 2.1 static slot descriptor

每个site由admission阶段生成一个`ProductionMutableSlotV1`等价对象，至少冻结：

```text
slot_ordinal
native_preactivation
provider_activation / provider_preactivation / start_node
alpha_semantic_path / beta_semantic_path
alpha_source_shape                 # [2,1,6,width]
alpha_optimizer_buffer_shape       # [6,width]
alpha_mutable_slice                # exact [0,0]
alpha_preserved_slice_hash         # exact [1,0]
feature_shape / feature_indices
beta_source_shape                  # [6,q]
beta_locations / beta_signs
parameter_group                    # alpha or beta
```

slot按topology/semantic path确定性排序。模型名、固定node id和固定shape只能出现在formal instance，不得进入通用
schema legality。

### 2.2 dynamic optimizer owner

prepared runtime持有：

- 六个contiguous lower-α parameter buffer，总4,248元素；prepare时从source精确pack，commit时按slot精确copy-out；
- 一条active β parameter buffer，共6元素；
- 五个empty β slot（不得为便于实现伪造非空参数）；
- preserved α direction存在于original live full-source Tensor并由private lease强引用，同时由immutable snapshot/receipt绑定
  digest；它不进入candidate GPU optimizer；
- S4-2由sealed policy driver建立两个Adam param group，`lrα=0.01/lrβ=0.05`；
- scheduler decay=`0.98`；
- S4-1A先建立persistent gradient views，optimizer moments在S4-2 prepare。

这里的parameter buffer不要求是原始`[2,1,6,width]`source tensor的PyTorch leaf view；生产合同只要求
pack/optimizer/copy-out三段在semantic path、slice和identity上完全闭合。这样避免把非leaf view误当Adam参数，
也让compiled evaluator获得稳定连续地址。

按当前fixture的设计账：

| 项 | 元素 | bytes(float32) |
|---|---:|---:|
| active lower α | 4,248 | 16,992 |
| active β | 6 | 24 |
| dα+dβ | 4,254 | 17,016 |
| Adam m+v（active参数） | 8,508 | 34,032 |

preserved α另有4,248元素/16,992 logical bytes；source live lease实际强引用12条existing CUDA Tensor、8,502元素/
34,008 logical bytes，incremental allocation=0。该账不计入candidate new parameter/gradient/moment，但必须披露
lifetime retention并在S4-3后close。这是静态设计账，不是实测显存claim。step scalar、allocator元数据、module workspace和terminal
arena另行披露。ordered buffer、empty β token、lease/version与DLPack纪律见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_ORDERED_BUFFER_ABI_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`。

### 2.3 terminal dense bridge

ordinal 9后，candidate把compressed terminal α/β一次性scatter到预分配dense native state，供existing KFSB和
atomic copy-out使用。桥必须证明：

- compressed→dense→compressed round-trip exact；
- unowned dense位置为冻结默认值，不参与production copy-out；
- preserved α direction digest未变；
- active β location/sign exact；
- bridge count=`1`，ordinal 0—8为0；
- dynamic allocation与D2H copy为0。

S4 correctness先保留existing dense terminal consumer。后续若KFSB改为直接消费compressed state，必须另立门禁，
不能在S4中顺手改语义。

## 3. sealed evaluator ABI

### 3.1 request

逻辑接口冻结为：

```text
AllStateCrownEvaluationRequestV1:
    evaluation_ordinal
    schedule_action_hash
    mutable_state_version_hash
    plan_instance_hash
    terminal_mode              # NONE or LOWER_ADJOINT_HANDOFF
```

`terminal_mode`不是任意bool：只能由sealed schedule action推导。ordinal 0—8必须为`NONE`，ordinal 9必须为
`LOWER_ADJOINT_HANDOFF`；调用方不能跳过或提前请求handoff。

### 3.2 input/output ABI

prepared evaluator持有固定ordinal buffer，不在每次调用接收dict：

```text
PreparedAllStateCrownEvaluatorV1.evaluate(request)
  -> AllStateCrownEvaluationResultV1

result:
    composite_result_lease:
        lower_view                    # [6,1]
        alpha_gradient_views          # ordered six [6,width]
        beta_gradient_slots           # one physical + five typed token
        terminal_child_transfer       # terminal only, one-shot
        execution_receipt
```

gradient view直接交给host optimizer，禁止clone后再赋值。empty β保持typed zero-width metadata token，不创建物理
Tensor/view，也不得用`None`或补零tensor代替。
所有view在prepare时完成DLPack/TVM绑定，warm invocation只更新版本与launch counters。

这里的result是一个composite lease：任一子view都不能独立释放后被下一evaluation重写。S4-1D formal每个fresh owner只执行
一次evaluation；terminal child只可transfer一次，parent可先close但child close之前arena仍存活。read-only request拒绝不改
state；一旦进入`EVALUATING`，任何失败均为`POISONED_NO_RETRY`，不得reset generation后复用半写arena。精确状态机见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_EVALUATOR_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`。

### 3.3 policy driver

不能把任意callback塞进`execute_rvir_v4_native_optimizer_trace`。应抽出sealed production policy driver，只有两个
exact evaluator实现：

1. `NativeDenseOracleEvaluatorV1`：现有RVIR dense reference；
2. `CompiledCompressedEvaluatorV1`：S4 candidate。

driver唯一拥有10/9 ordinal、two-param-group Adam、scheduler、clamp、stop/prune/keep-best语义。candidate evaluator
只产生lower/gradient/handoff，不修改parameter、moment、scheduler或live provider tensor。

S4-2比较的是两个driver execution的每步投影结果，不要求两个内部representation相同；要求production-visible
compressed α/β、lower、policy decision和terminal state逐步等价。

## 4. terminal handoff exact contract

### 4.1 现有B4-A事实

B4-A已经验证ordinal 9同一次evaluation返回lower与六个native lA：

| native site | lA shape | 元素 |
|---|---:|---:|
| 17 | `[6,1,8,16,16]` | 12,288 |
| 19 | `[6,1,16,8,8]` | 6,144 |
| 23 | `[6,1,16,8,8]` | 6,144 |
| 25 | `[6,1,16,8,8]` | 6,144 |
| 28 | `[6,1,16,8,8]` | 6,144 |
| 31 | `[6,1,100]` | 600 |

合计37,464 float32/149,856 bytes。冻结artifact证明：handoff count=`1`、terminal export CROWN rerun=`0`、
one-shot lease与lineage hash均存在。

### 4.2 phase-tagged arena复用

S4 pass B产生的六个effective preactivation与上述lA逐site同shape。推荐每slot状态机：

```text
EMPTY
  → EFFECTIVE_VALUE_READY
  → GRADIENT_CONSUMED
  → TERMINAL_LA_READY       # 仅ordinal 9
  → LEASED
  → RELEASED
```

pass C按`31→28→25→23→19→17`到达site。只有该site gradient已经消费coefficient adjoint后，terminal模式才可把
incoming coefficient写回同slot。非terminal模式不得生成或泄漏lA。

必须有结构测试证明任何site都没有“先覆盖coefficient adjoint、后计算gradient”的别名错误，也没有post-transform
lA copy或spec-axis丢失。若lifetime分析失败，先使用
独立预分配terminal arena保证正确性，并如实披露额外149,856 bytes；不得以错误alias换memory数字。

### 4.3 intermediate bounds不属于candidate输出

`NativeBackwardExportV4.intermediates`来自optimizer入口的shared external `relu_pre`，α/β mutation期间不变化。
existing no-CROWN assembly把这些bounds与terminal lower/lA组合成export。因此candidate terminal handoff无需重算或返回
12个intermediate lower/upper tensor，只需绑定其source/version hash。

### 4.4 handoff消费顺序

```text
ordinal 9 compiled evaluation
  → lower + six lA one-shot lease
  → no-CROWN NativeBackwardExport assembly
  → KFSB score/candidate evaluation
  → live core return assembly
  → 12-path α/β atomic commit + host packet
```

lA只供KFSB读取；最终provider return的`batched_lA`在当前fixed path中是empty。lA lease可在KFSB/assembly后释放，
不得延长到solver queue生命周期。

hot assembly不得对lower/lA重复`clone()`；若existing B4-A assembly的clone仍存在，S4-3必须新增pointer-safe immutable
lease路径或把clone成本单独计入，不能声称zero dynamic allocation。

## 5. KFSB是S4之后仍存在的性能边界

existing KFSB对3个candidate分别执行一次child CROWN；每次batch=`6×4=24`，共得到3个`[24,1]`child lower，
合计72个child结果。它目前调用BoundFlow PyTorch/native CROWN，而不是S4 compiled evaluator。

S4 correctness保持KFSB不变是正确的ownership切分，但S4-P必须单独测量：

- optimizer 10次compiled evaluation；
- terminal handoff/assembly；
- KFSB score；
- 3次batch-24 child CROWN；
- atomic commit与queue/post。

如果KFSB child CROWN成为新瓶颈，合法后继应是“three-candidate child batch compiled evaluation”，复用S4 evaluator
但使用expanded batch/state。它不是S4-1/S4-2的隐含工作，也不得在S4 headline里被排除。

## 6. 新增fail-closed原因

除主预注册已有reason外，S4实现至少新增：

1. `ALPHA_MUTABLE_DIRECTION_MISMATCH`；
2. `ALPHA_PRESERVED_DIRECTION_DRIFT`；
3. `COMPRESSED_NATIVE_ROUND_TRIP_MISMATCH`；
4. `DENSE_BRIDGE_BEFORE_TERMINAL`；
5. `TERMINAL_SLOT_PHASE_MISMATCH`；
6. `EFFECTIVE_VALUE_OVERWRITTEN_BEFORE_GRADIENT`；
7. `TERMINAL_LA_INVENTORY_INCOMPLETE`；
8. `TERMINAL_LA_LEASE_REUSED`；
9. `INTERMEDIATE_SOURCE_VERSION_MISMATCH`；
10. `HOT_HANDOFF_CLONE_OR_ALLOCATION_OBSERVED`。

## 7. 开工顺序修正

S3外审批准后：

1. S4-0先交付tensor-free compressed slot receipt及不可序列化strong-ref live lease；
2. S4-1A从current provider mapping逐对象重验并单次接管lease，然后绑定独立leaf lower-α/active-β buffers、empty β
   token、persistent gradients和ordered ABI；prepare采用validation→local staging→single-transfer，异常逆序清理并
   `FAILED_CLOSED`；lease持续到S4-3 commit/abort，不在pack后丢弃；
3. S4-1B0关闭Ainput zero→center三元endpoint，S4-1B/1C完成六V与六路gradient；
4. S4-1D完成single-evaluation closure；
5. S4-2A抽出sealed policy driver，以native dense evaluator回归原行为；
6. S4-2B接入compiled compressed evaluator，逐step比较production-visible state；
7. S4-3接terminal dense bridge、phase-tagged lA lease、existing KFSB/commit；
8. S4-4 artifact/replay/tamper；
9. 另立S4-P timing，再决定是否开放compiled KFSB child batch。

S4-1D唯一prepared evaluator、修正后的438,726-byte logical correctness ledger、5+5 fresh/full-IEEE raw/replay/tamper
门禁见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_ALL_STATE_EVALUATOR_CLOSURE_BLUEPRINT_2026_08_28.md`。
事务实施冻结与旧账`52,014 B`漏项纠正见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_EVALUATOR_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`。

S4-2 sealed driver的live keep-best/stop/patience/pruning、functional Adam、`10/9/10`
evaluation/update/scheduler-call与trajectory artifact合同见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_SEALED_PRODUCTION_POLICY_DRIVER_BLUEPRINT_2026_08_28.md`。

S4-3 whole-core exact-call的provider return constructor、official post、host packet、intermediate-container、device commit
以及`POISONED_NO_RETRY`失败语义见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_EXACT_CALL_TRANSACTION_BLUEPRINT_2026_08_28.md`。existing KFSB/
commit可复用其数学与owner，但不能直接继承“完全原子”措辞。

provider net α/intermediate/lA scratch的consumer、terminal 36-path transfer、B0 post-KFSB batch-24 residue、R/C
post-native-KFSB 36-path normalization由`ProviderNetScratchFinalizationPlanV2`表达；storage alias/unique bytes、β variant policy及
query-scoped exclusive core-owner latch见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_3A_PROVIDER_NET_SCRATCH_CONSUMER_AUDIT_2026_08_28.md`。这些是logical
lifetime transaction，不是terminal ABI的第13+条production数值path。

S4-4正式证据链、18 fresh B0/R/C六全排列、stdlib tensor raw/replay、68类fully re-signed tamper及
`COMMITTED_POST_FAILED_POISONED`见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_4_FORMAL_ARTIFACT_REPLAY_TAMPER_CLOSURE_BLUEPRINT_2026_08_28.md`。

## 8. 当前门禁

本文只是implementation blueprint。S3 exchange仍为`ready_for_audit`，没有external approval；因此S4代码、GPU
correctness和timing仍关闭。本稿不形成性能、same-solver或complete-query claim。
