---
status: implementation-construction-ready-gate-closed-v1
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-1d-all-state-evaluator-construction
stage: s04
depends-on: s3-external-approval-and-s4-0-s4-1a-s4-1b0-s4-1b-s4-1c-closure
execution-authority: false
code-change-open: false
gpu-correctness-open: false
formal-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1D all-state single-evaluation evaluator 实施施工包

## 0. 直接结论

S4-1D不增加新的bound算法、solver IR或TIR数学；它把S4-0—S4-1C已经冻结的owner、buffer、selector、
selected-value、compressed gradient与terminal lA组合成唯一的single-evaluation prepared runtime事务。

本次逐文件施工审计纠正了旧S4-1D readiness的四个实质问题：

1. **raw预算口径错误**：`919,680 B`只是5+5 candidate output，不是A/B/C三方完整raw；
2. **worker拓扑不足**：每类5个worker无法覆盖A/B/C六全排列，应改成nonterminal/terminal各6个fresh；
3. **lease状态机把两个动作混成一步**：terminal child transfer不等于parent close，必须正交表达child-first与
   parent-first释放；
4. **raw Tensor无法形成可撤销capability**：一旦getter把Tensor交给外部，lease close后外部引用仍可访问；
   production API不得公开raw tensor getter，必须由opaque lease调用sealed consumer。

修正后的冻结结果：

- Pass A固定19 logical actions；Pass B固定一个selected graph invocation及其真实内部call receipt；
- Pass C固定nonterminal/terminal=`17/23` logical actions；
- evaluator lifecycle采用9 states、9 events、14 legal transitions、67 invalid combinations；
- component→execution→artifact seal DAG固定15个节点且无环；
- argument DLPack=`110`，result-facing额外普通Torch view=`6`，但这些view保持private；
- conditional known logical memory=`389,574 B`；selected-input alias失败则=`463,302 B`；
- formal改为12 fresh workers：六全排列×nonterminal/terminal；
- candidate output raw=`1,103,616 B`，A/B/C三方output raw=`3,310,848 B`；
- 六个terminal candidate的V-pre-overwrite sidecar=`899,136 B`；
- minimum mandatory numeric raw=`4,209,984 B = 4.01495361328125 MiB`；
- projection只能附加，不能替代完整IEEE payload；
- 当前S3仍`ready_for_audit`，本文没有开放任何S4代码、formal或性能门禁。

## 1. 本阶段的唯一问题

S4-1D回答的是：

> 已分别冻结的S4-0 admission、S4-1A storage、S4-1B coefficient/selected-value与S4-1C gradient/lA，能否在
> 一次真实production-shaped evaluation里由一个owner原子组合，成功只发布一份opaque result capability，失败不泄漏
> 半写结果，并能被第三方从完整raw重放？

它不回答：

- 10 evaluation/9 mutation trajectory；
- Adam、scheduler、keep-best、stop/prune；
- provider commit/post/queue；
- same-solver latency；
- complete-query或总体10x；
- multi-model generalization。

这些分别留给S4-2、S4-3、S4-P和最终formal。

## 2. 复用资产与禁止旁路

### 2.1 直接复用

| 上游 | S4-1D消费内容 | hash/receipt边界 |
|---|---|---|
| S4-0 | prepared admission、live source lease、exact-call identity | admission/lease receipt |
| S4-1A | 7 parameter、7 gradient、lower/upstream、16 base DLPack、single resource owner | buffer receipt |
| S4-1B0 | ternary endpoint classifier/select module | module/selector receipt |
| S4-1B | 19-action Pass A、six selector、49-arg Pass B、six V arena slot | pass/module/arena receipt |
| S4-1C | 7 gradient symbol、6 terminal copy symbol、17/23 Pass C | phase/module/gradient receipt |
| B4-A | terminal topology、shape、one-shot KFSB handoff oracle | native oracle only |
| RVIR-v4 | ordinal/version/exact-call与production state identity | adapter boundary |

### 2.2 禁止

- S2/R31B2/B4-B2 per-site wrapper直接成为production entry；
- generic registry或arbitrary callback选择consumer；
- result公开Tensor/dict/list getter；
- terminal额外第11次CROWN；
- post-begin reset/retry/fallback；
- warm DLPack、output allocation或content D2H；
- 把component hash串成循环receipt；
- 只保存candidate、hash或projection却宣称three-way raw；
- 在S4-1D内部修改parameter、moment、LR或provider state。

## 3. 逐文件实现边界

门禁开放后建议新增：

```text
boundflow/runtime/asplos27_s4_all_state_evaluator.py
boundflow/runtime/asplos27_s4_evaluation_receipt.py
boundflow/runtime/asplos27_s4_result_capability.py
tests/test_asplos27_s4_all_state_evaluator.py
tests/test_asplos27_s4_evaluation_receipt.py
tests/test_asplos27_s4_result_capability.py
scripts/run_asplos27_s4_1d_worker.py
scripts/run_asplos27_s4_1d_artifact.py
scripts/replay_asplos27_s4_1d_artifact.py
scripts/probe_asplos27_s4_1d_tamper.py
tests/test_asplos27_s4_1d_artifact.py
```

不新增`boundflow/ir`文件。原因是：

- bound/legality/plan/task/schedule identity已经由前序compiler对象拥有；
- 本阶段新增的是runtime transaction与capability，不是可优化的数学程序表示；
- Pass A/B/C action inventory通过typed runtime dataclass表达即可；
- 强行为state machine再造solver IR只会产生第二个owner。

## 4. public prepare/evaluate API

### 4.1 prepare

```python
prepare_s4_all_state_evaluator_v1(
    prepared_buffers: PreparedS4MutableBuffersV1,
    prepared_component_modules: S4PreparedComponentModulesV1,
    fixed_runtime_inputs: S4FixedRuntimeInputsV1,
    *,
    exact_call_id: str,
) -> PreparedS4AllStateCrownEvaluatorV1
```

不接受caller传device、stream、allocator、tensor override、plan override、fault callback或consumer callback。

prepare第一步必须从`PreparedS4MutableBuffersV1`发起module-private single-transfer adoption；不能通过public getter
取得16个buffer再组装。新evaluator的resource owner一次性收养：

```text
S4-0 private live source lease
S4-1A resource owner
compiled module handles/immutable receipts
two coefficient storages
six selector storages
one V/lA storage
all 110 argument DLPack descriptors
six result-facing ordinary view descriptors
phase/counter device buffers
```

adoption前任一失败由原owner继续持有；adoption成功后旧prepared wrapper不可再用。禁止逐字段move造成double-owner或
no-owner窗口。

### 4.2 evaluate

```python
PreparedS4AllStateCrownEvaluatorV1.evaluate(
    request: S4AllStateEvaluationRequestV1,
) -> S4AllStateResultCapabilityV1
```

request是tensor-free frozen canonical dataclass：

```text
schema_version
exact_call_identity_hash
evaluation_ordinal
expected_state_version
mode = NONTERMINAL | TERMINAL
schedule_action_hash
prepared_identity_hash
```

S4-1D只准：

```text
(ordinal=0, version=0, mode=NONTERMINAL)
(ordinal=9, version=9, mode=TERMINAL)
```

version9 parameter内容来自冻结production/native fixture，不由本evaluator执行九次mutation得到。

## 5. read-only admission与事务开始点

### 5.1 admission必须完全无副作用

依次验证：

1. evaluator state=`PREPARED_READY`；
2. owner PID/thread/device/current non-default stream exact；
3. request exact-call/prepared/schedule identity exact；
4. ordinal/version/mode tuple exact；
5. 无live parent/child capability；
6. S4-0 admission与live source lease仍有效；
7. S4-1A buffer owner与parameter state version exact；
8. immutable module/cache receipts独立validate；
9. 110 argument descriptor pointer/offset/shape/stride/dtype/device/generation exact；
10. 六ordinary result view仍指向预期storage/offset；
11. component plan/action hashes exact；
12. static metadata identity exact。

任一reject：

```text
state stays PREPARED_READY
generation delta=0
counter/phase/buffer delta=0
launch/copy=0
capability published=0
```

### 5.2 原子开始

全部通过后，唯一线性化点执行：

```text
PREPARED_READY
  -> allocate evaluation generation
  -> state=EVALUATING
  -> reset counters/phase tags
```

generation分配与state transition必须在同一owner-critical section；counter reset属于post-begin，不得提前。

## 6. 完整evaluation动作DAG

### 6.1 Pass A

复用S4-1B冻结19-action：

```text
seed
Linear16
ReLU31
Linear14
pack A29
ReLU28
residual11 stage1
pack A26
residual11 stage2
pack A24
ReLU23
residual6 stage1
pack A20
residual6 stage2
pack A18
ReLU17
Conv0-right
pack Ainput ternary
box concretize lower
```

Pass A结束必须sealed：lower full-write、six selector legal、Ainput reader count=0、coefficient descriptor revoked。

### 6.2 Pass B

固定一次selected graph invocation：

```text
coefficient arena Ainput phase
  -> selected-input alias activation
  -> selected graph 42 read + 7 write arguments
  -> six V full-write
  -> VM/token completion
  -> six V receipt seal
  -> selected-input reader count=0
  -> coefficient recompute descriptor reactivation
```

内部operation envelope为6 selected TIR、6 persistent copy、6 Conv、1 Gemm；实际device kernel/call count必须从
compiled/runtime receipt观察，不能把one VM invocation写成one kernel。

### 6.3 Pass C

复用S4-1C construction：

```text
coefficient actions = 10
dalpha emitters     = 6
dbeta emitters      = 1
nonterminal total   = 17
terminal copies     = 6
terminal total      = 23
```

reverse site order=`31,28,25,23,19,17`。site31严格：

```text
A31/V31 ready
  -> emit dα31
  -> emit dβ31
  -> terminal-only copy A31 over V31
  -> ReLU31 transform
```

其他site是`emit dα→terminal-only copy→transform/reuse`。最后site17后停止，不执行ReLU17/Conv0/concretize。

### 6.4 final gate与发布

顺序固定：

```text
all Pass A/B/C stream work enqueued
  -> same-stream completion boundary for final gate
  -> final finite/discrete/device counter gate
  -> output content root
  -> execution semantic root
  -> execution receipt independent validate
  -> construct private result capability
  -> publish exactly once
  -> state enters result state
```

receipt validate前不得把capability写入public/local return slot。failure保留最小tensor-free failure receipt并进入
`POISONED_NO_RETRY`。

## 7. 纠正后的parent/child capability状态机

### 7.1 为什么旧状态机不够

旧图写成：

```text
RESULT_LEASED -- terminal child transferred --> PARENT_CLOSED_CHILD_LIVE
```

但transfer与parent close是两个独立动作。真实合法顺序至少包括：

- child transfer→child close→parent close；
- child transfer→parent close→child close；
- parent在child尚未transfer时直接close并连同embedded child一起释放。

把它们混在一起会让duplicate transfer、child-first close与embedded cleanup无法机械判断。

### 7.2 九个状态

```text
PREPARED_READY
EVALUATING
NT_PARENT_OPEN
T_PARENT_OPEN_CHILD_EMBEDDED
T_PARENT_OPEN_CHILD_LIVE
T_PARENT_OPEN_CHILD_CLOSED
PARENT_CLOSED_CHILD_LIVE
POISONED_NO_RETRY
CLOSED
```

### 7.3 九个event

```text
admission_reject
admit_start
post_begin_fail
publish_nonterminal
publish_terminal
transfer_child
close_parent
close_child
close_owner
```

### 7.4 十四条合法transition

| from | event | to |
|---|---|---|
| PREPARED_READY | admission_reject | PREPARED_READY |
| PREPARED_READY | admit_start | EVALUATING |
| PREPARED_READY | close_owner | CLOSED |
| EVALUATING | post_begin_fail | POISONED_NO_RETRY |
| EVALUATING | publish_nonterminal | NT_PARENT_OPEN |
| EVALUATING | publish_terminal | T_PARENT_OPEN_CHILD_EMBEDDED |
| NT_PARENT_OPEN | close_parent | CLOSED |
| T_PARENT_OPEN_CHILD_EMBEDDED | transfer_child | T_PARENT_OPEN_CHILD_LIVE |
| T_PARENT_OPEN_CHILD_EMBEDDED | close_parent | CLOSED |
| T_PARENT_OPEN_CHILD_LIVE | close_child | T_PARENT_OPEN_CHILD_CLOSED |
| T_PARENT_OPEN_CHILD_LIVE | close_parent | PARENT_CLOSED_CHILD_LIVE |
| T_PARENT_OPEN_CHILD_CLOSED | close_parent | CLOSED |
| PARENT_CLOSED_CHILD_LIVE | close_child | CLOSED |
| POISONED_NO_RETRY | close_owner | CLOSED |

`9×9-14=67`种其他state/event组合全部稳定拒绝且不改状态。canonical model hash：

```text
963e723f7bc722a51f728527d599f1fc246065fc1eb700416810de2ad108599d
```

### 7.5 release语义

- nonterminal parent close：撤销lower/gradient capability并close evaluator；
- terminal embedded parent close：未transfer child随parent一起撤销；
- child-first：先撤销lA capability，parent仍可消费lower/gradient；
- parent-first：先撤销lower/gradient，child继续持有lA arena与owner strong ref；
- 最后一个capability close后owner进入CLOSED；
- capability release不等于CUDA reserved memory立即下降；
- S4-2 controlled re-arm是下阶段新合同，S4-1D success不回READY。

## 8. raw Tensor getter不可作为lease

### 8.1 反例

只要API曾返回raw Tensor：

```python
escaped = lease.view
lease.close()
escaped.sum()  # 仍然合法
```

现场反例中lease property在close后会拒绝，但此前逃逸的Tensor仍可读出sum=`28.0`。清空wrapper字段不能撤销外部Tensor
引用，PyTorch也没有通用的raw Tensor capability revocation。

### 8.2 production API

所以三个类都必须是non-dataclass private-physical/public-opaque设计：

```text
S4AllStateResultCapabilityV1
S4TerminalAdjointCapabilityV1
_S4EvaluationResourceOwnerV1
```

公开API只允许：

```python
result.receipt                       # tensor-free
result.consume_into_policy(driver)  # exact sealed type, once
result.transfer_terminal()           # returns opaque child, once
result.serialize_into_formal(sink)   # formal-only exact sealed type, once
result.close()

terminal.consume_into_kfsb(consumer) # exact sealed type, once
terminal.serialize_into_formal(sink) # formal-only exact sealed type, once
terminal.close()
```

不提供：

```text
.lower / .gradients / .lA
dict/list/tuple of Tensor
__iter__ / __getitem__
generic callback
to_dict/asdict/pickle/deepcopy
DLPack export
```

sealed consumer必须为repo内exact class，module receipt绑定consumer implementation hash；不得允许subclass、duck typing或
arbitrary callable。consumer执行结束后做owner-reference audit，确认其字段/return没有保留source Tensor/storage。

对外能诚实claim的是“close后拒绝新的consume/transfer/serialize，API禁止raw Tensor escape”；不能claim Python能够撤销
任意已经逃逸的raw Tensor。

## 9. object ownership与cleanup

### 9.1 physical owner

`_S4EvaluationResourceOwnerV1`是唯一physical owner：

```text
_live_source_lease
_mutable_buffer_owner
_component_modules
_coefficient_storages
_selector_storages
_value_la_storage
_argument_views
_ordinary_result_views
_counter_phase_buffers
_state/generation
_parent_capability_count
_terminal_capability_count
```

wrapper、receipt和artifact均不持raw Tensor。

### 9.2 post-begin failure cleanup

不恢复半写buffer。固定：

```text
capture stable failure detail/counters/generation
clear exception/traceback tensor locals
revoke all unpublished capabilities
state=POISONED_NO_RETRY
retain physical owner only until explicit close
close: views -> phase buffers -> V/lA -> selectors -> coefficient -> modules -> mutable owner -> source lease
```

close幂等，但能力不可恢复。禁止failure后native fallback、retry或queue continue。

## 10. component与artifact seal DAG

### 10.1 十五节点拓扑

```text
source_static_root
  -> request_root
  -> pass_a_root
  -> pass_b_root
  -> pass_c_root
  -> output_root
  -> terminal_v_sidecar_root
  -> final_gate_root
  -> execution_semantic_root
  -> execution_receipt_hash
  -> result_lease_runtime_ref
  -> raw_worker_root
  -> summary_root
  -> artifact_files_root
  -> manifest_hash
```

实际依赖不是单链；冻结拓扑规则：

- output root依赖Pass A lower与Pass C gradients/lA；
- terminal V sidecar依赖Pass B且在覆盖前形成；
- final gate依赖output及A/B/C component roots；
- execution semantic root依赖request、A/B/C、output、final gate；
- result runtime ref只引用receipt hash，不反向进入execution semantic root；
- raw worker依赖receipt、output、sidecar；
- manifest封装其他文件但排除自身。

canonical DAG hash：

```text
444a98d8f91dddf471475a21b9f984251d9f1b6e19a7d47b4234131a3314bca9
```

### 10.2 禁止的循环

- execution receipt包含result lease hash，同时result lease hash包含execution receipt；
- component receipt依赖最终receipt；
- summary hash进入raw row而raw row又进入summary；
- manifest把自身digest列入自身files；
- cache hit counter进入immutable module identity。

## 11. execution receipt

`S4AllStateEvaluationReceiptV1`只含tensor-free canonical字段：

```text
schema/version
source/admission/exact-call/plan/trace/buffer identity
hardware/device/current-stream/TVM-FFI-stream identity
Pass A/B/C module/template/schedule/device-source roots
request ordinal/version/mode/generation
19 / one-graph / 17-or-23 logical action counts
actual VM/library/TIR/copy/kernel counts
selector/value/gradient/terminal inventories
110 argument descriptors / 6 private ordinary result views
storage/offset/generation/phase summary
lower/gradient/terminal output content roots
conditional memory ledger and allocator observations
fallback/eager/native-shadow/retry/queue-continue = 0
timing_recorded/performance_claimed/same_solver_claimed = false
execution_semantic_root
receipt_hash
```

静态module、descriptor和metadata在prepare验证；warm evaluate只读O(1) identity/generation/counter。formal content raw/hash
位于计时外，不得回流production receipt。

## 12. final gate

lease构造前核验：

- lower `[6,1]` finite；
- six dα与active dβ finite；
- five empty β exact token，无physical tensor/view/launch；
- terminal mode six lA finite、shape/order/spec-axis exact；nonterminal lA inventory=0；
- selector legal：ternary=`{-1,0,+1}`、binary=`{0,1}`，invalid sentinel count=0；
- six α index与β location/sign identity exact；
- Pass A/B/C generation一致；
- 19/17-or-23 action inventory exact；
- actual writes/launch/copy无缺失重复；
- lower/gradient/V/lA storage ranges合法；
- fallback/eager/native-shadow/retry=0；
- claim flags全部false。

kernel poison产生canonical qNaN，final gate转为稳定reason；禁止NaN→0后继续。

## 13. descriptor与memory账

### 13.1 argument/view

```text
S4-1A base argument DLPack          = 16
S4-1B union                         = 90
S4-1A/B/C full union                = 110
private ordinary result views       = 6
warm DLPack/view construction       = 0
```

六ordinary view不进入110，也不新增storage。

### 13.2 conditional logical memory

```text
active parameters                    17,016 B
gradient outputs                     17,016 B
six selectors                        55,296 B
V/terminal-lA arena                 149,856 B
two coefficient arenas             147,456 B
residual scratch additional              0 B
lower/upstream/bias                      72 B
compressed metadata                  2,862 B
total                               389,574 B
```

它依赖selected input复用coefficient storage。若S4-1B alias/liveness失败：

```text
389,574 + 73,728 = 463,302 B
```

logical ledger、Torch allocated/reserved、CUDA driver free delta与compiled VM/cuDNN workspace必须四栏披露；任何一栏不得
替代其他栏。S4-1D不作memory improvement claim。

## 14. formal worker拓扑修正

### 14.1 为什么不是5+5

每个worker比较三个独立实现：

```text
A = pinned production/provider
B = provider-independent full PyTorch/autograd oracle
C = compiled S4 evaluator candidate
```

三者有六种执行顺序：

```text
ABC ACB BAC BCA CAB CBA
```

每类5个worker无法覆盖六全排列。correctness虽不claim timing，但全排列能暴露跨owner状态污染、cache/stream残留和
candidate依赖先执行oracle的问题。因此冻结为：

```text
6 nonterminal fresh workers × six permutations
6 terminal fresh workers    × six permutations
total = 12 subprocess
```

每个实现使用从同一source identity独立克隆的输入state，不共享mutable tensor/storage；执行顺序不能通过共享输入污染
另一个实现。

### 14.2 每worker numeric payload

每个实现：

```text
nonterminal = lower24 + gradient17016 = 17,040 B
terminal    = nonterminal17040 + lA149856 = 166,896 B
```

### 14.3 修正后的最低raw预算

```text
candidate 6+6 outputs      = 1,103,616 B
A/B/C three-way outputs    = 3,310,848 B
candidate terminal V sidecar
  6 × 149,856              =   899,136 B
minimum mandatory numeric  = 4,209,984 B
                            = 4.01495361328125 MiB
```

该预算不含JSON metadata、component receipts、source identity、hash或environment；manifest必须分别披露numeric raw与
total artifact bytes。旧`919,680 B`保留为“5+5 candidate-only历史估算”，不得再称完整formal raw。

### 14.4 为什么只为terminal保存V sidecar

S4-1D integration closure依赖已关闭的S4-1B与S4-1C component artifact。nonterminal V数学由component artifact证明；
terminal新增风险是V被lA覆盖，因此六个terminal candidate必须在覆盖前保存完整V sidecar，并在覆盖后保存lA。

若外审要求S4-1D本身重新证明所有10/12 worker的V数学，则把V sidecar扩到12 worker：

```text
12 × 149,856 = 1,798,272 B
```

但不得在不改protocol/manifest的情况下悄悄扩大或缩小。

## 15. artifact文件与raw-first

建议：

```text
manifest.json
protocol.json
source_identity.json
dependency_artifacts.json
component_receipts.jsonl
workers.jsonl
raw/index.jsonl
raw/blobs/<sha256>.bin
summary.json
stdout.txt
replay.py
```

规则：

1. 创建空artifact目录，manifest intent先写；
2. worker每完成一个implementation，先fsync content-addressed raw blob，再追加index row；
3. 三方同worker完成后才写worker completion row；
4. 12/12完成前summary不存在；
5. partial result拒绝resume；
6. raw blob保留dtype/shape/stride语义、endianness、signed zero、NaN payload；
7. dependency artifact绑定immutable manifest hash，不复制篡改历史artifact；
8. manifest最后写并排除自身digest；
9. artifact不得含本机绝对路径。

## 16. independent replay

replayer只能用stdlib，禁止import BoundFlow、Torch、TVM、NumPy或αβ-CROWN。必须重算：

- 12 worker与两fixture六全排列；
- A/B/C input state identity与storage independence metadata；
- blob length/dtype/shape/element/byte/hash；
- lower/dα/dβ/lA max abs/max rel/sign；
- empty β token；
- terminal V pre-overwrite与lA post-copy inventory；
- 19/17/23 action sequence；
- 9-state/14-transition capability trace；
- 15-node seal DAG；
- 110/6 descriptor/view projection；
- 389,574/463,302 conditional ledger；
- component/dependency/source roots；
- summary与stdout；
- claim flags。

replay不得调用production validator或只比较expected summary hash。

## 17. failure reason

新增或复用稳定detail：

```text
S4_EVALUATOR_PREPARED_IDENTITY_MISMATCH
S4_EVALUATOR_REQUEST_TUPLE_MISMATCH
S4_EVALUATOR_OWNER_CONTEXT_MISMATCH
S4_EVALUATOR_COMPONENT_RECEIPT_MISMATCH
S4_EVALUATOR_DESCRIPTOR_INVENTORY_MISMATCH
S4_EVALUATOR_ALREADY_CONSUMED
S4_EVALUATOR_POST_BEGIN_POISONED
S4_EVALUATOR_GENERATION_MISMATCH
S4_EVALUATOR_PASS_A_INCOMPLETE
S4_EVALUATOR_PASS_B_INCOMPLETE
S4_EVALUATOR_PASS_C_INCOMPLETE
S4_EVALUATOR_ACTION_ORDER_MISMATCH
S4_EVALUATOR_FINAL_GATE_FAILED
S4_EVALUATOR_RECEIPT_DAG_CYCLE
S4_EVALUATOR_RESULT_PUBLISHED_EARLY
S4_RESULT_RAW_TENSOR_EXPORT_FORBIDDEN
S4_RESULT_CONSUMER_TYPE_MISMATCH
S4_RESULT_CONSUMER_RETAINED_SOURCE
S4_RESULT_DUPLICATE_CONSUME
S4_RESULT_PARENT_ALREADY_CLOSED
S4_TERMINAL_TRANSFER_IN_NONTERMINAL
S4_TERMINAL_TRANSFER_DUPLICATE
S4_TERMINAL_CHILD_ALREADY_CLOSED
S4_TERMINAL_PARENT_CHILD_STATE_MISMATCH
S4_TERMINAL_CONSUMER_RETAINED_SOURCE
S4_FORMAL_WORKER_PERMUTATION_MISMATCH
S4_FORMAL_THREE_WAY_RAW_INCOMPLETE
S4_FORMAL_TERMINAL_V_SIDECAR_MISSING
S4_FORMAL_NUMERIC_BUDGET_MISMATCH
S4_FORMAL_DEPENDENCY_ARTIFACT_MISMATCH
S4_CLAIM_TRUE_BEFORE_FORMAL
```

## 18. negative matrix

至少覆盖：

1. evaluator不是READY；
2. exact-call hash漂移；
3. schedule hash漂移；
4. ordinal0要求terminal；
5. ordinal9要求nonterminal；
6. state version与ordinal不一致；
7. current device漂移；
8. current stream为default；
9. Torch/FFI stream不一致；
10. live source lease失效；
11. buffer owner已close；
12. component module receipt漂移；
13. descriptor pointer漂移；
14. descriptor stride/offset漂移；
15. ordinary result view错storage；
16. admission reject仍增加generation；
17. admission reject仍reset counter；
18. Pass A缺action；
19. Pass A pack/concretize顺序交换；
20. Pass B在Ainput reader未归零时alias；
21. Pass B只写5个V；
22. Pass B VM token未完成即进入C；
23. Pass C用stale coefficient descriptor；
24. Pass C nonterminal不是17 action；
25. Pass C terminal不是23 action；
26. site31 copy早于dβ；
27. site31 copy早于dα；
28. residual scratch在emitter前复用；
29. final gate前stream工作未完成；
30. lower NaN；
31. dα NaN；
32. dβ Inf；
33. terminal lA NaN；
34. selector invalid sentinel残留；
35. empty β physicalized；
36. gradient emitter缺失/重复；
37. terminal copy缺失/重复；
38. receipt validate前publish result；
39. post-begin失败回READY；
40. post-begin失败retry；
41. poisoned generation复用；
42. failure后native fallback；
43. result公开`.lower` raw Tensor；
44. result公开gradient tuple/dict；
45. result允许generic callback；
46. consumer subclass/duck type；
47. sealed consumer保留Tensor字段；
48. sealed consumer return raw Tensor；
49. duplicate parent consume；
50. nonterminal transfer child；
51. duplicate terminal transfer；
52. embedded child未transfer而parent close泄漏；
53. child-first后parent不能close；
54. parent-first后child不能close；
55. close child两次；
56. close parent两次；
57. child close后再次transfer；
58. parent close后新consume；
59. 67 invalid state/event任一改变状态；
60. component receipt依赖final receipt成环；
61. result hash与receipt互相依赖；
62. manifest包含自身digest；
63. cache observation进入module identity；
64. warm构建DLPack；
65. warm分配output；
66. runtime receipt做content D2H；
67. S4-1D修改parameter；
68. S4-1D修改moment/LR；
69. S4-1D调用provider commit/post；
70. formal只有5+5；
71. formal缺任一六全排列；
72. A/B/C共享mutable tensor/storage；
73. formal只保存candidate；
74. formal把919,680称三方raw；
75. terminal V在覆盖后才抓取；
76. V sidecar缺任一site；
77. projection替代full bytes；
78. raw blob dtype/endianness漂移；
79. partial artifact resume；
80. dependency artifact manifest漂移；
81. full re-sign后篡改action count；
82. full re-sign后篡改capability trace；
83. full re-sign后篡改raw预算；
84. full re-sign后篡改A/B/C顺序；
85. full re-sign后篡改memory conditional；
86. full re-sign后performance/same-solver=true。

## 19. positive tests

### 19.1 state/capability

- pre-begin rejection保持READY；
- nonterminal publish→consume→close；
- terminal embedded直接parent close；
- terminal child-first close；
- terminal parent-first close；
- post-begin failure→poison→close；
- 全14 legal transition通过；
- 全67 invalid combination稳定拒绝。

### 19.2 transaction

- nonterminal 19/one graph/17；
- terminal 19/one graph/23；
- 110 pointer exact、6 private view exact；
- lower/grad/lA full-write；
- warm allocation/view/fallback=0；
- component roots与15-node DAG独立重算。

### 19.3 formal

- 12 subprocess；
- 两fixture各六全排列；
- A/B/C source-equivalent且storage independent；
- numeric raw恰`4,209,984 B`最低预算；
- replay summary逐字节一致；
- 至少12类fully outer-resigned tamper全部拒绝。

## 20. 逐提交顺序

前序门禁关闭后：

```text
feat(runtime): add S4-1D evaluator owner and read-only admission
test(runtime): close admission and post-begin poison transitions
feat(runtime): assemble Pass A/B/C and final gate
test(runtime): close 19/17/23 action and component receipt DAG
feat(runtime): add opaque result and terminal capabilities
test(runtime): close 9-state parent-child lifecycle and raw-export negatives
feat(formal): add 12-worker three-way full-IEEE artifact
test(formal): close replay dependency and fully re-signed tamper
docs: close S4-1D and only then open S4-2
```

S4-1D通过前，S4-2实现、same-solver接入、timing和性能仍关闭。

## 21. construction manifest

```json
{
  "schema": "boundflow.asplos27-s4-1d-construction/v1",
  "scope": {
    "execution_authority": false,
    "code_change_open": false,
    "formal_open": false,
    "timing_open": false,
    "performance_claimed": false
  },
  "actions": {
    "pass_a": 19,
    "pass_b_graph": 1,
    "pass_c_nonterminal": 17,
    "pass_c_terminal": 23
  },
  "capability_model": {
    "state_count": 9,
    "event_count": 9,
    "legal_transition_count": 14,
    "invalid_combination_count": 67,
    "hash": "963e723f7bc722a51f728527d599f1fc246065fc1eb700416810de2ad108599d"
  },
  "seal_dag": {
    "node_count": 15,
    "hash": "444a98d8f91dddf471475a21b9f984251d9f1b6e19a7d47b4234131a3314bca9"
  },
  "views": {
    "argument_dlpack": 110,
    "private_result_ordinary": 6,
    "warm_created": 0
  },
  "memory": {
    "conditional_known_logical_bytes": 389574,
    "alias_failure_known_logical_bytes": 463302
  },
  "formal": {
    "fixture_count": 2,
    "permutation_per_fixture": 6,
    "worker_count": 12,
    "candidate_output_bytes": 1103616,
    "three_way_output_bytes": 3310848,
    "terminal_v_sidecar_bytes": 899136,
    "minimum_numeric_raw_bytes": 4209984
  },
  "negative_minimum": 86,
  "raw_tensor_getter_allowed": false
}
```

canonical construction hash：

```text
76da18648d874dfec6e867deaf26122f093f8157c68967ef2d06afe362243cd1
```

## 22. 当前门禁

```text
S3 exchange = ready_for_audit
S4-0/1A/1B0/1B/1C/1D code = closed
S4-1D formal = closed
S4-2/3/4/P = closed
performance/same-solver/query claims = false
next executable repository action = external audit S3
next design-only action = refresh S4 design handoff with S4-1D construction corrections
```

本施工包只把未来实现变成可机械审计的transaction；它不把设计诊断冒充production功能或性能结果。
