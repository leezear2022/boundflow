# ASPLOS'27 S4-2：sealed production policy driver 实施施工包

status: implementation-construction-design-only
date: 2026-08-29
execution-authority: false
code-change-open: false
formal-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false

## 0. 结论

S4-2 的代码仍因 S3 外审未批准而关闭，但实现边界已经可以收束成一套可直接施工、可机械拒绝的合同：

```text
sealed production policy program
  + exact evaluator family
  + opaque single-generation result capability
  + functional Adam/clamp/scheduler shadow
  + deterministic commit cursor
  + controlled next-generation issuance
  + keep-best/stop/patience/pruning/terminal owner
  -> one sealed 10-evaluation policy run
```

本施工包修正旧 S4-2 蓝图的六个不足：

1. **不能从 S4-1D result 取 raw Tensor**：policy driver 必须成为 exact sealed consumer，消费过程不返回
   lower/gradient/lA view；
2. **re-arm 不是重新打开已经关闭的 one-shot evaluator**：S4-2 新增 run-level evaluator family owner，每次只发行
   一个 generation handle；旧 S4-1D public one-shot 语义保持不变；
3. **ordinal、optimizer mutation、storage generation 不能共用一个 version**：terminal restore 可能写参数但不是第 10 次
   Adam mutation，必须拆成独立计数；
4. **terminal lA 必须与最终 best state 同轮次**：当前 fixed workload 六个 domain 均以 ordinal 9 最优，允许零重跑
   handoff；若任一 domain 的 best 来自更早轮次，本阶段稳定拒绝，不得把 ordinal 9 lA 与更早 α/β 拼接；
5. **5 对 fresh 无法平衡顺序**：A/B 与 B/C 各改为 6 对，分别 3 个正序、3 个反序，共 24 个独立进程；
6. **旧 raw 下限漏掉 Adam 未投影参数和最终 restore state**：修正后的 24-worker mandatory transition-tensor floor
   至少为 `60,550,896 B = 57.74583435058594 MiB`，且仍不含 policy projection、receipt、source 和容器开销。

这些是 implementation contract，不是 S4 实现、correctness、memory peak、speedup 或 ASPLOS claim。

## 1. 权威输入与不变量

### 1.1 仓库内输入

- S4 same-solver 总预注册；
- S4 evaluator ABI 与 terminal handoff；
- S4-0 mutable admission/lease；
- S4-1A ordered buffer ABI；
- S4-1B effective-value graph；
- S4-1C compressed gradient emitter；
- S4-1D one-evaluation transaction 与 opaque capability；
- S4-2 blueprint 与 implementation-readiness；
- S4-3 whole-core exact-call transaction。

### 1.2 pinned external source

```text
alpha-beta-CROWN commit = e5c7e17bf0488843acb77b7519f59876717a49f4
auto_LiRPA commit       = 5a098e8f9fb5786a428a024981d833d303921f2d
optimized_bounds.py    = d07152c487a4c95ab25c39365f8b0a02e15f8a21c96bd5c435e1659957691492
operators/relu.py      = 71a576980988a650a866fe027e7941cdbdfeec54fb3215b43206fa3b198b565c
Python                  = alpha-beta-CROWN/.venv Python 3.11
torch                   = 2.11.0+cu130
device                  = RTX 4060 Laptop GPU
```

正式 artifact 必须重新核对 commit 与 blob hash；本稿数字只绑定上述 source。

### 1.3 fixed production control

```text
evaluation_limit            = 10
Adam parameter mutations    = 9
scheduler calls             = 10
consumed LR transitions     = 9
alpha LR                    = 0.01
beta LR                     = 0.05
ExponentialLR gamma         = 0.98
Adam betas                  = (0.9, 0.999)
Adam eps                    = 1e-8
weight_decay                = 0
keep_best                   = true
early_stop_patience         = 10
start_save_best             = 0.5
pruning_in_iteration        = true
pruning threshold           = 0.2
max_time                    = 60 s
fix_intermediate_bounds     = true
float64 terminal iteration  = false
bound side                  = lower
```

### 1.4 fixed live outcome

```text
checkpoint ordinals        = [0, 6, 7, 8, 9]
physical pruning active    = [false] * 10
timeout                    = [false] * 10
stop_all                   = [false] * 10
best iteration/domain      = [9, 9, 9, 9, 9, 9]
terminal is best/domain    = [true, true, true, true, true, true]
```

这些只约束当前 ResNet2B fixed fixture。synthetic fixture 必须使用不同 policy hash，不能修改上面事实。

## 2. 代码对象：runtime contract，不新增通用 IR

S4-2 不新增 Bound/Plan/Task/Schedule 之外的 compiler IR。新增对象属于 prepared runtime 与 audit receipt。

### 2.1 `S4SealedPolicyProgramV1`

immutable、frozen、canonical、stable-hash，至少包含：

```text
schema_version
program_hash
source_policy_hash
topology_hash
mutable_admission_hash
ordered_parameter_abi_hash
evaluator_family_abi_hash
evaluation_limit
Adam param-group ABI
Adam defaults
LR scheduler ABI
loss reduction ID
stop criterion ID
keep-best/checkpoint policy
patience policy
pruning policy
timeout policy
terminal handoff policy
early-best terminal admission policy
```

禁止字段：

- arbitrary callback；
- Python module/class object；
- expected trace reader；
- fallback function；
- workload-specific node id 或 shape 常数藏在通用 schema 中。

fixed ResNet program instance当然可以绑定冻结 shape/topology，但 schema 本身不得为 ResNet 特判。

### 2.2 `_S4PolicyEvaluatorFamilyV1`

它是 S4-2 新增的 run-level physical resource owner：

```text
exact evaluator kind       # NATIVE_DENSE_ORACLE_V1 / COMPILED_COMPRESSED_V1
prepared module/cache
prepared argument descriptors
stable parameter/gradient/lower/upstream views
coefficient/V/lA arena
current evaluator generation
current parameter input version
issued-generation flag
poison flag
closed flag
```

factory 只接受两个 exact enum；不接受 subclass、duck type 或 callable。

### 2.3 `S4EvaluationGenerationV1`

每个 generation 是 one-shot：

```text
generation_id
evaluation_ordinal
evaluation_input_version
request semantic root
component execution root
opaque result capability or failure receipt
```

它复用 S4-1D 的 transaction/phase/capability 语义，但不把旧 one-shot evaluator“重开”。generation close 后，只有
family owner 在满足本稿 re-arm gate 时才能发行下一代。

### 2.4 `_S4SealedProductionPolicyDriverV1`

driver 唯一拥有：

- 10/9/10 control sequence；
- stop/loss/keep-best/checkpoint/patience/pruning/time；
- functional Adam state；
- clamp；
- scheduler state；
- transition shadow；
- stable copy-commit cursor；
- evaluator re-arm authority；
- terminal best-state admission；
- terminal child capability；
- run receipt 与 poison 状态。

evaluator 不得修改这些 state；driver 不得改变 evaluator 的 Pass A/B/C 顺序或 kernel schedule。

### 2.5 `S4PolicyTransitionShadowV1`

每个非 terminal ordinal 只使用 prepared shadow：

```text
raw derivative projection
signed/masked optimizer gradient
unprojected next parameter
projected next parameter
next m
next v
next step
next alpha/beta LR
next scheduler counters
transition validation receipt
```

production warm path可用同一 parameter shadow 先保存 functional Adam 输出、完成 equation/finite/raw-observer gate后
原位 clamp；formal sink必须在 clamp 前保存 IEEE bytes。不能因只保留一个 physical shadow 就丢失 unclamped evidence。

### 2.6 `S4PolicyRunReceiptV1`

tensor-free receipt至少绑定：

```text
program/source/evaluator/family hashes
ten request/result/component roots
generation sequence
evaluation-input versions
optimizer mutation sequence
storage commit generations
scheduler call/LR sequence
checkpoint/best/prune/stop/time sequence
commit cursor sequence
terminal child lineage
terminal-best admission result
memory/storage inventory
provider/fallback/native-shadow counters
failure/poison state
performance_claimed=false
```

## 3. 三个版本轴必须分开

旧断言：

```text
ordinal == parameter_version == parameter_mutation_count
```

只在进入每个 fixed evaluation 前偶然成立，不能作为完整 run ABI。S4-2 冻结三个轴：

### 3.1 `evaluation_input_version`

- ordinal 0 输入 version 0；
- ordinal 1—9 输入 version 1—9；
- 只表示 evaluator 读到的是第几次 Adam mutation 后的参数；
- 与 request/generation/result root绑定。

### 3.2 `optimizer_mutation_count`

- 初始 0；
- ordinal 0—8 各成功增加 1；
- terminal ordinal 9 不增加；
- fixed run最终严格为 9。

### 3.3 `storage_commit_generation`

- 每次稳定参数/m/v/step copy-commit完成后增加；
- restore-to-earlier-best若真的写回参数，也增加；
- fixed workload terminal就是best，restore通过逐元素/identity验证后是 no-op，最终保持9；
- synthetic earlier-best允许最终为10，但仍必须满足`optimizer_mutation_count=9`。

### 3.4 其他独立计数

```text
evaluator_generation       = 1..10
scheduler_step_call_count  = 0..10
consumed_lr_transition     = 0..9
checkpoint_event_count     = 5      # fixed workload
terminal_handoff_count     = 1
duplicate_crown_count      = 0
```

任何 receipt 把这些字段折成一个 version 都拒绝。

## 4. opaque result 的精确消费

### 4.1 禁止 raw getter

S4-1D 已证明 Python/PyTorch raw Tensor 逃逸后无法由 close 撤销。因此 S4-2 不得调用：

```text
result.lower
result.gradients
result.lA
result.to_dict()
result.__iter__()
DLPack export
generic callback
```

### 4.2 exact consume

只允许 module-private exact 路径：

```text
result._consume_into_policy_exact(driver, consume_token)
```

要求：

1. `type(driver)`必须是 exact sealed class；
2. driver implementation hash、program hash、generation、ordinal、parameter version exact；
3. lower与derivative只写入driver已准备的policy/output/shadow view，不作为return返回；
4. nonterminal consume完成后parent result关闭且generation变成consumed；
5. terminal consume原子地把opaque terminal child移入driver terminal slot，再关闭parent；
6. consume return只能是tensor-free decision seed/receipt root；
7. consume后检查driver字段和return不持有source Tensor/storage；
8. 任一异常发生在evaluator begin之后，family与run同时poison。

### 4.3 raw derivative与optimizer gradient分离

evaluator给出的是：

```text
d lower / d active parameter
```

driver根据production：

```text
loss = -sum(lower where active and not stopped)
```

生成：

```text
optimizer_gradient = -active_loss_mask * raw_derivative
```

formal必须同时保存 raw derivative 与 signed/masked gradient，不能只保存后者后声称验证 evaluator VJP。

## 5. controlled re-arm：发行下一代，不复活旧对象

### 5.1 re-arm 前置条件

只有 nonterminal ordinal 0—8 且以下全部成立才可 re-arm：

```text
result parent closed
terminal child absent
generation consumed exactly once
policy decision validated
functional Adam/clamp/scheduler shadow validated
all 28 mutable copy items committed
scheduler state committed
optimizer_mutation_count == ordinal + 1
storage_commit_generation == ordinal + 1
next evaluation_input_version == ordinal + 1
family/run not poisoned
stream/device/module/cache/pointer identity unchanged
no outstanding source/result capability
```

### 5.2 28项 deterministic commit cursor

ordered physical parameters固定为六α+一active β：

```text
0..6    current parameter <- projected next parameter
7..13   current m         <- next m
14..20  current v         <- next v
21..27  current step      <- next step
```

每项commit前后记录：

```text
ordinal
slot
destination identity
source shadow identity
before content root
after content root
CUDA/CPU device
version before/after
```

中途失败不倒拷贝；cursor位置进入failure receipt，run=`POISONED_NO_RETRY`。

### 5.3 scheduler commit

scheduler transition先在host shadow中计算并验证，但只能在28项tensor commit全部成功后提交：

```text
current alpha LR
current beta LR
last epoch / call count
consumed transition count
```

如果scheduler commit失败，parameter state已经变更，必须poison；不得把数值copy回去冒充未执行。

### 5.4 next generation issuance

re-arm只做：

- 检查current stable pointer不变；
- 检查parameter content/version与已提交root一致；
- reset evaluator device counters/phase tags；
- 增加generation id；
- 发行新的one-shot generation handle。

禁止：

- 重建DLPack view；
- 重编译TIR/Relax/CUDA Graph；
- 新分配output/arena；
- 复用上代partial receipt；
- generation id回退或重用；
- re-arm terminal generation。

## 6. fixed 10/9/10 事务顺序

### 6.1 ordinal 0—8

```text
1. read-only preflight
2. family issues generation n
3. S4-1D evaluator begin
4. one Pass A/B/C evaluation
5. result independently validated and published
6. exact policy consume; no raw Tensor escape
7. stop/loss/best/patience/checkpoint decision
8. early-exit predicates
9. raw derivative -> signed/masked gradient
10. functional Adam into unprojected shadow
11. validate Adam equation/m/v/step and finite
12. beta projection and alpha projection in shadow
13. validate clamp, signed zero and nonfinite policy
14. stage next scheduler state
15. validate whole transition
16. deterministic 28-item copy-commit
17. commit scheduler state exactly once
18. commit policy/pruner next state
19. increment mutation/storage/input version
20. controlled re-arm and issue next generation
```

### 6.2 ordinal 9

```text
1. evaluate once in terminal mode
2. same evaluation emits lower, compressed derivative and six lA
3. exact consume transfers opaque terminal child into driver
4. update best/patience/checkpoint decision
5. evaluate early-exit predicates
6. Adam mutation count += 0
7. parameter projection count += 0
8. scheduler step call += 1
9. pruner next/final update
10. determine final best state
11. validate terminal-is-best admission
12. restore best state (fixed path exact no-op)
13. dense->compressed round-trip and preserved-source validation
14. publish terminal policy result + opaque child
```

### 6.3 LR sequence

```text
ordinal 0  alpha=0.01                 beta=0.05
ordinal 1  alpha=0.0098               beta=0.049
ordinal 2  alpha=0.009604             beta=0.04802
ordinal 3  alpha=0.00941192           beta=0.0470596
ordinal 4  alpha=0.0092236816         beta=0.046118408
ordinal 5  alpha=0.009039207968       beta=0.04519603984
ordinal 6  alpha=0.00885842380864     beta=0.0442921190432
ordinal 7  alpha=0.0086812553324672   beta=0.043406276662336
ordinal 8  alpha=0.008507630225817854 beta=0.04253815112908928
ordinal 9  alpha=0.008337477621301497 beta=0.04168738810650749
post       alpha=0.008170728068875466 beta=0.040853640344377336
```

formal按pinned Python/torch float行为逐位保存；上表展示值不能替代raw float64 scheduler receipt。

## 7. functional Adam 与投影

### 7.1 two-group ABI

```text
group alpha:
  parameters = six ordered α buffers
  lr = 0.01
  batch_dim = 2

group beta:
  parameters = one active β buffer
  lr = 0.05
  batch_dim = 0
```

共同字段逐项绑定：

```text
betas=(0.9,0.999)
eps=1e-8
weight_decay=0
amsgrad=false
maximize=false
foreach=null
capturable=false
differentiable=false
fused=null
decoupled_weight_decay=false
```

`batch_dim`不是标准Adam方程输入，但属于live param-group ABI，必须进入prepare receipt。

### 7.2 source binding

第一版允许使用pinned `torch.optim._functional.adam`，但必须绑定：

- torch version/build；
- function source/module identity；
- exact arguments；
- single-tensor/foreach/fused选择；
- CPU float32 step scalar ABI；
- 9×7 live-vs-functional bit-exact diagnostic。

若任一parameter/m/v/step不exact，S4-2 STOP，不允许只放宽lower容差。

### 7.3 clamp顺序

production顺序：

```text
Adam
-> beta nonnegative projection
-> alpha ReLU clip [0,1]
-> scheduler step
```

formal保存：

```text
unprojected next parameter
projected next parameter
negative/above-one/below-zero/nonfinite counts
signed-zero inventory
```

不能把clamp提前到Adam、把preserved α一起clamp，或用zero替换NaN继续。

## 8. keep-best、checkpoint与terminal一致性

### 8.1 best bound更新

每个ordinal保存：

```text
current lower
reduced lower/domain
best lower before/after
improved-domain mask
need_update
patience before/after
ret_0 before/after
checkpoint predicate
checkpointed-domain mask
best iteration before/after
```

ordinal 0初始化best，但live observer中`need_update=false`、patience变1；不能把初始化误写成数值improvement。

### 8.2 checkpoint谓词

production exact：

```text
i < 1
or i > int(iteration * start_save_best)
or deterministic
or stop_final
or patience == early_stop_patience
or time_spent > max_time
```

fixed值：

```text
int(10*0.5)=5
i>5
checkpoint=[0,6,7,8,9]
```

ordinal5不是checkpoint。

### 8.3 compressed checkpoint

candidate checkpoint只复制：

```text
six active lower α = 16,992 B
one active β       =     24 B
total              = 17,016 B
```

preserved α由immutable source lease提供，不重复复制。每次checkpoint必须按domain mask复制active slice；不能整buffer
覆盖后再声称per-domain等价。

### 8.4 fixed intermediate bounds

当前`fix_intermediate_bounds=true`，所以candidate不复制production `299,712 B` best-intermediate container；替代证据是：

- prepare object/storage/version/content root；
- every-evaluation O(1) identity/version guard；
- S4-3 precommit current-provider rebind；
- official post external container parity。

任一guard漂移则fail closed。

### 8.5 terminal-best/lA admission

terminal child中的六lA是ordinal9参数状态的adjoint。合法handoff要求：

```text
best_iteration_by_domain == [9,9,9,9,9,9]
terminal_is_best_mask     == [T,T,T,T,T,T]
restored compressed state == ordinal9 compressed state
```

如果某个domain best来自更早ordinal：

- policy control test可以验证checkpoint/restore；
- S4-2 fixed exact-call handoff必须拒绝`TERMINAL_ADJOINT_NOT_BEST_STATE`；
- 关闭terminal child；
- 不执行第11次CROWN；
- 不把ordinal9 lA与earlier α/β混合；
- 不进入S4-3。

未来若要支持该情况，必须单独预注册“每轮lA checkpoint”或“restore后一次显式terminal rerun”路线；两者都不属于本阶段。

## 9. policy run状态机

### 9.1 16 states

```text
PREPARED_READY
EVALUATING
RESULT_OPEN
POLICY_UPDATE
UPDATE_STAGED
UPDATE_VALIDATED
UPDATE_COMMITTING
REARM_PENDING
REARMING
POLICY_TERMINAL
RESTORE_STAGED
RESTORE_VALIDATED
RESTORE_COMMITTING
TERMINAL_READY
POISONED_NO_RETRY
CLOSED
```

### 9.2 16 events

```text
admission_reject
begin_evaluation
publish_result
consume_update
consume_terminal
stage_update
stage_restore
validate_transition
begin_commit
finish_update_commit
finish_restore_commit
begin_rearm
finish_rearm
handoff_terminal
post_begin_fail
close_owner
```

### 9.3 32 legal transitions

| from | event | to |
|---|---|---|
| PREPARED_READY | admission_reject | PREPARED_READY |
| PREPARED_READY | begin_evaluation | EVALUATING |
| PREPARED_READY | close_owner | CLOSED |
| EVALUATING | publish_result | RESULT_OPEN |
| EVALUATING | post_begin_fail | POISONED_NO_RETRY |
| RESULT_OPEN | consume_update | POLICY_UPDATE |
| RESULT_OPEN | consume_terminal | POLICY_TERMINAL |
| RESULT_OPEN | post_begin_fail | POISONED_NO_RETRY |
| POLICY_UPDATE | stage_update | UPDATE_STAGED |
| POLICY_UPDATE | post_begin_fail | POISONED_NO_RETRY |
| UPDATE_STAGED | validate_transition | UPDATE_VALIDATED |
| UPDATE_STAGED | post_begin_fail | POISONED_NO_RETRY |
| UPDATE_VALIDATED | begin_commit | UPDATE_COMMITTING |
| UPDATE_VALIDATED | post_begin_fail | POISONED_NO_RETRY |
| UPDATE_COMMITTING | finish_update_commit | REARM_PENDING |
| UPDATE_COMMITTING | post_begin_fail | POISONED_NO_RETRY |
| REARM_PENDING | begin_rearm | REARMING |
| REARM_PENDING | post_begin_fail | POISONED_NO_RETRY |
| REARMING | finish_rearm | PREPARED_READY |
| REARMING | post_begin_fail | POISONED_NO_RETRY |
| POLICY_TERMINAL | stage_restore | RESTORE_STAGED |
| POLICY_TERMINAL | post_begin_fail | POISONED_NO_RETRY |
| RESTORE_STAGED | validate_transition | RESTORE_VALIDATED |
| RESTORE_STAGED | post_begin_fail | POISONED_NO_RETRY |
| RESTORE_VALIDATED | begin_commit | RESTORE_COMMITTING |
| RESTORE_VALIDATED | post_begin_fail | POISONED_NO_RETRY |
| RESTORE_COMMITTING | finish_restore_commit | TERMINAL_READY |
| RESTORE_COMMITTING | post_begin_fail | POISONED_NO_RETRY |
| TERMINAL_READY | handoff_terminal | CLOSED |
| TERMINAL_READY | post_begin_fail | POISONED_NO_RETRY |
| TERMINAL_READY | close_owner | CLOSED |
| POISONED_NO_RETRY | close_owner | CLOSED |

其余：

```text
16 * 16 - 32 = 224 invalid state/event pairs
```

canonical model hash：

```text
75e0c1b7aa4fc9bd439d15af41f7c1b86c8c4c7f732ca6bb55108488fa743279
```

### 9.4 poison传播

`post_begin_fail`适用于：

- evaluator；
- result validate/publish/consume；
- policy decision；
- functional Adam；
- clamp；
- scheduler shadow；
- transition validate；
-任何commit item；
- re-arm；
- terminal restore/round-trip/handoff。

poison后：

```text
retry=0
resume=0
fallback=0
native shadow=0
next generation=0
S4-3 handoff=0
queue continue=0
```

只允许保存tensor-free failure receipt并close。

## 10. early exit 与 synthetic policy

### 10.1 fixed patience不可达

从0开始、每次最多+1、条件`patience > 10`：

```text
10 evaluations -> max patience 10 -> branch unreachable
11 evaluations -> ordinal10 -> patience 11 -> branch reachable
```

因此fixed ResNet program不得声称覆盖patience exit。

### 10.2 sealed synthetic programs

至少：

1. `evaluation_limit=12, patience=10`，触发`>10`；
2. ordinal3 stop-all；
3. sealed scripted clock timeout；
4. partial prune preserve mask；
5. different domains select different best ordinal；
6. ordinal0 no-gradient；
7. best来自更早ordinal，验证restore但terminal handoff稳定拒绝；
8. evaluator/result/Adam/commit/re-arm分阶段故障注入。

每个variant有独立program hash，不能进入ResNet formal/performance artifact。

## 11. memory与storage账

### 11.1 已知base subtotal

```text
S4-1D evaluator state                         389,574 B
current Adam m/v                               34,032 B
current CPU step scalars                           28 B
compressed best checkpoint                     17,016 B
best lower                                         24 B
ret_0                                              24 B
next parameter/m/v/step shadow                  51,076 B
known base subtotal                            491,774 B
```

CUDA/CPU known base=`491,718/56 B`。

### 11.2 `491,774 B`只能叫base lower bound

实现仍必须显式落地并测量：

- active/prune/current/next masks；
- stop/improved/checkpoint masks；
- best-iteration vector；
- policy counters/flags；
- evaluator-family owner metadata；
- formal observer staging；
- TVM/cuDNN/module workspace；
- allocator rounding；
- fixed model/input/intermediate source；
- result/terminal capability Python objects。

因此本施工包把旧“known S4-2 logical subtotal”收紧为：

```text
491,774 B = known tensor/base lower bound
complete policy-driver logical subtotal = pending implementation storage receipt
peak allocated/reserved = pending measurement
```

任何后续文档直接把491,774写成完整S4-2 footprint、peak或memory improvement均拒绝。

### 11.3 shadow证据与physical storage

一个physical next-parameter shadow足够：

1. functional Adam写入unprojected值；
2. equation/finite gate；
3. formal sink保存unprojected IEEE raw；
4. 原位projection；
5. formal sink保存projected IEEE raw；
6. transition validate；
7. copy-commit。

production非formal run不得为了保存raw永久增加第二份parameter shadow；formal observer开销单列且不进入性能计时。

## 12. 三层correctness比较

### 12.1 A/B/C定义

```text
A = pinned live production optimizer observer
B = sealed policy driver + NativeDenseOracleEvaluatorV1
C = sealed policy driver + CompiledCompressedEvaluatorV1
```

- A/B关闭policy extraction；
- B/C关闭representation/evaluator replacement；
- 禁止直接C对历史simplified loop。

### 12.2 A/B比较

必须逐ordinal比较：

- production-visible lower；
- full/native α/β与active projection；
- raw/masked gradient（ordinal0—8）；
- Adam parameter/m/v/step；
- clamp；
- LR/scheduler；
- stop/best/checkpoint/patience/pruning/time；
- final restore；
- terminal lA；
- counters与exit reason。

B terminal额外derivative不是A policy输入，不得伪造为A官方autograd输出；只在B/C比较中使用。

### 12.3 B/C比较

逐ordinal比较：

- lower；
- six active lower-α derivative；
- active β derivative；
- signed/masked gradient；
- compressed parameter/m/v/step；
- preserved α source identity/content；
- five empty β token；
-全部policy projection；
- terminal six lA；
- restored dense projection与round-trip。

### 12.4 容差

```text
lower/state      max abs/rel <= 2e-4
gradient/m/v     max abs/rel <= 2e-5
sign             exact
discrete fields  exact
order/shape/dtype/device exact
empty/preserved tokens exact
```

functional Adam在pinned环境目标是bit exact；上方容差不得用来掩盖Adam equation漂移。

## 13. formal worker topology

### 13.1 为什么不是5对

5对只能形成3/2顺序分布，不能同时满足：

```text
AB == BA count
BC == CB count
```

所以每组改为6对：

```text
A/B: 6 pairs = 12 fresh workers
  AB, BA, AB, BA, AB, BA

B/C: 6 pairs = 12 fresh workers
  BC, CB, BC, CB, BC, CB

total = 24 fresh subprocesses
```

每个pair成员独立加载source、prepare、运行和落raw；部分结果不得resume。

### 13.2 mandatory transition-tensor floor

尺寸：

```text
compressed parameter P_c = 17,016 B
dense parameter      P_d = 34,008 B
compressed m/v       M_c = 34,032 B
dense m/v            M_d = 68,016 B
step state                 = 28 B
lower                      = 24 B
terminal six lA            = 149,856 B
```

每run至少保存：

```text
parameter before/after        10 each
gradient                       A:9, B/C:10
m/v before/after              10 each
step before/after             10 each
lower                         10
unprojected parameter shadow   9
restored terminal parameter    1
terminal lA                    1
```

由此：

```text
A minimum = 2,837,288 B
B minimum = 2,871,296 B
C minimum = 1,511,936 B

6*A + 12*B + 6*C
= 60,550,896 B
= 57.74583435058594 MiB
```

A terminal无官方optimizer gradient，故9份；B/C evaluator均保留terminal raw derivative，故10份。

### 13.3 仍未包含

上述只是transition tensor floor，不包含：

- policy masks/projection；
- best lower/ret0 before/after；
- checkpoint state sidecars；
- current/post LR raw；
- counters/receipts；
- preserved α source raw；
- empty β tokens；
- model/property/input/intermediate source；
- S4-1D component artifact引用；
- JSON/index/base64/filesystem开销；
- environment/source manifest。

formal manifest必须报告实际每path、每worker、每tensor bytes，不能把`60,550,896 B`宣传成最终artifact大小。

## 14. replay

replayer必须stdlib-only，不import BoundFlow/PyTorch/TVM/NumPy/αβ-CROWN，并独立完成：

1. 24-worker与pair-order inventory；
2. A/B/C path和fresh process identity；
3. raw shape/dtype/endianness/byte count；
4. lower/gradient/state最大差、sign与finite；
5. loss mask与gradient符号；
6. functional Adam parameter/m/v/step equation；
7. unprojected→projected clamp；
8. LR/scheduler 10-call与9-consumed transition；
9. best/checkpoint/patience/stop/pruning/time；
10. version/generation/commit cursor；
11. terminal-best admission、restore和round-trip；
12. terminal child lineage与duplicate CROWN=0；
13. source/program/evaluator/component/execution/artifact hash chain；
14. summary canonical重算。

replay只比较summary/hash而不重算Adam与policy语义，不算通过。

## 15. negative与fully re-signed tamper

### 15.1 prepare/admission

1. 第三方evaluator kind；
2. evaluator subclass；
3. arbitrary callback；
4. expected-trace reader；
5. program/source/topology/admission hash漂移；
6. α/β group交换；
7. `batch_dim`漂移；
8. Adam default漂移；
9. evaluator family pointer/cache/module漂移；
10. ordinal/version/generation不匹配。

### 15.2 capability/re-arm

11. raw Tensor getter；
12. duplicate consume；
13. terminal child重复transfer；
14. parent close后再次consume；
15. result capability跨generation；
16. outstanding result时re-arm；
17. commit未完成时re-arm；
18. generation重用/回退；
19. terminal generation re-arm；
20. re-arm重建DLPack/分配storage。

### 15.3 policy/optimizer

21. gradient符号反转；
22. stop/loss mask错误；
23. ordinal9发生第10次Adam mutation；
24. m/v/step重置或跨run复用；
25. unclamped shadow缺失；
26. clamp先于Adam；
27. preserved α被clamp；
28. beta负值通过；
29. nonfinite被替换为0；
30. scheduler少/多一次；
31. terminal post LR错误；
32. consumed LR写成10；
33. checkpoint ordinal含5或缺6；
34. patience fixed分支伪造可达；
35. pruning inactive被删除为无分支；
36. earlier-best terminal lA被错误handoff。

### 15.4 commit/poison

37. 28项commit乱序；
38. 某个m/v/step漏commit；
39. mid-copy失败后rollback/retry；
40. scheduler commit先于tensor commit；
41. partial commit后next generation；
42. poison后S4-3 handoff；
43. storage generation冒充optimizer mutation；
44. terminal no-op restore伪造成第10次mutation；
45. earlier-best restore漏掉storage generation。

### 15.5 artifact

46. 5-pair伪装balanced；
47. B process跨A/B与B/C复用；
48. raw只保存hash；
49. 缺unprojected shadow；
50. 缺restored terminal state；
51. A terminal伪造官方gradient；
52. 20-worker旧预算冒充新formal；
53. 修改raw并重签外层digest；
54. 修改policy projection并重签；
55. 修改generation/commit cursor并重签；
56. 修改terminal-best mask并重签；
57. `performance_claimed=true`。

至少57类是本阶段minimum；实现时若新增长路径或状态，negative数只能增加。

## 16. 文件与提交施工顺序

S3外审批准且S4-0→1A→1B0→1B→1C→1D依次关闭后：

1. `feat(runtime): add sealed S4 policy program and evaluator family`；
2. `feat(runtime): add exact opaque result consumer and generation re-arm`；
3. `feat(runtime): add functional Adam and scheduler transition shadow`；
4. `feat(runtime): add deterministic commit cursor and poison semantics`；
5. `feat(runtime): add keep-best checkpoint and terminal-best admission`；
6. `test(runtime): close policy, re-arm, commit and synthetic state machines`；
7. `test(runtime): close live A/B extraction parity`；
8. `feat(runtime): bind compiled compressed evaluator`；
9. `test(runtime): close B/C 10/9/10 trajectory`；
10. `artifact: freeze 24-worker raw replay and fully re-signed tamper`；
11. `docs: close S4-2 and only then open S4-3 implementation`。

建议生产文件：

```text
boundflow/runtime/asplos27_s4_policy_program.py
boundflow/runtime/asplos27_s4_evaluator_family.py
boundflow/runtime/asplos27_s4_policy_driver.py
boundflow/runtime/asplos27_s4_functional_adam.py
boundflow/runtime/asplos27_s4_policy_receipt.py
```

建议测试：

```text
tests/test_asplos27_s4_policy_program.py
tests/test_asplos27_s4_policy_state_machine.py
tests/test_asplos27_s4_evaluator_rearm.py
tests/test_asplos27_s4_functional_adam.py
tests/test_asplos27_s4_keep_best_terminal.py
tests/test_asplos27_s4_policy_negative.py
tests/test_asplos27_s4_policy_artifact.py
```

## 17. GO/STOP

### 17.1 S4-2关闭条件

全部成立：

- A/B 6-pair与B/C 6-pair通过；
- 10/9/10/9计数exact；
- 10 generation全部one-shot且9次re-arm exact；
- functional Adam pinned bit-exact；
- lower/state/gradient/sign满足冻结门禁；
- keep-best/checkpoint/patience/stop/prune/time exact；
- fixed terminal-best六domain成立；
- terminal child同轮次、one-shot、duplicate CROWN=0；
- 28-item commit与poison negative关闭；
- 24-worker完整raw可stdlib replay；
- fully re-signed tamper全部拒绝；
- memory receipt给出complete logical与allocated/reserved实测；
- no fallback/native shadow/expected trace；
- `performance_claimed=false`。

### 17.2 STOP

任一成立：

- raw Tensor必须逃逸才能驱动policy；
- re-arm必须复活已关闭one-shot或重建warm view；
- functional Adam无法复刻live state；
- compressed checkpoint无法恢复production visible state；
- terminal best不是当前轮却仍需零重跑handoff；
- evaluator/commit失败后只能靠retry/fallback继续；
- 24-worker artifact不能独立重算；
- S4-2必须执行KFSB、provider commit或queue mutation才能通过。

STOP后不得开放S4-3。

## 18. 当前门禁

```text
S3 external audit approval       pending
S4-0 implementation              closed
S4-1A/1B0/1B/1C/1D implementation closed
S4-2 construction design         complete
S4-2 implementation/formal       closed
S4-2 timing/performance          closed
S4-3/S4-4/S4-P                   closed
```

当前下一外部动作仍是审计：

```text
.docops/exchange/asplos27-s3-optimizer-runtime-20260828/request.md
.docops/exchange/asplos27-s3-optimizer-runtime-20260828/r001/delivery.md
```

本施工包不改变`.docops/s.md`的active stage/next，也不授权S4代码。
