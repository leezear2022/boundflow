---
status: diagnostic-complete-code-closed
date: 2026-08-28
type: implementation-blueprint
topic: boundflow
slug: asplos27-s4-3-whole-core-exact-call-transaction
stage: s04
execution-authority: false
code-change-open: false
correctness-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
---

# ASPLOS'27 S4-3：whole-core exact-call事务与回滚实施蓝图

> 2026-08-29实施就绪修订：本稿的路线仍有效，但memory、working-β、rollback、lease和post/queue细节已由
> `BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_29.md`精确化。实现时以
> 后者为准；旧`608,942 B`、blanket restore与粗粒度post状态机不再是当前合同。

## 0. 直接结论

S4-3不是“把S4-2的terminal tensor塞回provider”的薄adapter。真实`update_bounds_core`事务同时改变或消费：

1. 六条production α与六条β live tensor；
2. `d`中的history/depths/thresholds及被prune掉的其他字段；
3. `pre_result.interm_bounds`的容器生命周期；
4. terminal lower、六条lA和六组shared intermediate bounds；
5. KFSB的三组candidate、72个child lower及最终六domain decision；
6. provider core-return兼容对象；
7. official `update_bounds_post`对queue-visible lower/upper/α/β的最终转换。

因此S4-3必须实现一个**prepared whole-core transaction**，而不是继续沿用CPU snapshot copy-out，也不能把现有
device commit v1直接宣称为完全原子。推荐路线冻结为：

```text
live pre-state + host packet + intermediate container
  → validate all identities/policy/topology/version before mutation
  → S4-2 sealed 10/9/10 driver
  → ordinal-9 one-shot terminal lower/lA handoff（不重跑CROWN）
  → existing native KFSB（3 × batch-24）
  → assemble provider-compatible core return（只构造类型，不调用bound callback）
  → PreparedWholeCoreTransactionV2
  → last-step device commit + host packet prune + intermediate-container clear
  → exactly one official provider update_bounds_post
```

本稿只冻结接口、ownership、失败语义、artifact与测试门禁。S3 exchange仍为`ready_for_audit`，所以S4-3代码、
GPU correctness、timing和性能claim全部关闭。

## 1. 为什么必须单独设计S4-3

### 1.1 S4-2的输出还不是solver事务结果

S4-2只应闭合sealed production policy内的10次evaluation、9次parameter mutation和10次scheduler call，并返回
terminal compressed α/β、lower与terminal handoff。它不拥有branch、queue、host domain packet或provider post。

如果在S4-2通过后直接返回lower，会遗漏：

- 6条lA对KFSB score的贡献；
- 3次child CROWN后的branch选择；
- 12条live state提交；
- `d`的字段prune；
- `pre_result.interm_bounds.clear()`；
- official post中的CPU materialization、beta conversion和`max(lb, lb_last)`语义。

因此“optimizer trajectory parity”不能自动升级为“whole-core same-solver correctness”。

### 1.2 现有device atomic v1只有content rollback

`fsg4_b3_device_atomic_commit.py`在commit前记录每个tensor的`_version`，commit成功后要求版本恰增加1；中途异常时
把12份backup内容copy回live tensor。这个机制能恢复tensor内容和host packet，但第二次copy会再次增加PyTorch
`_version`。PyTorch不提供合法API把该counter写回旧值。

所以当前能诚实声明的是：

- validation/staging失败：发生在任何live mutation之前，内容、object identity和`_version`均不变；
- mid-commit失败：live内容和host packet可恢复，但tensor `_version`可能漂移；
- mid-commit失败后不能透明fallback、重试同一事务或继续solver；必须标记
  `POISONED_NO_RETRY`并终止当前query/process边界。

除非另行证明pointer-swap不会破坏shared storage、alias、provider consumer和optimizer引用，S4-3不得使用“完全
可回滚”或“失败后状态完全不变”措辞。

### 1.3 provider callback为0不等于provider对象调用为0

现有live-return路径禁止provider `compute_bounds`和`update_bounds` callback，但仍通过provider dataclass/type factory
构造兼容返回对象。按当前固定路径，assembly调用以下12个constructor/factory：

| factory | 调用数 |
|---|---:|
| `BranchingDecisions` | 1 |
| `UpdateBoundCoreReturn` | 1 |
| `AlphaValueData` | 1 |
| `WorkingIntermediateBoundsInfo` | 1 |
| `IntermediateBoundsInfo` | 6 |
| `BatchedLAs` | 1 |
| `SubDomainClipDecisions` | 1 |
| 合计 | 12 |

之后host solver仍恰一次调用official `update_bounds_post`。因此receipt必须把以下计数分开：

```text
provider_bound_callback_count = 0
provider_return_constructor_call_count = 12
provider_postprocess_call_count = 1
```

把三者合并成`provider_call_count=0`会造成错误claim。

## 2. 已核对的真实reference事务

### 2.1 pinned source identity

本设计基于以下现场source：

- αβ-CROWN：`e5c7e17bf0488843acb77b7519f59876717a49f4`；
- auto_LiRPA：`5a098e8f9fb5786a428a024981d833d303921f2d`；
- BoundFlow本轮设计基线：`2c6dde8a8dd675573ddd5fb040e354ad07369dea`。

正式artifact必须重新绑定届时actual source，不能把上述哈希硬编码成通用schema条件。

### 2.2 reference timeline

真实provider `update_bounds_core`的语义顺序可归纳为：

```text
attach α/β to net
  → optimized bounds 10/9 trajectory
  → clear local interm_bounds container
  → clear pre_result.interm_bounds
  → update_bounds_precompute_extract
       - compute verified/unverified mask
       - prune d to history/depths/thresholds
       - assemble working α/β/intermediate/lA/branch fields
  → return UpdateBoundCoreReturn
  → official update_bounds_post
       - materialize queue-visible tensors
       - apply post lower/upper policy
       - convert α/β representation
       - return queue packet
```

RVIR reference把bound evaluation、terminal export和branch从provider callback中替换出去，但solver-visible事务结果仍应
与上述路径等价。

### 2.3 当前formal fixture的固定边界

本轮formal fixture具有：

- domain batch=`6`；
- 六个domain在core阶段均unverified；
- `n_splits=6`、`n_verified=0`；
- `x_Ls/x_Us=None`；
- clip decision empty；
- branching方法=`KfsbBranching`；
- final decision固定为：

```text
[[5,27], [5,32], [5,90], [5,90], [5,32], [5,90]]
```

这些数字是formal instance事实，不进入通用transaction schema。若verified mask不再全false、domain数量变化、clip开启、
multitree开启或brancher变化，v1必须fail closed，不能把“六domain happy path”写成一般solver支持。

## 3. ownership映射

| 对象 | reference owner | S4-3 candidate owner | 生命周期/提交点 |
|---|---|---|---|
| live source α/β | provider/net + solver | solver live targets | 最后一步device commit |
| compressed candidate α/β | 无 | S4-2 prepared runtime | prepare后到commit完成 |
| preserved α direction | pre snapshot | immutable source receipt | whole transaction |
| optimizer moment/LR/policy | native optimizer | S4-2 sealed driver | 10/9/10 trajectory |
| terminal lower | native final evaluation | ordinal-9 handoff lease | KFSB/core assembly前 |
| terminal lA | native export rerun或handoff | ordinal-9 one-shot handoff | KFSB后release |
| intermediate tensor values | `relu_pre`/pre_result | shared external immutable source | core assembly；不重算 |
| intermediate container | provider pre-result | transaction host target | commit时clear |
| KFSB children | native KFSB | existing native KFSB | 3 candidate调用 |
| final branch decision | KFSB | existing native KFSB | core return |
| provider return types | provider module | compatibility factory bridge | assembly only |
| `d` host packet | solver | transaction host target | commit时prune/replace |
| official post result | provider post | unchanged official post | commit成功后恰一次 |
| net内部scratch α/β | provider optimizer | non-authoritative scratch | 必须完成consumer audit |

关键约束：S4-3不能让同一对象同时被candidate runtime和provider optimizer拥有。provider net scratch若在后续阶段仍被读取，
必须纳入commit；若后续只消费core/post返回对象，则应由consumer audit证明其dead，而不是靠推测忽略。

## 4. `PreparedWholeCoreTransactionV2`

### 4.1 static plan

建议新增的runtime对象不是solver IR，而是typed transaction plan：

```text
WholeCoreTransactionPlanV2:
    source_identity
    mutable_path_inventory          # ordered 12 paths
    target_pointer_and_alias_map
    expected_target_versions
    host_packet_schema
    intermediate_container_identity
    provider_factory_identity
    official_post_identity
    rollback_order
    failure_policy                  # PRECOMMIT_CLEAN / POISONED_NO_RETRY
```

它只表达动态提交责任，不重复Bound/Plan/Task/Schedule/TIR。

### 4.2 prepared buffers

warm exact-call前至少预分配：

- 12条production candidate target的完整shape buffer；
- 12条rollback content buffer；
- host candidate packet；
- host rollback packet；
- intermediate-container rollback descriptor；
- provider-compatible working β对象所需storage，或证明constructor不产生headline动态GPU allocation；
- terminal bridge、KFSB和post所需固定views。

现有v1在stage时通过`clone()`构造full candidate，在commit时再为12条target动态`clone()`备份。S4-3 correctness可以
先保留并披露，但S4-P不得声称warm allocation为0；性能版必须改为prepared persistent buffers或逐项计入。

### 4.3 已知logical memory账

production full mutable candidate：

```text
α source = 8,496 float32 = 33,984 bytes
active β = 6 float32 = 24 bytes
candidate subtotal = 34,008 bytes
rollback subtotal  = 34,008 bytes
candidate + rollback = 68,016 bytes
```

S4-2 implementation-readiness已补齐step、compressed best、`ret_0`和validate-before-commit shadow，修正后的
known subtotal为`540,926 bytes`；S4-3还必须prepared持有upper `[6,1]`与depths `[6]`。加入candidate/rollback后：

```text
known S4-3 CUDA subtotal = 540,870 + 68,016 + 24 = 608,910 bytes
known S4-3 CPU subtotal  = 56 + 24 = 80 bytes
known S4-3 logical subtotal = 608,990 bytes
```

这**不是peak memory claim**。provider β location/sign的72 B是external retained liveness，不重复计入new allocation；
hot path必须以prepared bridge消除working-beta deepcopy。该账仍不含policy/pruner masks、KFSB child buffers、model/fixed
inputs、cuDNN/TVM workspace、allocator metadata、post D2H输出及shared intermediate source storage。

## 5. terminal export：禁止第11次CROWN

### 5.1 唯一合法来源

S4-3只能消费S4-2 ordinal 9同一次compiled evaluation生成的：

- terminal lower `[6,1]`；
- 六条terminal lA，共37,464 float32/149,856 bytes；
- intermediate source/version receipt。

不得调用`export_rvir_v4_native_backward`的full-CROWN rerun模式。合法assembly应等价于existing
`assemble_native_backward_from_terminal_handoff_v1`：只组装lower、lA和shared intermediate，不执行bound kernel。

### 5.2 one-shot lease

handoff receipt至少包含：

```text
terminal_ordinal = 9
terminal_lower_lease_count = 1
terminal_lA_lease_count = 1
terminal_lA_element_count = 37464
terminal_duplicate_crown_count = 0
intermediate_source_version_hash
lease_state = READY -> LEASED -> RELEASED
```

KFSB消费结束后lease必须release；最终core return的`batched_lA`在当前路径仍是empty，不能把terminal lA延长到queue
生命周期来隐藏clone或ownership问题。

## 6. KFSB保持existing owner，但必须完整计入

S4-3 correctness不重写KFSB。冻结行为为：

```text
candidate count = 3
per-candidate child batch = 6 domains × 4 children = 24
child CROWN count = 3
child lower count = 3 × 24 = 72
final decision count = 6
```

历史artifact中child lower最大差为`3.0994415283203125e-06`且sign exact；该数字只作门禁设计输入，S4-3必须从
新five-fresh raw独立重算。

KFSB receipt必须绑定每个candidate的：candidate ordinal、decision inventory、child state lineage、child lower raw、
score、winner及final branch decision。不得只存最终decision而无法判断三次child CROWN是否真的执行。

S4-P必须把KFSB三次native child CROWN单独计时。如果它成为瓶颈，后继是另行预注册的compiled expanded-batch KFSB，
不是从S4-3 scope中删除它。

## 7. core return兼容边界

### 7.1 exact fields

固定v1 core return必须逐字段比较：

- `lower_bounds`、`upper_bounds`及last layer shape；
- `lAs` inventory和shape；
- `working_alpha`、`working_beta`；
- 6组`interm_bounds`；
- `history`、`depths`、`thresholds`；
- `branching_decision`、`n_splits`、`split_depth`；
- `n_verified=0`；
- empty `batched_lA`；
- empty clip decision；
- `x_Ls/x_Us=None`。

不能只比较lower和decision。

### 7.2 provider compatibility桥

第一版允许使用provider return dataclass constructor，因为solver下游已经绑定这些types。允许的边界是“构造兼容
数据对象”，不允许调用provider bound computation。receipt必须记录12次constructor调用及其type identity/hash。

这部分是host integration overhead，S4-P必须计入。未来若替换为BoundFlow-owned typed return，需要单独证明所有downstream
consumer兼容；不能为减少英文计数或显示“provider-free”而越过验证。

## 8. host packet与intermediate container也属于事务

### 8.1 `d` packet

reference extraction把`d`裁剪为：

```text
history
depths
thresholds
```

其他字段（例如formal probe中的`discard_after_core`）必须在commit后消失。candidate packet、pre packet和rollback packet
都要绑定canonical hash。只更新三个字段但保留reference会删除的extra field，不是等价事务。

### 8.2 `pre_result.interm_bounds`

reference在core内部清空本地`interm_bounds`和`pre_result.interm_bounds`。当前device atomic transaction只覆盖12条tensor和
host packet，没有receipt化这个容器副作用。

S4-3必须新增：

```text
input_intermediate_container_identity
input_intermediate_container_pre_hash
candidate_intermediate_container_state = EMPTY
commit_clear_count = 1
rollback_restore_count
post_container_hash
```

container clear与host packet replace必须和12-path device commit处于同一个logical commit。若container不能安全rollback，
其mutation应排在所有可能失败的检查之后，并沿用`POISONED_NO_RETRY`失败语义。

## 9. device commit V2与失败分类

### 9.1 commit之前必须完成的检查

以下全部应在第一次live copy前完成：

1. source/code/model/property identity；
2. plan/policy/topology/lineage hash；
3. 调用existing `live_targets_from_pre_result_v4()`从current provider mapping重新枚举12条target；
4. current target逐path必须`is` S4-0 strong-ref lease中的原Tensor，并验证raw storage/pointer/alias；
5. tensor `_version`、shape/dtype/device/stride/offset与pre digest；
6. preserved α digest；
7. S4-2 terminal state和handoff lease；
8. KFSB inventory/decision；
9. provider factory和official post identity；
10. host packet candidate schema/hash；
11. intermediate-container identity/pre hash；
12. all finite/numeric/discrete gates；
13. claim flags全部false。

若provider mapping在S4-1A pack后被整体或局部替换，即使新Tensor内容、稳定group与`_version=0`完全相同，也必须以
`LIVE_LEASE_PROVIDER_REBIND`在copy前clean abort。commit必须写入current provider仍持有的原对象，不能写入lease保存的
detached旧对象后声称solver state已更新。事务结束后lease必须close；mid-commit poisoned路径也不得复用lease fallback/retry。

### 9.2 commit顺序

第一版建议冻结唯一顺序：

```text
12 device targets（stable rollback ordinal）
  → host d packet replacement/prune
  → pre_result.interm_bounds clear
  → commit receipt seal
```

所有target candidate和rollback buffer必须在commit前准备完毕。commit区域不允许编译、lazy import、planner decision、
provider bound callback或动态GPU allocation。

rollback只能恢复已经candidate-write的prefix，不能为了形式上的“12/12 restored”回写untouched suffix。即使prefix内容
恢复exact，`_version`仍不可逆，terminal仍为poisoned。

### 9.3 失败状态机

```text
UNCLAIMED -> PREPARED -> COMMITTING -> CORE_COMMITTED
          -> POSTPROCESSING -> POST_READY -> QUEUEING -> COMPLETED

pre-begin fault  -> PRECOMMIT_ABORTED_CLEAN
commit fault     -> COMMIT_POISONED
post fault       -> POST_POISONED
queue-add fault  -> QUEUE_POISONED
```

语义：

- `ABORTED_CLEAN`：可以安全报告失败；live内容、identity、version均未变；
- `COMMITTED`：可以进入official post；
- `POISONED_NO_RETRY`：不能调用native fallback、不能重新commit、不能继续queue；只能终止并保留fault artifact。
- `POST_POISONED`：12-path/host/container提交已经发生，official post没有形成合法queue result；
- `QUEUE_POISONED`：post已经完成，但candidate child queue insertion失败或部分发生；两者都不得伪装成precommit clean
  abort，也不得自动回滚、重调post或重试queue。当前query必须终止并冻结fault raw。

不得把`POISONED_NO_RETRY`伪写成“rollback success”。

### 9.4 pointer-swap只作为后续实验

理论上pointer/reference swap可避免in-place `_version`漂移，但会改变object identity和alias关系，并可能让provider net、
optimizer或downstream consumer继续持有旧tensor。只有完成以下审计才能开放：

- 所有live target consumer清单；
- shared-storage/alias清单；
- provider net scratch引用；
- optimizer引用；
- queue/post引用；
- swap后DLPack/module pointer合同。

在此之前S4-3不采用pointer-swap，也不把它列成已实现优化。

## 10. provider net scratch consumer audit

provider attach过程可能让net内部α通过`detach().requires_grad_(True)`共享原storage，β对象也被附着到net。candidate路径
绕开provider optimizer后，net scratch的consumer与lifetime已由
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_3A_PROVIDER_NET_SCRATCH_CONSUMER_AUDIT_2026_08_28.md`完成read-only
源码审计。固定v1结论为：

| consumer phase | 读取core return | 读取live source | 读取net scratch | 结论 |
|---|---:|---:|---:|---|
| BoundFlow native KFSB | 是 | typed terminal state | 否 | provider-independent |
| `update_bounds_post` | 是 | 否 | 否 | 函数无net参数 |
| queue insert | post结果 | 写domain storage | 否 | domain list成为owner |
| next pick/pre | 否 | 是 | 否 | 从domain packet重建 |
| next candidate core | typed pre | 是 | 否 | closed-world安全 |
| next provider core / all-node LP | 是 | 是 | 是/可能 | S4-v1必须拒绝 |

live probe证明reference terminal extraction中途move/gc
`6 α + 12 intermediate attributes + 18 all-node lA = 36 attributes`；只有六条split-layer lA进入terminal
`BatchedlA`。但B0随后三次provider KFSB child CROWN会重填batch-24 scratch，约`2,829,600 B` unique storage并保持到
solver return；当前R则把batch-12 stale scratch约`1,414,752 B`原样保留。两者都不是post/queue owner。

因此candidate必须使用`ProviderNetScratchFinalizationPlanV2`：B0只观测KFSB residue，R/C在native KFSB后把live枚举的
36 path规范化为sentinel，provider net β inventory必须为0；B0/R/C差异以
`NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`显式准入，不要求虚假disposal parity。首次candidate commit后锁定
query-scoped exclusive owner，禁止provider fallback/reentry。net scratch不增加production mutable tensor path的12计数，
但phase、logical/unique storage、alias、finalization必须进入transaction、memory disclosure、raw/replay/tamper；attribute
clear本身不构成即时CUDA free或memory claim。

## 11. official post是correctness scope的一部分

S4-3不能在core return处停止比较。official `update_bounds_post`还执行queue-visible转换，包括CPU materialization、α/β
conversion以及非deterministic配置下的lower合并策略。因此formal必须同时保存：

```text
core_result_raw
post_result_raw
solver_status / success / visited
queue-visible lower / upper / alpha / beta / history / depth / threshold
provider_postprocess_call_count = 1
query_total_domain_add_count = 2
candidate_post_domain_add_count = 1
```

query total包含一次initial unverified-domain add和一次candidate post add；这三个counter必须分别实测，不能从彼此推断。

历史live-return证据曾比较451个tensor-derived对象、213,060个sign元素，最大有限浮点差
`1.0669231414794922e-05`且sign exact；这些只作为覆盖规模参考。S4-3必须用新R/C raw重算，不能复制历史summary。

## 12. five-fresh correctness协议

### 12.1 对照

```text
R = RVIR provider-independent native whole-core reference
C = RVIR + S4-2 compiled evaluator + S4-3 transaction
B0 = original provider（额外semantic control，可选但推荐）
```

R/C必须在同一个`ABCrownSolver.verify`边界、同一model/property、seed、dtype、device、branch配置、timeout和pre-state下运行。

### 12.2 运行顺序

预注册五个fresh pair并交替顺序，例如：

```text
R/C, C/R, R/C, C/R, R/C
```

每个worker独立进程；部分结果不能resume成formal。raw先落盘，再从raw生成summary。

### 12.3 数值门禁

- lower/state：`atol=rtol=2e-4`；
- compiled internal gradient：`atol=rtol=2e-5`；
- KFSB child lower：`atol=rtol=2e-4`；
- 所有有限浮点逐tensor记录max abs/rel diff；
- lower、gradient、child lower sign exact；
- NaN/Inf inventory exact，未经预注册不得忽略。

### 12.4 离散门禁

以下必须exact：

- solver status/success/visited；
- evaluation/update/scheduler=`10/9/10`；
- mutable path、shape、dtype、device、preserved digest；
- terminal handoff/lease/rerun=`1/1/0`；
- KFSB candidate/child CROWN/child lower=`3/3/72`；
- final decisions；
- n_splits/n_verified=`6/0`；
- provider bound callbacks=`0`；
- provider return constructors=`12`；
- official postprocess=`1`；
- query total domain add=`2`；
- candidate post domain add=`1`；
- committed paths=`12`；
- host packet字段恰为history/depths/thresholds；
- intermediate-container clear=`1`；
- fallback/native shadow/eager=`0`；
- claim flags全部false。

## 13. raw、replay与tamper

### 13.1 raw必须保留

- source/code blob/external repo/model/property identity；
- R/C每个fresh的pre snapshot与policy/topology hashes；
- 10-step S4-2 trajectory；
- terminal lower/lA/intermediate source receipt；
- KFSB三候选raw与72 child lower；
- core return和post return投影；
- transaction pre/candidate/post/rollback hashes和versions；
- host packet/container before/after；
- provider constructor/callback/post counters；
- solver verdict与queue-visible结果；
- fault-injection raw。

replay必须只从raw重算summary、numeric/discrete parity、memory ledger、receipt链和claim flags，不import运行时对象来复用
production verifier。

### 13.2 minimum fully re-signed tamper

至少覆盖以下26类；修改raw后同步重签外层digest仍必须被语义重算拒绝：

1. source commit；
2. code blob；
3. model/property digest；
4. policy/topology hash；
5. missing mutable path；
6. swapped α path；
7. active β location/sign；
8. preserved α drift；
9. terminal ordinal；
10. terminal lA element inventory；
11. terminal lease reuse；
12. duplicate terminal CROWN；
13. intermediate source version；
14. KFSB candidate count；
15. KFSB child batch/count；
16. child lower numeric value；
17. final branch decision；
18. provider bound callback counter；
19. provider constructor counter；
20. official post counter；
21. committed path count/order；
22. tensor pre/post version；
23. host packet extra field；
24. intermediate container not cleared；
25. poisoned failure被改写成clean rollback；
26. performance/same-solver/complete-query flag提前true。

## 14. negative/fault-injection门禁

除主预注册已有reason外，S4-3至少新增或精确化：

1. `PROVIDER_BOUND_CALLBACK_OBSERVED`；
2. `PROVIDER_RETURN_CONSTRUCTOR_COUNT_MISMATCH`；
3. `PROVIDER_POSTPROCESS_COUNT_MISMATCH`；
4. `HOST_PACKET_SCHEMA_OR_PRUNE_MISMATCH`；
5. `INTERMEDIATE_CONTAINER_IDENTITY_MISMATCH`；
6. `INTERMEDIATE_CONTAINER_NOT_CLEARED`；
7. `LIVE_TARGET_VERSION_STALE`；
8. `LIVE_TARGET_ALIAS_MISMATCH`；
9. `TERMINAL_HANDOFF_MISSING_OR_REUSED`；
10. `TERMINAL_DUPLICATE_CROWN`；
11. `KFSB_CHILD_INVENTORY_MISMATCH`；
12. `KFSB_BRANCH_DRIFT`；
13. `CORE_RETURN_SCHEMA_MISMATCH`；
14. `OFFICIAL_POST_RESULT_DRIFT`；
15. `MID_COMMIT_FAILURE_POISONED`；
16. `RETRY_AFTER_POISONED_FORBIDDEN`；
17. `FALLBACK_AFTER_PARTIAL_COMMIT_FORBIDDEN`；
18. `NET_SCRATCH_CONSUMER_UNRESOLVED`；
19. `OFFICIAL_POST_FAILURE_AFTER_COMMIT_POISONED`；
20. `QUEUE_CONTINUE_AFTER_POST_FAILURE_FORBIDDEN`。

fault injection至少覆盖：第1/6/12条device copy、host packet replacement、container clear、post entry、post中段和
post return前边界。每案必须
记录内容、identity和`_version`的实际恢复范围，不得只断言tensor值相等。

## 15. 实现切分

S3外审批准且S4-0—S4-2依次关闭后，按短提交推进：

1. `test(adapter): inventory whole-core provider transaction and consumers`；
2. `feat(runtime): prepare S4 whole-core candidate and rollback buffers`；
3. `feat(runtime): add terminal handoff-only backward export assembly`；
4. `feat(runtime): bind existing native KFSB with exact receipts`；
5. `feat(runtime): add host packet and intermediate-container transaction`；
6. `feat(runtime): add device atomic transaction v2 and poisoned failure state`；
7. `feat(adapter): assemble provider-compatible core return and official post`；
8. `test(adapter): close whole-core five-fresh and fault injection`；
9. `artifact: add S4-3 raw replay and fully re-signed tamper`；
10. `docs: close S4-3 and open S4-4 artifact closure`。

不得把transaction、same-solver timing、compiled KFSB和pointer-swap实验放进同一提交。

## 16. GO / STOP

### GO

只有下列全部成立才允许状态升级为`VALIDATED-S4-SAME-SOLVER-CORRECTNESS`：

- S4-2 compiled 10/9/10 trajectory已独立关闭；
- R/C five-fresh whole-core和official post均通过；
- terminal lower/lA来自ordinal 9且duplicate CROWN=0；
- KFSB 3/3/72及final decision exact；
- 12 live paths、host packet和intermediate container完成同一logical commit；
- provider bound callbacks=0、constructor=12、post=1；
- precommit失败clean，mid-commit失败明确poisoned且禁止fallback/retry；
- official post/queue add失败分别明确为`POST_POISONED/QUEUE_POISONED`，禁止继续或伪装clean rollback；
- provider net scratch consumer audit无未决读取；
- replay PASS，minimum 26类tamper全拒绝；
- timing/performance/same-solver headline flag仍false。

### STOP

任一情况停止S4-3：

- 需要第11次CROWN补terminal lA；
- KFSB或post必须回到provider bound callback；
- verified mask、clip/multitree或domain形态超出formal v1却未fail closed；
- `pre_result.interm_bounds`副作用无法纳入transaction；
- net scratch后续consumer无法证明或同步；
- mid-commit失败后仍尝试native fallback/继续queue；
- official post失败后回滚并重试、继续queue或报告clean abort；
- 把content rollback误报为版本/identity完全回滚；
- 为过门禁隐藏provider constructor、post、KFSB或host integration成本。

## 17. 当前停止点

```text
S3 exchange = ready_for_audit / no audit result
S4-0..S4-2 implementation = closed
S4-3 implementation/correctness/artifact = closed
S4-P timing = closed
```

本文不改变DocOps当前`next=external-audit-asplos27-s3-optimizer-runtime`。它只把S4-3开工前最容易造成错误
correctness claim的事务边界提前冻结。
