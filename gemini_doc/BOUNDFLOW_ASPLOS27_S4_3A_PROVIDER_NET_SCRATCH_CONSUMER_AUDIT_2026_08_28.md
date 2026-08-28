---
status: diagnostic-complete-corrected-v2-code-closed
date: 2026-08-28
type: consumer-and-lifetime-audit
topic: boundflow
slug: asplos27-s4-3a-provider-net-scratch-consumer
stage: s04
execution-authority: false
code-change-open: false
correctness-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
---

# ASPLOS'27 S4-3A：provider net scratch consumer与lifetime审计

## 0. 直接结论

S4-3蓝图中的“provider net scratch consumer待核”已经从源码层闭合，但结论不是简单的“需要第13条数值commit”或
“全部dead”二选一：

1. 在当前固定S4-v1路径中，KFSB由BoundFlow provider-independent实现，official `update_bounds_post`只读取
   `core_result`，`BatchedDomainList.add/pick_out`只读写domain storage，下一次`update_bounds_pre`从domain packet重建
   α/β；因此post/queue/pre不会直接读取net内部α、sparse β、lA或intermediate lower/upper。
2. 但reference core会把terminal α、intermediate bounds和lA从net转移或清空。candidate若完全不处理net，会留下
   stale GPU object，造成生命周期和显存偏差；它们不能被称为“无关”。
3. `last_update_preserve_mask`只在auto_LiRPA pruner存在时被覆写；若candidate之后允许provider core重新进入，旧mask和
   未清理alpha key可能污染下一次provider计算。因此S4-v1必须在首次candidate commit后锁定
   `exclusive_core_owner=CANDIDATE`，同一query不允许provider fallback/mixed execution。
4. formal v1还必须fail closed拒绝all-node-split LP、cuts/BICCOS、clip、BFS/multitree和非固定KFSB；这些路径会在
   core之后或core内部重新读取net scratch。
5. terminal extraction阶段确实会move/gc
   `6 α entries + 12 intermediate lower/upper attributes + 18 lA attributes = 36 attributes`；其中
   `BatchedlA.from_net`只导出六条split-layer lA，不能替代18条all-node GC inventory。
6. 但36个sentinel不是B0 `update_bounds_core`的最终状态：随后三次provider KFSB child CROWN会重新填充net，留下
   batch-24 α/intermediate/lA residue并一直存活到solver return。当前provider-independent R则把core-entry已有的
   batch-12 stale scratch原样保留；两者都不是post/queue的数值owner。
7. `ProviderNetScratchFinalizationPlanV2`因此取代V1：B0只观测`PROVIDER_KFSB_RESIDUE`；R/C在native KFSB后把
   36个live α/intermediate/lA attribute规范化为sentinel，并要求provider net β inventory为0。R/C互相exact，B0/R/C
   的net scratch差异必须以`NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`显式准入，不能伪写成disposal parity。

所以production α/β数值commit仍是12条semantic path；net scratch作为**phase-aware lifetime/finalization transaction**
单独receipt化，
不能混入12-path coverage数字，也不能在S4-P显存测量中忽略。

## 1. pinned事实基础

本次read-only审计基于：

- BoundFlow probe执行HEAD：`6648859`；runtime capture代码自`3ca4d5c`未改，二者之间只有S4文档/DocOps变更；
- αβ-CROWN：`e5c7e17bf0488843acb77b7519f59876717a49f4`；
- auto_LiRPA：`5a098e8f9fb5786a428a024981d833d303921f2d`。

关键external source：

- `complete_verifier/activation_split/stage_preprocess.py`；
- `complete_verifier/activation_split/stage_solve.py`；
- `complete_verifier/activation_split/stage_postprocess.py`；
- `complete_verifier/activation_split/update_bounds_phases.py`；
- `complete_verifier/activation_split/decision_precompute.py`；
- `complete_verifier/branching_domains.py`；
- `complete_verifier/state/alpha.py`；
- `complete_verifier/state/beta.py`；
- `complete_verifier/state/intermediate_bounds.py`；
- `complete_verifier/state/lA.py`；
- `auto_LiRPA/auto_LiRPA/optimized_bounds.py`。

正式实现/证据必须重新绑定当时source，不得把上述commit或绝对路径写入通用runtime schema。

### 1.1 live reference core纠错探针

在固定ResNet2B property、CUDA、`max_iterations=1`、batch 64、β-CROWN 10 evaluations下，复用现有
`run_rvir_v4_production_state_capture.py` reference worker，并只在以下provider extraction入口外包只读observer：

- `WorkingIntermBoundsInfo.from_net(move=True)`；
- `BatchedlA.from_net(move=True)`与`BatchedlA.gc_lA_from_net`；
- `AlphaValueData.from_net(move=True)`；
- `BetaFullData.from_net`。

单次运行得到`core_count=1`，事实如下：

| category | live tensor/attribute | reference结果 | logical bytes |
|---|---:|---|---:|
| terminal part-scope α | 6 | 六个mapping entry均变为`ValueError` | 33,984 |
| intermediate lower/upper | 12 | 六层×lower/upper均变为`EmptiedTensor` | 299,712 |
| exported split-layer lA | 6 | 进入`BatchedlA`，总37,464 float32 | 149,856（属于下行18项） |
| all-node lA disposal | 18 | `gc_lA_from_net`把18个tensor attribute全部变为`EmptiedTensor` | 471,984 |
| sparse β containers | 6×1 | before/after容器与count不变 | 本探针未冻结tensor bytes |

18条lA disposal path为：`/input-1`、`/input`、`/input-4`、`/input-8`、`/input-12`、`/37`、`/38`、
`/39`、`/input-16`、`/input-20`、`/input-24`、`/43`、`/44`、`/45`、`/46`、`/input-28`、`/48`、`/49`。
第二次幂等GC仍枚举18个attribute，但其中tensor count已经为0。

本次pre-disposal logical tensor bytes合计：

```text
33,984 + 299,712 + 471,984 = 805,680 bytes
```

该数是逻辑tensor字节相加，不是unique-storage、peak allocated/reserved或性能claim；诊断worker结果位于临时目录并已
自动清理。S4-4 formal必须用冻结raw重新枚举、记录storage identity/alias并独立重放。本探针只用于纠正“六条terminal
lA等于六个GC attribute”的错误假设。

### 1.2 storage alias与terminal transfer事实

第二个只读probe在第一次terminal extraction前同时冻结tensor object/storage/data pointer：

- 36个disposal tensor逻辑合计`805,680 B`，但只有34个nonempty unique storages、`756,528 B`；
- lA `/37`与`/38`共享storage，`/45`与`/46`共享storage，所以logical bytes不能当unique storage；
- 六α return全部是新Tensor object，但`6/6`与net source共享storage/data pointer；
- 12 intermediate return同样是新Tensor object，但`12/12`共享source storage/data pointer；
- 六条export lA是transpose/view object，`6/6`共享source storage/data pointer；
- 因而把net attribute换成sentinel并不会立即释放这些return仍持有的storage。36项清理是owner/reference transfer，
  不是`805,680 B`即时free证明。

terminal sparse β只有`/input-28`的一组`val/loc/sign`非空，logical/unique storage均为`96 B`；其余layer的
`[6,0]` empty tensors不应因`data_ptr=0`被误判成跨layer alias。`BetaFullData.from_net`返回原list和原SparseBeta
object；field tensor可在后续KFSB中被替换。当前probe同时确认`last_update_preserve_mask=None`、`cut_used=false`、
`constraints_optimized=None`。

### 1.3 B0 post-KFSB residue与当前R stale scratch

第三、第四个probe分别覆盖B0 provider core和现有RVIR provider-independent R：

| phase/variant | α unique | intermediate unique | lA logical / unique | β unique | 合计unique |
|---|---:|---:|---:|---:|---:|
| B0 terminal pre-extract(batch 6) | 33,984 | 299,712 | 471,984 / 422,832 | 96 | 756,624 |
| B0 core/solver return residue(batch 24) | 135,936 | 1,198,848 | 1,887,936 / 1,494,720 | 96 | 2,829,600 |
| current R core-entry/return stale(batch 12) | 67,968 | 599,424 | 943,968 / 747,360 | 0 | 1,414,752 |

B0在terminal extraction之后执行三次batch-24 provider child CROWN；最终六α、12 intermediate和18 lA全部换成
batch-24新storage，并从core return到official post、queue和solver return保持object/storage exact。它是最后一次KFSB
候选计算残留，不是`core_result`或queue owner。

当前R的native KFSB/provider-independent exact-call则完全不读写provider net：core entry已有的batch-12 α/intermediate/lA
在core return和solver return保持`36/36` object与storage identity；net β inventory为0，provider compute/update callback和
fallback均为0。也就是说当前R尚未做scratch normalization，只是把旧值原样留在net。

这些数字来自临时诊断run，不是formal artifact或memory claim。尤其不能用B0 residue与R stale scratch之差宣称节省；
S4-4必须同步测量allocated/reserved、storage alias/lifetime和R/C finalization后的真实状态。

## 2. reference事务对net的读写

### 2.1 core入口前

`branch_and_bound_preprocess`从`BatchedDomainList.pick_out`获得domain packet，执行split/history更新，再调用
`update_bounds_pre`。这一段：

- 从domain storage恢复alphas、betas、history、split history、bounds、thresholds、c；
- `AlphaValueData.from_domain_dict(d)`只包装`d["alphas"]`；
- `BetaFullData.from_domain_dict(d)`按history重建SparseBeta；
- 尚未把动态α/β写入net。

### 2.2 provider core attach

`update_bounds_core`先：

```text
betas_by_layer.attach_to_net(net)
alphas_by_layer.attach_to_net(net)
net.net.set_bound_opts(...)
net.set_crown_bound_opts("beta")
```

其中：

- α attach用`alpha.detach().requires_grad_(True)`；新tensor与domain source共享storage；
- β attach调用`net.net.accept_beta`；
- cuts开启时还写`net.net.cut_used`并可能`set_cut_params`；
- clip/final-A路径可能修改`needed_A_dict`、intermediate bounds和domain state。

### 2.3 optimized CROWN期间

auto_LiRPA在10次evaluation中写或使用：

- activation `.alpha`；
- activation `.sparse_betas`；
- node `.lower/.upper/.lA`；
- `bound_opts`；
- `constraints_optimized`；
- `last_update_preserve_mask`；
- cuts/output constraints相关scratch；
- timer/profiler状态。

注意`last_update_preserve_mask`只在`if pruner:`分支中赋值；源码没有在每次optimized bounds入口无条件reset为None。
因此旧mask对mixed provider reentry不是天然安全的。

### 2.4 reference extract/disposal

固定decision-precompute路径执行：

1. `WorkingIntermBoundsInfo.from_net(move=True)`：把每个`layers_requiring_bounds`的lower/upper放入return owner，随后
   把原attribute替换为`EmptiedTensor`；
2. `BatchedlA.from_net(...)`读取lA，再由`BatchedlA.gc_lA_from_net`把所有非None lA替换为`EmptiedTensor`；
3. `AlphaValueData.from_net(..., move=True)`读取part-scope terminal α，并把对应`m.alpha[spec]`替换为ValueError sentinel；
4. `BetaFullData.from_net(...)`返回指向net sparse β的对象，**不清空β**；
5. 读取`last_update_preserve_mask`完成mask/pruning；
6. 读取`cut_used`决定new split history；
7. 组装branch/clip/core return。

这说明reference的net scratch生命周期本身是core语义的一部分，尤其会影响GPU live allocation。

## 3. core返回后的consumer闭包

### 3.1 `stage_solve` post-core读取

`branch_and_bound_solve`在core return后只直接使用：

- `prePacket.preResults.d_dict["depths"]`；
- `net.tot_ambi_nodes`；
- `net.solver_model_initialized`；
- all-node-split LP分支中的net/working intermediate/core bounds。

正常fixed path的`all_node_split_flag=false`，所以不会进入LP；但一旦为true，net又变成语义输入。S4-v1必须在
candidate launch前确认所有depth `< net.tot_ambi_nodes`且all-node-split LP不可达，否则拒绝。

### 3.2 KFSB

current candidate的`evaluate_rvir_v4_native_kfsb`只消费：

- BoundFlow `BFTaskModule`及parameters；
- `InputSpec`与linear spec；
- terminal `NativeAlphaBetaOptimizationState`；
- terminal backward export的six lA/intermediates；
- typed topology和thresholds。

它不接收provider net，也以profile hook禁止provider `compute_bounds/update_bounds`。所以S4 fixed KFSB不读net scratch。

### 3.3 official post

`update_bounds_post`函数签名没有net参数。它只读取`UpdateBoundCoreReturn`和static names/masks，并执行：

- lb/ub/lA transfer；
- working α转CPU/可能float16；
- working β转domain dict；
- intermediate unstable projection；
- `max(lb, lb_last)`/`min(ub, ub_last)`；
- branching/clip return type转换。

因此只要candidate core return逐字段完整，post不需要net内部α/β/lA/intermediate。

### 3.4 queue add/pick

`BatchedDomainList.add`只消费post `ret`和core history/depth/threshold，随后把：

- lower/upper/lA；
- alphas/betas；
- unstable bounds；
- c/threshold；
- decision/clip/history/depth

写入自身storage。它没有net引用。

下一轮`pick_out`再从这些storage恢复domain packet；`update_bounds_pre`也没有net参数。只有下一次
`update_bounds_core`才把domain α/β attach到net。

### 3.5 consumer判定表

| phase | 读core return | 读domain storage | 读net α/β | 读net lA/bounds | 判定 |
|---|---:|---:|---:|---:|---|
| candidate KFSB | 是（typed export） | 否 | 否 | 否 | net scratch不参与 |
| stage_solve normal path | 是 | depths | 否 | 否 | all-node false时安全 |
| all-node LP | 是 | 是 | 可能 | 是 | S4-v1拒绝 |
| official post | 是 | 否 | 否 | 否 | core return是owner |
| domains.add | post结果 | 写入 | 否 | 否 | domain list成为owner |
| next pick/pre | 否 | 是 | 否 | 否 | 从domain重建 |
| next candidate core | typed pre | 是 | 否 | 否 | closed-world安全 |
| next provider core | provider pre | 是 | 是/覆写不完备风险 | 是 | v1禁止mixed reentry |

## 4. 为什么不能简单称net scratch为dead

### 4.1 GPU lifetime偏差

reference extract会释放net对α/intermediate/lA GPU tensor的引用；candidate不做时，这些旧tensor仍可被net持有。即使
它们不再影响数值，也会：

- 增大allocated/reserved memory；
- 改变allocator reuse和后续workspace availability；
- 让S4-P memory/headline与B0不可比；
- 保留旧state，扩大意外consumer和debug泄漏面。

因此“没有semantic read”不等于“可以无限保留”。

### 4.2 mixed executor污染

provider next core的α attach只覆写domain packet里存在的key；candidate未执行reference `alpha_drop_unused/move`时，net中
额外key可能继续存在。`last_update_preserve_mask`也可能保留旧值。允许C→provider切换会引入难以证明的隐式state。

### 4.3 object identity与rollback

alpha/intermediate/lA disposal只是把provider object的attribute/reference换成sentinel，不需修改原tensor内容。它与12条
production tensor copy不同，可以通过保存旧Python object reference实现identity-exact rollback。但仍必须属于同一logical
commit，不能在candidate correctness尚未完成时提前clear。

## 5. S4-v1 closed-world owner合同

### 5.1 exclusive latch

建议adapter增加query-scoped：

```text
ExclusiveCoreOwnerLatchV1:
    query_identity
    state = UNCLAIMED | CANDIDATE_ACTIVE | COMPLETED | POISONED
    first_candidate_commit_ordinal
    provider_reentry_count
    fallback_count
```

规则：

- adapter安装后provider bound core callback已经禁止；
- 第一次candidate transaction开始前从`UNCLAIMED`进入commit state；
- 成功commit后=`CANDIDATE_ACTIVE`；
- 后续same-query core只能再次走同一admitted candidate signature；
- 任一unsupported signature、provider reentry或fallback请求立即fail closed；
- query成功post/termination后=`COMPLETED`；
- mid/post commit故障=`POISONED`；
- latch不能跨solver/query复用。

formal ResNet2B v1现有证据只有`core_count=1`。因此S4 correctness claim只覆盖一个whole-core exact-call；多core BaB、
signature变化和candidate→provider切换都需要后续独立artifact，不能从exclusive latch推断已支持。

### 5.2 禁止的fallback语义

在S4-v1中：

- admission失败不是“退回provider”；而是query fail closed；
- candidate launch后任何异常都不能调用provider core；
- candidate成功后下一core unsupported也不能provider fallback；
- external user若需要B0，应使用独立fresh solver process，而不是同query切换。

这与S4-4的B0/R/C独立subprocess协议一致。

## 6. `ProviderNetScratchFinalizationPlanV2`

### 6.1 plan不是新IR

V2取代只描述terminal move/gc的`ProviderNetScratchDisposalPlanV1`。它是prepared runtime lifetime plan，只包含
provider attribute binding、phase和variant policy：

```text
ScratchAttributeBindingV2:
    category                 # ALPHA / INTERMEDIATE_LOWER / INTERMEDIATE_UPPER / LA
    provider_object_identity
    attribute_or_mapping_key
    pre_object_identity
    pre_tensor_identity_or_sentinel
    storage_identity / alias_group
    terminal_transfer_sentinel_kind
    finalization_sentinel_kind
    rollback_ordinal

ProviderNetScratchFinalizationPlanV2:
    variant_policy           # B0_OBSERVE_RESIDUE / R_C_NORMALIZE
    core_entry_inventory
    terminal_transfer_audit  # B0-only source observation
    post_kfsb_inventory
    finalization_bindings
    beta_admission
    policy_mirrors
    exclusive_owner_latch
```

不表达solver control、图、TIR或优化策略。

### 6.2 terminal transfer inventory

由live reference core probe得到当前formal fixture静态下界：

| category | expected formal最低数 | reference动作 |
|---|---:|---|
| terminal part-scope α entries | 6 | mapping entry→ValueError sentinel |
| intermediate lower | 6 | attribute→EmptiedTensor |
| intermediate upper | 6 | attribute→EmptiedTensor |
| exported split-layer lA | 6 | 进入`BatchedlA`，不作为全GC计数替代物 |
| all-node nonempty lA | 18 | attribute→EmptiedTensor |
| 合计transfer attributes | 36 | terminal owner/reference transfer |

实现必须运行时枚举：

- `get_enabled_opt_act()`中的actual α part-scope entries；
- `layers_requiring_bounds`中actual nonempty lower/upper；
- `net.nodes()`中actual nonempty lA。

该36项描述B0 terminal extraction和R/C finalization的attribute path集合，不表示B0 core return为36个sentinel。
若actual path不是formal预注册inventory，必须在任何mutation前拒绝并输出extra/missing path；不得截断到36。六条
terminal lA export与18条all-node lA path必须使用不同字段和计数器；storage统计还必须保留alias group。

### 6.3 variant-specific final state

| variant | post-KFSB provider net policy | 当前fixture expectation |
|---|---|---|
| B0 | `B0_OBSERVE_RESIDUE` | batch-24 α/intermediate/lA residue + 6 sparse β containers |
| R | `R_C_NORMALIZE` | native KFSB后36 path全部sentinel；provider β tensor/container inventory=0 |
| C | `R_C_NORMALIZE` | 与R exact；active β只在typed/core result owner中 |

B0 residue不复制到R/C，因为它既不是queue-visible state，也没有后续consumer；伪造batch-24 residue会额外制造约2.83 MB
unique storage并重新引入provider scratch owner。R/C也不能继续沿用当前R的batch-12 stale值，而必须在logical commit中
规范化。

允许差异必须同时满足：

- source/动态probe证明post、queue和固定next-pre不读net scratch；
- query-scoped exclusive owner阻止provider reentry/reuse；
- B0、R、C raw分别记录entry、post-KFSB和solver-return inventory；
- R/C finalization exact，B0差异reason固定为`NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`；
- 不用该差异自动形成memory claim。

### 6.4 β admission与identity

reference B0的`BetaFullData.from_net`返回原list/SparseBeta object，KFSB期间非空`val/loc/sign`field storage可被替换，
最终仍留下六个container。R/C不模仿该residue：当前provider-independent入口的net β inventory为0，active β由typed
production state/core result拥有。

R/C admission必须：

- launch前要求provider net sparse β tensor/container inventory=0；
- 若非0则fail closed，不通过“顺手清理”接管未知alias；
- core result持有candidate working β并供official post转换；
- B0 raw披露六container、18 field tensor（仅3个nonempty、96 B）及field identity变化；
- R/C raw证明net β保持0，exclusive latch阻止重新attach/read。

若后续选择把candidate β attach到net以消除双份存储，必须另证alias、version、rollback和post owner，不能在S4-3A
顺手升级。

### 6.5 preserve mask与policy scratch

`last_update_preserve_mask`不是disposal tensor，但应作为policy mirror attribute：

- S4-2必须产生candidate preserve mask或exact None；
- S4-3 commit把net field设置为该projection，便于debug/provenance并消除stale值；
- receipt比较reference/candidate mask；
- attribute assignment可identity rollback；
- 它不计入12 production mutable paths或36 finalization attributes。

`constraints_optimized`在fixed output-constraint-disabled路径应被验证/镜像为None；`cut_used`必须为false；`bound_opts`只保存
canonical policy hash并由candidate prepare确定性设置。

## 7. transaction集成

### 7.1 expanded logical components

S4-3 logical transaction包含：

```text
12 production tensor content paths
1 host d packet
1 pre_result.interm_bounds container
N provider scratch attribute finalizations  # current formal fixture expected 36
1 variant finalization policy               # B0 observe / R-C normalize
policy mirrors                              # preserve mask / constraints / cut state
1 exclusive owner latch
```

计数必须分开。`committed_path_count=12`只能表示production α/β tensor paths，不能把scratch attributes混入后仍沿用12。

### 7.2 commit时序

推荐：

1. 完成S4-2、terminal handoff、KFSB、core return assembly；
2. validate 12 targets、host/container、scratch inventory、policy mirror与latch；
3. 准备tensor rollback contents和Python reference rollback table；
4. commit 12 tensor contents；
5. replace/prune host packet；
6. clear pre_result intermediate container；
7. apply R/C post-KFSB scratch finalization sentinels（B0只observe，不修改）；
8. apply preserve/constraint/cut mirrors；
9. seal latch与receipt；
10. official post。

步骤4之后任一失败仍为`POISONED_NO_RETRY`，即使Python references能exact恢复，因为tensor `_version`不能恢复。

### 7.3 success/post failure

successful R/C core commit后scratch normalization已生效。official post失败时：

- 不恢复net scratch；
- 不重attach旧α/β；
- 不调用provider core；
- latch=`POISONED`；
- failure=`COMMITTED_POST_FAILED_POISONED`；
- query终止。

## 8. unsupported optional path gates

S4-v1必须在candidate launch前固定拒绝：

1. cuts enabled；
2. BICCOS enabled；
3. CPLEX cut fetch/update；
4. `net.net.cut_used=true`或cut module active；
5. clip domains enabled；
6. domain BFS/precompute BFS；
7. multitree BaB；
8. branching method非fixed KFSB；
9. input-and-activation branching；
10. output constraints/invprop；
11. all-node-split depth reachable；
12. all-node-split LP enabled+reachable；
13. solver model initialization side effect required；
14. multiple core calls未预注册；
15. query/solver object reuse；
16. provider fallback/mixed executor request。

这些不是永久不支持，而是证明net scratch可不作为数值owner的前提。后续每开放一类，都要重新做consumer/lifetime审计。

## 9. formal evidence补充

### 9.1 B0/R/C scratch projection

S4-4每个worker新增：

```text
net_scratch_core_entry_inventory
net_scratch_terminal_pre_extract_inventory        # B0 only
net_scratch_terminal_post_transfer_inventory      # B0 only
net_scratch_post_kfsb_inventory
net_scratch_post_finalization_inventory
net_scratch_solver_return_inventory
alpha_value/sentinel counts
intermediate_tensor/sentinel counts
lA_tensor/sentinel counts
beta_object/value inventory
logical_bytes / unique_storage_bytes / alias_groups
tensor object / storage / data-pointer lineage
scratch_finalization_policy
last_update_preserve_mask identity/content
constraints_optimized state
cut_used state
bound_opts policy hash
exclusive_owner_latch transitions
```

### 9.2 parity规则

- B0/R/C的queue-visible语义必须一致；
- B0 terminal transfer 36项、post-KFSB batch-24 residue和solver-return identity逐phase自洽；
- R/C core-entry stale inventory如实记录，post-finalization 36个sentinel exact；
- B0 residue与R/C normalized差异必须命中固定allowed-difference reason，不能要求虚假lifetime parity；
- R/C provider net sparse β必须为0；active β只在typed/core result owner中；
- C provider reentry/fallback=0；
- actual phase/finalization inventory必须来自raw，不从expected summary复制；
- logical与unique storage分列，empty tensor `data_ptr=0`不得形成alias group；
- 不能把terminal transfer或R/C normalization写成即时free/allocated下降。

### 9.3 dynamic consumer probe

除了AST/source audit，formal建议用Python profile/attribute proxy observer记录core return之后到queue add完成之间对：

- activation `.alpha/.sparse_betas`；
- layer `.lower/.upper/.lA`；
- `last_update_preserve_mask`；
- `constraints_optimized/cut_used/bound_opts`

的读写。observer只做计数和path，不改变值；control/profile语义必须exact，扰动只作correctness诊断，不形成timing share。

## 10. 新增negative/tamper reasons

至少新增：

1. `PROVIDER_NET_SCRATCH_INVENTORY_MISMATCH`；
2. `PROVIDER_ALPHA_SCRATCH_NOT_DISPOSED`；
3. `PROVIDER_INTERMEDIATE_SCRATCH_NOT_DISPOSED`；
4. `PROVIDER_LA_SCRATCH_NOT_DISPOSED`；
5. `PROVIDER_SCRATCH_DISPOSED_BEFORE_COMMIT`；
6. `PROVIDER_SCRATCH_ROLLBACK_IDENTITY_DRIFT`；
7. `PRESERVE_MASK_MIRROR_MISMATCH`；
8. `STALE_PRESERVE_MASK_OBSERVED`；
9. `CUT_OR_CONSTRAINT_SCRATCH_ACTIVE`；
10. `ALL_NODE_SPLIT_PATH_UNSUPPORTED`；
11. `MULTI_CORE_QUERY_UNSUPPORTED`；
12. `SOLVER_OBJECT_REUSE_UNSUPPORTED`；
13. `EXCLUSIVE_CORE_OWNER_REENTRY`；
14. `PROVIDER_FALLBACK_AFTER_CANDIDATE_FORBIDDEN`；
15. `UNEXPECTED_NET_SCRATCH_READ_AFTER_CORE`；
16. `STALE_NET_BETA_CONSUMER_OBSERVED`；
17. `SCRATCH_LIFETIME_MEMORY_UNDISCLOSED`；
18. `SCRATCH_PATH_MIXED_INTO_PRODUCTION_12_COUNT`；
19. `SCRATCH_PHASE_ORDER_MISMATCH`；
20. `B0_KFSB_RESIDUE_PROJECTION_MISMATCH`；
21. `RC_FINALIZATION_NOT_NORMALIZED`；
22. `SCRATCH_LOGICAL_UNIQUE_STORAGE_CONFLATED`；
23. `SCRATCH_ALIAS_GROUP_MISMATCH`；
24. `EMPTY_STORAGE_FALSE_ALIAS`；
25. `BETA_FIELD_IDENTITY_LINEAGE_MISMATCH`；
26. `SCRATCH_IMMEDIATE_FREE_FALSE_CLAIM`。

fully re-signed tamper必须覆盖至少：删除一个finalization path、把lA sentinel改为tensor、改preserve mask、伪造
exclusive latch、把provider reentry从1改0、把multi-core从2改1、隐藏B0 β/residue、把36 scratch错误并入12-path、
交换terminal-transfer/post-KFSB phase、把B0 batch24改12、把R/C normalized改stale、删除alias group、用logical替代
unique storage、把empty `data_ptr=0`伪装成alias、声称attribute clear即时释放storage。

## 11. tests与实现切分

S3批准且S4-0—S4-2关闭后，S4-3A按短提交：

1. `test(adapter): inventory provider post-core scratch consumers`；
2. `feat(adapter): add exclusive core owner latch`；
3. `feat(runtime): compile provider scratch finalization plan v2`；
4. `test(runtime): audit B0 residue and normalize R/C scratch`；
5. `feat(runtime): bind preserve-mask and constraint scratch mirrors`；
6. `test(adapter): reject optional/mixed/multi-core paths`；
7. `artifact: record B0/R/C scratch lifetime projection`；
8. `docs: close S4-3A consumer/lifetime gate`。

不得与same-solver timing、compiled KFSB、pointer-swap或general multi-core支持合并。

## 12. GO / STOP

### GO

S4-3只有在以下新增条件也成立时才能关闭：

- normal fixed path从core return到post/queue没有net dynamic scratch read；
- B0 terminal transfer、post-KFSB residue与solver-return phase lineage完整；
- R/C finalization categories/keys/sentinel exact，且不再保留core-entry stale batch-12 scratch；
- formal最低36 attributes逐项解释，extra/missing fail closed；六条export lA与18条GC lA分开核对；
- B0 net β residue有显式consumer=0和memory披露，R/C provider net β inventory=0；
- preserve mask/constraints/cut mirrors一致；
- exclusive owner latch阻止provider fallback/reentry；
- all-node/cut/clip/BFS/multitree/multi-core/reuse均在launch前拒绝；
- R/C scratch finalization只在commit发生，precommit failure完全不变；
- logical/unique storage、alias和transfer-return shared storage均由raw重放；
- mid/post failure进入对应poisoned状态；
- raw/replay/tamper覆盖lifetime而非只比较numeric result。

### STOP

任一情况停止：

- downstream实际读取未同步net scratch；
- candidate必须在post前调用provider core提取scratch；
- 不能枚举actual α/intermediate/lA owner；
- 为了fallback允许candidate后provider reentry；
- all-node LP或cuts在formal中可达；
- R/C未规范化stale scratch却声称已清理；
- 把B0 KFSB residue当authoritative state，要求R/C额外重建；
- 把attribute sentinel replacement写成即时CUDA free或allocated下降；
- 把net scratch disposal伪写为第13条production α/β数值path；
- 通过硬编码36绕过动态inventory。

## 13. 当前停止点

```text
S3 exchange = ready_for_audit / no audit result
S4 code = closed
S4-3A source/consumer diagnostic = complete
S4-3A implementation/runtime evidence = closed
S4-P timing = closed
```

本文解决设计未决项，不改变DocOps当前`next=external-audit-asplos27-s3-optimizer-runtime`，不形成same-solver、
complete-query、memory或performance claim。
