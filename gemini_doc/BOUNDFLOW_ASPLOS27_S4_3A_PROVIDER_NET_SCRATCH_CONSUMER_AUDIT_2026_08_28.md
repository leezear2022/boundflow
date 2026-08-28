---
status: diagnostic-complete-corrected-code-closed
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
5. S4-3新增`ProviderNetScratchDisposalPlanV1`，镜像reference的move/gc生命周期。现场reference core探针纠正了
   初稿计数：`BatchedlA.from_net`只导出六条split-layer lA，但`gc_lA_from_net`会清空18个nonempty node lA。
   formal静态最低inventory因此是
   `6 α entries + 12 intermediate lower/upper attributes + 18 lA attributes = 36 attributes`；实际实现必须从live
   provider object枚举并冻结，不能把36硬编码为通用schema。

所以production α/β数值commit仍是12条semantic path；net scratch作为**lifetime/disposal transaction**单独receipt化，
不能混入12-path coverage数字，也不能在S4-P显存测量中忽略。

## 1. pinned事实基础

本次read-only审计基于：

- BoundFlow source probe HEAD：`3ca4d5c`（相对`34ae567`只有S4文档变更，runtime capture路径未改）；
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

## 6. `ProviderNetScratchDisposalPlanV1`

### 6.1 plan不是新IR

它是prepared runtime lifetime plan，只包含provider attribute binding：

```text
ScratchAttributeBindingV1:
    category                 # ALPHA / INTERMEDIATE_LOWER / INTERMEDIATE_UPPER / LA
    provider_object_identity
    attribute_or_mapping_key
    pre_object_identity
    pre_tensor_identity_or_sentinel
    disposal_sentinel_kind
    expected_reference_owner
    rollback_ordinal
```

不表达solver control、图、TIR或优化策略。

### 6.2 formal最低inventory

由live reference core probe得到当前formal fixture静态下界：

| category | expected formal最低数 | reference动作 |
|---|---:|---|
| terminal part-scope α entries | 6 | mapping entry→ValueError sentinel |
| intermediate lower | 6 | attribute→EmptiedTensor |
| intermediate upper | 6 | attribute→EmptiedTensor |
| exported split-layer lA | 6 | 进入`BatchedlA`，不作为全GC计数替代物 |
| all-node nonempty lA | 18 | attribute→EmptiedTensor |
| 合计disposal attributes | 36 | reference release/move mirror |

实现必须运行时枚举：

- `get_enabled_opt_act()`中的actual α part-scope entries；
- `layers_requiring_bounds`中actual nonempty lower/upper；
- `net.nodes()`中actual nonempty lA。

若actual不是formal预注册inventory，必须在任何mutation前拒绝并输出extra/missing path；不得截断到36。六条terminal
lA export与18条provider lA disposal必须使用不同字段和计数器。

### 6.3 β为何不在disposal list

reference `BetaFullData.from_net`不move/clear sparse β；working β可继续引用net对象直到post转换。因此candidate不应为了
“清干净”自行删除net β，否则反而偏离reference生命周期。

candidate必须如实披露：

- net仍持有pre/stale sparse β；
- core return持有candidate working β；
- 两者的logical/allocated bytes；
- exclusive latch保证stale net β不被读取。

若后续选择把candidate β attach到net以消除双份存储，必须另证alias、version、rollback和post owner，不能在S4-3A
顺手升级。

### 6.4 preserve mask与policy scratch

`last_update_preserve_mask`不是disposal tensor，但应作为policy mirror attribute：

- S4-2必须产生candidate preserve mask或exact None；
- S4-3 commit把net field设置为该projection，便于debug/provenance并消除stale值；
- receipt比较reference/candidate mask；
- attribute assignment可identity rollback；
- 它不计入12 production mutable paths或36 disposal attributes。

`constraints_optimized`在fixed output-constraint-disabled路径应被验证/镜像为None；`cut_used`必须为false；`bound_opts`只保存
canonical policy hash并由candidate prepare确定性设置。

## 7. transaction集成

### 7.1 expanded logical components

S4-3 logical transaction包含：

```text
12 production tensor content paths
1 host d packet
1 pre_result.interm_bounds container
N provider scratch attribute disposals      # current formal fixture expected 36
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
7. apply scratch sentinels；
8. apply preserve/constraint/cut mirrors；
9. seal latch与receipt；
10. official post。

步骤4之后任一失败仍为`POISONED_NO_RETRY`，即使Python references能exact恢复，因为tensor `_version`不能恢复。

### 7.3 success/post failure

successful core commit后scratch disposal已生效。official post失败时：

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
net_scratch_pre_inventory
net_scratch_post_core_inventory
net_scratch_post_post_inventory
alpha_value/sentinel counts
intermediate_tensor/sentinel counts
lA_tensor/sentinel counts
beta_object/value inventory
last_update_preserve_mask identity/content
constraints_optimized state
cut_used state
bound_opts policy hash
exclusive_owner_latch transitions
```

### 9.2 parity规则

- B0/R/C的queue-visible语义必须一致；
- R/C scratch disposal categories/keys/sentinel kinds exact；
- B0可有provider内部object id差异，但lifetime category/count应与R/C对齐；
- C net sparse β可保持stale，但必须披露且consumer count=0；
- C provider reentry/fallback=0；
- actual disposal inventory必须来自raw，不从expected summary复制；
- post后没有unexpected CUDA tensor retained byα/intermediate/lA paths。

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
18. `SCRATCH_PATH_MIXED_INTO_PRODUCTION_12_COUNT`。

fully re-signed tamper必须覆盖至少：删除一个disposal path、把lA sentinel改为tensor、改preserve mask、伪造
exclusive latch、把provider reentry从1改0、把multi-core从2改1、隐藏stale beta memory、把36 scratch错误并入12-path
commit count。

## 11. tests与实现切分

S3批准且S4-0—S4-2关闭后，S4-3A按短提交：

1. `test(adapter): inventory provider post-core scratch consumers`；
2. `feat(adapter): add exclusive core owner latch`；
3. `feat(runtime): compile provider scratch disposal plan`；
4. `test(runtime): mirror alpha/intermediate/lA disposal and rollback refs`；
5. `feat(runtime): bind preserve-mask and constraint scratch mirrors`；
6. `test(adapter): reject optional/mixed/multi-core paths`；
7. `artifact: record B0/R/C scratch lifetime projection`；
8. `docs: close S4-3A consumer/lifetime gate`。

不得与same-solver timing、compiled KFSB、pointer-swap或general multi-core支持合并。

## 12. GO / STOP

### GO

S4-3只有在以下新增条件也成立时才能关闭：

- normal fixed path从core return到post/queue没有net dynamic scratch read；
- actual scratch inventory完整且reference/candidate disposal parity通过；
- formal最低36 attributes逐项解释，extra/missing fail closed；六条export lA与18条GC lA分开核对；
- net β保留有显式consumer=0和memory披露；
- preserve mask/constraints/cut mirrors一致；
- exclusive owner latch阻止provider fallback/reentry；
- all-node/cut/clip/BFS/multitree/multi-core/reuse均在launch前拒绝；
- scratch disposal只在commit发生，precommit failure完全不变；
- mid/post failure进入对应poisoned状态；
- raw/replay/tamper覆盖lifetime而非只比较numeric result。

### STOP

任一情况停止：

- downstream实际读取未同步net scratch；
- candidate必须在post前调用provider core提取scratch；
- 不能枚举actual α/intermediate/lA owner；
- 为了fallback允许candidate后provider reentry；
- all-node LP或cuts在formal中可达；
- 未清理scratch却声称显存不恶化；
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
