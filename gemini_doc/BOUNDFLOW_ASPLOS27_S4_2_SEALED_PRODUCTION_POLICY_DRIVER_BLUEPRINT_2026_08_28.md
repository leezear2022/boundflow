---
topic: boundflow
slug: asplos27-s4-2-sealed-production-policy-driver
stage: s04
status: implementation-blueprint-gate-closed
execution-authority: false
code-change-open: false
correctness-claimed: false
performance-claimed: false
date: 2026-08-28
---

# ASPLOS'27 S4-2：sealed production policy driver与10/9轨迹实施蓝图

## 0. 结论

S4-2不能直接复用`execute_rvir_v4_native_optimizer_trace`、S3 P-anchor循环或
`execute_terminal_optimizer_with_lower_adjoint_handoff_v1`。这些路径已经证明10次CROWN evaluation、9次Adam
mutation、two-group LR、clamp和lower数值可以对齐，但它们没有完整拥有production
`auto_LiRPA._get_optimized_bounds`的下列事务：

- stop predicate与逐domain loss mask；
- iteration pruning的preserve-mask状态机；
- keep-best bound、checkpointed α/β、patience和restore-best；
- 60秒termination predicate；
- terminal iteration无参数update但仍发生的production scheduler step；
- optimizer step/m/v、clamp前后与terminal bridge的完整可重放receipt。

S4-2应新增一个**sealed、无任意callback、representation-neutral的host policy driver**。它只接受两个精确
evaluator实现：

1. `NativeDenseOracleEvaluatorV1`：production full α/sparse β的独立oracle；
2. `CompiledCompressedEvaluatorV1`：S4-1D lower-direction compressed α + active sparse β candidate。

两者共享同一个policy transition实现。这样比较的是evaluator差异，不是两套host loop差异。

本文只冻结实现和correctness门禁。S3 exchange仍为`ready_for_audit`；S3外审批准并关闭前，S4-0—S4-2代码、
GPU correctness和timing全部保持关闭。

## 1. 为什么现有10/9循环还不够

### 1.1 三条现有路径的覆盖边界

| 路径 | 已覆盖 | 未覆盖/不应承担 |
|---|---|---|
| `rvir_v4_native_optimizer.py` | six-state、2 param groups、10/9、Adam、clamp、9个可见LR transition | keep-best、stop、patience、pruning、restore、terminal lA、production terminal scheduler call |
| `native_alpha_beta_optimizer_schedule.py` | typed Task/Schedule、select-best原型 | frozen production controls、two-LR decay、pruning/stop、compressed state、terminal bridge |
| `fsg4_b4a_terminal_lower_adjoint_handoff.py` | ordinal 9 lower+lA、无第11次CROWN | keep-best/stop/prune receipts、all-state compiled VJP、compressed optimizer owner |
| S3 optimizer wrapper | P-only compiled VJP、10/9 trajectory、host Adam | 其余五α、active β、production exact policy、same-solver state/terminal export |

因此“把S4 evaluator传给现有函数”会把未实现的policy语义藏在函数名后面，无法形成production exact-call证据。

### 1.2 pinned production源码的真实顺序

权威production源码为：

```text
/home/lee/Codes/alpha-beta-CROWN/auto_LiRPA @5a098e8
auto_LiRPA/optimized_bounds.py::_get_optimized_bounds
```

固定lower-only路径每个iteration的语义顺序是：

```text
evaluate bound
→ recover full shape if prior pruning active
→ compute stop predicate / loss mask
→ update best bound and patience
→ conditionally checkpoint best α/β
→ evaluate stop / patience / max-time exits
→ zero_grad
→ backward + Adam step                 # 非terminal iteration
→ beta nonnegative projection
→ alpha [0,1] projection
→ scheduler.step()                     # production在terminal iteration也调用
→ pruner.next_iter()
```

循环退出后还有：

```text
pruner.update_best(...)
→ restore best α/β/intermediate state
→ return best bound / terminal state
```

S4-2必须按此owner顺序建模，不允许只用`for ordinal in range(10)`无条件展开后声称“production policy”。

## 2. 当前冻结workload的独立事实

### 2.1 policy identity

冻结artifact：

```text
artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1
```

其production policy hash为：

```text
d36c40126cb8e518c095203381ddc4365ff29198167dde47a3be7af71cc3c679
```

固定controls：

| 字段 | 值 |
|---|---:|
| evaluations / parameter updates | `10 / 9` |
| α LR / β LR | `0.01 / 0.05` |
| scheduler gamma | `0.98` |
| objective | lower-only |
| loss reduction | production `reduction_sum` |
| Adam | enabled，two param groups |
| keep-best | true |
| early-stop patience | 10 |
| start-save-best | 0.5 |
| pruning-in-iteration | true |
| pruning threshold | 0.2 |
| max time | 60 s |
| float64 final iteration | false |
| fixed intermediate bounds | true |
| α / β enabled | true / true |
| shared α / cuts / gamma / directly-optimize | false / false / absent / absent |

### 2.2 raw重算出的实际分支结果

从`production_capture.pt`的10份raw lower独立重算：

```text
best_iteration_by_domain = [9, 9, 9, 9, 9, 9]
terminal_is_best_mask     = [T, T, T, T, T, T]
post mutable == step 9    = 12 / 12 paths
```

terminal lower为：

```text
[-0.3619601727, -0.4186353683, -0.4712548256,
 -0.3577542305, -0.4386272430, -0.4879784584]
```

这说明formal workload上六个domain均在ordinal 9取得最好lower，production restore-best最终恰好选择last state。
它**不说明keep-best可以删除**；相反，S4-2需要显式证明candidate得到同一winner和restore decision。

### 2.3 scheduler的两个计数口径

production在ordinal 0—8之后各执行一次update和`scheduler.step()`，因此ordinal 9 evaluation看到：

```text
lr_alpha = 0.01 * 0.98^9 = 0.0083374776213014975
lr_beta  = 0.05 * 0.98^9 = 0.041687388106507489
```

ordinal 9不做Adam update，但pinned production源码仍在loop尾执行第10次`scheduler.step()`，其不可再消费的post
LR为：

```text
post_lr_alpha = 0.01 * 0.98^10 = 0.0081707280688754665
post_lr_beta  = 0.05 * 0.98^10 = 0.040853640344377336
```

所以S4-2 receipt必须分开记录：

```text
evaluation_count              = 10
parameter_mutation_count      = 9
consumed_lr_transition_count  = 9
scheduler_step_call_count     = 10
terminal_post_lr_observable   = false
```

不得把“scheduler call 10”误写成“parameter update 10”，也不得因terminal post LR不再被消费就静默省略production
源码事实。

## 3. S4-2新增对象：不是新编译IR

本阶段不再引入Bound/Plan/Task/Schedule之外的新通用IR。新增的是prepared runtime contract与typed receipt：

### 3.1 `S4SealedPolicyProgramV1`

immutable、hashable，字段至少包括：

```text
policy_hash
topology_hash
mutable_admission_hash
evaluator_program_hash
ordered_parameter_abi_hash
evaluation_limit = 10
alpha_lr = 0.01
beta_lr = 0.05
lr_decay = 0.98
adam_hyperparameters
loss_reduction_id
stop_criterion_id
keep_best / patience / start_save_best
pruning policy + threshold
max_time
terminal_handoff_policy
```

它是compiled program的runtime policy descriptor，不表达kernel循环，也不lower到TIR。

### 3.2 `S4PolicyRuntimeStateV1`

每次run独占、不可跨query共享：

```text
evaluation_ordinal
parameter_mutation_count
scheduler_step_call_count
alpha_lr / beta_lr
active_domain_mask
pruning_active / next_pruning_active
best_lower[6,1]
best_state_checkpoint
best_iteration_by_domain[6]
patience
stop_predicate[6,1]
stop_all
timeout_predicate
optimizer step/m/v
parameter_version
terminal_decision
```

所有state transition由driver拥有；evaluator不得写这些字段。

### 3.3 `S4EvaluationRequestV1`

```text
evaluation_ordinal
parameter_version
active_domain_mask
ordered six-alpha views
ordered active-beta view + five empty tokens
fixed input/spec/intermediate/split/history identities
loss_upstream_mask
terminal_mode
```

`terminal_mode`在当前fixed protocol只有ordinal 9可为true。若stop/patience/time在0—8提前触发，driver必须真实走
termination分支并以稳定reason拒绝当前S4 fixed-workload admission；不得继续无条件执行到9。早停workload的零重跑
terminal handoff泛化另立门禁，不能暗中在本阶段扩大scope。

### 3.4 `S4EvaluationResultV1`

```text
lower[6,1]
d_lower_d_alpha[6 ordered buffers]
d_lower_d_active_beta[6,1]
empty_beta_gradient_tokens[5]
terminal_lA_lease | None
component_receipt
```

返回的是`d lower / d parameter`。driver根据production stop mask和`loss=-sum(lower_active)`在persistent gradient
view上施加符号与mask；evaluator不得自行决定loss、zero_grad、step、clamp或scheduler。

### 3.5 两个且仅两个evaluator

接口构造函数不接受任意callable。factory只接受closed enum：

```text
NATIVE_DENSE_ORACLE_V1
COMPILED_COMPRESSED_V1
```

任何第三种provider callback、expected-trace reader、fallback或native shadow均在prepare时拒绝。

## 4. optimizer representation与内存owner

### 4.1 candidate参数

延续S4-1A：

- 六个independent contiguous lower-α leaf：合计4,248 float32；
- 一个active β leaf：6 float32；
- 五个empty β只保留typed token；
- preserved α方向4,248元素来自immutable source snapshot，不是optimizer parameter；
- ordered parameter count=`7`，logical parameter/gradient bytes各`17,016`。

### 4.2 Adam prepared state

每个physical parameter拥有：

```text
step scalar
exp_avg (m)
exp_avg_sq (v)
```

candidate m+v为8,508 float32=`34,032 bytes`。7个step scalar的device/dtype/bytes取决于pinned PyTorch
functional Adam contract，必须在prepare receipt中逐项记录，不能把它们偷偷并入或排除memory claim。

correctness实现优先使用一个固定版本的functional Adam transition或显式等价方程，prepared时创建m/v/step；禁止用
“dummy zero-gradient step后再复位”的方式预热，因为那会制造不可审计的隐藏mutation。

必须用同一初始参数和gradient对照`torch.optim.Adam`，逐step比较parameter/m/v/step，证明以下默认项exact：

```text
betas=(0.9, 0.999)
eps=1e-8
weight_decay=0
amsgrad=false
maximize=false
foreach/capturable/differentiable/fused = live production values
```

最后四项不能靠当前环境默认值猜测；S4-2A必须从live production optimizer param groups/state冻结。

### 4.3 correctness logical ledger

S4-1D修正ledger为`438,726 bytes`。加入candidate m+v后，已知静态logical subtotal为：

```text
438,726 + 34,032 = 472,758 bytes
```

该数仍排除model/fixed input、cuDNN/TVM workspace、allocator metadata、7个step scalar、best-state checkpoint、pruner
mask和terminal bridge scratch。S4-2必须分项测量这些新增项，`472,758`不是peak显存claim。

## 5. sealed driver精确状态机

### 5.1 prepare

1. exact type检查program、evaluator、admission、topology和policy；
2. 绑定六α/active β/empty token的ordered ABI；
3. 创建persistent gradients、Adam m/v/step与best-state checkpoint；
4. 绑定evaluator prepared arenas、module/cache/pointer identity；
5. 生成oracle或candidate evaluator hash；
6. 验证live production optimizer defaults与policy hash；
7. 初始化ordinal/version/counters，不做evaluation或mutation。

### 5.2 每个ordinal

```text
assert ordinal == parameter_version == parameter_mutation_count
assert current LR == base_lr * decay^ordinal
evaluate once
validate result before any mutation
compute stop mask and masked loss
update best bound / need_update / patience
if production checkpoint predicate:
    checkpoint improved α/β domains
evaluate stop_all / patience / max_time
if early terminal:
    take admitted termination branch or fail closed for current fixed scope
if ordinal < 9:
    write signed/masked gradients
    Adam update exactly once
    clamp α to [0,1], β to [0,+inf)
    increment parameter version
production scheduler.step exactly once
advance pruning state
```

ordinal 9 additionally要求：

- evaluation内部同一pass产生lower和six-lA one-shot lease；
- Adam update=`0`；
- projection mutation=`0`；
- production scheduler call=`1`，但post LR不得被下一evaluation消费；
- terminal dense bridge只能在S4-3消费；S4-2只验证round-trip和lease identity；
- terminal duplicate CROWN=`0`。

### 5.3 loop exit

1. 应用pruner final best-bound恢复；
2. restore checkpointed best α/β；
3. 生成compressed terminal state；
4. 一次compressed→native dense→compressed round-trip；
5. preserved α方向与source snapshot逐元素/hash不变；
6. terminal lease交给后继或释放；
7. sealed run不可再次执行。

## 6. keep-best、stop、patience与pruning的比较口径

### 6.1 每ordinal必须冻结的policy projection

```text
reduced_lower_by_domain
stop_mask / stop_all
loss_mask / loss_scalar
best_lower_before / after
improved_domain_mask
need_update
patience_before / after
checkpoint_predicate
checkpointed_domain_mask
best_iteration_by_domain
pruning_active / next_pruning_active
preserve_mask / next_preserve_mask
timeout_predicate
termination_reason
```

wall-clock elapsed值只披露，不要求oracle/candidate相等；两边的`timeout_predicate`必须同为false。任何一边触发60秒上限，
本轮pair无效且不得resume。

### 6.2 fixed workload的expected decision

S4-2 formal只在raw再次证明下列事实时准入：

```text
evaluations = 10
updates = 9
stop_all[0:9] = false
timeout[0:9] = false
best_iteration_by_domain = [9,9,9,9,9,9]
post_state == restored_best_state == ordinal9 state
```

pruning mask不能沿用旧artifact的推断；S4-2A新增observer必须从live pruner直接捕获。若oracle出现非identity preserve mask，
candidate必须用static D=6 mask语义逐步等价，不能因compiled shape固定而禁用production pruning。

### 6.3 不把inactive branch当作“已删除”

formal workload上stop/timeout/restore-to-earlier可能不触发。因此另加synthetic policy fixtures：

- ordinal 3 stop-all；
- patience超过10；
- timeout predicate；
- partial preserve mask；
- best winner来自不同ordinal/domain；
- no-gradient early exit。

这些fixture验证driver control，不形成ResNet性能或泛化claim。

## 7. trajectory parity

### 7.1 三层比较

1. **A：live production observer**：冻结真实controls、Adam defaults、pruner、keep-best与scheduler calls；
2. **B：sealed driver + native dense oracle evaluator**：必须先独立恢复A；
3. **C：sealed driver + compiled compressed evaluator**：只在B关闭后接入。

禁止直接用C对现有simplified native loop并据此升级production correctness。

### 7.2 每ordinal数值字段

- lower；
- six active lower-α before/gradient/after；
- active β before/gradient/after；
- five empty β token identity；
- α/β Adam m/v/step；
- current LR和post-scheduler LR；
- clamp-before/after；
- best-state checkpoint与restored terminal state；
- ordinal 9 six terminal lA。

### 7.3 容差与离散字段

```text
lower/state max abs/rel <= 2e-4
gradient/m/v max abs/rel <= 2e-5
sign exact
key/order/shape/dtype/device exact
all masks/ordinals/counters/decisions exact
empty token/preserved α exact
```

若functional Adam与live Adam在m/v或parameter trajectory超出门禁，STOP；不得只放宽lower容差掩盖optimizer漂移。

## 8. atomicity、异常与terminal bridge

- evaluator异常发生在Adam mutation前：parameter/m/v/LR/best/pruner/version全部rollback；
- Adam transition或clamp异常：整ordinal rollback，不能留下部分parameter group更新；
- scheduler异常：parameter update也必须rollback，保持ordinal原子性；
- result hash/shape/sign/nonfinite异常在loss与mutation前拒绝；
- terminal bridge只读terminal compressed state与immutable preserved source；
- bridge输出必须dense→compressed exact round-trip；
- 非terminal bridge、重复bridge、lease重复消费、preserved direction改写均拒绝；
- S4-2不执行KFSB、live commit或queue mutation，这些留给S4-3。

## 9. receipt与artifact

### 9.1 per-step raw

每step保存原始tensor或可独立重算payload：

```text
request/result/component hashes
parameter/gradient/m/v/step before/after
lower + policy projection
LR before/after scheduler
clamp delta counts
best/pruner/stop/patience/time predicates
terminal lA if and only if terminal
launch/provider/fallback/native-shadow counters
```

只保存hash而不保存raw tensor，不能关闭S4-2。

### 9.2 run summary

至少包括：

```text
evaluation/update/scheduler-call = 10/9/10
candidate evaluator call = 10
provider/fallback/native-shadow = 0/0/0
best iteration vector
pruning transition ledger
termination reason
terminal handoff count = 1
terminal duplicate CROWN = 0
policy/program/evaluator/source hashes
performance_claimed = false
```

### 9.3 five-fresh与replay

- A/B先five-fresh关闭driver extraction；
- B/C再five-fresh关闭compiled trajectory；
- pair顺序预注册并交替，部分结果不得resume；
- replay从raw重算全部数值最大差、policy decisions、Adam equations、scheduler计数和terminal round-trip；
- source绑定BoundFlow commit、TVM submodule、production αβ-CROWN/auto_LiRPA、model/property与所有code blobs。

## 10. minimum negative/tamper门禁

至少覆盖：

1. 任意第三方evaluator/callback；
2. evaluator读取expected trace；
3. policy hash或live Adam defaults漂移；
4. α/β group交换、缺失或LR错误；
5. ordinal/version/update cardinality漂移；
6. terminal发生第10次parameter mutation；
7. scheduler call少1/多1或terminal post LR错误；
8. gradient符号或stop mask错误；
9. m/v/step缺失、重置或跨run复用；
10. clamp顺序错误或preserved α被clamp；
11. keep-best improvement/checkpoint/restore错误；
12. patience、stop或timeout predicate错误；
13. pruning preserve-mask或full/local domain映射错误；
14. active β location/sign/token漂移；
15. nonfinite lower/gradient/moment；
16. evaluator launch重复或缺失；
17. mutation失败后部分state残留；
18. nonterminal lA出现或terminal lA缺失；
19. terminal dense bridge重复/提前/round-trip不等；
20. preserved α direction漂移；
21. provider/fallback/native-shadow/eager非零；
22. raw删改后只重签外层digest；
23. performance/timing/same-solver flag提前true。

所有tamper必须在重新签名外层manifest后仍由语义重算拒绝。

## 11. 实现切分

S3外审批准、S4-0与S4-1D依次关闭后，S4-2按以下短提交执行：

1. `feat(runtime): capture live production policy decisions and Adam defaults`；
2. `feat(runtime): add sealed representation-neutral policy driver`；
3. `test(runtime): prove driver parity with native dense oracle`；
4. `feat(runtime): bind compiled compressed evaluator to sealed driver`；
5. `test(runtime): close all-state 10/9 optimizer trajectory`；
6. `artifact: add S4-2 raw replay and semantic tamper closure`；
7. `docs: close S4-2 and open S4-3 exact-call correctness`。

不得把A/B driver extraction、B/C compiled comparison与S4-3 whole-core接入合成一个提交。

## 12. GO / STOP

### GO

只有下列全部成立才关闭S4-2：

- A/B live-policy parity与B/C compiled parity均five-fresh通过；
- 10/9/10 evaluation/update/scheduler-call cardinalityexact；
- lower/state/gradient/moments通过冻结容差且sign exact；
- best/prune/stop/patience/timeout/restore decisions exact；
- terminal bridge round-trip exact、preserved α无漂移、duplicate CROWN=0；
- provider/fallback/native-shadow/eager=0；
- raw replay PASS，minimum 23类tamper全部拒绝；
- `performance_claimed=false`。

通过后只开放S4-3 whole-core exact-call correctness；timing/S4-P仍关闭。

S4-3的精确后继合同见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_EXACT_CALL_TRANSACTION_BLUEPRINT_2026_08_28.md`。它把terminal
handoff、existing KFSB、provider-compatible core return、official post、12-path device commit、host packet及
`pre_result.interm_bounds`副作用纳入同一个logical transaction，并诚实区分precommit clean abort与mid-commit
`POISONED_NO_RETRY`。

### STOP

任一情况停止：

- sealed driver不能独立恢复live production policy；
- 需要禁用pruning/keep-best/stop才能得到parity；
- functional Adam与live Adam trajectory无法满足门禁；
- compiled evaluator需要修改optimizer state或读取expected trace；
- preserved α进入optimizer或发生漂移；
- terminal handoff需要第11次CROWN；
- 为通过门禁放宽容差、隐藏scheduler terminal call或删除未触发policy分支。

## 13. 当前停止点

```text
S3 exchange = ready_for_audit / no audit result
S4-0..S4-1D implementation = closed
S4-2 implementation/correctness/artifact = closed
S4-3/S4-P = closed
```

本文是提前消除实现歧义的蓝图，不改变DocOps当前`next=external-audit-asplos27-s3-optimizer-runtime`，不形成
performance、same-solver、complete-query或ASPLOS-ready claim。
