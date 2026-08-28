---
status: diagnostic-complete-code-closed
date: 2026-08-29
type: implementation-readiness
topic: boundflow
slug: asplos27-s4-3-whole-core-transaction-readiness
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4-3：whole-core事务实现就绪审计

## 0. 直接结论

S4-3的计算语义和provider consumer已经足够清楚，但旧蓝图还不能直接照着实现。对真实B3-C same-solver路径做
live只读诊断后，本稿关闭了五个会造成错误实现或错误receipt的缺口：

1. `pre_result.interm_bounds`在当前B3-C candidate中没有被清空，六个key一直存活到worker结束；
2. current working-β assembly会在热路径执行`deepcopy`，产生allocator-visible动态分配；
3. current v1 fault rollback会回写全部12条target，使11条未被写过的tensor也推进`_version`；
4. official post之后的候选queue insertion只是本轮query第二次`BatchedDomainList.add`，不能把query-total=2误写成
   candidate-post=2；
5. host packet不是一步从20个key裁到3个key，而是先保留`betas/history/depths`供decision，再最终保留
   `thresholds/history/depths`。

因此S4-3 implementation必须使用prepared return bridge、prefix-only best-effort rollback和覆盖post/queue的exclusive
latch。S3 external exchange仍未批准，所以本稿只提供实现合同和可复核诊断，**不开放S4 production代码**。

## 1. 审计范围与证据边界

### 1.1 亲读源码

本轮核对的真实路径包括：

- BoundFlow `fsg4_b3_device_atomic_commit.py`、`fsg4_b3_device_live_return.py`、
  `rvir_v4_live_return.py`、`rvir_v4_atomic_copy_out.py`；
- pinned αβ-CROWN `activation_split/update_bounds_phases.py`、`decision_precompute.py`、
  `stage_solve.py`、`stage_postprocess.py`、`return_types.py`；
- pinned auto_LiRPA β state的`to_domain_dict()`消费点；
- actual B3-C same-solver worker和prepared executor context。

### 1.2 live probe纪律

probe只包裹已有函数并记录对象身份、tensor逻辑bytes、CUDA allocator delta、`_version`和调用ordinal；不改变
candidate数值、branch或solver配置。首次有两次失败必须保留为诊断边界：

- 一次导入了错误worker symbol，在solver前以`ImportError`停止；
- 一次只patch device assembly、未进入B3-C prepared context，因此没有device transaction记录。

后续结论只来自修正后的真实B3-C context。失败尝试不能混入formal evidence，也不能被写成candidate通过。

### 1.3 本稿不形成的claim

- 不形成S4 correctness closure；
- 不形成latency或memory peak claim；
- 不形成same-solver performance claim；
- live allocator数字只用于发现hidden allocation，不作为未来实现后的headline；
- 本稿不替代S3外审，也不改变`next=external-audit-asplos27-s3-optimizer-runtime`。

## 2. reference whole-core真实时序

reference事务应展开为：

```text
update_bounds_pre
  -> production optimizer evaluations/mutations
  -> terminal lower/lA/intermediate materialization
  -> first host projection keeps history/depths/betas
  -> native KFSB decision consumes betas/history/depths
  -> final host projection keeps thresholds/history/depths
  -> clear local interm_bounds
  -> clear pre_result.interm_bounds
  -> build core_result
  -> official update_bounds_post
  -> BatchedDomainList.add(candidate children)
```

`decision_precompute.py`的两阶段host裁剪必须进入合同。S4 candidate可以在内部采用更紧凑的结构，但formal raw必须
同时记录`host_intermediate_keys`和`host_final_keys`，不能只记录最终三键然后声称整个过程等价。

## 3. live B3-C事实

### 3.1 intermediate container没有被candidate清空

修正后的true B3-C run中，`pre_result.interm_bounds`从assembly前、assembly后、device commit后到worker结束始终是同一
dict对象，且始终含六个key：

```text
/39
/44
/input
/input-20
/input-28
/input-8
```

这证明旧device transaction只关闭了12条tensor target和host packet，没有关闭reference的container副作用。
S4-3必须把该dict纳入logical commit；“core numeric相同”不能替代container lifecycle等价。

### 3.2 host packet从20键最终裁到3键

commit前观察到20个key：

```text
alphas, batch_first_branching_decisions, betas, cs, depths,
final_lb, final_ub, global_lb, history, input_split_idx, lAs,
lower_bounds, split_history, sub_domain_clip_decisions, thresholds,
unstable_bounds, upper_bounds, x_Ls, x_Us
```

commit后exact为：

```text
depths, history, thresholds
```

最终schema与旧文档一致，但实现不得在native KFSB前删掉decision仍需要的`betas`。

### 3.3 current working-β bridge有hidden allocation

active β的逻辑inventory为`96 B`：value=`24 B`、location=`48 B`、sign=`24 B`。current assembly先
`copy.deepcopy(pre_result.betas_by_layer)`再替换`.val`：

- candidate `.val`最终正确alias transaction candidate；
- live allocated delta=`1,024 B`；
- transient peak delta=`2,048 B`；
- 删除临时对象后allocated delta回到0。

逻辑上只有location/sign共`72 B`需要保留，allocator却因deepcopy和size class产生更大动态开销。S4-3 hot path禁止
继续deepcopy。

### 3.4 current assembly另有dynamic upper allocation

真实assembly中：

- 6条working α全部alias transaction full candidate；
- 12条intermediate lower/upper全部exact alias terminal export；
- lower alias terminal export；
- upper `[6,1]`由assembly临时新建，逻辑`24 B`；
- assembly CUDA allocated delta总计`1,536 B`。

结合β bridge的`1,024 B` allocator delta，余下一个CUDA allocation size class与临时upper一致。S4-3应把upper
做成prepared persistent buffer，不在whole-core热路径创建。

### 3.5 provider constructor净计数仍是12

单次production assembly净构造：

| constructor | count |
|---|---:|
| BranchingDecisionData | 1 |
| UpdateBoundsCoreReturn | 1 |
| AlphaFullData | 1 |
| WorkingIntermediateBounds | 1 |
| IntermediateBounds | 6 |
| BatchedlAs | 1 |
| ClipDomainsInfo | 1 |
| 合计 | 12 |

probe自身为检查intermediate额外构造的对象必须从raw count扣除，不能把observer扰动写成production count。未来prepared
working-β wrapper在prepare阶段构造，hot assembly仍冻结为12个provider return constructor。

### 3.6 official post与queue事实

真实B3-C worker中：

- official post恰执行1次；
- post entry core tensor为CUDA unique/logical=`334,152 B`、CPU=`144 B`；
- post output为CPU unique/logical=`50,736 B`，另保留一个existing CUDA `c`=`240 B`；
- post阶段CUDA allocated delta=`0`；
- 当前execution schema没有可靠的`provider_postprocess_call_count`字段，不能从缺失/null推断为1。

query内`BatchedDomainList.add`总计2次：

1. ordinal 0发生在candidate post之前，把初始未验证domains从0加入到3；
2. ordinal 1发生在official post完成后，把candidate children从0加入到6。

所以receipt必须同时写：

```text
query_total_domain_add_count = 2
candidate_post_domain_add_count = 1
provider_postprocess_call_count = 1
```

三者必须来自真实observer/counter，禁止相互推断。

### 3.7 current v1 blanket rollback会污染未写target的version

故障注入点固定在第一条candidate copy完成后。current v1异常路径随后恢复全部12条target：

- copy seam总调用13次：1次candidate copy + 12次restore；
- 第一条已写target `_version`增量=`+2`；
- 其余11条从未candidate-write的target也因restore增量=`+1`；
- tensor content可恢复，host packet仍未改变。

因此“content恢复”不等于clean rollback。V2只能best-effort恢复**已提交prefix**，不得回写未触碰target；且发生第一条
device copy后无论恢复是否成功都必须进入poisoned terminal。

## 4. 修正后的prepared return bridge

### 4.1 owner划分

```text
PreparedS4WholeCoreTransactionV2
  owns:
    persistent upper [6,1]
    persistent depths [6]
    prepared working-beta Python wrapper
    12-path rollback buffers
    exclusive transaction latch
    tensor-free receipt builder
  borrows under leases:
    terminal lower
    terminal six-lA
    terminal 12 intermediate tensors
    full candidate alpha/beta values
    immutable beta location/sign
```

working-β wrapper只把candidate `.val`绑定到persistent candidate value；location/sign共享immutable provider source，要求
object/storage/version/content在post完成前保持有效。`BetaFullData.to_domain_dict()`只读取`.val`这一源码事实允许该设计，
但location/sign仍是验证语义和下一轮state identity的一部分，不能从receipt删除。

### 4.2 禁止的热路径行为

```text
hot_beta_deepcopy_count = 0
hot_dynamic_cuda_allocation_count = 0
hot_dynamic_cpu_allocation_count = 0
hot_tensor_clone_count = 0
prepared_beta_bridge_build_count = 1
```

Python小对象创建可单独披露；上述“allocation=0”只指tensor storage/allocator，不伪称Python runtime零分配。

### 4.3 terminal lease不能在KFSB后整体释放

lease按consumer拆分：

- terminal lA sublease可在native KFSB完成后释放；
- terminal lower必须保持到official post完成；
- fixed intermediate source/bridge必须保持到official post完成；
- working α/β candidate value必须保持到official post完成；
- rollback buffer在device commit成功后可释放；
- S4-0 live source lease在current-provider revalidation、commit与post consumer结束后关闭。

“KFSB consume后释放所有terminal state”是错误边界，会让core_result在post前悬空。

## 5. 修正后的memory ledger

修正后的S4-2 subtotal：CUDA=`491,718 B`、CPU=`56 B`、总计=`491,774 B`。S4-3新增：

| item | device | new logical bytes |
|---|---|---:|
| candidate + rollback buffers | CUDA | 68,016 |
| persistent upper `[6,1]` | CUDA | 24 |
| persistent depths `[6]` | CPU | 24 |
| immutable β location/sign lease | external retained | 72 |

因此新的known-new subtotal为：

```text
S4-3 CUDA = 491,718 + 68,016 + 24 = 559,758 B
S4-3 CPU  = 56 + 24                  =      80 B
S4-3 total-new logical              = 559,838 B
```

external retained的β location/sign `72 B`不重复计入new allocation，但必须计入liveness披露。该subtotal仍不是peak：
S4-2 scratch、candidate、rollback和post output不一定同时达到最大live set，CUDA allocator size class/workspace也未包含。
formal必须按phase记录`logical live / allocated / reserved / peak`，禁止把静态subtotal冒充峰值。

## 6. container与host logical commit

### 6.1 intermediate container

prepare阶段保存：

- exact built-in dict object identity；
- exact六key顺序与key set；
- 每个value的object/storage/version/content projection；
- shallow reference table，不clone tensor。

所有numeric、lease、candidate、KFSB、target和host preflight通过后，logical commit才可：

1. 对12条device target做stable copy；
2. 写最终host三键；
3. 对同一`pre_result.interm_bounds`执行exactly-once `clear()`。

candidate将clear从reference的KFSB前移动到KFSB后，必须在receipt中写成allowed reordering，并证明native KFSB无该
container consumer。该重排只为安全事务边界，不形成memory优化claim。

### 6.2 prefix-only best-effort rollback

```text
committed_prefix_count = k
restored_prefix_count  = k
untouched_suffix_restore_count = 0
```

每条path保存pre/candidate/post-fault version。发生第`k+1`条copy故障时，只恢复前`k`条已成功candidate-write路径；
当前故障path若copy可能部分写入，按backend copy原子性合同决定是否纳入prefix，不能凭异常位置猜测。

即使`restored_prefix_count == committed_prefix_count`且content exact，terminal仍是`COMMIT_POISONED`，因为PyTorch
`_version`不可逆。不得fallback、retry、继续post或继续queue。

## 7. 覆盖post/queue的exclusive latch

旧`UNCLAIMED/CANDIDATE_ACTIVE/COMPLETED/POISONED`太粗，无法阻止commit/post期间重入。冻结状态：

```text
UNCLAIMED
  -> PREPARED
  -> COMMITTING
  -> CORE_COMMITTED
  -> POSTPROCESSING
  -> POST_READY
  -> QUEUEING
  -> COMPLETED
```

failure terminals：

```text
PRECOMMIT_ABORTED_CLEAN
COMMIT_POISONED
POST_POISONED
QUEUE_POISONED
```

只有第一条state transition前的fail可进入clean terminal。一旦进入`COMMITTING`，任何失败都禁止native fallback/retry。
official post失败进入`POST_POISONED`；queue add失败进入`QUEUE_POISONED`，不能降级成post success或clean abort。

14条静态transition与修正memory ledger的canonical model hash为：

```text
833e8a9bfc72cfa4856d72765f888fe8d4416deb08cd46fdda57aedd406ccaf5
```

该hash只是设计模型的确定性指纹，未来实现必须从代码重新生成，不能硬编码成通过条件。

## 8. receipt最低字段

```text
transaction_state_before
transaction_state_after
failure_terminal_or_none
exclusive_generation

prepared_beta_bridge_build_count
hot_beta_deepcopy_count
hot_dynamic_cuda_allocation_count
hot_dynamic_cpu_allocation_count

device_target_count
committed_prefix_count
restored_prefix_count
untouched_suffix_restore_count
device_path_versions_before
device_path_versions_after_commit
device_path_versions_after_fault

host_intermediate_keys
host_final_keys
intermediate_container_size_before
intermediate_container_size_after
intermediate_container_clear_count

provider_core_call_count
provider_compute_bounds_call_count
provider_update_bounds_call_count
provider_fallback_call_count
provider_postprocess_call_count
query_total_domain_add_count
candidate_post_domain_add_count
check_worst_domain_call_count

terminal_lower_lease_state
terminal_la_lease_state
terminal_intermediate_lease_state
working_state_lease_state
```

receipt保持tensor-free；raw object/storage token只存在于进程内lease和worker raw投影，不能进入canonical artifact。

## 9. 新增negative与tamper门禁

在旧S4-3集合上至少新增：

1. intermediate container不是exact built-in dict；
2. container key set/order漂移；
3. success后container仍为6；
4. clear count不是1；
5. KFSB仍需container却提前clear；
6. hot working-β deepcopy非0；
7. hot tensor allocation非0；
8. dynamic upper allocation非0；
9. terminal lower lease在post前释放；
10. intermediate lease在post前释放；
11. post counter缺失、null或由其他counter推断；
12. query-total add=2被误写为candidate-post add=2；
13. candidate-post add不是1；
14. commit故障后恢复untouched suffix；
15. content恢复后错误标记clean；
16. `COMMITTING`期间第二次claim；
17. post故障后继续queue；
18. queue故障后retry/fallback；
19. host `betas`在decision前删除；
20. final host keys不是exact三键；
21. working β location/sign source替换或版本漂移；
22. prepared bridge重复build；
23. retained external bytes从liveness披露消失；
24. fully re-signed raw把post count 1改0；
25. fully re-signed raw把prefix-only rollback改blanket rollback；
26. fully re-signed raw把queue poisoned改clean abort。

## 10. formal run合同修正

S4-3自身仍采用R/C 5对、10个fresh subprocess做implementation correctness；S4-4再运行B0/R/C六全排列18
worker。每个S4-3 worker必须原始记录：

- core entry/exit、post entry/exit、queue add entry/exit ordinal；
- 12条device path pre/candidate/final/fault version；
- host两阶段key投影；
- container identity/size/clear ordinal；
- provider core/compute/update/fallback/post真实counter；
- query total add与candidate post add；
- phase memory与lease acquire/release；
- lower/lA/intermediate/state及post output完整IEEE numeric payload。

R/C correctness只能在official post和candidate queue insertion都成功后判PASS，不能在core return处提前结束。

## 11. 实现短提交顺序

S3 external approval且S4-0—S4-2依序关闭后，S4-3只按以下顺序开放：

1. `feat(runtime): add S4 whole-core exclusive latch`；
2. `feat(runtime): prepare provider working-beta bridge and persistent upper/depths`；
3. `feat(runtime): add lease-aware whole-core return assembly`；
4. `feat(runtime): add prefix-only device rollback receipt`；
5. `feat(runtime): commit host packet and intermediate container`；
6. `feat(adapter): observe official post and candidate queue insertion`；
7. `test(runtime): add S4-3 state/container/rollback/post negatives`；
8. `artifact: add five-pair whole-core correctness replay`；
9. `docs: close S4-3 and open S4-4`。

每一刀都必须保持provider compute/update/fallback为0；第6刀之前不得宣称whole-core complete。

## 12. GO / STOP

### GO

- 12-path、host final keys和container clear构成一个exclusive logical transaction；
- hot β deepcopy、dynamic upper和tensor allocation为0；
- prefix-only rollback不写untouched suffix，故障仍诚实poison；
- official post=1、candidate post queue add=1、query total add=2分别实测；
- terminal/working leases覆盖各自最后consumer；
- R/C 5对从core到queue完整闭合且raw可重放。

### STOP

- 需要通过blanket restore才能继续query；
- post仍依赖已经释放或复用的terminal storage；
- container clear无法纳入exclusive transaction；
- queue add失败后host solver无法停止或隔离poisoned state；
- receipt只能从summary推断post/queue，而没有实际counter；
- 为消除72 B metadata copy而放松identity/version/content guard。

## 13. 当前状态

```text
S4-3 source/consumer audit             = complete
S4-3 live transaction diagnostic       = complete
S4-3 implementation-readiness design   = complete
S4-3 production implementation         = closed by S3 external gate
S4-3 formal correctness                = closed
S4-3 timing/performance                = closed
```

下一外部动作仍是审计S3。S3无blocker后，也必须依S4-0→S4-1→S4-2→S4-3逐级关闭，不能因本稿已经详细就
跳过前置实现门禁。
