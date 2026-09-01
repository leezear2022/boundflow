# ASPLOS'27 S4-3：whole-core exact-call transaction 实施施工包

status: implementation-construction-design-only
date: 2026-08-29
execution-authority: false
code-change-open: false
formal-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false

## 0. 结论

S4-3 的生产代码仍因 S3 外审未批准而关闭，但 whole-core exact-call 的实现边界已经可以收束为一个可直接施工、
可故障注入、可从 raw 独立重放的 runtime transaction：

```text
S4-2 terminal capability
  -> native KFSB: 3 candidates / 3 child evaluations / 72 child lower
  -> provider-net scratch finalization
  -> provider-compatible core return assembly
  -> 12 device-path + host packet + intermediate-container logical commit
  -> exactly one official update_bounds_post
  -> exactly one candidate BatchedDomainList.add
  -> exactly one check_worst_domain completion
```

本施工包在旧 blueprint/readiness 基础上做八项机械修正：

1. **旧 14-state latch 只覆盖 commit/post/queue，不覆盖 terminal claim、KFSB 和 provider scratch**；新模型为
   `23 states / 22 events / 40 legal / 466 invalid`，canonical hash 为
   `6ed3d2fd946aaa0f6342f637a4754cc50eeec96e24392ed3b42adbbf92a3388a`；
2. **5 对 R/C 无法平衡运行顺序**；改为 6 对、12 个 fresh subprocess，`RC/CR=3/3`；
3. **logical commit 不是“12-path atomic copy”**；固定为 14 个 mutation ordinal：12 个 device target、一次 host
   final packet replace、一次 intermediate-container clear；
4. **provider-net scratch normalization 也是可失败 mutation**；36 个 α/intermediate/lA attribute 的 finalization
   独立 receipt 化，发生部分清理后只能 `STAGING_POISONED`，不得 clean retry；
5. **queue add 不是原子容器操作**；pinned provider 会依次修改 Python 列表、clip decision、TensorStorage、
   unstable bounds、α、threshold、C 和 `num_domains`。任一异常均进入 `QUEUE_POISONED`；只允许事后盘点 changed
   units，不声称 provider 内部 rollback；
6. **rollback 只恢复成功写入的 device prefix**；host/container 已开始 mutation 后即使 best-effort 恢复内容，
   `_version` 和对象历史也不可逆，仍为 `COMMIT_POISONED`；
7. **`559,838 B` 降级为 known base lower bound**；它不是完整 prepared transaction、provider scratch、post、
   queue storage 或 peak-memory 账；
8. **formal raw 重新机械计数**：在当前 fixed fixture 和冻结 semantic snapshot schema 下，R/C 每 worker 的
   mandatory tensor-occurrence floor 分别为 `3,897,248 / 2,537,888 B`；12 workers 总计至少
   `38,610,816 B = 36.8221435546875 MiB`。它不是物理文件下限，content-addressed sidecar 可去重。

这些结论只形成 implementation contract，不形成 S4 correctness、memory peak、same-solver speedup、complete-query
或 ASPLOS 性能 claim。

## 1. 权威输入与 source closure

### 1.1 仓库内权威输入

实现时按以下优先级读取：

1. S4 same-solver exact-call prereg；
2. S4 evaluator ABI 与 terminal handoff；
3. S4-0 V4 live admission/strong-ref lease；
4. S4-1A V5 ordered prepared buffer；
5. S4-1B/1C effective-value、compressed gradient、shared V/lA arena；
6. S4-1D opaque single-evaluation transaction；
7. S4-2 sealed policy driver construction package；
8. S4-3 blueprint、implementation readiness、provider-net scratch audit；
9. 本施工包。

若旧稿与本施工包在状态机、fresh topology、rollback、queue failure、raw floor 或 memory label 上冲突，以本稿为准。

### 1.2 pinned external source

```text
alpha-beta-CROWN commit = e5c7e17bf0488843acb77b7519f59876717a49f4
auto_LiRPA commit       = 5a098e8f9fb5786a428a024981d833d303921f2d

update_bounds_phases.py = 26d57bda8be6dbc690bfecf9723f467ec72f357c1765c38e18cfa42e00892dc7
decision_precompute.py   = 28ebd957d7c9e88061a576c466e39b201df4b37e1550818bc798a9d9016c1f26
stage_postprocess.py     = beea3e0c5ea1b18cfc4989d24935cba8841b2c0a85c012749dfa0a83bd95dd42
stage_solve.py           = 621ff0b51d226045b81d72e641aa90153a2752912cce7415dde320ba6e08a4cd
stage_preprocess.py      = c4ba8941de0fd8e0ef0c5ee0bbc08bc2d75f536e0885777ee289f03c37a022b3
return_types.py          = 23df4c57afa0c9725a94d5217b5298c7ec5447c90d586f7c863d843ad32ab33e
branching_domains.py     = d2726d32d7f0e347c97673aecc4efb9fa0e2ecbcccbdd3a3761cd7597c248b16
```

正式 artifact 必须从当次 checkout 重算 commit/blob；通用 runtime schema 禁止硬编码绝对路径或上述具体 hash。

### 1.3 fixed production scope

本版只承认：

```text
model/property              = fixed ResNet2B property 0
device/dtype                = CUDA / float32
bound side                  = lower
domains                     = 6
policy evaluations/mutation = 10 / 9
KFSB candidates             = 3
child batch per candidate   = 24
child lower total           = 72
split mode                  = fixed activation KFSB
all-node LP                 = false
cuts/BICCOS                 = false
clip                        = false
BFS/multitree               = false
provider fallback           = forbidden
exclusive core owner        = CANDIDATE after first candidate claim
```

任一开关、shape、topology 或 provider source 不同必须以稳定 reason 在 terminal claim 前拒绝；不得在运行中选择别的
owner 或回退 provider core。

## 2. 真实 whole-core 时序

### 2.1 reference 时序

```text
domain pick_out
  -> update_bounds_pre
  -> d has 20 keys
  -> provider core / optimizer
  -> first host projection: betas, history, depths
  -> KFSB decision consumes betas/history/depths
  -> final host projection: thresholds, history, depths
  -> core return
  -> update_bounds_post
  -> BatchedDomainList.add
  -> optional sort
  -> check_worst_domain
```

S4 candidate 可以用 typed private objects 重组内部顺序，但 queue-visible 语义、post 次数、add 次数、decision 输入、
host final keys 和 solver accounting 必须保持 exact。

### 2.2 candidate 时序

```text
preflight all static/live identities
  -> acquire process-global exact-call latch
  -> claim S4-2 terminal capability once
  -> run native KFSB once
  -> release terminal-lA sublease
  -> finalize 36 provider-net scratch attributes
  -> assemble provider-compatible core result from prepared owners
  -> validate all 14 commit mutation inputs
  -> commit 12 device paths in stable order
  -> replace final host packet
  -> clear pre_result.interm_bounds exactly once
  -> seal core commit
  -> call official update_bounds_post exactly once
  -> release post-consumed leases
  -> call candidate domains.add exactly once
  -> run check_worst_domain exactly once
  -> complete and close all owners
```

任何 post-claim 异常都不能回到 native/provider path。只有 terminal claim 前、live state identity/version/content 未变的
拒绝才是 clean abort。

## 3. runtime 对象，而不是新的 solver IR

S4-3 不新增 Bound/Plan/Task/Schedule 之外的通用 IR。以下对象属于 prepared runtime、transaction state 与 audit
receipt。

### 3.1 `S4WholeCoreTransactionPlanV1`

immutable、canonical、stable-hash，至少包含：

```text
schema_version
plan_hash
exact_call_contract_hash
S4_0_live_plan_hash
S4_1A_buffer_plan_hash
S4_2_policy_program_hash
topology_hash
provider_source_identity
device_path_specs[12]
device_commit_order[12]
host_intermediate_key_set
host_final_key_set
intermediate_container_key_set
provider_constructor_contract
provider_scratch_attribute_contract[36]
post_function_identity
queue_class_identity
queue_success_projection_schema
state_model_hash
claim_flags=false
```

plan 只描述合法事务，不持有 live Tensor、provider object、queue instance 或 host packet。

### 3.2 `PreparedWholeCoreTransactionV2`

在 exact-call 之前构造，独占：

```text
12 private candidate buffers                       34,008 B
12 private rollback buffers                        34,008 B
persistent upper [6,1]                                 24 B
persistent CPU depths [6]                              24 B
prepared working-beta wrapper
prepared provider return shells/descriptors
14-step commit cursor storage
23-state transaction latch
provider scratch finalization descriptor
fault/poison receipt builder
```

它借用但不拥有：

```text
S4-0 live source strong-ref lease
S4-2 terminal lower lease
S4-2 terminal lA lease
S4-2 terminal intermediate lease
S4-2 terminal working-state lease
provider pre_result / d packet
BatchedDomainList instance
```

### 3.3 `S4WholeCoreRunV1`

一次 exact-call 的唯一 mutable owner：

```text
current_state
transition_ordinal
terminal_capability_state
KFSB state
scratch_finalization state
commit cursor
committed_device_prefix_count
host mutation state
container mutation state
post state
queue state
lease states
poison reason
```

同一个 prepared plan 可以跨独立 exact-call 复用 immutable static 部分，但 mutable run、candidate content、rollback
content、capability 和 latch 不得复用。

### 3.4 `S4WholeCoreReceiptV1`

canonical receipt 必须 tensor-free、pointer-free、path-free，至少绑定：

```text
plan_hash
source_identity_hash
exact_call_id_hash
state_model_hash
transition_trace_hash
terminal_ordinal
terminal_is_best_all_domains
KFSB candidate/child/final-decision counters
provider scratch finalization counters
provider constructor counters
14 mutation outcomes
device prefix write/restore counts
host/container outcomes
post/add/check_worst counters
queue before/input/accepted/pruned/after counts
lease terminal states
poison reason
fallback/provider callback counters
performance_claimed=false
same_solver_claimed=false
```

raw tensor payload、object identity、storage pointer 和 `_version` 只进入进程内 observer 与 formal sidecar，不进入
canonical receipt。

## 4. 23-state exact-call 状态机

### 4.1 states

```text
UNCLAIMED
PREPARED
TERMINAL_CLAIMED
KFSB_RUNNING
KFSB_READY
SCRATCH_FINALIZING
SCRATCH_READY
CORE_STAGED
DEVICE_COMMITTING
HOST_COMMITTING
CONTAINER_COMMITTING
CORE_COMMIT_VALIDATING
CORE_COMMITTED
POSTPROCESSING
POST_READY
QUEUEING
COMPLETED
PRECOMMIT_ABORTED_CLEAN
STAGING_POISONED
COMMIT_POISONED
POST_POISONED
QUEUE_POISONED
CLOSED
```

### 4.2 events

```text
prepare
preflight_reject
claim_terminal
begin_kfsb
finish_kfsb
begin_scratch_finalize
finish_scratch_finalize
stage_core
begin_commit
finish_device
finish_host
finish_container
seal_core
begin_post
finish_post
begin_queue
finish_queue
staging_fail
commit_fail
post_fail
queue_fail
close_owner
```

### 4.3 40 legal transitions

主成功链：

```text
UNCLAIMED --prepare--> PREPARED
PREPARED --claim_terminal--> TERMINAL_CLAIMED
TERMINAL_CLAIMED --begin_kfsb--> KFSB_RUNNING
KFSB_RUNNING --finish_kfsb--> KFSB_READY
KFSB_READY --begin_scratch_finalize--> SCRATCH_FINALIZING
SCRATCH_FINALIZING --finish_scratch_finalize--> SCRATCH_READY
SCRATCH_READY --stage_core--> CORE_STAGED
CORE_STAGED --begin_commit--> DEVICE_COMMITTING
DEVICE_COMMITTING --finish_device--> HOST_COMMITTING
HOST_COMMITTING --finish_host--> CONTAINER_COMMITTING
CONTAINER_COMMITTING --finish_container--> CORE_COMMIT_VALIDATING
CORE_COMMIT_VALIDATING --seal_core--> CORE_COMMITTED
CORE_COMMITTED --begin_post--> POSTPROCESSING
POSTPROCESSING --finish_post--> POST_READY
POST_READY --begin_queue--> QUEUEING
QUEUEING --finish_queue--> COMPLETED
COMPLETED --close_owner--> CLOSED
```

失败链：

```text
UNCLAIMED/PREPARED --preflight_reject--> PRECOMMIT_ABORTED_CLEAN
TERMINAL_CLAIMED/KFSB_RUNNING/KFSB_READY/
SCRATCH_FINALIZING/SCRATCH_READY/CORE_STAGED --staging_fail--> STAGING_POISONED
DEVICE_COMMITTING/HOST_COMMITTING/CONTAINER_COMMITTING/
CORE_COMMIT_VALIDATING --commit_fail--> COMMIT_POISONED
CORE_COMMITTED/POSTPROCESSING --post_fail--> POST_POISONED
POST_READY/QUEUEING --queue_fail--> QUEUE_POISONED
```

`UNCLAIMED`、`PREPARED`、`COMPLETED`与五个 clean/poison terminal 可通过 `close_owner` 到 `CLOSED`；其余
state/event 组合全部拒绝。

机械模型：

```text
states  = 23
events  = 22
legal   = 40
invalid = 23*22 - 40 = 466
hash    = 6ed3d2fd946aaa0f6342f637a4754cc50eeec96e24392ed3b42adbbf92a3388a
```

### 4.4 clean 与 poisoned 的分界

- terminal capability 未 claim、live identity/content/version 未变：可 `PRECOMMIT_ABORTED_CLEAN`；
- terminal 已 claim，即使尚未写 provider live Tensor：失败也为 `STAGING_POISONED`，因为 one-shot terminal 和
  KFSB execution 不能重放为同一次 exact-call；
- 第一条 device copy 开始后：`COMMIT_POISONED`；
- core seal 后：post 失败为 `POST_POISONED`；
- post 完成后：add 或 check-worst 失败为 `QUEUE_POISONED`。

任何 poisoned state 都禁止 fallback、retry、second post、second add 或继续 solver。

## 5. terminal capability、KFSB 与 scratch finalization

### 5.1 terminal admission

claim 前必须验证：

```text
terminal capability owner/type/hash
terminal ordinal == 9
terminal is best == true for all 6 domains
handoff count == 1
terminal CROWN rerun == 0
lower/lA/intermediate/state source hashes
all leases READY
exact-call latch unclaimed
```

任一 domain best 来自更早 ordinal 时稳定拒绝；不得把 ordinal-9 lA 与 earlier-best α/β 拼成 core return。

### 5.2 native KFSB

KFSB 保持 existing provider-independent owner，必须完整计数：

```text
candidate count           = 3
child evaluation count    = 3
child batch               = 24 each
child lower elements      = 72
final decision domains    = 6
provider compute callback = 0
provider update callback  = 0
```

raw 保存每个 candidate 的 split inventory、child lineage、24 个 child lower、reduction、winner 和 final decision；
不得只保存 winner。

### 5.3 provider-net scratch finalization

KFSB 完成后，R/C 必须把 provider net 中固定 36 个 live attribute 规范化为 sentinel：

```text
6 alpha mapping entries
12 intermediate lower/upper attributes
18 all-node lA attributes
```

并验证 provider net β inventory 为 0。正式 receipt：

```text
scratch_attribute_expected = 36
scratch_attribute_seen     = 36
scratch_attribute_finalized= 36
scratch_beta_inventory     = 0
finalization_variant       = CANDIDATE_NORMALIZED
```

不能把 B0 的 batch-24 KFSB residue 与 R/C normalized sentinel 写成 scratch parity。B0 只记录
`PROVIDER_KFSB_RESIDUE`；差异标记为 `NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`。

若第 k 项 finalization 后异常，记录实际 prefix/changed inventory，进入 `STAGING_POISONED`；不尝试用旧 object 重建
provider scratch，也不允许 provider core reentry。

## 6. provider-compatible core return assembly

### 6.1 prepared bridge

热路径禁止：

- `deepcopy(working_beta)`；
- `torch.full_like` 动态构造 upper；
- `torch.as_tensor` 动态构造 depths；
- lazy import、lazy compile、planner decision；
- raw content hash或全量D2H；
- per-field fallback。

prepared owner 预先持有 upper `[6,1]`、CPU depths `[6]`、working-β wrapper 和 provider return shell。

### 6.2 exact provider constructor inventory

固定生产路径净计数为 12：

```text
BranchingDecisionData      1
UpdateBoundsCoreReturn     1
AlphaFullData              1
WorkingIntermediateBounds  1
IntermediateBounds         6
BatchedlAs                 1
ClipDomainsInfo            1
total                     12
```

observer 自身构造对象必须从计数排除。prepared shell 若改变 constructor 时点，只能改变 prepare/run 分段，不能改变
semantic object inventory；receipt 同时披露 prepare/run count。

### 6.3 alias/lifetime

- 6 working α 必须 alias transaction full candidate；
- active β `.val` 必须 alias candidate β value；
- β location/sign 以 immutable lease 保留到 post；
- 12 intermediate lower/upper 必须 alias terminal export；
- terminal lower 进入 core result；
- persistent upper 进入 core result；
- current path `batched_lA` 为 empty provider-compatible object；terminal lA 只供 KFSB；
- core result 及其借用 storage 必须活到 official post 完成。

## 7. 14-step logical commit

### 7.1 stable mutation order

device ordinal 0—11：

```text
0  alpha/%2F45/%2F49
1  alpha/%2F48/%2F49
2  alpha/%2Finput-12/%2F49
3  alpha/%2Finput-16/%2F49
4  alpha/%2Finput-24/%2F49
5  alpha/%2Finput-4/%2F49
6  beta/%2F39/0/value
7  beta/%2F44/0/value
8  beta/%2Finput-20/0/value
9  beta/%2Finput-28/0/value
10 beta/%2Finput-8/0/value
11 beta/%2Finput/0/value
```

logical ordinal 12—13：

```text
12 HOST_PACKET_FINAL_REPLACE
13 INTERMEDIATE_CONTAINER_CLEAR
```

host final packet 的 exact key set 为：

```text
depths
history
thresholds
```

KFSB 的 intermediate host projection 为：

```text
betas
depths
history
```

### 7.2 commit 前置检查

必须在 ordinal 0 前完成：

```text
all candidate/rollback buffers ready
12 current targets still `is` S4-0 leased objects
storage/alias/shape/stride/offset/dtype/device exact
current `_version` exact
current content exact
candidate finite/content exact
host 20-key source identity/version exact
host intermediate/final projections exact
pre_result.interm_bounds exact six-key identity/content/version projection
all terminal/working leases live
provider callbacks/fallback zero
```

preflight 不得在 ordinal 0 后补做。

### 7.3 device prefix rollback

每个成功 candidate write 后立刻推进：

```text
committed_device_prefix_count += 1
commit_cursor += 1
```

若第 `k` 条 copy 抛异常，只对 `[0,k)` 已成功写入 prefix 做逆序 best-effort content restore；未触碰 suffix 不写。

```text
candidate writes              = k
rollback writes attempted     = k
untouched suffix writes       = 0
final state                   = COMMIT_POISONED
retry/fallback/post/queue      = 0
```

即使 content 恢复 exact，已写 prefix 的 `_version` 至少增加 2，不能称 atomic rollback success。rollback 异常追加
stable secondary reason，但不能覆盖原始 commit fault。

### 7.4 host/container failure

12 device path 成功后，host replace 或 container clear 失败时：

- device live state已经 mutation；
- host/container可能部分 mutation；
- 可以为 fault artifact 做 best-effort host/container恢复；
- 不得把恢复成功升级成 clean；
- 最终一定是 `COMMIT_POISONED`。

core commit receipt 只有在 14 项全部完成、current provider revalidation通过、host exact keys、container empty 时才 seal。

## 8. official post

### 8.1 post 是 correctness scope

official `update_bounds_post` 还执行：

- GPU→CPU lower/upper/lA transfer；
- working α 转 CPU/可能转 float16；
- working β 转 domain dict；
- intermediate unstable projection；
- `max(lb, lb_last)`与`min(ub, ub_last)`策略；
- branching/clip object 转换。

所以 core return 通过不等于 S4-3 通过。

### 8.2 exact counters

```text
official_post_enter_count = 1
official_post_exit_count  = 1
post_shadow_count         = 0
post_retry_count          = 0
```

post 入口 live probe 的历史 fixed facts：CUDA core unique/logical `334,152 B`、CPU `144 B`；post output CPU
tensor logical `50,736 B`，另借用 existing CUDA `c=240 B`。历史 serialized semantic projection 的 tensor
occurrence为 `50,976 B = 50,736 + 240`。这些用于 raw schema设计，不是新实现 memory peak claim。

### 8.3 post failure

core 已 seal 后任何 post exception：

```text
state = POST_POISONED
queue add count = 0
post retry = 0
fallback = 0
query terminates = true
```

不得回滚 14-step commit 后重调 post。

## 9. queue insertion 不是原子操作

### 9.1 pinned add mutation surfaces

fixed candidate `BatchedDomainList.add` 至少触及：

```text
Python sequence owners:
  histories
  all_betas
  split_histories
  depths
  all_decision_split_depths
  all_decision_branching_decisions
  all_decision_branching_points

decision owner:
  all_clip_decisions

TensorStorage owners:
  all_global_lbs
  all_global_ubs
  all_lb_alls[/49]
  all_ub_alls[/49]
  unstable_interm_bounds[6 layers][lower/upper]
  all_alphas[6 activation/spec entries]
  all_thresholds
  Cs

scalar owner:
  num_domains
```

fixed path 的 lA、x_L/x_U/input-split storage 不新增；若现场出现，contract不匹配并拒绝。

### 9.2 success accounting

query 内 add 次数必须分开：

```text
initial domain add          = 1
candidate post domain add   = 1
query total add             = 2
candidate input domains     = 6
candidate accepted domains  = 6
candidate pruned domains    = 0
candidate before/after      = 0/6 for fixed fixture
```

`query total=2` 不能误写成 candidate add=2。

### 9.3 production 与 formal observer 分层

production success hot path只做 O(1) counter、before/after domain count和 pinned function identity；不对全部 queue storage
逐项hash。

formal observer 保存每个 mutation unit 的 before/after长度、shape、dtype、logical path与IEEE payload。发生 exception 后，
fault path允许执行昂贵的 best-effort snapshot，计算 changed unit inventory；无论 changed count 是0还是部分，状态都是
`QUEUE_POISONED`。

不得声称知道 provider 内部“最后成功 cursor”，除非 fault-injection wrapper明确绑定了注入 ordinal。生产 receipt 只记录：

```text
queue_call_started
queue_call_returned
queue_changed_unit_inventory_if_fault
queue_before_count
queue_after_count_if_observable
```

### 9.4 `check_worst_domain`

queue 阶段直到 `check_worst_domain` 与 rhs offset 完成才可 `finish_queue`。若 add 已成功但 check-worst 失败：

```text
candidate_add_exit_count = 1
check_worst_exit_count   = 0
state                    = QUEUE_POISONED
```

queue 已改变，不能重试 add 或继续下一轮 solver。

## 10. version、cursor 与 counter 分离

以下字段不得复用：

```text
evaluation_input_version       # S4-2 evaluator generation input
optimizer_mutation_count       # 9 Adam updates
storage_commit_generation      # S4-2 parameter storage commits
exact_call_transaction_ordinal # 23-state transition ordinal
device_commit_cursor           # 0..12 successful device writes
logical_commit_cursor          # 0..14 device+host+container
provider_scratch_changed_count # 0..36
official_post_count            # 0..1
candidate_queue_add_count      # 0..1
query_total_add_count          # fixed success=2
```

Tensor `_version` 是外部 PyTorch mutation history，只能观察，不能当作可回滚的 transaction generation。

## 11. lease 与最后 consumer

| lease | acquire | last consumer | release |
|---|---|---|---|
| terminal lower | terminal claim | official post | post success/fault close |
| terminal lA | terminal claim | native KFSB | KFSB success后立即 |
| terminal intermediate | terminal claim | official post | post success/fault close |
| terminal working α/β | terminal claim | official post | post success/fault close |
| S4-0 live source | exact-call prepare | current-provider revalidate/post | post success/fault close |
| candidate buffers | prepare | official post | post success/fault close |
| rollback buffers | prepare | core commit | core seal或commit fault close |
| provider core result | core stage | official post | post success/fault close |
| post result | post success | queue add | queue success/fault close |

terminal lA 可在 KFSB 后释放，但不能因此释放 terminal lower/intermediate/working state。provider post 使用后者。

任何 consumer 异常必须由 `finally` 关闭其仍拥有的 lease；poisoned state 不意味着允许泄漏。

## 12. memory 与 storage ledger

### 12.1 known base lower bound

沿用 S4-2 construction 的已知 base：

```text
S4-2 CUDA known base = 491,718 B
S4-2 CPU known base  =      56 B
S4-2 total           = 491,774 B
```

S4-3 新增：

```text
candidate buffers      CUDA = 34,008 B
rollback buffers       CUDA = 34,008 B
persistent upper       CUDA =     24 B
persistent depths      CPU  =     24 B
```

所以：

```text
S4-3 CUDA known base = 559,758 B
S4-3 CPU known base  =      80 B
S4-3 total           = 559,838 B
```

immutable β location/sign `72 B` 是 external retained lease，不重复计入 new allocation。

### 12.2 不能从 559,838 B 推断什么

它不包含：

- 完整 S4-2 policy object/storage；
- transaction/latch/receipt Python object；
- provider return shell metadata；
- KFSB workspace；
- provider net scratch live residue/finalization瞬态；
- post CPU output；
- queue TensorStorage增长；
- CUDA allocator size class、reserved、TVM workspace、cuDNN workspace；
- formal observer/raw sidecar。

所以它只能叫 **known tensor/base lower bound**，不能叫 total memory、peak allocated/reserved或 memory reduction。

### 12.3 liveness 测量点

formal至少记录：

```text
pre-terminal
terminal claimed
KFSB peak
scratch finalized
core staged
device commit peak
core sealed
post peak
post ready
queue add peak
solver return
```

每点同时保存 torch allocated/reserved、CUDA driver free、logical/unique storage和alias projection。不得用逻辑相加替代
allocator peak。

## 13. correctness comparison

### 13.1 variants

```text
R = RVIR provider-independent native whole-core reference
C = RVIR + S4-2 compiled evaluator + S4-3 transaction
B0 = original provider whole-core semantic control，S4-4使用
```

S4-3 closure只做 R/C；B0/R/C formal与timing仍属于 S4-4。

### 13.2 numerical parity

逐 worker、逐 semantic path：

```text
lower/state/core/post/KFSB child lower  atol=rtol=2e-4
compiled internal gradient/m/v          atol=rtol=2e-5
sign                                    exact
NaN/Inf inventory                       exact
shape/dtype/order                       exact
```

容差不能掩盖离散 decision、history、depth、threshold、split、counter 或 owner 漂移。

### 13.3 discrete parity

成功 worker 必须 exact：

```text
evaluation/mutation/scheduler = 10/9/10
terminal handoff/rerun        = 1/0
terminal ordinal/best         = 9/all true
KFSB candidate/eval/lower     = 3/3/72
final decisions               = 6 exact
provider callbacks/fallback   = 0/0
scratch finalized/beta        = 36/0
provider constructors         = 12
device/logical commit         = 12/14
host final keys               = depths,history,thresholds
container clear               = 1
official post                 = 1
candidate add/query add       = 1/2
check_worst completion        = 1
performance/same-solver flags = false/false
```

## 14. formal worker topology 与 raw floor

### 14.1 6-pair topology

5 pairs 会造成 3/2 顺序偏差，所以冻结：

```text
pair 0: R then C
pair 1: C then R
pair 2: R then C
pair 3: C then R
pair 4: R then C
pair 5: C then R

total = 6 pairs = 12 fresh subprocesses
RC/CR = 3/3
```

每个 worker 独立加载 source、prepare、verify、落 raw；partial run 不得 resume 为 formal。

### 14.2 semantic snapshot tensor-occurrence floor

S4-2 每 worker mandatory transition tensor floor：

```text
R native dense policy = 2,871,296 B
C compiled policy     = 1,511,936 B
```

S4-3 下游每 worker 至少保存以下独立 semantic occurrence：

```text
whole-core semantic projection = 821,976 B
  fields                         408 B
  terminal/KFSB branch trace 521,736 B
    candidate child lowers       288 B
  working intermediate       299,712 B
  history                       120 B

transaction pre/candidate/final = 102,024 B
  3 * 34,008

official post projection        = 50,976 B
queue-visible projection        = 50,976 B

downstream subtotal          = 1,025,952 B
```

于是：

```text
R per worker = 2,871,296 + 1,025,952 = 3,897,248 B
C per worker = 1,511,936 + 1,025,952 = 2,537,888 B

6*R + 6*C
= 38,610,816 B
= 36.8221435546875 MiB
```

这叫 tensor-occurrence floor：同一 storage 在不同阶段出现时按 semantic occurrence 计数。content-addressed binary
sidecar 可以让相同 IEEE payload 只存一次，因此 `38,610,816 B` 不是物理 artifact 文件大小下限。

### 14.3 floor 尚未包含

- queue fault snapshots；
- provider scratch pre/post raw；
- policy/checkpoint非tensor字段；
- source/model/property/input；
- object/storage/version/alias index；
- receipts、JSONL、manifest、stdout；
- negative/fault workers；
- filesystem padding/compression；
- environment/source trust anchor。

正式 artifact 必须披露 semantic occurrence、unique content bytes和physical file bytes三个口径。

## 15. replay 与 seal

replay 使用 stdlib reader，不 import BoundFlow runtime 或 provider verifier，至少重算：

1. source/code/model/property identity；
2. 23-state transition legality和hash；
3. terminal/KFSB/scratch counters；
4. 14-step commit order、prefix restore和versions；
5. host/container exact state；
6. core/post/queue numerical parity；
7. post/add/check-worst call accounting；
8. lease acquire/release；
9. raw occurrence/unique/physical byte ledgers；
10. claim flags。

seal DAG：

```text
source/model/property roots
  -> static plan + state model
  -> worker raw tensor index + binary sidecars
  -> worker semantic hash
  -> pair comparison
  -> summary
  -> tamper report
  -> replay stdout
  -> manifest
  -> external trust anchor
```

任何边都只能指向前序 immutable node，禁止循环地让 manifest hash进入其自身 source root。

## 16. negative 与 fully re-signed tamper

### 16.1 admission/terminal

至少覆盖：

1. source commit/blob mismatch；
2. unsupported solver switch；
3. exact-call latch重复claim；
4. terminal capability type/hash错误；
5. terminal ordinal非9；
6. earlier-best domain；
7. terminal lease missing/reused/released；
8. provider live rebind；
9. current target content/version漂移。

### 16.2 KFSB/scratch/assembly

10. KFSB candidate count非3；
11. child lower count非72；
12. provider callback非0；
13. final decision漂移；
14. terminal lA提前释放；
15. scratch attribute少/多/重排；
16. scratch partial finalization fault；
17. provider β inventory非0；
18. working-β hot deepcopy非0；
19. dynamic upper/depth allocation非0；
20. provider constructor count非12；
21. core alias/lifetime错误。

### 16.3 commit

22. candidate path缺失/重复/交换；
23. alias/stride/offset错误；
24. device copy fault ordinal 0—11各一；
25. rollback写 untouched suffix；
26. rollback顺序非逆prefix；
27. content restore被伪报为version restore；
28. host intermediate/final key错误；
29. host replace fault；
30. container key/content漂移；
31. container clear fault；
32. seal前漏做current-provider revalidation；
33. commit fault后retry/fallback/post。

### 16.4 post/queue

34. official post count 0或2；
35. post shadow调用；
36. post fault后继续add；
37. candidate add count 0或2；
38. query total add误写；
39. queue accepted/pruned错误；
40. queue partial mutation fault；
41. add成功后check-worst fault；
42. queue fault后retry/add/fallback；
43. queue-visible tensor/decision/history/depth/threshold漂移；
44. post/queue前lease释放。

### 16.5 artifact full resign

至少同步重签外层 digest 后仍拒绝：

45. state model hash；
46. legal transition改写；
47. terminal ordinal/best；
48. KFSB child lower；
49. scratch finalized count；
50. device commit order；
51. prefix restore count；
52. host final keys；
53. container clear；
54. post count；
55. candidate/query add count；
56. queue accepted count；
57. poisoned state改clean；
58. tensor sidecar bytes；
59. tensor occurrence/unique bytes；
60. `performance_claimed`或`same_solver_claimed`翻true。

fault injection 必须隔离进程；每个 poisoned worker退出后不得复用CUDA context、provider net或queue。

## 17. 文件与短提交施工顺序

S3外审批准后，建议严格按以下顺序，每刀独立测试：

1. `feat(runtime): add S4 whole-core plan and 23-state latch`；
2. `feat(runtime): prepare working-beta upper depths and rollback owners`；
3. `feat(runtime): add terminal and lease-aware KFSB staging`；
4. `feat(adapter): finalize provider net scratch for candidate owner`；
5. `feat(adapter): assemble prepared provider-compatible core return`；
6. `feat(runtime): add 14-step commit and prefix-only rollback`；
7. `feat(adapter): observe official post and queue transaction`；
8. `test(runtime): add S4-3 state commit post queue fault matrix`；
9. `artifact: add six-pair S4-3 raw replay and tamper`；
10. `docs: close S4-3 correctness and open S4-4 only`。

禁止把以下内容混入 S4-3：

- timing；
- compiled KFSB；
- pointer swap；
- CUDA Graph/multistream；
- B0/R/C three-way formal；
- complete-query claim。

## 18. GO / STOP

### 18.1 S4-3关闭条件

- 23-state/22-event/40-legal模型与hash exact；
- terminal capability/KFSB/scratch完整闭合；
- provider callbacks/fallback为0；
- hot working-β deepcopy、dynamic upper/depth allocation为0；
- 12 device + host + container 14-step logical commit成功；
- prefix rollback不写untouched suffix，所有post-claim failure诚实poison；
- official post、candidate add、check-worst分别exact一次；
- queue partial failure可诊断且query强制终止；
- lease覆盖最后consumer并最终全部closed；
- R/C 6对、12 fresh从terminal到queue全部通过；
- raw/replay/full-resign tamper通过；
- `performance_claimed=false`、`same_solver_claimed=false`保持。

### 18.2 STOP

任一项发生即停止，不开放 S4-4：

- terminal earlier-best仍被拼接；
- provider core/KFSB fallback进入；
- scratch partial finalization被当clean；
- host/container不在14-step commit；
- rollback写未触碰suffix或声称version恢复；
- post失败后重试；
- queue partial mutation后继续solver；
- formal只比较core、不比较post和queue；
- 5-pair顺序偏斜仍作为formal；
- 用`559,838 B`宣称完整memory或peak；
- 用历史summary替代新raw独立重算。

## 19. 当前门禁

当前 S3 exchange 仍为 `ready_for_audit`，所以：

```text
S4-0 production code closed
S4-1A/1B0/1B/1C/1D production code closed
S4-2 production code closed
S4-3 production code closed
S4 formal/timing closed
```

本施工包只把“批准后怎么实现”冻结到机械可执行程度。唯一外部下一动作仍是完成 S3 optimizer-runtime 外审；
无 blocker 后才按 S4-0→1A→1B0→1B→1C→1D→2→3→4 顺序实施。
