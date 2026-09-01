# ASPLOS'27 S4-4：formal artifact、stdlib replay、tamper与外部锚实施施工包

> **2026-08-29 challenge/witness所有权修订**：本稿16-node artifact seal DAG、96-case registry和external
> anchor字段继续有效，但单独由executor在Git/DocOps中写入的anchor最多形成`CHALLENGE_BOUND`，不能自动称为
> independent authenticity。正式关闭还必须有auditor预先发行challenge、auditor-controlled fresh run、独立重算和
> execution witness，达到`INDEPENDENTLY_WITNESSED`。新增外部流程节点不改artifact内部16-node/36-edge hash。
> 详细schema、状态机与W01—W24 negative suite由
> `BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`拥有。

status: implementation-construction-design-only
date: 2026-08-29
execution-authority: false
code-change-open: false
formal-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false

## 0. 结论

S4-4 的代码和 formal run 仍因 S3 外审及 S4-0—S4-3 顺序门禁而关闭，但 evidence pipeline 已经可以冻结成
一套可直接施工、可由纯标准库独立重放、不会把内部自洽冒充外部真实性的合同：

```text
clean tracked source + frozen protocol
  -> 18 positive subprocess: six B0/R/C permutations
  -> 15 isolated fault subprocess
  -> per-worker tensor index + content-addressed binary sidecar
  -> semantic root
  -> raw-derived summary
  -> layered tamper report
  -> deterministic replay stdout
  -> final manifest
  -> artifact-external Git/DocOps anchor
  -> anchored external replay
```

本施工包在旧 blueprint/readiness 上做十项机械修正：

1. **S4-3 状态机已经变化**：fault replay必须绑定新的
   `23 states / 22 events / 40 legal / 466 invalid`模型和hash
   `6ed3d2fd946aaa0f6342f637a4754cc50eeec96e24392ed3b42adbbf92a3388a`，不能继续使用旧粗粒度模型；
2. **15个fault点重新分层**：只有terminal claim前的live validation可clean；KFSB、scratch finalization与provider
   constructor失败统一为`STAGING_POISONED`；新增`CHECK_WORST_DOMAIN`故障；
3. **旧positive raw预算使用过时的`1,341,776 B` C trajectory**：改为S4-2施工包冻结的B0/R/C policy floor
   `2,837,288 / 2,871,296 / 1,511,936 B`；
4. **B0不应伪造candidate snapshot**：B0只保存pre/final两份12-path state=`68,016 B`；R/C保存
   pre/candidate/final=`102,024 B`；
5. **KFSB已在whole-core semantic projection中**：不再重复加旧`38,040 B`独立KFSB估算；
6. **18 positive mandatory tensor-occurrence floor**修正为
   `61,586,208 B = 58.733184814453125 MiB`；
7. **15 fault不能只算transaction**：每个故障worker都必须真实执行并保存完整C policy trajectory；仅此最低为
   `24,209,400 B`；positive+fault合计至少
   `85,795,608 B = 81.8210678100586 MiB`；
8. **tamper registry从历史71类扩为96类**：补入S4-2 version/cursor/terminal-best、S4-3 staging/scratch/
   14-step commit/check-worst、raw occurrence与anchor边界；其中95类必须由对应层拒绝，1类fresh-process
   attestation必须诚实输出`OFFLINE_UNATTESTABLE`并要求现场执行证据；
9. **seal DAG机械化为16 nodes / 36 edges**，hash=
   `01e179ea504f94c3e9720d5f63b318e34e912738d30c21d690f283b857ac491c`；
10. **真实性anchor必须在artifact信任域外**：artifact内便利副本永不自动采信；内部PASS只能写
    `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT`。

这些数字都是设计期 mandatory semantic occurrence、registry 与证据拓扑，不是实际压缩文件大小、correctness、memory
peak、speedup、same-solver performance、complete-query或ASPLOS-ready claim。

## 1. 权威输入与版本边界

### 1.1 仓库内输入

按以下顺序读取：

1. S4 same-solver prereg；
2. S4 evaluator ABI/terminal handoff；
3. S4-0 V4 construction package；
4. S4-1A V5、S4-1B0、S4-1B、S4-1C、S4-1D construction packages；
5. S4-2 policy construction package；
6. S4-3 whole-core construction package；
7. S4-4 blueprint/readiness；
8. 本施工包。

旧稿在fault terminal、71类tamper、raw budget或S4-3 state hash上与本稿冲突时，以本稿为准。

### 1.2 本施工包不开放实现

当前S3 exchange仍为：

```text
status         = ready_for_audit
round          = 1
approved_round = null
```

因此当前合法动作仅为设计、read-only source审计和既有测试；禁止新增S4 artifact生产代码、运行33-worker formal、
生成performance结果或创建“已验证”状态。

### 1.3 不是新的compiler IR

本阶段新增的protocol、tensor index、receipt、manifest和anchor都是artifact schema/runtime evidence，不是求解器IR，
也不进入Bound/Plan/Task/Schedule/TIR优化决策。

## 2. `S4FormalProtocolV1`

### 2.1 immutable protocol字段

```text
schema_version
protocol_hash
source_contract_hash
model/property/config hashes
positive_orders[6][3]
fault_registry_hash
tamper_registry_hash
seal_dag_hash
S4-3 state_model_hash
numeric_policy_registry_hash
tensor_index_schema_hash
positive_worker_count = 18
fault_worker_count = 15
total_worker_count = 33
performance_claimed = false
same_solver_claimed = false
complete_query_claimed = false
```

本稿另冻结一个不含未来source/input实值的`protocol_structure_projection`：positive orders、三个registry/DAG/state
hash、三类worker count和三个false claim flag。该结构投影hash为：

```text
71fcf182ef64198946099bce6acc3a96cb7574e5308b25b53ce7faaee9b24b8d
```

它不是未来formal的完整`protocol_hash`。完整hash必须在实现时加入source/input/numeric/tensor schema实值；不能把上方
structure hash硬编码成admission通过条件。

### 2.2 variant定义

```text
B0 = pinned original provider whole-core
R  = RVIR provider-independent native whole-core reference
C  = RVIR + S4-2 compiled evaluator + S4-3 transaction
```

比较边：

```text
B0 -> R  closes provider replacement semantics
R  -> C  closes compiled representation/evaluator semantics
B0 -> C  derived cross-check，不替代前两条归因边
```

B0 provider callback与scratch residue是variant-specific；不得把它们强行与R/C的callback=0、normalized scratch做
exact parity。queue-visible/core/post/state/decision语义仍须比较。

## 3. positive worker拓扑

### 3.1 六全排列

```text
triplet 0: B0, R,  C
triplet 1: B0, C,  R
triplet 2: R,  B0, C
triplet 3: R,  C,  B0
triplet 4: C,  B0, R
triplet 5: C,  R,  B0
```

exact worker ordinals：

```text
w00 B0   w01 R    w02 C
w03 B0   w04 C    w05 R
w06 R    w07 B0   w08 C
w09 R    w10 C    w11 B0
w12 C    w13 B0   w14 R
w15 C    w16 R    w17 B0
```

每个成员必须由stdlib parent使用显式interpreter启动独立`exec`；禁止在一个Python进程里顺序运行triplet成员，禁止
fork已初始化CUDA的parent。

### 3.2 positive cardinality

```text
triplet_count  = 6
order_count    = 6
B0_count       = 6
R_count        = 6
C_count        = 6
positive_count = 18
```

部分成功不得resume为formal；整批从新的空staging目录重新生成。

## 4. 15个isolated fault worker

### 4.1 registry

```text
F01 PRECLAIM_LIVE_VALIDATION       -> PRECOMMIT_ABORTED_CLEAN
F02 KFSB_AFTER_BEGIN               -> STAGING_POISONED
F03 SCRATCH_FINALIZATION_MID       -> STAGING_POISONED
F04 PROVIDER_CONSTRUCTOR           -> STAGING_POISONED
F05 DEVICE_COPY_01                 -> COMMIT_POISONED
F06 DEVICE_COPY_06                 -> COMMIT_POISONED
F07 DEVICE_COPY_12                 -> COMMIT_POISONED
F08 HOST_PACKET_REPLACE            -> COMMIT_POISONED
F09 INTERMEDIATE_CONTAINER_CLEAR   -> COMMIT_POISONED
F10 CORE_SEAL_REVALIDATION         -> COMMIT_POISONED
F11 OFFICIAL_POST_ENTRY            -> POST_POISONED
F12 OFFICIAL_POST_MATERIALIZATION  -> POST_POISONED
F13 OFFICIAL_POST_RETURN           -> POST_POISONED
F14 QUEUE_ADD_MID                  -> QUEUE_POISONED
F15 CHECK_WORST_DOMAIN             -> QUEUE_POISONED
```

canonical fault registry hash对
`[{"id": "Fxx", "name": "...", "terminal_state": "..."}]`按编号顺序做
`sort_keys=True,separators=(",", ":")`的canonical JSON SHA256：

```text
4b69d50391ff84d42a0d6ea5fb8c43d7b6f8040db4de5cd43d56cc2848256330
```

### 4.2 device ordinal语义

`DEVICE_COPY_01/06/12`是人类one-based第1/6/12项，对应zero-based ordinal `0/5/11`。receipt同时保存：

```text
fault_copy_ordinal_zero_based
committed_prefix_count
rollback_attempt_count
rollback_success_count
untouched_suffix_write_count = 0
versions_before/after_fault
```

异常可能发生在copy前或copy后；fault seam必须明确自己保证的边界，不能只凭Python异常位置猜测目标是否已经写入。

### 4.3 fault进程纪律

```text
w18 F01 ... w32 F15
```

每个worker只运行一个fault；poison后不得运行第二case、fallback、retry、second post或second queue add。非预期crash、
timeout或缺raw不是“预期fault通过”，而是整批formal失败。

### 4.4 fault raw最小内容

每个fault至少保存：

- 完整C policy trajectory；
- 12-path pre/candidate/fault三个semantic occurrence；
- terminal/KFSB/scratch/core/commit/post/queue已到达阶段的raw；
- host/container/queue changed-unit projection；
- state transition trace；
- lease terminal状态；
- original/secondary fault reason；
- fallback/retry/post/add/check-worst counters。

## 5. stdlib-only parent与process receipt

### 5.1 parent边界

parent只可import标准库；启动前验证：

```text
torch not in sys.modules
tvm not in sys.modules
CUDA_VISIBLE_DEVICES contract exact
explicit interpreter executable/hash exact
source/input/protocol pre-snapshot exact
output staging directory empty
```

不得静默使用PATH里的`python`。

### 5.2 worker receipt

```text
worker_ordinal
variant_or_fault_id
parent_nonce_hash
worker_nonce_hash
pid
parent_pid
OS process start-time tick
interpreter file digest
argv projection
sanitized env projection
CUDA initialization receipt
start/end monotonic ordinal
exit code
stdout/stderr digest
```

PID、start tick和nonce能发现重复/结构错误，但不是不可伪造attestation。offline replayer只能验证structure；fresh OS process
最终依赖parent执行纪律、external anchor和审计现场复跑。

### 5.3 timeout与失败

timeout必须按worker终止并使整批non-formal；不得把超时worker的已有raw与后续resume合并。非formal failure保留在
formal tree之外的sibling diagnostic目录。

## 6. source closure与clean scope

### 6.1 五层source identity

```text
L0 trusted revisions
   BoundFlow HEAD, TVM/TVM-FFI gitlinks, alpha-beta-CROWN, auto_LiRPA

L1 loaded Python inventory
   worker结束时sys.modules内trusted roots的真实文件

L2 declared execution inventory
   runner/worker/replayer/tamper + S4-0—S4-3 explicit entrypoints

L3 loaded native inventory
   /proc/self/maps relevant repo/native libraries, SHA256, size, build receipt

L4 compiled artifact identity
   Plan/TIR/module/device-source/cache receipts
```

旧诊断观察到101个BoundFlow core、4个repo script、559个TVM/TVM-FFI Python和3个repo native文件；这些是
旧worker的观测，不是未来S4-4 exact expected count。正式manifest记录33-worker variant-specific union和intersection。

### 6.2 clean字段分离

```text
full_worktree_clean
source_scope_clean
unrelated_dirty_paths
untracked_importable_code_count
```

`.docops`审计文档或PDF可作为披露后的unrelated dirty，不得因此伪写full clean；trusted import/build roots中的untracked
`.py/.pyc/.so`一律拒绝。

### 6.3 pre/post race closure

- worker 0 前冻结revision、gitlinks、external commits、declared code和inputs；
- 每worker保存loaded/native/compiled inventory；
- worker 32 后重新核对全部source-scope bytes；
- 任一worker看到不同source/native digest，整批non-formal；
- 禁止runtime-generated Python source；generated TIR/device code必须进入compiled receipt。

## 7. artifact目录与原子生成

### 7.1 tree

```text
artifact/
  protocol.json
  environment.json
  source_identity.json
  model_property_config.json
  numeric_policy_registry.json
  tamper_registry.json
  raw/
    positive/w00-b0/ ... w17-b0/
    fault/w18-f01/ ... w32-f15/
  worker_source_union.json
  semantic_root.json
  summary.json
  tamper_report.json
  replay_stdout.txt
  README.md
  manifest.json
```

每worker目录至少包含：

```text
worker.json
tensor_index.jsonl.gz
payload_index.jsonl.gz
tensor_payloads.bin.gz
trajectory.jsonl.gz
kfsb.jsonl.gz
transaction.jsonl.gz
post_queue.jsonl.gz
source_inventory.json
stdout.txt
stderr.txt
```

### 7.2 staging与publish

生成器使用目标同父目录下的唯一临时目录；全部33 workers、derive、自检、tamper和manifest完成后才原子rename到final
artifact path。final path预存在时fail closed，不覆盖、不续跑。

失败目录移到明确的non-formal sibling或保留供诊断；不得让partial tree出现在正式artifact路径。

## 8. `TensorIndexRecordV3`

### 8.1 canonical字段

```text
worker_ordinal
variant_or_fault_id
phase
semantic_path
ordinal
comparison_policy_id
dtype
shape
stride
storage_offset_elements
layout = strided
byte_order = little
payload_id
payload_nbytes
tensor_value_sha256
source_device_class
materialization_reason
object_group
storage_group
storage_nbytes
view_min_offset_bytes
view_max_offset_bytes_exclusive
```

唯一键：

```text
(worker_ordinal, phase, semantic_path, ordinal)
```

duplicate、missing、extra、unsorted或noncanonical path拒绝。

### 8.2 logical value与storage identity分层

payload保存tensor logical contiguous value，不保存整个backing storage。stride/offset/storage group记录原view/alias投影。
因此本schema可证明本次worker报告的alias/view合同，不能证明未被任何view覆盖的storage gap内容。

负stride、越过`storage_nbytes`、shape product overflow、unsupported layout拒绝。empty tensor即使payload相同也不能仅凭
`data_ptr=0`合并storage group。

### 8.3 stable group labels

`object_group/storage_group`只在单worker内稳定：按该group lexicographically-first semantic path派生标签。raw pointer、
Python `id()`、HOME、hostname或用户名不进入canonical artifact。

## 9. content-addressed payload sidecar

### 9.1 payload定义

```text
payload_id = sha256(raw logical bytes)
```

`payload_index.jsonl.gz`按payload_id排序，记录`offset/nbytes`。解压后的`tensor_payloads.bin`是所有unique payload按
payload_id排序直接连接：无header、无gap、无overlap、无orphan、无trailing bytes。

不同dtype/shape允许引用同一raw payload；semantic value hash另绑定dtype、shape和payload：

```text
tensor_value_sha256 = sha256(canonical(dtype, shape, payload_id))
```

### 9.2 worker局部去重

只在同一worker内去重；不同worker禁止共享payload文件或硬链接。删除任一worker目录必须破坏semantic root。

### 9.3 dtype

支持：

```text
bool, uint8, int8, int16, int32, int64,
float16, bfloat16, float32, float64
```

complex、quantized、sparse、non-strided或platform-native endian拒绝。writer规范为little-endian；raw完整保留±0、subnormal、
NaN payload与Inf sign。

## 10. strict canonical parser

### 10.1 JSON

- duplicate key拒绝；
- UTF-8、key顺序、分隔符、换行和integer spelling canonical；
- JSON中禁止float scalar；容差用decimal string，测量值用IEEE hex或decimal string；
- 拒绝NaN/Inf token、`1e999`、NUL/control character、超长line；
- record count与解压大小预设上限。

### 10.2 gzip

使用stdlib zlib/gzip解码，要求单member、完整EOF、无unused/trailing bytes、固定资源上限。跨zlib版本只验证解压canonical
stream hash和语义，不要求重新压缩得到相同bytes；manifest仍绑定生成时的compressed file SHA256。

### 10.3 tree

拒绝：

```text
symlink
hardlink across workers
absolute path
.. component
backslash
duplicate normalized path
unlisted file
device/FIFO/socket
```

## 11. numeric comparison registry

### 11.1 policies

```text
EXACT_BITS
EXACT_DISCRETE
FINITE_ATOL_RTOL
FINITE_SIGN_EXACT
SIGNED_ZERO_DISCLOSE
CANONICAL_QNAN_BITS
NONFINITE_FORBIDDEN
```

每个semantic path必须有唯一policy；generic codec不得硬编码ResNet node/shape，fixture protocol可以冻结具体path inventory。

### 11.2 torch-compatible tolerance

保持历史有向比较：reference=`lhs`、observed=`rhs`，逐元素：

```text
abs(lhs-rhs) <= atol + rtol * abs(rhs)
```

```text
B0 -> R lower/state/core/post/KFSB  atol=2e-4, rtol=2e-4
R  -> C lower/state/core/post/KFSB  atol=2e-4, rtol=2e-4
R  -> C compiled gradient/m/v       atol=2e-5, rtol=2e-5
sign                               exact
equal_nan                          false
```

stdlib comparator用`Decimal.from_float`或等价exact-bit算法计算有限差；float64减法overflow不能自动通过。summary记录
max abs/rel、argmax path/index、sign、signed-zero与finite class。

### 11.3 variant-specific exclusions

以下只做结构/披露，不做B0/R/C exact parity：

- provider callback内部计数；
- B0 post-KFSB residue vs R/C normalized scratch；
- implementation-specific constructor时点；
- process/PID；
- compressed physical bytes。

## 12. positive mandatory tensor-occurrence floor

### 12.1 fixed per-stage尺寸

S4-2 policy floors：

```text
B0 policy = 2,837,288 B
R policy  = 2,871,296 B
C policy  = 1,511,936 B
```

共同whole-core semantic projection：

```text
fields                        408 B
terminal/KFSB branch trace 521,736 B
working intermediate       299,712 B
history                       120 B
whole-core total          821,976 B
```

其他stage：

```text
B0 pre/final state            68,016 B
R/C pre/candidate/final      102,024 B
official post projection      50,976 B
queue-visible projection      50,976 B
```

KFSB 72 child lower的`288 B`已经包含在`521,736 B` branch trace，不能再次加旧独立KFSB预算。

### 12.2 per-variant

```text
B0 = 2,837,288 + 821,976 + 68,016  + 50,976 + 50,976
   = 3,829,232 B

R  = 2,871,296 + 821,976 + 102,024 + 50,976 + 50,976
   = 3,897,248 B

C  = 1,511,936 + 821,976 + 102,024 + 50,976 + 50,976
   = 2,537,888 B
```

### 12.3 18 positive

```text
6 * (3,829,232 + 3,897,248 + 2,537,888)
= 61,586,208 B
= 58.733184814453125 MiB
```

这是semantic tensor-occurrence floor：相同payload跨phase仍按不同证据发生计数。worker-local content-addressed sidecar可
物理去重，所以该数不是compressed/uncompressed文件大小下限。

## 13. fault mandatory floor

### 13.1 每fault最低

15个fault都必须先产生真实S4-2 C terminal state：

```text
C policy trajectory            = 1,511,936 B
pre/candidate/fault 12-path     =   102,024 B
minimum per fault              = 1,613,960 B
```

### 13.2 15 fault

```text
15 * 1,613,960
= 24,209,400 B
= 23.08788299560547 MiB
```

这仍不含各fault到达阶段的terminal/KFSB/scratch/core/post/queue额外raw，所以只是真正下限，不是规划上界。

### 13.3 全33 worker最低

```text
61,586,208 + 24,209,400
= 85,795,608 B
= 81.8210678100586 MiB
```

manifest必须分别报告：

```text
semantic_tensor_occurrence_bytes
unique_payload_bytes
decompressed_stream_bytes
compressed_file_bytes
total_tree_bytes
```

不得用任一口径冒充另一个口径或memory peak。

## 14. 16-node seal DAG

### 14.1 nodes

```text
protocol
environment
source_declarations
model_property_config
positive_raw
fault_raw
worker_source_union
semantic_root
derived_summary
tamper_registry
tamper_report
replay_stdout
readme
final_manifest
external_anchor
anchored_replay_record
```

### 14.2 dependencies

- protocol/environment/source/model/positive/fault/worker-union都指向semantic root；
- semantic root指向summary、tamper report、replay stdout；
- summary指向tamper report、replay stdout和README；
- tamper registry指向tamper report；
- tamper report与replay stdout指向README；
- artifact内13个上游node都指向final manifest；
- final manifest、semantic root、protocol/source/model指向external anchor；
- final manifest和external anchor指向anchored replay record。

canonical DAG payload固定为
`{"schema":"boundflow.s4-formal-seal-dag/v2","nodes":[...],"edges":[[from,to],...]}`：nodes按
§14.1顺序，edges按§14.2规则生成顺序，之后用`sort_keys=True,separators=(",", ":")`编码。机械结果：

```text
nodes = 16
edges = 36
acyclic = true
hash = 01e179ea504f94c3e9720d5f63b318e34e912738d30c21d690f283b857ac491c
```

summary不得绑定后生成的tamper report；tamper report不得绑定final manifest；replay stdout不输出final manifest hash。
它们由final manifest共同绑定。

## 15. semantic root与manifest

### 15.1 semantic root

只绑定上游decompressed canonical streams与schema roles：

```text
protocol hash
environment hash
source declarations hash
model/property/config hash
18 positive worker semantic hashes
15 fault worker semantic hashes
worker source union hash
```

不绑定summary、tamper、README、manifest或external anchor。

### 15.2 final manifest

canonical manifest绑定自身之外artifact tree的每个regular file：

```text
relative path
role
compressed bytes
file SHA256
decompressed canonical bytes/hash where applicable
```

`manifest_hash`对去掉自身字段的canonical payload计算。manifest不得包含external anchor或anchored replay record；二者
在artifact外。

## 16. external anchor

### 16.1 schema

```text
S4ExternalAuditAnchorV2:
  artifact_schema
  artifact_relative_path
  artifact_commit
  final_manifest_sha256
  semantic_root_sha256
  source_revision
  protocol_sha256
  model_sha256
  property_sha256
  config_sha256
  replayer_git_blob_sha256
  expected_positive_worker_count = 18
  expected_fault_worker_count = 15
  expected_claim_flags = false
```

### 16.2 ownership

anchor位于artifact result commit之后创建的DocOps audit request/delivery并由后续Git commit绑定。artifact目录内可有
`UNTRUSTED_ANCHOR_COPY.json`供阅读，但replayer默认拒绝它；anchored-check必须显式接收artifact tree之外路径。

### 16.3 两层结论

```text
SELF_CONSISTENT
  manifest + semantic derivation内部一致

ANCHORED_AUTHENTIC
  SELF_CONSISTENT + explicit external anchor exact
```

formal closure要求第二层；第一层不能升级validated。

## 17. replayer三种模式

### 17.1 derive

从raw/declarations生成summary；仅generator内部使用，不读取现有summary作为输入。

### 17.2 self-check

验证tree/manifest、codec、semantic root、raw-derived summary、tamper registry/report和claim flags。它证明artifact内部
自洽，不证明source/model真实性。

### 17.3 anchored-check

在self-check后验证external anchor、artifact commit与replayer blob。只有该模式可形成外审formal PASS输入。

replayer只能importstdlib，禁止Torch、TVM、NumPy、BoundFlow helper或provider validator。

## 18. 96-case layered tamper registry

### 18.1 registry hash与层数

```text
case count = 96
registry hash = 5fdfa8bcbc41516807f7eef220ede181253ade7b0c42fa31a4620dcdf37f7d05

EXTERNAL_ANCHOR         = 13
FROZEN_PROTOCOL         = 5
RAW_SEMANTIC_DERIVATION = 77
EXECUTION_EVIDENCE_ONLY = 1
```

enforcement mapping固定为：`T01—T10/T93/T94/T96=EXTERNAL_ANCHOR`；
`T11/T12/T20/T91/T92=FROZEN_PROTOCOL`；`T19=EXECUTION_EVIDENCE_ONLY`；其余77项=
`RAW_SEMANTIC_DERIVATION`。registry hash对`[{id,name,enforcement_layer}]`按编号顺序做canonical JSON SHA256。

### 18.2 T01—T10 source/authenticity

```text
T01 boundflow_revision
T02 boundflow_executed_blob
T03 tvm_gitlink
T04 tvm_ffi_native
T05 abcrown_revision
T06 auto_lirpa_revision
T07 model_digest
T08 property_digest
T09 config_digest
T10 replayer_interpreter_blob
```

### 18.3 T11—T20 protocol/process

```text
T11 tolerance_policy
T12 claim_flag
T13 worker_delete
T14 worker_duplicate
T15 permutation_order
T16 variant_assignment
T17 parent_nonce_duplicate
T18 process_lineage_structure
T19 fresh_process_attestation_overclaim
T20 parent_cuda_initialized
```

T19不能由offline JSON不可伪造证明。正确结果是`OFFLINE_UNATTESTABLE`并要求parent/OS/现场复跑证据；不得伪写
“semantic replayer cryptographically rejected freshness forgery”。其余case必须由对应层拒绝。

### 18.4 T21—T30 S4-0 live admission

```text
T21 missing_alpha_path
T22 swapped_alpha_path
T23 preserved_alpha_drift
T24 active_beta_location_sign
T25 empty_beta_alias
T26 storage_stride_offset
T27 live_tensor_version
T28 live_provider_rebind
T29 mapping_snapshot_hash_confusion
T30 beta_width_history
```

### 18.5 T31—T40 S4-1 backend/ABI

```text
T31 ternary_endpoint_policy
T32 ainput_zero_inventory
T33 derived_center_formula
T34 selector_poison_qnan
T35 compressed_gradient
T36 site31_reader_order
T37 terminal_la_transform_phase
T38 terminal_la_spec_axis
T39 arena_alias_group
T40 argument_module_receipt
```

### 18.6 T41—T52 S4-2 policy

```text
T41 evaluation_cardinality
T42 optimizer_mutation_cardinality
T43 scheduler_cardinality
T44 generation_rearm
T45 evaluation_input_version
T46 storage_commit_generation
T47 adam_28_item_cursor
T48 adam_moment
T49 projection_clamp
T50 checkpoint_keep_best
T51 terminal_best_state
T52 terminal_handoff_reuse_rerun
```

### 18.7 T53—T64 terminal/KFSB/scratch

```text
T53 terminal_ordinal
T54 terminal_output_inventory
T55 kfsb_candidate_count
T56 kfsb_child_lower
T57 kfsb_final_decision
T58 provider_bound_callback
T59 scratch_missing_path
T60 scratch_partial_finalization
T61 provider_beta_inventory
T62 scratch_phase_ordinal
T63 scratch_logical_unique_alias
T64 exclusive_owner_reentry
```

### 18.8 T65—T78 commit/post/queue

```text
T65 device_commit_order_count
T66 candidate_tensor_value
T67 tensor_version_after_commit
T68 prefix_restore_scope
T69 untouched_suffix_write
T70 host_final_keys
T71 container_clear
T72 core_seal_state
T73 provider_constructor_count
T74 official_post_count
T75 official_post_value
T76 domain_add_counters
T77 queue_partial_mutation
T78 check_worst_failure_state
```

### 18.9 T79—T90 raw/replay/seal

```text
T79 payload_value
T80 payload_offset_stream
T81 tensor_index_key
T82 dtype_shape_byte_order
T83 signed_zero_nonfinite
T84 object_storage_group
T85 gzip_member_trailing
T86 json_canonical_duplicate
T87 artifact_path_symlink
T88 semantic_root
T89 summary_forgery
T90 tamper_replay_manifest_dag
```

### 18.10 T91—T96 status/security/anchor

```text
T91 premature_validated_status
T92 performance_claim_flag
T93 source_scope_clean_lie
T94 untracked_importable_source
T95 privacy_path_token_leak
T96 artifact_internal_anchor_autoselect
```

### 18.11 fully re-signed规则

每案必须：

1. copy完整baseline；
2. 修改目标semantic/protocol/source；
3. 重算payload/index/stream/file/summary/manifest全部受影响digest；
4. 证明外层integrity合法；
5. 使用该case预注册mode和enforcement layer；
6. 得到stable semantic/anchor/protocol reason，或T19的explicit limitation；
7. tamper report绑定baseline semantic root，但不依赖final manifest。

不能把未重签导致的file digest mismatch算作semantic tamper通过。

## 19. status与claim

artifact内部只允许：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT
FORMAL-NO-GO-S4-SAME-SOLVER-CORRECTNESS
```

只有anchored external audit批准、exchange关闭、closure commit落地后，权威文档才可升级：

```text
VALIDATED-S4-SAME-SOLVER-CORRECTNESS
```

无论哪种状态，S4-P timing前：

```text
performance_claimed = false
same_solver_performance_claimed = false
complete_query_claimed = false
queue_speedup_claimed = false
tenx_claimed = false
asplos_ready = false
```

## 20. security与隐私

- 扫描HOME、用户名、绝对repo/external路径、SSH/token-like值；
- source路径只保存owner+relative path；
- environment只保存allowlist与value class；
- successful stdout/stderr结构化并扫描；
- uncontrolled traceback只放non-formal sibling；
- expected fault reason使用stable enum，不存任意异常全文；
- parser有file/line/record/decompressed/payload/tree总量上限；
- artifact tree拒绝链接、设备文件、path traversal和zip bomb。

## 21. 实现对象与文件建议

### 21.1 artifact code

```text
boundflow/artifacts/s4_protocol.py
boundflow/artifacts/s4_tensor_codec.py
boundflow/artifacts/s4_source_inventory.py
boundflow/artifacts/s4_semantic_compare.py
boundflow/artifacts/s4_manifest.py
boundflow/artifacts/s4_anchor.py
scripts/run_asplos27_s4_formal_worker.py
scripts/run_asplos27_s4_formal_artifact.py
scripts/replay_asplos27_s4_formal_artifact.py
scripts/tamper_asplos27_s4_formal_artifact.py
```

这些文件只在S4-3关闭后创建；当前不提前落空壳代码。

### 21.2 tests

```text
tests/test_asplos27_s4_tensor_codec.py
tests/test_asplos27_s4_source_inventory.py
tests/test_asplos27_s4_protocol.py
tests/test_asplos27_s4_fault_registry.py
tests/test_asplos27_s4_replay.py
tests/test_asplos27_s4_tamper.py
tests/test_asplos27_s4_anchor.py
```

## 22. acceptance tests

### 22.1 codec/parser

- 10 dtype roundtrip；
- ±0/subnormal/max finite/NaN payload/±Inf；
- empty same payload但distinct storage；
- noncontiguous logical payload + original stride/offset；
- gap/overlap/orphan/unreferenced/trailing payload；
- duplicate JSON、noncanonical integer、1e999、gzip multistream/trailing；
- symlink/hardlink/path traversal/device file。

### 22.2 protocol/source

- 18/15/33 exact；
- 六全排列与ordinals exact；
- parent无Torch/TVM/CUDA；
- source pre/post race；
- untracked importable code/native；
- submodule/external/native/compiled receipt drift；
- full clean与source-scope clean分开。

### 22.3 fault

- 15 registry exact/hash exact；
- 每fault独立process；
- clean/staging/commit/post/queue terminal exact；
- poison后zero retry/fallback/illegal next action；
- prefix restore/untouched suffix/version exact；
- queue partial和check-worst区别可重放。

### 22.4 replay/tamper/anchor

- 16-node DAG无环/hash exact；
- summary全部raw-derived；
- replay stdout逐字重建；
- 96 registry exact/hash exact；
- 95 case按层拒绝，T19输出诚实限制；
- artifact内伪anchor不自动采信；
- self-check PASS + anchored-check FAIL反例；
- external anchor exact后anchored-check PASS。

## 23. 实现与提交顺序

仅在S3外审批准且S4-0—S4-3依序关闭后：

1. `docs: freeze S4-4 formal protocol and registries`；
2. `feat(artifact): add strict tensor index and binary sidecar codec`；
3. `feat(artifact): add source scope native compiled inventory`；
4. `feat(artifact): add stdlib semantic comparator and summary derivation`；
5. `feat(artifact): add 18-worker positive parent and worker projection`；
6. `feat(artifact): add 15 isolated fault workers`；
7. `feat(artifact): add seal DAG manifest and replay modes`；
8. `test(artifact): add 96 layered tamper cases`；
9. `artifact: generate S4 formal candidate from clean source`；
10. `docs: commit artifact result and pending-external status`；
11. `docs: create external anchor and DocOps audit request`；
12. `docs: respond to audit and close correctness or NO-GO`；
13. `docs: preregister S4-P timing only after correctness closure`。

artifact infrastructure、clean source、formal raw、artifact result、external anchor/audit request和closure必须分提交，
避免第一次结果与其生成代码共享dirty source。

## 24. GO / STOP

### 24.1 GO

- 18 positive + 15 fault=`33` independent subprocess；
- B0/R/C六全排列与variant count exact；
- source/loaded/native/compiled/input closure完整；
- indexed binary raw由stdlib恢复IEEE、path、view和alias projection；
- S4-3状态机/fault terminal/14-step commit/post/queue/check-worst可重放；
- positive/fault semantic occurrence达到冻结floor或解释合法额外项；
- 16-node seal DAG无环；
- derive/self-check/anchored-check分层正确；
- 96 tamper registry完整，95拒绝+1诚实限制；
- internal status仍pending external、所有performance flag false；
- external audit批准后才升级validated。

### 24.2 STOP

- 仍使用旧`1,341,776 B` trajectory预算；
- B0伪造candidate state或KFSB重复计账；
- fault worker不保存完整C trajectory；
- KFSB/scratch fault仍写clean abort；
- 未覆盖check-worst failure；
- fault复用positive/poisoned process；
- `.pt`或summary成为唯一truth；
- artifact内hash冒充source/model真实性；
- freshness字段被宣称cryptographic attestation；
- summary/tamper/manifest形成hash cycle；
- raw occurrence、unique payload、compressed bytes混写；
- external audit前写VALIDATED或任何performance claim。

## 25. 当前门禁

```text
S3 external audit             = pending
S4-0—S4-3 implementation     = closed
S4-4 artifact code           = closed
S4-4 formal run              = closed
S4-P timing/performance      = closed
```

本施工包只冻结“未来批准后怎么造可独立审计的证据”。当前唯一外部下一动作仍是完成S3 optimizer-runtime外审；
无blocker后按S4-0→1A→1B0→1B→1C→1D→2→3→4顺序施工。
