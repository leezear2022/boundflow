---
status: diagnostic-complete-construction-ready-code-closed
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-1a-implementation-construction-package
stage: s04
depends-on: validated-s4-0-mutable-state-admission
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4-1A：ordered mutable buffer逐文件施工包与V5事务修正

## 0. 直接结论

S4-1A原方向成立：把S4-0承认的六lower-α和唯一active β复制到16个独立persistent CUDA storage，建立
16个prepare-time DLPack view，并把12条live source strong-ref lease继续保留到S4-3。但旧V4蓝图不能直接编码，仍有
八个可复现的施工歧义：

1. 旧public API让caller传`device/stream_identity`，会把应由runtime观察的事实变成可伪造输入；
2. 旧S4-1A同时定义evaluation/result lease、10-step version、Adam smoke和terminal handoff，跨进S4-1D/S4-2职责；
3. 现有R31 DLPack cache key只有`(data_ptr, shape)`，同pointer/shape不同stride会碰撞并绕过TVM的noncontiguous拒绝；
4. source/candidate content hash会调用CUDA Tensor `.cpu()`，所以“success prepare不新增同步”不成立；
5. 只清空resource list不能保证失败后显存释放，Python loop local和retained traceback会继续持有最后一个Tensor/TVM view；
6. 多字段逐个移动lease与staging会产生短暂double-owner/no-owner窗口，`try/finally`本身不能提供原子ownership；
7. gradient/lower是`empty`未初始化buffer，不能放进content hash，也不得在第一次full-write前读取；
8. S4-1A仍没有process-global provider latch；它只能证明lease内source和candidate local owner，不能声称provider mapping
   在整个query没有rebind。

V5冻结为：

```text
PreparedS4MutableStateAdmissionV1
  → begin single-attempt private ticket
  → entry source capture
  → 7 parameter D2D clone
  → 7 gradient + lower + upstream storage
  → 16-way storage/view validation
  → exit source capture + initialized-content validation
  → tensor-free preparation receipt
  → one resource-owner adoption
  → PreparedS4MutableBuffersV1
```

S3 exchange仍为`ready_for_audit`，S4-0尚未实现/关闭。因此本文只关闭S4-1A第一行代码前的文件、API、资源、
故障与formal歧义，**不开放S4-1A production code、TIR、optimizer、formal或timing**。

## 1. 权威源码与现场诊断事实

### 1.1 S4-0 V4交接边界

S4-0 construction package已经冻结：

- public wrapper为`PreparedS4MutableStateAdmissionV1`；
- receipt是tensor-free canonical对象；
- private lease保存12条source Tensor strong ref、raw storage/version/content token和raw `exact_call_id`；
- wrapper只公开`begin_buffer_prepare(current_sources, *, exact_call_id)`；
- wrapper状态为`OPEN→PREPARING→TRANSFERRED/FAILED_CLOSED`；
- S4-0只能证明local single-transfer，process-global exclusivity留给S4-3 latch。

S4-1A必须消费该private ticket，不能通过public property取得lease，也不能重新从stable receipt group构造source owner。

### 1.2 现有R31 DLPack cache只能作机制参考

`boundflow/runtime/r3_compiled_p_alpha_vjp.py`与`r3_full_lower_forward_tir.py`当前实现：

```text
key = (tensor.data_ptr(), tuple(tensor.shape))
view = tvm.runtime.from_dlpack(tensor)
torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
```

它证明了prepare-time view和warm lookup可行，但不能原样复制：

- key没有storage identity、stride、offset、dtype或device；
- `_views`是dict，未按buffer ordinal冻结顺序；
- object没有显式`close()`，失败时依赖Python frame/GC；
- roundtrip只检查pointer，没有检查shape/stride/dtype/device；
- existing object还保留大量dict和runtime tensors，不是S4-1A的最小owner。

因此S4-1A复用DLPack机制，不复用该class的ownership/cleanup/key实现。

### 1.3 formal plan-order inventory

从冻结snapshot、TOPOLOGY、ResNet2B ONNX和R31 plan独立重建，plan hash为：

```text
39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f
```

plan order不是snapshot字符串排序：

| slot | native | active α shape/elements | β shape/elements |
|---:|---:|---:|---:|
| 0 | 17 | `[6,164]` / 984 | `[6,0]` / 0 |
| 1 | 19 | `[6,132]` / 792 | `[6,0]` / 0 |
| 2 | 23 | `[6,121]` / 726 | `[6,0]` / 0 |
| 3 | 25 | `[6,86]` / 516 | `[6,0]` / 0 |
| 4 | 28 | `[6,178]` / 1,068 | `[6,0]` / 0 |
| 5 | 31 | `[6,27]` / 162 | `[6,1]` / 6 |

所以：

```text
active α = 4,248 float32 = 16,992 B
active β = 6 float32 = 24 B
parameter = gradient = 4,254 float32 = 17,016 B
lower = upstream = 6 float32 = 24 B each
candidate storage = 16
candidate logical bytes = 17,016 + 17,016 + 24 + 24 = 34,080 B
leased source = 12 tensors / 8,502 elements / 34,008 B
```

五个empty β保留五个typed token，physical Tensor/view/storage/optimizer ordinal均为0。

### 1.4 GPU owner与allocator诊断

RTX 4060 Laptop / torch `2.12.1+cu132` / CUDA `sm_89`现场探针：

```text
S4_1A_CONSTRUCTION_DIAGNOSTIC_PASS
parameter/gradient = 4,254 / 4,254 elements
parameter/gradient = 17,016 / 17,016 B
candidate storage unique = 16/16
candidate/source nonempty storage overlap = 0
base DLPack pointer exact = 16/16
candidate logical bytes = 34,080
allocated delta after buffers = 39,936
allocated delta after views = 39,936
all parameter leaf = true
all buffers contiguous = true
```

`39,936 B`是本机allocator rounding诊断，不进入canonical receipt或跨环境GO门槛；formal raw只披露entry/peak/exit
allocated/reserved，不要求其他PyTorch/CUDA版本逐位相等。

五个empty CUDA Tensor均`data_ptr=0/nbytes=0`，但Python object和storage `_cdata`各自不同。empty token不得依赖
zero pointer分组。

### 1.5 DLPack shape-only key反例

构造同一`[2,2]` storage：

```text
x.stride = (2,1)
y=x.T.stride = (1,2)
x.data_ptr == y.data_ptr
x.shape == y.shape
```

结果：

```text
old_key_equal=true
new_key_equal=false
tvm_direct_noncontiguous_rejected=true
cache_reused_wrong_stride=(2,1) instead of (1,2)
cache_reused_wrong_content=[[0,1],[2,3]] instead of [[0,2],[1,3]]
```

即：TVM本会拒绝`y`，但old key先命中`x`会绕过拒绝并静默返回错误view。S4-1A必须在cache lookup前验证
contiguous/stride，并将完整view identity纳入private key。

### 1.6 retained traceback清理反例与修正

第一次诊断只执行：

```text
views.clear(); del buffer lists; gc.collect(); stream.synchronize()
```

仍得到`candidate_allocated_delta_after_cleanup=1,024 B`。原因是loop local `parameter/tensor/view`仍强引用最后两个
resource；如果caller保留异常，traceback还会保留整个frame。

修正探针使用单一`Staging` owner，`close()`清空全部字段，并在raise前把所有resource local置`None`；stable error在
退出`except`后抛出，使`__context__ is None`。parameter/buffer/view三阶段在外部保留异常对象时均得到：

```text
allocated_delta_with_exception_retained=0
context_is_none=true
```

因此这不是测试卫生，而是production failure ownership合同。

## 2. S4-1A精确scope

### 2.1 本阶段拥有

S4-1A只拥有：

- S4-0 private lease到prepared runtime的single-owner transfer；
- 6 α leaf parameter + 1 active β leaf parameter；
- 6 dα + 1 active dβ persistent output；
- lower与fixed upstream persistent buffer；
- 5 empty β typed token；
- 16 base DLPack view与private exact view key；
- buffer preparation receipt；
- prepare failure cleanup与deterministic close。

### 2.2 本阶段明确不拥有

以下旧蓝图内容移出S4-1A closure：

- `S4EvaluationRequestV1`与`S4EvaluationResultLeaseV1`；
- evaluation ordinal/generation、gradient lease和terminal one-shot；
- CROWN forward/VJP或任何TIR launch；
- `.grad`绑定、Adam param group、scheduler或one-step Adam；
- 10 evaluation / 9 mutation trajectory；
- terminal lA handoff、commit/post/queue；
- performance/timing/complete-query claim。

evaluation/result lease属于S4-1D，optimizer属于S4-2，terminal/commit属于S4-3。旧one-step Adam probe只保留为历史
PyTorch可行性诊断，不能成为S4-1A positive acceptance。

### 2.3 本阶段能声称什么

S4-1A通过最多声称：

> exact-call-bound local buffer ownership、16-storage/view prepare和failure cleanup validated；CROWN math、optimizer、
> provider mapping stability、process-global exclusivity与performance unvalidated。

provider如果在prepare期间把container rebind到same-content clone，lease内旧object不一定变化。S4-1A不会错误commit，
因为S4-3 current-provider precommit必须拒绝；但S4-1A不能把这段时间的mapping stability写成已证明。

## 3. 文件施工边界

### 3.1 S4-0批准后第一批代码

只新增：

```text
boundflow/runtime/asplos27_s4_ordered_buffer_abi.py
tests/test_asplos27_s4_ordered_buffer_abi.py
```

允许import：

```text
torch
tvm（只在prepare helper局部import）
rvir_v4_production_state.production_tensor_sha256
S4-0 admission receipt/private ticket type
R31FullRegionPlanV1的tensor-free layout projection type
VerificationRejectionReason
```

禁止import/调用：

```text
provider solver/pre_result/net
torch.optim / scheduler
S4-1B/1C/1D evaluator
S4-2 policy driver
atomic commit/post/queue
TIR compiler/module cache
CUDA Graph/timing event/global registry
torch.cuda.empty_cache
```

### 3.2 formal evidence后续独立提交

代码/单元关闭后才新增：

```text
scripts/run_asplos27_s4_1a_buffer_worker.py
scripts/run_asplos27_s4_1a_buffer_artifact.py
scripts/replay_asplos27_s4_1a_buffer_artifact.py
scripts/probe_asplos27_s4_1a_buffer_tamper.py
tests/test_asplos27_s4_1a_buffer_artifact.py
```

formal raw、implementation和第一次结果不得来自同一dirty commit。

## 4. public API V5

```python
prepare_s4_mutable_buffers_v1(
    prepared_admission: PreparedS4MutableStateAdmissionV1,
    current_live_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
) -> PreparedS4MutableBuffersV1
```

不接受caller提供：

- `device`：从receipt/lease/source exact推出；
- `stream_identity`：从当前thread/device的`torch.cuda.current_stream()`观察；
- allocator、view factory、fault callback或provider callback；
- plan/snapshot override；
- arbitrary tensor list/dict。

测试用故障通过monkeypatch module-private leaf helpers注入，不污染public signature。

函数第一项动作必须是：

```text
prepared_admission.begin_buffer_prepare(
    current_live_sources,
    exact_call_id=exact_call_id,
) -> _S4BufferPrepareTicketV1
```

ticket不可公开构造、复制或序列化；它成为唯一lease transaction owner。wrapper进入`PREPARING`后无论成功/失败都不能
回到`OPEN`。

## 5. 对象设计

### 5.1 `S4EmptyBetaSlotTokenV1`

frozen tensor-free dataclass：

```text
slot_ordinal
semantic_path
shape=[D,0]
source_content_hash
physical_buffer_present=false
physical_view_present=false
optimizer_ordinal=-1
```

### 5.2 `S4MutableBufferDescriptorV1`

frozen tensor-free dataclass，按16项resource order记录：

```text
buffer_ordinal
semantic_role
slot_ordinal_or_minus_one
shape / stride / storage_offset
dtype / device
element_count / logical_bytes
requires_grad / is_leaf / contiguous
initialized_at_prepare
initial_content_hash_or_none
view_ordinal
```

raw storage/pointer不进入descriptor；private resource owner另持raw token。

### 5.3 `S4MutableBufferPreparationReceiptV1`

frozen canonical dataclass，只含descriptor/token/counter/hash/false claim。其`validate()`必须从明细重算全部汇总，不能依赖
汇总字段自证。

### 5.4 `_S4BufferResourceOwnerV1`

module-private普通`__slots__` class，是唯一physical resource owner：

```text
_ticket
_parameters              # tuple/list staging: 6 α + 1 β
_gradients               # 6 dα + 1 dβ
_lower
_upstream
_views                    # ordered 16
_private_view_keys
_state = STAGING/PREPARED/CLOSED
```

所有resource先安装到该owner字段，不能散落在多个wrapper/tuple。`close()`固定清空view→output→gradient→parameter→ticket，
并把字段置空/`None`；幂等但不恢复能力。

### 5.5 `PreparedS4MutableBuffersV1`

普通`__slots__` class：

```text
receipt                   # public tensor-free
_resources                # private single owner
_state = PREPARED/CLOSED
```

本阶段只公开：

```python
close() -> None
```

未来S4-1B/1C/1D需要borrow时另按其已批准蓝图增加module-private typed borrow，不在S4-1A提前实现generic getter、dict、
callback或tensor override。

ticket/resource/prepared owner全部拒绝`copy/deepcopy/pickle/__reduce__/asdict`，不实现`to_dict/stable_hash`。receipt才允许
canonical serialization。

## 6. 固定prepare算法

### Phase A：single-attempt与零allocation admission

1. wrapper第一行`OPEN→PREPARING`并产生private ticket；
2. 检查PID/thread/raw exact-call、admission hash和construction model hash；
3. `type(current_live_sources) is dict`，key exact str、value exact `torch.Tensor`；
4. current device/stream从runtime观察并与lease policy匹配；
5. entry capture按plan slot、每slot α后β采集12条object/storage/layout/version/content；
6. 从receipt+plan projection导出16 buffer + 5 token manifest；
7. Phase A monkeypatch所有allocation/view helper为必抛仍必须通过。

Phase A source content为第一轮12条logical D2H、`34,008 B`。

### Phase B：local resource staging

固定allocation order：

```text
alpha parameter slot 0..5
  → active beta parameter slot 5
  → alpha gradient slot 0..5
  → active beta gradient
  → lower output
  → fixed upstream
```

parameter构造必须exact：

```python
active.detach().clone(memory_format=torch.contiguous_format).requires_grad_(True)
```

禁止`.to().contiguous()`冒充clone。gradient/lower使用`empty`，receipt标记`initialized_at_prepare=false`；在S4-1B/1C
full-write前任何read/hash/finite检查均拒绝。upstream用`full([D,1], -1.0)`并在prepare校验content。

resource owner在每次成功allocation后立即接管引用；function frame不得长期持有第二份list。

### Phase C：physical/storage validation

逐项检查：

1. 7 parameter exact plain Tensor、leaf、requires-grad、contiguous CUDA float32；
2. 7 gradient + lower/upstream exact plain Tensor、requires-grad false、contiguous CUDA float32；
3. 16个candidate storage token互异；
4. 16 candidate与12 source nonempty storage集合不相交；
5. empty β无physical owner；
6. candidate logical bytes=`34,080`；
7. 7 parameter content与source active/β exact；
8. upstream content为六个float32 `-1.0`；
9. gradient/lower保持uninitialized/read-forbidden状态。

parameter+upstream content validation为8条logical D2H、`17,040 B`。

### Phase D：16个base DLPack view

每个buffer在cache lookup前检查contiguous及expected stride。private key固定为：

```text
buffer ordinal
untyped_storage()._cdata
storage.data_ptr / storage.nbytes
tensor.data_ptr
shape / stride / storage_offset
dtype / device
```

key里的raw token只活在process内，receipt保存去pointer后的descriptor hash。

view order与buffer ordinal完全一致。每个roundtrip必须同时检查pointer、shape、stride、offset、dtype、device；临时Torch
roundtrip Tensor立即释放并把local置`None`。empty β不建view。

### Phase E：exit source capture与receipt

全部allocation/view完成后，对lease内12条source再采集object/storage/layout/version/content：

- entry/exit任一变化报`BUFFER_PREPARE_SOURCE_READ_RACE`；
- provider mapping整体rebind仍不在本阶段claim；
- source第二轮再产生12条logical D2H、`34,008 B`；
- receipt从descriptor/token重算hash/count/bytes；
- tensor-free walker拒绝Tensor/module/view/callback/raw pointer/capsule。

### Phase F：single-owner adoption

不要逐字段移动parameter/view/lease。正确序列：

```text
ticket + all staging already owned by one _S4BufferResourceOwnerV1
  → create PreparedS4MutableBuffersV1(receipt, resource_owner)
  → resource_owner STAGING→PREPARED
  → wrapper PREPARING→TRANSFERRED
```

wrapper与prepared可以短暂同时持有resource-owner对象引用，但`resource_owner._state`只有一个logical owner transition；
失败时prepared先close、owner清理、wrapper进入`FAILED_CLOSED`。不得出现两份lease字段或逐Tensormove。

## 7. validation copy、allocation与同步账

### 7.1 S4-1A自己的成功账

```text
source entry content        = 12 copies / 34,008 B
source exit content         = 12 copies / 34,008 B
initialized candidate       = 8 copies  / 17,040 B
---------------------------------------------------
S4-1A logical D2H           = 32 copies / 85,056 B

parameter D2D clone         = 7 copies  / 17,016 B
candidate logical storage   = 16        / 34,080 B
base DLPack views           = 16
```

8 initialized candidate是7 parameter加fixed upstream；gradient/lower未初始化，不允许被hash。

### 7.2 与S4-0累计但不混账

S4-0已冻结`24 / 68,016 B`。到S4-1A prepared completion累计：

```text
logical D2H copies = 24 + 32 = 56
logical D2H bytes  = 68,016 + 85,056 = 153,072 B
```

receipt分别记录`prior_s4_0_*`与`s4_1a_*`，不能用累计数伪装本阶段数。

### 7.3 正确措辞

S4-1A不能声称：

- success prepare无同步；
- zero GPU work；
- 34,080 logical B等于allocator delta；
- D2H logical count等于CUPTI transaction count；
- DLPack view等于kernel launch。

content hash的`.cpu()`会同步对应CUDA work。S4-P必须把S4-0+1A validation D2H/sync完整计入wrapper；correctness阶段
不得为性能删除content guard。

## 8. failure与exception ownership

### 8.1 固定cleanup order

```text
roundtrip local Tensor/view
  → base DLPack views
  → lower/upstream
  → gradients
  → parameters
  → staging containers
  → lease ticket strong refs
  → device/stream restoration check
  → wrapper FAILED_CLOSED
```

不调用`empty_cache()`；reserved可以保留，allocated/live candidate必须回entry ledger。

### 8.2 retained traceback门禁

implementation必须满足：

1. resource都在`_S4BufferResourceOwnerV1`字段，不散落在多个list；
2. `finally`调用owner close；
3. 所有loop local `parameter/gradient/output/view/roundtrip`在raise前置`None`；
4. `except`内只构造stable error和debug type string；
5. 离开`except`后再`raise stable_error`，使`stable_error.__context__ is None`；
6. canonical/debug receipt均不保存原exception或traceback。

外部保留stable exception对象、调用`traceback.format_exception`后，candidate allocated delta仍必须0。

### 8.3 fault classification

prepare尚未mutate provider source，所有本阶段失败为：

```text
PREPARE_ABORTED_CLEAN
```

但同一exact call不得retry或fallback；不是因为source已poison，而是single-attempt receipt与公平路径已冻结。若cleanup不能
证明干净，升级为`PREPARE_CLEANUP_UNPROVEN_NO_RETRY`，仍不得fallback。

## 9. preparation receipt

最低字段：

```text
schema_version
construction_model_hash
admission_hash / snapshot_hash / plan_hash
exact_call_identity_hash
device / dtype
buffer_descriptors[16]
empty_beta_tokens[5]
parameter/gradient/token/storage/view counts
parameter/gradient/candidate/leased logical elements/bytes
source entry/exit projection hashes
initialized candidate projection hash
private view descriptor hash
source/initialized/cumulative D2H count/bytes
parameter D2D count/bytes
warm_dlpack_view_count=0
full_alpha_device_copy_count=0
dense_alpha_materialization_count=0
dense_beta_materialization_count=0
prepare_retry_count=0
prepare_fallback_count=0
empty_cache_call_count=0
provider_mapping_stability_validated=false
process_global_exclusivity_validated=false
crown_numeric_semantics_validated=false
optimizer_trajectory_validated=false
timing_recorded=false
performance_claimed=false
receipt_hash
```

不进入canonical receipt：

- raw exact-call ID、PID/thread/stream/pointer/storage `_cdata`；
- actual allocator address或reserved block；
- Tensor/TVM view/lease/ticket/provider object；
- exception/traceback；
- 本机绝对路径。

allocator `entry/peak/exit allocated/reserved`进入formal worker raw diagnostic，不进入fresh-process exact receipt hash。

## 10. stable detail code V5

V5 construction model冻结20项S4-1A精化detail：

| detail code | VerificationRejectionReason | 边界 |
|---|---|---|
| `BUFFER_PREPARE_EXACT_CALL_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | raw ID/receipt hash不一致 |
| `BUFFER_PREPARE_ALREADY_ATTEMPTED` | `UNSAFE_ALIAS_OR_LIFETIME` | wrapper非OPEN |
| `BUFFER_PREPARE_OWNER_CONTEXT_MISMATCH` | `UNSAFE_ALIAS_OR_LIFETIME` | PID/thread/current stream漂移 |
| `BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH` | `STATE_VERSION_MISMATCH` | object/storage/layout/version/content不符 |
| `BUFFER_PREPARE_SOURCE_READ_RACE` | `STATE_VERSION_MISMATCH` | entry/exit source变化 |
| `BUFFER_PREPARE_MANIFEST_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | slot/resource/token投影不符 |
| `PARAMETER_SOURCE_STORAGE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` | candidate/source storage相交 |
| `PARAMETER_GRADIENT_STORAGE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` | parameter/gradient相交 |
| `CANDIDATE_STORAGE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` | 16项非互异 |
| `BUFFER_INITIAL_CONTENT_MISMATCH` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` | parameter/upstream初值不符 |
| `BASE_DLPACK_VIEW_KEY_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | key/roundtrip metadata不符 |
| `BASE_DLPACK_VIEW_COUNT_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | 非16/16 |
| `BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | D2H/D2D账篡改 |
| `BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED` | `UNSAFE_ALIAS_OR_LIFETIME` | cleanup后Tensor/view仍活 |
| `BUFFER_PREPARE_ERROR_CONTEXT_RETAINED` | `UNSAFE_ALIAS_OR_LIFETIME` | stable error保留原异常/frame |
| `BUFFER_PREPARE_CLEANUP_INCOMPLETE` | `UNSAFE_ALIAS_OR_LIFETIME` | allocated/live owner未回entry |
| `BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH` | `UNSAFE_ALIAS_OR_LIFETIME` | double/no owner |
| `BUFFER_PREPARE_SCOPE_ESCAPE` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` | Adam/evaluator/TIR/provider越界 |
| `BUFFER_PREPARE_FALLBACK_OR_RETRY_FORBIDDEN` | `RUNTIME_FALLBACK_REQUIRED` | retry/fallback/empty-cache |
| `BUFFER_PREPARE_SERIALIZATION_FORBIDDEN` | `UNSAFE_ALIAS_OR_LIFETIME` | ticket/owner copy/pickle |

不修改Verification IR枚举；这些是runtime refinement。

## 11. 单元测试施工包

### 11.1 positive最低24项

1. plan-order六slot exact；
2. 6 α parameter；
3. 1 active β parameter；
4. 5 empty β token；
5. parameter/gradient=`4,254/4,254`；
6. parameter/gradient=`17,016/17,016 B`；
7. candidate storage=`16`、logical=`34,080 B`；
8. 16 storage互异；
9. candidate/source overlap=0；
10. all parameter exact plain Tensor/leaf/requires-grad；
11. gradient/lower/upstream requires-grad=false；
12. all buffer contiguous CUDA float32；
13. parameter initial content exact；
14. upstream六个`-1.0` exact；
15. gradient/lower未初始化且pre-write read拒绝；
16. base view=`16/16` full metadata exact；
17. empty β view=0；
18. warm view creation=0；
19. leased source=`12/8502/34008`且incremental allocation=0；
20. S4-1A D2H=`32/85056`、cumulative=`56/153072`；
21. parameter D2D=`7/17016`；
22. receipt递归tensor/pointer/provider/error-free；
23. close幂等且不恢复prepare能力；
24. all claim/timing/performance false。

### 11.2 negative minimum 68

固定首要reason顺序后至少覆盖：

1. prepared admission类型错误；
2. top-level live mapping为dict subclass/custom Mapping；
3. exact-call ID非法；
4. exact-call ID与lease不同；
5. admission hash篡改；
6. 同一wrapper第二次attempt；
7. wrapper状态非OPEN；
8. owner PID漂移；
9. owner thread漂移；
10. current stream漂移；
11. source path缺失；
12. source path多余；
13. source object替换；
14. same-object storage rebind；
15. source `_version`漂移；
16. `.data`/DLPack content drift绕过version；
17. source shape漂移；
18. source dtype漂移；
19. source device漂移；
20. source stride/offset漂移；
21. source noncontiguous；
22. source cross-path alias漂移；
23. lease inventory少/多/乱序；
24. lease logical bytes篡改；
25. slot/resource reorder；
26. α parameter count错误；
27. active β parameter count错误；
28. empty β token count错误；
29. empty β创建zero-width physical Tensor；
30. parameter非leaf；
31. parameter requires-grad false；
32. parameter Tensor subclass；
33. parameter shape错误；
34. parameter dtype/device错误；
35. parameter noncontiguous；
36. parameter initial content错误；
37. preserved α plane被复制进candidate；
38. parameter/source storage alias；
39. parameter/parameter storage alias；
40. gradient/parameter storage alias；
41. gradient/gradient storage alias；
42. lower/upstream与其他buffer alias；
43. gradient/lower在full-write前被读取；
44. candidate logical bytes全重签篡改；
45. base view count不是16；
46. pointer exact不是16；
47. same pointer/shape不同stride key collision；
48. noncontiguous buffer在cache lookup前未拒绝；
49. dtype/device/storage/offset从view key删除；
50. warm path新建view；
51. receipt含Tensor/TVM view；
52. receipt含raw pointer/storage token；
53. D2H/D2D count或bytes全重签篡改；
54. claim/timing/performance任一true；
55. parameter allocation fault；
56. gradient allocation fault；
57. lower/upstream allocation fault；
58. TVM view creation fault；
59. torch roundtrip validation fault；
60. receipt construction/validation fault；
61. source exit hash fault或entry/exit read race；
62. adoption fault不得double-owner/no-owner；
63. failure后device/stream未恢复；
64. failure后TVM view仍强引用storage；
65. failure后candidate Tensor/local仍强引用storage；
66. retained stable exception/traceback使allocated delta非0；
67. stable error `__context__`保留底层异常/frame；
68. failure后retry/fallback/empty-cache或ticket/owner copy/pickle恢复能力。

每项断言exact detail和VerificationRejectionReason。复合攻击只允许§6顺序的第一个reason。

## 12. formal evidence设计

S4-1A code/unit closure后生成：

```text
5 fresh positive workers
7 isolated fault workers
total = 12 fresh processes
```

七fault点：parameter、gradient、output、TVM view、roundtrip、receipt、adoption。每个poisoned/fault process不复用。

positive raw：

- source/commit/protocol/model/property/config hash；
- S4-0 admission receipt与S4-1A preparation receipt；
- 7 source-active + 7 candidate parameter + upstream的indexed raw-binary sidecar；
- 16 descriptor、5 token、view/storage/D2H/D2D projection；
- entry/peak/exit allocated/reserved diagnostic；
- exact-call hash，不保存raw ID或本机路径；
- claim flags全false。

fault raw：

- stable fault point/detail/reason；
- wrapper/ticket/resource final state；
- source entry/exit content/version；
- candidate refs alive、allocated delta、stream/device restoration；
- error context/traceback resource retention；
- retry/fallback/empty-cache counters。

stdlib replayer不import BoundFlow/PyTorch/TVM/Numpy：

- 解码float32 raw并逐元素比较source-active与candidate parameter；
- 重算content/descriptor/receipt/root hash；
- 重算全部逻辑count/bytes；
- 验证fault state/counter envelope；
- 不把allocator physical cleanup伪装成stdlib可重演事实，外审需现场重跑fault worker。

internal通过状态只能是：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1A
FORMAL-NO-GO-S4-1A-BUFFER-PREPARE
```

外审批准后才能写`VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE`。

## 13. 提交顺序

依赖门禁关闭后：

1. `docs: activate S4-1A construction contract`；
2. `feat(runtime): add S4 ordered mutable buffer prepare`；
3. `test(runtime): close S4-1A 68 negative ownership gates`；
4. `artifact: generate S4-1A twelve-process evidence`；
5. `docs: deliver S4-1A external audit`；
6. `docs: close S4-1A or formal no-go`；
7. `docs: activate S4-1B0 ternary preparation`。

不得把S4-1D evaluator、S4-2 optimizer或timing塞进S4-1A实现提交。

## 14. 可重算construction model

canonicalization：UTF-8、JSON `sort_keys=True`、`separators=(',', ':')`、原生boolean。完整JSON：

```json
{"claims":{"buffer_ownership_validated":true,"crown_numeric_semantics":false,"local_single_owner_transfer":true,"optimizer_trajectory":false,"performance":false,"process_global_exclusivity":false,"provider_mapping_stability":false},"cleanup_order":["roundtrip_locals","base_dlpack_views","lower_and_upstream","gradients","parameters","staging_containers","lease_ticket","device_stream_check"],"counts":{"active_beta_parameter":1,"alpha_parameter":6,"base_dlpack_view":16,"candidate_logical_bytes":34080,"candidate_storage":16,"empty_beta_token":5,"gradient_bytes":17016,"gradient_elements":4254,"leased_source_bytes":34008,"leased_source_elements":8502,"leased_source_tensor":12,"parameter_bytes":17016,"parameter_elements":4254},"detail_codes":["BUFFER_PREPARE_EXACT_CALL_MISMATCH","BUFFER_PREPARE_ALREADY_ATTEMPTED","BUFFER_PREPARE_OWNER_CONTEXT_MISMATCH","BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH","BUFFER_PREPARE_SOURCE_READ_RACE","BUFFER_PREPARE_MANIFEST_MISMATCH","PARAMETER_SOURCE_STORAGE_ALIAS","PARAMETER_GRADIENT_STORAGE_ALIAS","CANDIDATE_STORAGE_ALIAS","BUFFER_INITIAL_CONTENT_MISMATCH","BASE_DLPACK_VIEW_KEY_MISMATCH","BASE_DLPACK_VIEW_COUNT_MISMATCH","BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH","BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED","BUFFER_PREPARE_ERROR_CONTEXT_RETAINED","BUFFER_PREPARE_CLEANUP_INCOMPLETE","BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH","BUFFER_PREPARE_SCOPE_ESCAPE","BUFFER_PREPARE_FALLBACK_OR_RETRY_FORBIDDEN","BUFFER_PREPARE_SERIALIZATION_FORBIDDEN"],"formal_processes":{"isolated_fault":7,"positive":5,"total":12},"negative_minimum":68,"phase_order":["begin_single_attempt_ticket","validate_owner_and_exact_call","validate_live_source_envelope","entry_source_capture","derive_ordered_manifest","allocate_alpha_parameters","allocate_active_beta_parameter","allocate_gradients","allocate_lower","allocate_upstream","validate_storage_and_leaf","create_base_dlpack_views","validate_dlpack_roundtrip","exit_source_capture","validate_initialized_content","build_and_validate_receipt","single_owner_adoption"],"resource_order":["alpha_parameter_0_5","active_beta_parameter","alpha_gradient_0_5","active_beta_gradient","lower_output","fixed_upstream","base_dlpack_view_0_15"],"scope":{"buffer_prepare":true,"crown_math":false,"evaluator":false,"optimizer":false,"terminal_handoff":false,"timing":false},"signature":["prepared_admission","current_live_sources","exact_call_id"],"validation_accounting":{"cumulative_d2h_bytes":153072,"cumulative_d2h_copies":56,"initialized_candidate_d2h_bytes":17040,"initialized_candidate_d2h_copies":8,"parameter_d2d_bytes":17016,"parameter_d2d_copies":7,"prior_s4_0_d2h_bytes":68016,"prior_s4_0_d2h_copies":24,"s4_1a_d2h_bytes":85056,"s4_1a_d2h_copies":32,"source_d2h_bytes":68016,"source_d2h_copies":24,"source_passes":2}}
```

SHA256：

```text
8ad25c2abf1eb98c3b1097bf7acb46aba227f7e94f0c7c03169f39e8da409a9d
```

实现必须从代码对象重建该model并比较，不能硬编码hash为PASS。

## 15. GO / STOP

### GO

- S4-0已外审批准并关闭；
- 3输入API、17-phase order和single resource owner实现；
- formal `6/1/5`、`4254/4254`、`17016/17016`、16 storage/view成立；
- S4-1A D2H=`32/85056`、cumulative=`56/153072`诚实披露；
- 68 negative与retained-traceback cleanup通过；
- 5 positive + 7 fault raw/replay通过；
- external audit批准；
- 全部claim false边界无漂移。

### STOP

- 用caller device/stream token替代runtime观察；
- 复用`(data_ptr, shape)`view key；
- cleanup只clear container而不清loop local/traceback；
- stable error保存原exception/context；
- gradient/lower未full-write就读取；
- 把Adam/evaluator/terminal带入S4-1A；
- 把validation D2H/sync写成0；
- 把local lease冒充provider mapping stability或process-global latch；
- S3/S4-0未关闭就写production代码。

## 16. 当前状态

```text
S3 exchange = ready_for_audit
S4-0 implementation = closed
S4-1A construction = diagnostic complete
S4-1A implementation/formal = closed
S4-1B/1C/1D/S4-2/S4-3/S4-4 = closed
S4 timing/performance = closed
```

本文使S4-1A在上游批准后可以按明确API、owner、resource order、68项negative与12-process formal直接施工；它没有
生成S4 runtime module、GPU result或性能claim。
