---
status: diagnostic-complete-construction-ready-code-closed
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-0-implementation-construction-package
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

# ASPLOS'27 S4-0：mutable-state admission逐文件施工包与V4边界修正

## 0. 直接结论

S4-0原V3方向成立：不新增solver IR，用一个tensor-free canonical receipt加一个进程内strong-ref lease，绑定
production六α+六β value的12条live mutable source。但把现有三份蓝图直接交给实现者，仍会留下六个可复现的歧义：

1. API没有绑定exact-call identity，却声称lease不能跨query/core call；
2. existing `live_targets_from_pre_result_v4()`接受任意`MutableMapping/Mapping`并调用`.get()`，与蓝图“读取前拒绝
   custom Mapping”矛盾；
3. `snapshot.validate()`只抛宽泛`ValueError`，不能天然提供44类稳定detail code；
4. live content hash对CUDA Tensor执行`.cpu()`，所以不是“零GPU活动”，而是零candidate kernel/零CUDA allocation，
   同时存在必须披露的validation D2H copy；
5. 一次live token capture无法区分admission期间的source read race；
6. S4-0最多证明一个prepared wrapper在本地single-transfer，不能单独证明process-global query exclusivity；真正的
   reentry closure仍属于S4-3 exact-call latch。

本稿把这些修成可直接编码的V4施工合同：

```text
pinned provider pre_result
  → strict built-in container extraction
  → first live-token capture
  → snapshot/topology/plan/policy/α/β exact validation
  → tensor-free receipt construction + independent recompute
  → second live-token capture（read-race check）
  → exact-call-bound strong-ref lease
  → nonserializable prepared wrapper
```

S3 external exchange仍为`ready_for_audit`，无audit result。本文只关闭第一行代码开工前的接口/文件/测试歧义，
**不开放S4-0 implementation、GPU formal或timing**。

## 1. 权威源码事实

### 1.1 snapshot是语义truth，不是live owner

`boundflow/runtime/rvir_v4_production_state.py`当前事实：

- `OwnedProductionTensorV4.own()`执行`detach().cpu().contiguous().clone()`；
- `ProductionStateBuilderV4`按capture时`id(value)`分配稳定ordinal alias group；
- snapshot保存`source_device`，不保存live object、storage、stride、offset或`_version`；
- `production_tensor_sha256()`会把输入移到CPU并按dtype/shape/content计算SHA256；
- `validate_beta_history_consistency()`只要求history长度不大于β width，并比较前缀，不要求exact width。

因此S4-0必须复用snapshot的content/history/policy truth，但不能从snapshot alias group推断当前provider object/storage。
β/history exact width也必须由S4 pass另行关闭，不能修改历史V4 artifact schema。

### 1.2 当前live target helper过宽

`boundflow/runtime/rvir_v4_live_return.py`：

```text
_raw_data()                         accepts isinstance(MutableMapping)
live_targets_from_pre_result_v4()  accepts nested isinstance(Mapping)
                                    invokes .get()
                                    returns built-in dict with 12 tensors
```

它适合作为已验证RVIR历史路径的兼容helper，但不能同时满足S4-0 negative“dict subclass/custom Mapping在读取前
拒绝”。S4不得修改它的历史行为；新增strict extraction helper，只供S4路径使用。

### 1.3 pinned provider容器可以严格检查

亲读pinned αβ-CROWN源码：

```text
complete_verifier/state/alpha.py
  AlphaValueData._data: dict
  inner activation value: dict

complete_verifier/state/beta.py
  BetaFullData._data: dict[str, list[SparseBeta]]

auto_LiRPA/beta_crown.py
  SparseBeta.val = torch.zeros(...).detach().to(...).requires_grad_()
```

所以S4 strict adapter要求以下exact built-in type不会排除当前formal provider：

```text
type(alpha_wrapper._data) is dict
type(beta_wrapper._data) is dict
type(alpha_wrapper._data[layer]) is dict
type(beta_wrapper._data[layer]) is list
len(beta_wrapper._data[layer]) == 1
type(alpha_tensor) is torch.Tensor
type(sparse_beta.val) is torch.Tensor
```

不import provider class，不保留provider object/callback；external repo commit/blob由artifact source identity绑定。

### 1.4 R31 plan只作当前formal specialization

`R31FullRegionPlanV1`已经提供canonical layout/tensor spec和static plan hash，但：

- `source_state_hash`绑定dense native mapping provenance，不等于snapshot hash；
- validator含ResNet2B/P-anchor formal specialization；
- 它不是新的通用Plan IR，也不拥有live Tensor。

S4-0只写一个adapter把R31 layout投影到通用slot receipt。未来Plan实现只要提供同一metadata contract，不改变
receipt schema。

### 1.5 existing commit不能替代lease

`fsg4_b3_device_atomic_commit.py`已经验证target path/shape/dtype/device/alias、入口`_version`和copy-out结果，可复用
算法和negative语料。但它在transaction staging时才拿到targets，不能证明S4-0 admission到S4-3 precommit之间provider
mapping没有整体rebind；所以lease仍是必要的时间性owner。

## 2. V4 claim边界

### 2.1 S4-0可以证明

- snapshot/topology/plan/live mapping在admission时完整覆盖六α+六β value；
- exact-call identity进入canonical receipt并与lease绑定；
- live source object/storage/layout/version/content在两次capture之间稳定；
- receipt完全tensor/provider/pointer-free并可跨fresh process重放；
- lease保留原Tensor强引用、只能本地single-transfer、不能copy/pickle/artifact walk；
- revalidation时same-content clone、same-storage view、普通in-place、`.data`/DLPack mutation分别稳定拒绝；
- admission不调用dense initializer、TVM compile/launch、provider bound callback或optimizer。

### 2.2 S4-0不能单独证明

- process-global同一query只创建一个admission；
- provider不会用第二个相同`exact_call_id`构造另一个wrapper；
- S4-1A到S4-3期间不存在reentry；
- commit/post/queue全事务exclusive；
- CUDA上不存在未由当前stream排序的外部writer。

这些属于RVIR adapter/S4-3 exclusive latch。S4-0 closure措辞必须从“关闭cross-query exclusivity”改为：

> exact-call identity bound + local single-transfer validated；process-global reentry pending S4-3 latch。

### 2.3 GPU活动的准确表述

S4-0禁止：

- candidate CUDA kernel；
- TVM compile/launch；
- candidate CUDA allocation；
- dense α/β/A materialization；
- optimizer mutation。

但live CUDA content hash会执行D2H validation copy。正式receipt必须分列：

```text
candidate_kernel_launch_count = 0
candidate_cuda_allocation_count = 0
device_to_host_validation_copy_count
device_to_host_validation_bytes
dense_materialization_observed = false
timing_recorded = false
performance_claimed = false
```

旧`gpu_execution_observed=false`可以为artifact兼容保留，但定义必须明确为“candidate execution”，不能用它隐去D2H。

## 3. 唯一代码边界

S3批准后，S4-0第一批只允许：

```text
boundflow/runtime/asplos27_s4_mutable_state_admission.py
tests/test_asplos27_s4_mutable_state_admission.py
scripts/run_asplos27_s4_0_admission_artifact.py       # 仅formal closure批次，非首个实现提交
scripts/replay_asplos27_s4_0_admission_stdlib.py      # 仅formal closure批次
gemini_doc/BOUNDFLOW_ASPLOS27_S4_CHANGE_LOG_2026_08_28.md
必要的README/claims/status同步
```

首个runtime模块是leaf：可以import已有IR/runtime类型，但任何已有production模块不得反向import它，直到S4-0单元关闭。
S4-0关闭后，RVIR exact-call adapter才在独立提交中调用它。

首批禁止：

- 修改`ProductionStateSnapshotV4`或R31 schema；
- 修改历史`live_targets_from_pre_result_v4()`语义；
- S4-1A buffer、TIR evaluator、policy、commit或post代码；
- registry/global cache；
- timing/performance字段；
- provider fallback。

## 4. 公共API

### 4.1 exception

```python
class S4MutableStateAdmissionError(RuntimeError):
    detail_code: str
    verification_reason: VerificationRejectionReason
    slot_ordinal: int | None
    semantic_path: str | None
```

构造时冻结canonical四字段；底层异常类型/文本只可放`__cause__`，不得进入artifact或stable hash。

### 4.2 strict extraction

```python
extract_s4_live_mutable_sources_v1(
    pre_result: object,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> dict[str, torch.Tensor]
```

职责只有：

1. 用`object.__getattribute__`读取已pinned wrapper的`_data`；
2. 在任何`.items/.get/__iter__`前检查exact built-in type；
3. 按topology构造12个path；
4. 检查nested dict/list和exact Tensor type；
5. 返回新的built-in dict，不保存pre_result或wrapper。

它不验证snapshot/plan，不计算hash，不创建lease。

### 4.3 prepare入口V4

```python
prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
) -> PreparedS4MutableStateAdmissionV1
```

`exact_call_id`由RVIR adapter生成，至少绑定：

```text
solver query/run identity
core ordinal
exclusive generation
```

receipt只保存`exact_call_identity_hash`，避免artifact泄漏用户query字符串；lease私有保存原字符串用于同进程phase比较。
空字符串、控制字符、绝对路径式本机信息和超过冻结长度的ID在读取live mapping前拒绝。

## 5. 对象设计

### 5.1 `S4MutableSlotV1`

保持原V3字段，补充：

```text
alpha_live_requires_grad
alpha_live_is_leaf
beta_live_requires_grad
beta_live_is_leaf
entry_content_capture_ordinal
exit_content_capture_ordinal
```

frozen dataclass只能保存canonical scalar/tuple/string/bool。`validate()`不能访问Tensor，也不能从汇总字段自证。

每个slot仍保存：

- topology四方identity；
- α/β semantic path、source metadata/hash；
- live object/storage稳定group；
- live shape/dtype/device/stride/offset/contiguous/version/content hash；
- α lower-active `[0,0]`和upper-preserved `[1,0]` slice projection；
- feature shape/flat indices/layout hash；
- β location/sign/history hash和active/empty状态。

### 5.2 `S4MutableStateAdmissionV1`

原字段保留，并新增：

```text
exact_call_identity_hash
live_content_capture_pass_count = 2
device_to_host_validation_copy_count
device_to_host_validation_bytes
candidate_kernel_launch_count = 0
candidate_cuda_allocation_count = 0
process_global_query_exclusivity_validated = false
```

`validate()`必须从slot重算：

- slot/path顺序和path-set hash；
- snapshot/plan/topology/policy/projection hash；
- α stored/active/preserved；
- β slot/active/element；
- D2H copy count/bytes；
- exact-call hash格式；
-全部claim/execution flag。

`process_global_query_exclusivity_validated=false`不是失败；它是诚实的阶段边界，直到S4-3 external latch关闭后也不能
回写或篡改S4-0 receipt。

### 5.3 `S4LiveMutableLeaseV1`

普通`__slots__` class，私有字段最低为：

```text
_admission_hash
_exact_call_id
_owner_process_id
_owner_thread_id
_entry_device
_entry_stream_token
_state
_source_rows             # tuple[path, strong Tensor ref, raw tokens]
```

raw row：

```text
path
strong_tensor_reference
id(tensor)
empty/nonempty ownership discriminator
device + untyped_storage()._cdata + storage.data_ptr + storage.nbytes
shape/dtype/device/stride/storage_offset/contiguous
requires_grad/is_leaf
entry _version/content hash
```

PID/thread/stream/raw pointer/token不得进入canonical receipt。`_cdata`不存在时fail closed，不得降级为`Tensor.data_ptr()`。

方法：

```python
revalidate_current_mapping(
    current_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
    phase: str,
    require_content_unchanged: bool,
) -> None

transfer_to_prepared_runtime(
    *, expected_admission_hash: str, exact_call_id: str
) -> None

mark_commit_started(*, exact_call_id: str) -> None
mark_committed_or_aborted(*, exact_call_id: str, outcome: str) -> None
close() -> None
```

每个非`close`方法先检查PID/thread/exact-call/state；`close()`幂等但不恢复能力。

### 5.4 `PreparedS4MutableStateAdmissionV1`

普通`__slots__` class：

```text
receipt                       # public, tensor-free
_live_lease                   # private
_state = OPEN/PREPARING/TRANSFERRED/FAILED_CLOSED/CLOSED
```

只公开：

```python
begin_buffer_prepare(
    current_sources,
    *,
    exact_call_id,
) -> private adoption payload
close()
```

adoption payload只能由S4-1A模块消费；不提供public lease property，不返回`tuple[receipt, lease]`。

lease/wrapper都实现`__copy__`、`__deepcopy__`、`__getstate__`、`__reduce__`、`__reduce_ex__`并抛稳定
`LIVE_LEASE_SERIALIZATION_FORBIDDEN`。不得实现`to_dict/stable_hash`，不得是dataclass。

## 6. 固定验证算法

### Phase A：输入envelope

1. claim/execution flags由构造路径固定false/0；
2. `type(topology) is tuple`；
3. `type(live_mutable_sources) is dict`；
4. keys全部`type(key) is str`；
5. values全部`type(value) is torch.Tensor`；
6. exact-call ID格式和hash；
7. 记录PID、thread和current CUDA stream到ephemeral local state。

任何custom dict/Tensor subclass在读取其用户可覆写方法前拒绝。

### Phase B：snapshot/plan/policy

稳定错误不能靠解析`ValueError`文本。实现顺序：

1. 显式检查snapshot type/schema/id，失败`SNAPSHOT_SCHEMA_VERSION_MISMATCH`；
2. 调用`snapshot.validate()`；任何未被更细S4 precheck覆盖的异常统一包装为`SNAPSHOT_SEMANTIC_INVALID`；
3. 显式检查plan type/schema，随后`plan.validate()`；剩余异常包装`PLAN_SEMANTIC_INVALID`；
4. lower-only policy exact：lower=true、upper=false、fix_intermediate=true、deterministic=true；
5. topology item type/field/nonempty/unique；按plan layout canonicalize。

不要复制整个V4/R31 validator；只加S4需要稳定分类的envelope和projection检查。

### Phase C：semantic projection

1. 从plan layout解析六slot和12个mutable path；
2. snapshot `MUTABLE_COPY_OUT` path集合exact；
3. α role/axes/dtype/source device/content/finite；
4. lower active/preserved slice和feature layout；
5. β value/location/sign role/shape/dtype/finite/nonnegative；
6. 每domain每site要求history width与β physical width exact；empty β width必须0；
7. `plan_binding_projection_hash`从snapshot+topology+plan fields独立重算；
8. `production_plan.source_state_hash`只写`oracle_mapping_provenance_hash`，不与snapshot hash等同。

### Phase D：第一次live token capture

按plan slot顺序、每slot α后β：

1. path coverage exact；
2. object identity；
3. nonempty raw storage token；empty以object identity为owner；
4. shape/dtype/device；
5. stride/storage offset/contiguous；
6. requires-grad/is-leaf；
7. `_version`；
8. full content hash；
9. cross-path object/storage alias projection。

distinct nonempty shared storage一律拒绝；当前formal不自动接受合法view alias。empty zero pointer不得互相成组。

### Phase E：receipt独立重算

1. object/storage group按plan首次出现顺序分配稳定ordinal；
2. 构造slot；
3. 从slots重算计数和hash；
4. 构造receipt；
5. 调用receipt `validate()`；
6. 递归检查receipt对象图无Tensor/module/callback/provider/raw pointer/capsule。

递归walker只遍历明确允许的frozen dataclass/list/tuple/dict/scalar；遇到未知对象立即拒绝，不能调用任意`__iter__`。

### Phase F：第二次live token capture与lease bind

在receipt完成后，对原12条Tensor再按固定顺序采集一次：

- `is` object exact；
- storage/layout/version/content exact；
- cross-path alias exact；
- PID/thread/current stream exact。

任一变化报`LIVE_SOURCE_READ_RACE`，不返回receipt/lease。通过后才把同一Tensor tuple转入lease并返回wrapper。

这不是通用并发锁；它只关闭admission函数自身的观察窗口。S4-1A和S4-3仍必须再次revalidate。

## 7. D2H与同步账

当前formal live source：

```text
6 α stored elements = 8,496
6 β value elements  = 6
total live elements = 8,502
float32 bytes/pass  = 34,008
```

两次full-content capture的静态最低：

```text
12 tensors/pass × 2 = 24 D2H validation copies
34,008 B/pass × 2   = 68,016 transferred logical bytes
```

empty五β copy bytes为0，但仍各计一个logical validation record。实际CUDA runtime可能合并或采用不同copy实现，所以：

- count/bytes是semantic validation账，不冒充CUPTI物理transaction count；
- S4-0不计时；
- S4-P必须把content guard D2H/sync归入wrapper；
- success路径禁止额外whole-device synchronize；
- hash产生的CPU临时对象在返回前释放，不进入persistent memory claim。

若以后为了性能删除第二轮hash或只检查`_version`，必须重新预注册；`.data`/DLPack反例已经证明version-only不安全。

## 8. stable reason补充

在原V3 detail code上新增12项：

| detail code | VerificationRejectionReason | 触发边界 |
|---|---|---|
| `EXACT_CALL_IDENTITY_INVALID` | `RECEIPT_IDENTITY_MISMATCH` | ID空/非法/泄漏式格式 |
| `EXACT_CALL_IDENTITY_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | phase调用ID与lease不同 |
| `LIVE_SOURCE_CONTAINER_TYPE_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | top-level非built-in dict |
| `LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | provider nested非dict/list |
| `LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED` | `DTYPE_OR_DEVICE_MISMATCH` | Parameter/custom subclass |
| `LIVE_SOURCE_OWNER_THREAD_MISMATCH` | `UNSAFE_ALIAS_OR_LIFETIME` | 跨thread使用lease |
| `LIVE_SOURCE_STREAM_MISMATCH` | `UNSAFE_ALIAS_OR_LIFETIME` | current stream漂移 |
| `LIVE_SOURCE_READ_RACE` | `STATE_VERSION_MISMATCH` | 两次capture之间变化 |
| `SNAPSHOT_SEMANTIC_INVALID` | `RECEIPT_IDENTITY_MISMATCH` | snapshot residual validator失败 |
| `PLAN_SEMANTIC_INVALID` | `RECEIPT_IDENTITY_MISMATCH` | plan residual validator失败 |
| `RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | D2H账被篡改 |
| `S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN` | `QUEUE_OR_TERMINATION_EFFECT_CROSSED` | 试图把S4-0局部lease升级为全局exclusive claim |

detail code是runtime binding精化，不修改`VerificationRejectionReason`枚举。

## 9. 状态与ownership

### 9.1 S4-0 wrapper

```text
OPEN
  → PREPARING
      → TRANSFERRED
      → FAILED_CLOSED
  → CLOSED
```

`begin_buffer_prepare`第一行先`OPEN→PREPARING`，然后revalidate；失败也不得回OPEN。`close()`从任意未terminal状态清理
lease并进入CLOSED。

### 9.2 lease

```text
OPEN
  → TRANSFERRED_TO_PREPARED_RUNTIME
  → COMMITTING
  → COMMITTED | ABORTED_CLEAN | POISONED_NO_RETRY
  → CLOSED
```

S4-0只实现并测试OPEN/single-transfer/close；后续transition由S4-1A/S4-3调用，但枚举和值现在冻结，避免跨文件同名
异义。

### 9.3 process-global latch

S4-3已有：

```text
UNCLAIMED → PREPARED → COMMITTING → ... → COMPLETED/POISONED
```

V4修订要求adapter在S4-0 admission前预留generation，并将`exact_call_id`绑定该generation；但真正“同query第二次core/
fallback/retry拒绝”仍须S4-3 latch实现后才claim。S4-0 tests只证明ID/hash/phase mismatch与local duplicate transfer拒绝。

## 10. 逐函数施工清单

### 10.1 pure canonical helpers

```text
_canonical_json
_canonical_hash
_is_sha256
_reject
_validate_exact_call_id
_topology_projection
_mutable_path_projection
_plan_snapshot_projection
_tensor_free_walk
```

这些helper不能import TVM/provider，不读wall clock，不访问global registry。

### 10.2 ephemeral helpers

```text
_raw_storage_token
_current_stream_token
_live_tensor_token
_capture_live_rows
_validate_live_rows_equal
```

只在lease/admission内部使用；返回值不得进入receipt。

### 10.3 public objects/functions

```text
S4MutableStateAdmissionError
S4MutableSlotV1
S4MutableStateAdmissionV1
S4LiveMutableLeaseV1
PreparedS4MutableStateAdmissionV1
extract_s4_live_mutable_sources_v1
prepare_s4_mutable_state_admission_v1
```

模块`__all__`只导出这些名字；raw token/helper不导出。

## 11. 测试施工包

### 11.1 fixture层

同一个测试文件提供三层fixture：

1. **generic synthetic CPU**：2—3 slot、active/empty β混合，证明schema不写死ResNet/6/31；
2. **frozen formal projection**：从已冻结snapshot/topology/R31 plan构造六slot/12 path CUDA live clone，验证正式算术；
3. **provider adapter structural fixture**：最小AlphaValueData/BetaFullData-like plain objects，验证strict built-in extraction，
   不importexternal provider。

正式closure另跑一个真实αβ-CROWN worker，不能只靠synthetic fixture。

### 11.2 positive minimum

至少12项：

1. generic 2-slot成功，schema无model/site/shape常数；
2. formal slot/path=`6/12`；
3. α stored/active/preserved=`8496/4248/4248`；
4. β slots/active/elements=`6/1/6`；
5. live tensors/elements/logical bytes=`12/8502/34008`；
6. content passes/D2H count/bytes=`2/24/68016`；
7. topology tuple和snapshot tensor输入置换不改变receipt；
8. 相同exact-call ID跨fresh process receipt/hash exact；
9. 不同exact-call ID只改变identity-bound投影；
10. receipt递归tensor/provider/pointer-free；
11. lease强引用在外部mapping删除+GC后仍保持原Tensor；
12. monkeypatch dense initializer/TVM/provider/allocator/clock为必抛，admission仍通过。

### 11.3 negative minimum 56

原44项全部保留；新增：

45. exact-call ID空/非法；
46. receipt exact-call hash全重签篡改；
47. 同一wrapper第二次prepare/transfer；
48. transfer或revalidate传入另一个exact-call ID；
49. 从非owner thread调用lease；
50. current CUDA stream token漂移；
51. top-level live mapping为dict subclass/custom Mapping；
52. provider alpha/beta `_data`非built-in dict；
53. nested alpha为custom Mapping；
54. beta collection为tuple/custom list；
55. live source为`torch.nn.Parameter`或Tensor subclass；
56. first/second capture之间object/storage/version/content任一漂移，稳定`LIVE_SOURCE_READ_RACE`。

每项同时断言exact detail code与`VerificationRejectionReason`。复合攻击按§6固定顺序只允许一个首要reason。

### 11.4 failure injection seam

实现只在测试构造器中接受private hook，不进入public API：

```text
after_input_envelope
after_snapshot_validation
after_first_live_capture
after_receipt_validation
before_second_live_capture
before_lease_publish
```

每点异常必须：

- 不返回partial receipt/lease；
- 不保留Tensor强引用；
- 不修改live source；
- 不调用fallback/retry；
- 不初始化TVM/candidate runtime。

## 12. formal closure设计

S4-0代码/单元关闭后，另一个提交生成5 fresh real provider worker：

```text
same model/property/config/source
same protocol exact-call id template
one admission per process
admit → receipt serialize → lease close
no buffer prepare / no candidate execute / no mutation
```

每workerraw至少保存：

- source/protocol/model/property/config hash；
- snapshot/topology/plan/projection hash；
- full canonical receipt；
- 12 path type/shape/dtype/device/stride/offset/version/content projection；
- D2H semantic copy count/bytes；
- candidate kernel/allocation/provider callback counters=`0`；
- lease state transition和close后strong-ref cleared evidence；
- exact-call identity hash，不保存本机raw ID。

stdlib replay从JSON重算全部canonical count/hash。raw不保存Tensor payload，因为S4-0只关闭admission metadata/content
digest；whole-core完整IEEE payload属于S4-4。外审仍可从pinned legacy `.pt`抽查digest对应性。

internal通过状态只能是：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-0
FORMAL-NO-GO-S4-0-MUTABLE-STATE-ADMISSION
```

外审批准后的closure commit才写`VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`。

## 13. 提交顺序

S3批准后唯一顺序：

1. `docs: activate S4-0 construction contract`；
2. `feat(runtime): add S4 mutable-state receipt and strict live admission`；
3. `test(runtime): close S4-0 local lease and 56 negative gates`；
4. `artifact: generate S4-0 five-fresh admission evidence`；
5. `docs: deliver S4-0 external audit`；
6. `docs: close S4-0 or formal no-go`；
7. `docs: activate S4-1A buffer prepare`。

代码、测试、formal raw、external audit和closure分提交。不得用同一dirty source同时生成代码和第一次formal result。

## 14. 设计指纹

为防实施时无意改顺序，本稿V4 construction model使用UTF-8、JSON key排序、紧凑分隔符
`(',', ':')`和JSON原生boolean。完整canonical JSON如下，不允许只按后面的自然语言摘要猜字段名：

```json
{"added_detail_codes":["EXACT_CALL_IDENTITY_INVALID","EXACT_CALL_IDENTITY_MISMATCH","LIVE_SOURCE_CONTAINER_TYPE_MISMATCH","LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH","LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED","LIVE_SOURCE_OWNER_THREAD_MISMATCH","LIVE_SOURCE_STREAM_MISMATCH","LIVE_SOURCE_READ_RACE","SNAPSHOT_SEMANTIC_INVALID","PLAN_SEMANTIC_INVALID","RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH","S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN"],"claims":{"local_single_transfer":true,"phase_identity_bound":true,"process_global_exclusivity":false},"formal_counts":[6,12,12,8502,34008,2,24,68016,56],"gpu_accounting":{"candidate_cuda_allocation":false,"candidate_kernel":false,"d2h_validation_disclosed":true},"live_token_order":["object","storage","layout","requires_grad_and_leaf","version","content","cross_path_alias","owner_context"],"signature":["snapshot","topology","production_plan","live_mutable_sources","exact_call_id"],"validation_order":["claim_flags","topology_container","live_source_container","live_source_keys","live_source_tensors","exact_call_identity","owner_context","snapshot_envelope","snapshot_semantics","plan_envelope","plan_semantics","policy","topology_projection","semantic_projection","entry_live_capture","receipt_recompute","receipt_object_graph","exit_live_capture_and_lease_bind"]}
```

SHA256：

```text
471424594fb4b6d017feac936a6005eb9d0451fd5579d026204ec952d0995239
```

模型冻结：

- 5个入口参数；
- 18阶段validation order；
- 8项live token order；
- 12项新增detail code；
- formal `6/12/12/8502/34008/2/24/68016/56`算术；
- local single-transfer=true、phase identity bound=true、process-global exclusivity=false；
- candidate kernel=false、D2H validation disclosed=true。

实现必须从代码对象重新构造并比较该模型；不能把hash常量硬编码成PASS。

## 15. GO / STOP

### GO

- exact-call identity进入receipt/lease及所有phase revalidation；
- strict provider extraction在读取自定义Mapping方法前拒绝；
- formal 6/12、α/β算术和exact β/history width全部关闭；
- 两次live capture与read-race门禁通过；
- receipt tensor/provider/pointer-free，lease/wrapper不可复制/序列化；
- D2H `24/68016 B`如实披露，candidate kernel/allocation=`0/0`；
- minimum 56 negative exact reason；
- 5 fresh real provider worker和stdlib replay通过；
- external audit批准；
- S3已经approved/closed，S4-1A仍closed。

### STOP

- 仍用历史宽松helper声称“custom Mapping读取前拒绝”；
- receipt无exact-call identity却声称不能跨query；
- 把S4-0 local lease冒充S4-3 process-global exclusivity；
- 解析`ValueError`英文文本生成stable reason；
- 只做一次content capture；
- 只看`_version`，删除content hash；
- 把D2H validation copy写成“零GPU活动”；
- Tensor/provider/raw token进入receipt/artifact；
- S4-0同时实现buffer/TIR/policy/commit/timing；
- external audit前写VALIDATED。

## 16. 当前状态

```text
S3 external audit                         = pending
S4-0 V4 source/API/test construction      = complete
S4-0 production implementation            = closed
S4-0 formal run                           = closed
S4-1A and later                           = closed
S4-P timing/performance                   = closed
```

当前唯一外部动作仍是S3审计。本文使S3一旦批准，S4-0可以按明确文件、API、validation order、56项negative和
five-fresh evidence直接施工，而不需要在编码时临时决定ownership或篡改分类。
