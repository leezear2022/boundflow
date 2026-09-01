---
status: diagnostic-complete-implementation-contract-frozen-code-closed
date: 2026-08-28
type: implementation-readiness
topic: boundflow
slug: asplos27-s4-1a-prepare-transaction-readiness
stage: s04
depends-on: validated-s4-0-mutable-state-admission
execution-authority: false
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1A：buffer prepare事务、lease转移与失败清理实施就绪结论

> **V5权威修订（2026-08-29）**：精确施工合同已由
> `BOUNDFLOW_ASPLOS27_S4_1A_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`取代。V5新增单一
> `_S4BufferResourceOwnerV1`，禁止逐字段move；cleanup除清owner字段外还必须清loop local，并在退出`except`后抛stable
> error，避免retained traceback继续持有CUDA Tensor/TVM view。现场反例中只clear container残留`1,024 B`，修正后
> parameter/buffer/view三阶段保留异常对象仍为allocated delta 0且`__context__ is None`。现有
> `(data_ptr,shape)`view key存在same-pointer/shape different-stride静默碰撞，V5在lookup前拒绝noncontiguous并绑定完整
> storage/layout identity。本文后文“success无同步”“36 negative”“one-step Adam positive”只保留历史语义。

## 0. 直接结论

S4-1A的数值buffer布局已经可行，但旧蓝图缺少两个production级所有权合同：

1. prepared runtime必须持有S4-0的private strong-ref lease到S4-3，因此“prepare后不得保留provider source Tensor”是
   错误门禁；正确边界是**恰好12条source Tensor只能存在于private lease，provider容器/callback和lease外额外引用为0**；
2. prepare会依次创建CUDA parameter、gradient/output与TVM DLPack view；任一步失败都必须按固定逆序清理、关闭lease、
   禁止重试，而不能留下半prepared owner或回退native路径。

冻结实现形态：

```text
PreparedS4MutableStateAdmissionV1(OPEN, owns lease)
  → begin_prepare: OPEN → PREPARING
  → Phase A: validate current provider + environment, allocation=0
  → Phase B: stage independent leaf/gradient/output/views locally
  → Phase C: validate receipt, single-transfer lease+staging
      ├─ success: wrapper=TRANSFERRED, PreparedS4MutableBuffersV1=PREPARED
      └─ failure: wrapper=FAILED_CLOSED, lease=CLOSED, staged refs=0
```

这不是新IR。它是S4-0 ephemeral lease到S4-1A prepared runtime之间的runtime transaction。S3外审和S4-0实现门禁
仍未关闭，因此本文不授权production代码、TIR launch或timing。

## 1. 可复用资产与缺口

### 1.1 R31B2 prepared DLPack cache

`PreparedR31B2CompiledCustomBackwardV1`已经证明：

- prepare时以`(tensor.data_ptr(), shape)`注册TVM view；
- empty tensor跳过；
- `torch.from_dlpack(view).data_ptr()==tensor.data_ptr()`可作pointer-exact gate；
- warm path只查prepared view，不调用`from_dlpack`。

S4-1A复用该机制，但必须额外验证buffer storage彼此独立、source storage不alias，以及prepare失败时先销毁TVM/roundtrip
view再销毁Tensor。

### 1.2 B4-B2 compressed parameter构造

现有sparse Linear/Conv TIR使用compressed production α/β直接构造leaf，证明无需dense scatter。S4-1A必须固定为：

```text
parameter = (
    source_active.detach()
    .clone(memory_format=torch.contiguous_format)
    .requires_grad_(True)
)
gradient = torch.empty_like(parameter, requires_grad=False)
```

禁止只用`.to(device).contiguous()`：当source已在目标device且contiguous时，它可能返回alias。每个parameter必须与12条
source、其他parameter、全部gradient/output拥有不同storage token。

### 1.3 S3 optimizer owner

S3已经验证host Adam、scheduler、`.grad`绑定和10/9 policy，但只覆盖P α。S4-1A只验证7个leaf与persistent gradient
owner；双param-group LR和moments由S4-2继续拥有，不能把一次probe升级成S4-2 closure。

## 2. formal buffer owner探针

从冻结`source_capture.pt`恢复12条mutable state，构造等价live CUDA source，然后只执行buffer pack、DLPack bind和一次
双param-group Adam owner smoke：

```text
S4_1A_FORMAL_BUFFER_OWNER_PROBE_PASS
alpha_parameter_count=6
active_beta_parameter_count=1
empty_beta_token_count=5
parameter_element_count=4254
gradient_element_count=4254
parameter_logical_bytes=17016
gradient_logical_bytes=17016
base_dlpack_view_count=16
dlpack_pointer_exact_count=16
all_parameter_leaf=true
all_parameter_storage_independent=true
source_hash_unchanged=true
source_version_unchanged=true
parameter_pointer_stable_after_adam=true
gradient_pointer_stable_after_adam=true
post_scheduler_lr=[0.0098,0.049]
```

结论：当前基础physical ABI恰为：

```text
6 alpha parameters + 1 active beta parameter
6 alpha gradients  + 1 active beta gradient
1 lower output     + 1 fixed upstream
= 16 base physical buffers/views
```

五个empty β只形成typed token，base DLPack view=`0`。S4-1D可以因TIR signature增加同storage reshape view，但必须
另记`additional_tir_view_count`，不得改变S4-1A `base_view_count=16`。

## 3. live source lease的真实内存口径

formal 12条source包括：

- 六α full source：8,496 float32；
- 唯一active β：6 float32；
- 五empty β：0元素；
- 合计：8,502元素、34,008 logical bytes。

现场强引用生命周期探针：

```text
S4_1A_LIVE_SOURCE_LEASE_RETENTION_PROBE_PASS
leased_source_tensor_count=12
leased_source_logical_elements=8502
leased_source_logical_bytes=34008
lease_incremental_allocated_bytes=0
allocated_drop_after_external_owner_removed=0
lease_extends_lifetime=true
allocated_after_lease_close_vs_baseline=0
```

因此必须区分：

- lease新增Python引用和lifetime，不新增CUDA storage；
- source storage由provider本来就拥有，`lease_incremental_allocated_bytes=0`；
- 若provider外部owner消失，lease会合法延长这34,008 logical bytes的lifetime；
- S4-3结束必须确定性close，否则会把lifetime延长到query之后；
- 这34,008 B必须作为`leased existing source`披露，不能计入candidate new allocation，也不能完全从memory ledger消失。

## 4. 修正prepared owner边界

### 4.0 runtime对象与artifact对象分离

- `S4EmptyBetaSlotTokenV1`、`S4MutableBufferPreparationReceiptV1`可以是frozen dataclass，因为只含canonical metadata；
- `S4PhysicalMutableSlotV1`、`PreparedS4MutableBuffersV1`必须是带`__slots__`的普通class，持有Tensor/TVM view/private
  lease，并与S4-0 wrapper一样拒绝copy/deepcopy/pickle/asdict；
- artifact入口只能接收preparation receipt，禁止对prepared owner做递归serialization/introspection；
- `metadata()`只能返回receipt或由receipt生成的纯JSON payload，不能动态遍历runtime object graph。

### 4.1 唯一允许的source引用

`PreparedS4MutableBuffersV1`递归对象图允许：

```text
_live_source_lease
  └─ exact 12 strong Tensor refs in canonical path order
```

但禁止：

- `pre_result`、provider node、alpha/beta container；
- provider lookup callback/closure；
- lease之外第二份source Tensor tuple/dict；
- snapshot CPU Tensor被误当live source；
- arbitrary source override。

旧detail `PROVIDER_SOURCE_RETAINED_AFTER_PREPARE`删除，替换为：

| detail | verification reason |
|---|---|
| `LEASED_SOURCE_INVENTORY_MISMATCH` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `SOURCE_TENSOR_OUTSIDE_PRIVATE_LEASE` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `PROVIDER_CONTAINER_OR_CALLBACK_RETAINED` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `LEASED_SOURCE_BYTES_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |

### 4.2 candidate storage独立性

必须机械证明：

1. 7 parameter storage互不alias；
2. 7 gradient storage互不alias；
3. parameter与gradient交集为空；
4. lower/upstream与前14项交集为空；
5. 16项与12 source storage交集为空；
6. empty β无Tensor/storage/view/optimizer ordinal；
7. source hash/version在prepare前后exact。

raw token沿用S4-0 pinned contract，不进入receipt；receipt只保存stable buffer ordinal、shape/dtype/device、logical bytes和
pointer-exact计数。

## 5. 两阶段prepare事务

### 5.1 wrapper状态机

```text
OPEN
  → PREPARING
      → TRANSFERRED
      → FAILED_CLOSED
```

- `begin_prepare()`只能调用一次，并在任何validation/allocation前把wrapper置为`PREPARING`；
- 同一wrapper第二个调用者立即得到`BUFFER_PREPARE_ALREADY_ATTEMPTED`；
- `FAILED_CLOSED`不能重开、retry或native fallback；
- `TRANSFERRED`后wrapper不再持有lease，prepared runtime成为唯一owner；
- close允许幂等，但不恢复prepare能力。

### 5.2 Phase A：零allocation validation

1. wrapper `OPEN→PREPARING`；
2. receipt/lease shared admission hash；
3. current source必须是existing helper返回的exact built-in dict；
4. 按S4-0固定顺序重验object/storage/physical/layout/version/content/alias；
5. device=`cuda:0`、dtype=float32、current stream与entry policy记录；
6. 计算expected parameter/gradient/token/view manifest；
7. monkeypatch CUDA allocation/TVM view为必抛时，Phase A仍必须通过。

### 5.3 Phase B：本地staging

固定顺序：

```text
6 alpha leaf clone
  → 1 active beta leaf clone
  → 7 persistent gradient
  → lower + upstream
  → validate 16-way storage independence/content/leaf
  → create 16 base TVM DLPack views
  → create-and-release roundtrip validation tensors
  → build and validate tensor-free preparation receipt
```

staging期间lease仍由prepared admission wrapper拥有。候选资源只能在函数局部固定tuple/list中存在，不得提前发布到registry、
global cache或provider object。

### 5.4 Phase C：single-transfer adoption

所有可能调用PyTorch/TVM/hash/canonicalization的步骤都在Phase C前完成。最终adoption只允许固定字段赋值：

```text
install lease into prepared owner
  → install staged tensor/view tuples
  → clear wrapper lease field
  → wrapper state=TRANSFERRED
  → prepared state=PREPARED(version=0,generation=-1)
```

adoption不得调用callback、allocator、TVM、hash、provider或user-defined method。实现必须用`try/finally`覆盖字段转移之间的
异常；任一异常由当前实际owner关闭lease并清理staging，禁止出现wrapper和prepared都以为自己拥有lease或二者都不拥有。

## 6. 失败清理与分类

### 6.1 固定逆序

Phase A/B/C任一失败，清理顺序固定为：

```text
roundtrip torch validation views
  → TVM DLPack views
  → lower/upstream
  → gradients
  → parameters
  → local staging containers
  → lease strong refs
  → wrapper=FAILED_CLOSED
  → verify device/stream/policy restored
```

禁止隐式调用`torch.cuda.empty_cache()`，因为它影响全局allocator/cache并破坏公平计时；只释放本事务引用。PyTorch
allocator reserved bytes可以保留，但allocated/live tensor必须回到entry ledger。

formal fault receipt在清理引用后只同步**entry current stream**再采集allocated/source/device证据，不执行whole-device
`torch.cuda.synchronize()`；success prepare不因本门禁新增同步。reserved delta只披露，不作为cleanup pass/fail。

### 6.2 现场三点故障注入

在parameter、buffers、views三个阶段分别注入异常：

```text
S4_1A_PREPARE_FAILURE_CLEANUP_PROBE_PASS
wrapper_state=FAILED_CLOSED                  # 3/3
lease_released=true                         # 3/3
candidate_python_refs_alive=0               # 3/3
allocated_delta_after_cleanup=0             # 3/3
retry_reason=BUFFER_PREPARE_ALREADY_ATTEMPTED # 3/3
source_hash_unchanged=true
source_version_unchanged=true
device_stream_restored=true
empty_cache_called=false
```

这些失败发生在live mutation前，transaction outcome为`PREPARE_ABORTED_CLEAN`，不是S4-3的mid-commit
`POISONED_NO_RETRY`。但formal candidate仍必须终止，不得在同一query回退native或重试prepare。

## 7. 新增stable detail code

在原S4-1A reason上增加/修正：

| detail code | verification reason |
|---|---|
| `BUFFER_PREPARE_ALREADY_ATTEMPTED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `BUFFER_PREPARE_TRANSFER_STATE_MISMATCH` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `BUFFER_PREPARE_STAGING_FAILED` | `RUNTIME_FALLBACK_REQUIRED` |
| `BUFFER_PREPARE_CLEANUP_INCOMPLETE` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `PARAMETER_SOURCE_STORAGE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `PARAMETER_GRADIENT_STORAGE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `BASE_DLPACK_VIEW_COUNT_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `PREPARE_SOURCE_MUTATION_OBSERVED` | `STATE_VERSION_MISMATCH` |
| `PREPARE_DEVICE_STREAM_OR_POLICY_DRIFT` | `DTYPE_OR_DEVICE_MISMATCH` |
| `PREPARE_EMPTY_CACHE_FORBIDDEN` | `RUNTIME_FALLBACK_REQUIRED` |
| `PREPARE_FALLBACK_OR_RETRY_FORBIDDEN` | `RUNTIME_FALLBACK_REQUIRED` |

底层异常类型/文本不得直接成为artifact reason；应封装为stable detail并把原异常类别放在debug-only、非canonical诊断字段。

## 8. 修正后的测试门槛

S4-1A minimum negative冻结为至少36类。除旧蓝图用例外必须包含：

1. `.to().contiguous()`无clone导致source alias；
2. 两parameter共享storage；
3. parameter/gradient共享storage；
4. lower/upstream与parameter alias；
5. empty β创建zero-width physical Tensor/view/optimizer entry；
6. lease内source少/多/乱序或logical bytes不等34,008；
7. source Tensor在lease外被prepared owner再次保存；
8. provider container/callback/closure被保存；
9. base view不是16或pointer exact不是16；
10. parameter、buffer、view三个阶段分别故障注入；
11. cleanup后仍有candidate Tensor/TVM view强引用；
12. cleanup allocated bytes不回entry ledger；
13. cleanup调用`empty_cache()`；
14. failure后第二次prepare；
15. failure后native fallback；
16. adoption任一字段转移点注入异常，不得double-owner/no-owner；
17. source content/version在prepare前后漂移；
18. device/stream/deterministic policy漂移；
19. wrapper/prepared/lease copy/deepcopy/pickle；
20. receipt把raw pointer、Tensor或lease写入artifact。

positive至少证明：

- formal `6/1/5`、`4254/4254`、`17016/17016 B`；
- base DLPack `16/16` pointer exact；
- 16 candidate storage互异且与12 source storage集合不相交；
- lease source=`12 / 8502 elements / 34008 logical B / incremental allocation 0`；
- one-step Adam后parameter/gradient pointer稳定、source hash/version不变；
- three-stage failure cleanup全部clean；
- `fallback/retry/timing/performance=0/0/false/false`。

## 9. 文件与门禁

S3与S4-0批准后，S4-1A仍只新增：

```text
boundflow/runtime/asplos27_s4_ordered_buffer_abi.py
tests/test_asplos27_s4_ordered_buffer_abi.py
```

不新增prepare transaction IR、registry或provider adapter；两阶段事务作为同一runtime模块内的private implementation。

关闭S4-1A必须同时满足：

- S4-0 receipt/lease已validated；
- formal owner与memory ledger逐项一致；
- two-phase prepare single-transfer成立；
- minimum 36类negative exact reason；
- 三阶段failure cleanup、source不变和禁止retry/fallback成立；
- targeted/full/static/DocOps通过；
- S4-1B仍在S4-1A关闭前保持closed。

当前状态不变：

```text
S3 exchange = ready_for_audit
S4-0 implementation = closed
S4-1A implementation = closed
S4 timing/performance = closed
```
