---
status: corrected-v4-prepare-transaction-frozen-implementation-blueprint
date: 2026-08-28
type: implementation-plan
topic: boundflow
slug: asplos27-s4-1a-ordered-buffer-abi
stage: s04
depends-on: validated-s4-0-mutable-state-admission
execution-authority: false-pending-s3-external-audit-and-s4-0
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1A：ordered parameter/gradient buffer ABI实施蓝图

## 0. 直接结论

S4-1A应把S4-0的tensor-free slot admission与live version lease实例化为**稳定的GPU buffer ownership**，但仍不实现CROWN数学或
TIR launch。核心选择是：

1. 六个lower-α各自成为独立contiguous leaf parameter buffer，shape=`[domain, compressed_width]`；
2. 唯一non-empty β成为一个leaf parameter buffer；五个empty β只保留typed empty token，不伪造参数或指针；
3. preserved α direction继续由immutable production snapshot/commit receipt拥有，不复制到candidate GPU optimizer；
4. 六dα和active dβ使用persistent output buffer，Adam直接消费，不在每次evaluation分配或clone；
5. hot evaluator调用只接受ordinal/version/token，不接受dict、semantic-path lookup、任意callback或动态tensor列表；
6. 所有DLPack view只在prepare阶段建立，warm evaluation新建view=`0`。

这不是新的IR。S4-0现在返回`PreparedS4MutableStateAdmissionV1`：公开部分是tensor-free receipt，私有部分是强引用
原始source的ephemeral lease。S4-1A必须从current provider mapping复核同一Python object/storage/version后接管该lease，
不能仅凭receipt稳定group重新取得same-content clone。Plan/Bound/Verification Graph仍由已有owner负责。V3反例与生命周期见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_LIVE_LEASE_IMPLEMENTATION_READINESS_2026_08_28.md`。

prepare两阶段事务、strong-ref retention账和失败清理的最终实施合同见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_PREPARE_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`。

## 1. 为什么不能沿用完整α source作为optimizer参数

production α source shape为`[2,1,D,W]`：

- lower-only mutable plane=`source[0,0]`；
- preserved plane=`source[1,0]`；
- atomic copy-out只更新mutable plane并保留另一方向。

两个看似简单的实现都不合格：

### 1.1 把完整source交给Adam

这会让Adam为8,496个元素建立参数与moments，其中4,248个preserved元素不应优化。它重新引入无意义状态和显存，
也模糊production ownership。

### 1.2 把`source[0,0]` view直接交给Adam

普通slice view不是独立leaf parameter。即使某个PyTorch版本暂时接受，也会把optimizer ownership绑定到full source
storage，破坏pointer/lifetime和terminal copy-out合同。

### 1.3 冻结方案

prepare时为每个slot创建独立buffer：

```text
source[0,0]
  → validate source/slice hash
  → device contiguous clone
  → detach().requires_grad_(True)
  → leaf lower-alpha parameter buffer
```

terminal时不回填full GPU source；由existing atomic copy-out以immutable source为模板，只更新active slice。

## 2. formal buffer inventory

### 2.1 physical parameter buffers

| 类型 | 数量 | 元素 | logical bytes(float32) |
|---|---:|---:|---:|
| lower-α parameter | 6 | 4,248 | 16,992 |
| active β parameter | 1 | 6 | 24 |
| empty β token | 5 | 0 | 0 |
| 合计physical parameter | 7 | 4,254 | 17,016 |

### 2.2 persistent evaluation/optimizer buffers

| 类型 | 数量 | 元素 | logical bytes(float32) |
|---|---:|---:|---:|
| dα output | 6 | 4,248 | 16,992 |
| active dβ output | 1 | 6 | 24 |
| lower output | 1 | 6 | 24 |
| fixed upstream | 1 | 6 | 24 |
| Adam m+v（S4-2才创建） | 14 tensors | 8,508 | 34,032 |

S4-1A本身只创建parameter/gradient/lower/upstream与typed tokens；Adam moments由S4-2 sealed policy driver prepare。
这些是logical design bytes，不是allocator peak、reserved memory或性能claim。

### 2.3 leased existing source与preserved state

S4-0 live lease强引用六个full α、六个β value source，共`8,502 elements / 34,008 logical bytes / 12 tensors`；其中
4,248 preserved α为16,992 B。lease新增CUDA allocation=`0`，但会延长existing source lifetime，因此必须单列
`leased existing source`，不能算作candidate new allocation或完全省略。

## 3. 建议新增模块和类型

建议新增：

```text
boundflow/runtime/asplos27_s4_ordered_buffer_abi.py
```

它依赖S4-0 admission模块，但不依赖provider solver、TVM module或optimizer loop。

### 3.1 `S4EmptyBetaSlotTokenV1`

frozen metadata：

```text
slot_ordinal
semantic_path
shape                       # [D,0]
source_hash
physical_buffer_present=false
```

token不得提供`data_ptr`、DLPack、fake scalar或zero-width optimizer parameter。

### 3.2 `S4PhysicalMutableSlotV1`

runtime对象按admission顺序持有：

```text
slot_ordinal
alpha_parameter             # leaf CUDA float32 [D,W]
alpha_gradient              # persistent CUDA float32 [D,W]
beta_parameter_or_none      # only non-empty
beta_gradient_or_none
alpha_source_active_hash
alpha_device_initial_hash
beta_source_hash
beta_device_initial_hash
```

runtime对象可以持有tensor，但其`metadata()`/artifact receipt不得序列化tensor或raw pointer。
该对象必须是非dataclass `__slots__` class并拒绝copy/deepcopy/pickle；不能让`dataclasses.asdict()`递归进入Tensor。

### 3.3 `PreparedS4MutableBuffersV1`

职责：

- 验证S4-0 admission hash与snapshot hash；
- 从prepared admission单次transfer live lease，并从current provider mapping验证path、Python object、raw storage、version、
  shape/dtype/device/stride/offset/content；
- 一次性pack 6 α + active β；
- 创建persistent gradient/lower/upstream；
- 建立slot ordinal→physical buffer的tuple，不建立hot dict；
- 建立prepare-only DLPack views；
- 维护state/evaluation/result lease版本；
- 提供tensor-free preparation receipt。
- 只在private lease内保留恰好12条provider source Tensor；禁止provider container/callback和lease外source引用；
- prepare任一步失败时按view→buffer→lease固定逆序清理并进入`FAILED_CLOSED`，禁止retry/fallback。

建议属性：

```text
admission
live_source_lease              # private, nonserializable, kept through S4-3
physical_slots                  # tuple，严格admission order
alpha_parameters                # tuple[6]
alpha_gradients                 # tuple[6]
beta_parameters                 # tuple[1] for current formal
beta_gradients                  # tuple[1]
beta_slot_to_physical_ordinal   # tuple[6]，empty=-1
empty_beta_tokens               # tuple[5]
lower_output
fixed_upstream                  # [D,1], frozen -1 for -lower.sum()
state_version
evaluation_generation
lease_state
prepare_state                  # OPEN/PREPARING/TRANSFERRED/FAILED_CLOSED
```

`beta_slot_to_physical_ordinal`属于static tuple，不允许hot path按semantic path查dict。
`PreparedS4MutableBuffersV1`同样是非dataclass `__slots__` class并拒绝copy/deepcopy/pickle；artifact只接受独立frozen
preparation receipt，不递归遍历prepared owner。

## 4. ordered evaluator ABI

### 4.1 request

```text
S4EvaluationRequestV1:
    evaluation_ordinal
    expected_state_version
    require_terminal_handoff
    schedule_action_hash
```

parameter tensor不出现在request；prepared evaluator已经持有固定buffer。request不得携带dict、callback、provider
object或任意tensor override。

### 4.2 result lease

```text
S4EvaluationResultLeaseV1:
    evaluation_ordinal
    input_state_version
    evaluation_generation
    lower                       # persistent [D,1] view
    alpha_gradient_leases       # exact ordered tuple[6]
    beta_gradient_leases        # ordered tuple[6]: physical or empty token
    terminal_handoff_or_none
    execution_receipt
```

result是lease，不是owned clone。driver在lease有效期内读取lower/gradient；optimizer mutation完成前evaluator不得重写
这些buffer。capture/debug需要CPU clone时必须在formal/debug scope显式计数，不能进入candidate headline。

### 4.3 不使用PyTorch autograd graph

S4 evaluator直接调用compiled forward+VJP并返回persistent gradients；sealed driver手工把这些buffer绑定到leaf
parameter的`.grad`。不得使用：

- `torch.autograd.Function.apply`；
- executor global registry；
- `save_for_backward(*tensors)`；
- `torch.autograd.grad`；
- higher-order gradient。

论文中的custom VJP指verification-specific compiled VJP，不要求保留PyTorch autograd history。

## 5. lifecycle与版本状态机

唯一合法状态转移：

```text
PREPARED(state_version=0)
  → EVALUATING(ordinal=0, version=0)
  → RESULT_LEASED(generation=0)
  → MUTATING
  → PREPARED(state_version=1)
  ...
  → EVALUATING(ordinal=9, version=9)
  → TERMINAL_RESULT_LEASED
  → TERMINAL_HANDOFF_CONSUMED
```

规则：

- ordinal必须连续0—9；
- ordinal 0—8要求一次mutation，state version各加1；
- ordinal 9不得mutation；
- 同一generation只能产生一个result lease；
- lease未消费时禁止下一evaluation或optimizer mutation之外的写；
- gradient generation必须等于lower generation；
- parameter pointer、gradient pointer、lower pointer在十次evaluation中保持不变；
- terminal handoff只允许ordinal 9且one-shot；
- exception必须回到最近一次已提交state version，不产生半写gradient/parameter状态。

S4-1A只实现/测试状态机与buffer owner，不执行真实evaluation；S4-1D才用compiled evaluator驱动它。

## 6. prepare入口与两阶段事务

```text
prepare_s4_mutable_buffers_v1(
    prepared_admission: PreparedS4MutableStateAdmissionV1,
    current_live_sources: dict[str, torch.Tensor],
    device: torch.device,
    stream_identity: ...,
) -> PreparedS4MutableBuffersV1
```

不能从snapshot CPU clone直接pack后声称接入live solver；必须使用current provider mapping中的原Tensor。mapping只在
prepare调用中读取，不保留provider lookup callback；prepared owner必须接管S4-0 strong-ref lease并保持到S4-3
commit/abort。prepared对象递归持有source Tensor是**必要行为**，但只能存在于private lease且恰好12条。

严格步骤分三phase：

**Phase A（allocation=0）**：

1. wrapper `OPEN→PREPARING`，第二次调用立即拒绝；
2. 校验receipt/lease identity、exact built-in dict/path coverage；
3. 按object/storage/physical/layout/version/content/alias重验current source；
4. 冻结device/stream/policy和expected manifest。

**Phase B（local staging）**：

5. 用`source[0,0].detach().clone(memory_format=contiguous).requires_grad_(True)`建立六α leaf；
6. active β同样显式clone为独立leaf，五empty β只建token；
7. 建立7 gradient、lower、upstream并验证16-way storage独立及与12 source storage不相交；
8. 建立恰好16个base DLPack view并验证16/16 pointer exact；roundtrip Tensor立即释放；
9. 构造并验证tensor-free preparation receipt。

**Phase C（single-transfer adoption）**：

10. 只用固定字段赋值把lease和staging移入prepared owner；
11. 清空wrapper lease并置`TRANSFERRED`；
12. prepared置`PREPARED/version=0/generation=-1`。

任一步失败必须逆序清除roundtrip→TVM view→output→gradient→parameter→lease，wrapper=`FAILED_CLOSED`；不调用
`torch.cuda.empty_cache()`，不retry，不native fallback，且source hash/version、device/stream/policy必须不变。
formal fault evidence只允许同步entry current stream后读取allocated/source证据；reserved delta仅披露、不作pass/fail，
success prepare不新增同步。

S4-1A不得创建full `[2,1,D,W]` device α copy、dense `[D,*feature_shape]` α/β、Adam、TIR module、CUDA Graph或
timing event。

## 7. DLPack与pointer纪律

现有R31B2已经证明prepare-time view cache与pointer-exact检查可行。S4-1A继承以下合同：

- view key至少包含`data_ptr + shape + dtype + device`；
- 同一storage的不同logical shape若确实被TIR需要，必须在prepare manifest显式列出；
- warm evaluator不得调用`from_dlpack`；
- pointer raw value只用于本进程runtime guard，不进入canonical artifact hash；
- artifact只记录buffer identity、shape/dtype、view count和`pointer_exact_count==view_count`；
- pointer drift在launch前拒绝；
- empty β不注册DLPack view。

S4-1A `base_dlpack_view_count`固定为16（7 parameter + 7 gradient + lower + upstream），base pointer exact也必须为16。
S4-1D因实际TIR signature增加的同storage reshape view另记`additional_tir_view_count`和最终total；不得把base 16伪写成
最终TIR总view count，也不得让additional view反向改变S4-1A owner。

## 8. preparation receipt

`S4MutableBufferPreparationReceiptV1`至少记录：

```text
admission_hash
snapshot_hash
device
dtype
stream_identity
alpha_parameter_count=6
physical_beta_parameter_count
empty_beta_token_count
parameter_element_count
gradient_element_count
parameter_logical_bytes
gradient_logical_bytes
pack_count
full_alpha_device_copy_count=0
dense_alpha_materialization_count=0
dense_beta_materialization_count=0
prepare_dlpack_view_count
prepare_dlpack_pointer_exact_count
warm_dlpack_view_count=0
base_dlpack_view_count=16
base_dlpack_pointer_exact_count=16
leaf_parameter_count
nonleaf_parameter_count=0
leased_source_tensor_count=12
leased_source_element_count=8502
leased_source_logical_bytes=34008
lease_incremental_allocated_bytes=0
prepare_outcome=PREPARED
prepare_retry_count=0
prepare_fallback_count=0
empty_cache_call_count=0
timing_recorded=false
performance_claimed=false
receipt_hash
```

所有汇总值从slot/buffer manifest重算。

## 9. fail-closed detail code

S4-1A至少冻结：

1. `ADMISSION_IDENTITY_MISMATCH`；
2. `LIVE_SOURCE_LEASE_MISMATCH`；
3. `SLOT_BUFFER_COUNT_MISMATCH`；
4. `EMPTY_BETA_PHYSICAL_BUFFER_FORBIDDEN`；
5. `PARAMETER_NOT_LEAF`；
6. `PARAMETER_REQUIRES_GRAD_MISMATCH`；
7. `BUFFER_DTYPE_OR_DEVICE_MISMATCH`；
8. `BUFFER_NONCONTIGUOUS`；
9. `BUFFER_INITIAL_CONTENT_MISMATCH`；
10. `PRESERVED_DIRECTION_DEVICE_COPY_OBSERVED`；
11. `PARAMETER_POINTER_DRIFT`；
12. `GRADIENT_POINTER_DRIFT`；
13. `WARM_DLPACK_VIEW_CREATED`；
14. `EVALUATION_ORDINAL_OR_VERSION_MISMATCH`；
15. `RESULT_LEASE_STILL_ACTIVE`；
16. `GRADIENT_GENERATION_MISMATCH`；
17. `DICT_CALLBACK_OR_TENSOR_OVERRIDE_ESCAPE`；
18. `AUTOGRAD_HISTORY_OR_REGISTRY_OBSERVED`；
19. `PROVIDER_CONTAINER_OR_CALLBACK_RETAINED`；
20. `CLAIM_FLAG_TRUE_BEFORE_FORMAL`。

并增加：`LEASED_SOURCE_INVENTORY_MISMATCH`、`SOURCE_TENSOR_OUTSIDE_PRIVATE_LEASE`、
`BUFFER_PREPARE_ALREADY_ATTEMPTED`、`BUFFER_PREPARE_TRANSFER_STATE_MISMATCH`、`BUFFER_PREPARE_CLEANUP_INCOMPLETE`、
`PARAMETER_SOURCE_STORAGE_ALIAS`、`PARAMETER_GRADIENT_STORAGE_ALIAS`、`BASE_DLPACK_VIEW_COUNT_MISMATCH`、
`PREPARE_SOURCE_MUTATION_OBSERVED`、`PREPARE_DEVICE_STREAM_OR_POLICY_DRIFT`、`PREPARE_EMPTY_CACHE_FORBIDDEN`和
`PREPARE_FALLBACK_OR_RETRY_FORBIDDEN`。

每个detail code映射到S4-0已有GC0 reason类别；不扩展Verification IR vocabulary。

## 10. 测试矩阵

建议测试文件：

```text
tests/test_asplos27_s4_ordered_buffer_abi.py
```

### 10.1 positive/structural

1. formal admission创建6 α leaf + 1 active β leaf + 5 empty token；
2. parameter/gradient elements均为4,254，logical bytes均17,016；
3. source active→device buffer数值/hash exact；
4. preserved source hash存在，但candidate device buffer不存在；
5. 两个optimizer group可绑定6 α与1 β，LR分别0.01/0.05；
6. persistent gradient赋给`.grad`后Adam step可执行，全部parameter pointer不变；
7. empty β从不进入optimizer param group；
8. snapshot tensor顺序变化不改变buffer ordinal；
9. parameter/gradient/lower/upstream全部contiguous、finite、CUDA float32；
10. prepare DLPack pointer全部exact，warm view creation=0；
11. 0—9/version 0—9状态机模拟通过；
12. metadata/receipt不含tensor/raw pointer/provider object；
13. no full α device copy/no dense αβ/no autograd registry；
14. two-fresh process descriptor/hash一致；
15. claim/timing/performance flags全false。

### 10.2 minimum 36 negative/tamper

逐项覆盖§9全部20类，并额外覆盖：

- 调换两个slot的physical buffer；
- 把empty β替换成`torch.empty(D,0)`并加入optimizer；
- 使用full α source或nonleaf slice作为parameter；
- evaluation lease未释放就重用gradient buffer；
- ordinal 9后执行mutation；
- 异常后state version或pointer manifest漂移；
- 全重签receipt后修改logical bytes、view count或claim flag。
- S4-0 admission后修改live source content或`_version`，prepare在任何allocation前拒绝；
- S4-0 admission后用same-content clone或same-storage view替换provider path，prepare在任何allocation前拒绝；
- 同一prepared admission/lease第二次prepare拒绝；prepared owner不得序列化lease；
- `.data`/DLPack alias绕过`_version`的content drift拒绝；hash同步成本留到S4-P单列，不在correctness阶段移除；
- 用snapshot CPU clone替换live CUDA mapping，拒绝；
- prepared owner在private lease外再次持有source Tensor，拒绝；private lease恰好12条是positive；
- provider container/callback/closure被保留，拒绝；
- parameter/buffer/view三阶段故障注入后candidate refs、allocated delta、retry/fallback必须为0；
- failure cleanup调用`empty_cache()`或第二次prepare，拒绝；
- adoption字段转移点异常不得产生double-owner/no-owner。

## 11. 当前原型验证

在冻结production snapshot上做了一次不入库的GPU owner原型：

```text
formal owner: 6 alpha + 1 active beta + 5 empty token
parameter/gradient = 4,254 elements / 17,016 B each
base DLPack = 16/16 pointer exact
all candidate storage independent from 12 source storage
one-step Adam parameter/gradient pointer stable
source hash/version unchanged

leased source = 12 tensors / 8,502 elements / 34,008 logical B
lease incremental allocated bytes = 0

failure injection = parameters/buffers/views 3/3 clean
candidate refs alive after cleanup = 0/3
allocated delta after cleanup = 0/3
retry/fallback/empty_cache = 0/0/0
```

该原型只验证PyTorch owner与内存算术可行，不是S4实现、correctness closure或性能证据。

## 12. 提交与门禁

S4-1A只能在“S3 external audit approved+closed”且“S4-0 validated”后开放。建议提交：

1. `feat(runtime): add S4 ordered mutable buffer ABI`；
2. `test(runtime): close S4-1A buffer ownership gates`；
3. `docs: close S4-1A and open S4-1B effective values`。

S4-1A通过只证明buffer/lease/version owner，不证明CROWN lower/VJP数值，不开放S4-2 optimizer trajectory或timing。

当前状态保持：

```text
S3 exchange = ready_for_audit
S4-0 implementation = closed
S4-1A implementation = closed
S4 timing/performance = closed
```
