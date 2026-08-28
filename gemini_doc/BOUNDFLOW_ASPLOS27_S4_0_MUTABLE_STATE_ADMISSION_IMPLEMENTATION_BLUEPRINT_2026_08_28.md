---
status: corrected-v3-implementation-blueprint
date: 2026-08-28
type: implementation-plan
topic: boundflow
slug: asplos27-s4-0-mutable-state-admission
stage: s04
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-execution-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-0：production mutable-state admission实施蓝图

## 0. 直接结论

S4-0不需要新增solver IR、execution IR或另一套verification graph。它需要一个**tensor-free typed runtime
binding receipt**和一个**不可序列化的ephemeral live lease**，分别关闭可审计语义与跨阶段对象身份：

```text
ProductionStateSnapshotV4
  + ProductionReluTopologyV4
  + R31FullRegionPlanV1
  + transient Mapping[path, live Tensor]
  + VerificationRejectionReason
  → PreparedS4MutableStateAdmissionV1
       ├─ S4MutableStateAdmissionV1
       └─ S4LiveMutableLeaseV1
```

receipt证明六个compressed lower-α slot、六个sparse β slot及layout/history的稳定投影；lease用强引用和raw token保证
S4-1A/S4-3操作的仍是同一批live object/storage/version。mapping和lease不进入receipt或artifact，lease不可跨query复用。
S4-0不创建dense α/β、不分配GPU buffer、不执行TIR、不计时，也不改变live solver state。

开工前源码审计已证明原三输入签名无法验证live storage alias和`_version`；详细反例与修正依据见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_ADMISSION_PREFLIGHT_CORRECTION_2026_08_28.md`。
tensor-free receipt无法独自排除same-content clone替换的反例及最终V3接口见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_LIVE_LEASE_IMPLEMENTATION_READINESS_2026_08_28.md`。

当前S3 external audit仍未返回，因此本文只有implementation blueprint地位；不能据此直接开始S4代码。

## 1. 已有事实与可复用owner

### 1.1 production snapshot

`boundflow/runtime/rvir_v4_production_state.py`已经提供：

- `ProductionStateSnapshotV4`：snapshot、history、optimizer policy及canonical stable hash；
- `OwnedProductionTensorV4`：semantic path、role、axes、dtype/shape、ownership、alias group、content hash；
- `ProductionTensorRole`与`ProductionTensorOwnership`；
- α layout、β location/sign/history一致性校验；
- finite、重复path、重复history key与content hash fail-closed。

因此S4-0不得重新定义tensor semantic metadata schema，也不得复制snapshot validator。但snapshot内部value是CPU
contiguous clone，`alias_group`按source Tensor object `id`分组；它不是live storage/version证据。S4-0必须在真实
boundary瞬时观察live source，而不是从snapshot推断CUDA alias或`_version`。

### 1.2 topology与native oracle

`boundflow/runtime/rvir_v4_pre_state_initializer.py`已经提供：

- `ProductionReluTopologyV4`：provider activation/preactivation、native preactivation、start node；
- `ProductionPreStateIdentityV4`：snapshot/topology/history/intermediate identities；
- compressed→dense→compressed round-trip oracle；
- 六α、六β、split/history、intermediate bound的owner一致性检查。

S4-0复用其topology语义，但**不能调用native initializer作为candidate admission实现**：该函数会创建dense
α/β/split tensor，正是S4 candidate热路径要避免的representation。它只保留为独立oracle。

### 1.3 compiled-region plan

`boundflow/runtime/r3_structured_owner_custom_backward.py`中的`R31FullRegionPlanV1`与`R31ReluLayoutV1`已经
包含：

- 六个ReLU的确定性顺序；
- provider/native/path映射；
- feature shape与compressed α flat indices；
- sparse β locations与split values；
- exact tensor specs、graph/state hash、domain/spec count。

S4-0只增加“production mutable slice如何绑定到这些layout”的runtime contract，不创建第二份Plan IR。当前
`R31FullRegionPlanV1.validate()`明确冻结六layout、domain=6、spec=1、start node和P-anchor，它是formal specialization，
不是通用Plan IR；S4 receipt schema保持通用，当前compiler通过adapter消费R31 plan。

### 1.4 legality vocabulary

`boundflow/ir/verification_graph.py`已经冻结`VerificationRejectionReason`。S4 runtime可以有更精确的
`detail_code`，但必须映射到已有verification legality类别，禁止让两个reason体系互不相认。

## 2. 当前formal inventory

冻结capture中六个α source为：

| native | source shape | stored | lower-active | preserved | compressed width |
|---|---:|---:|---:|---:|---:|
| 17 | `[2,1,6,164]` | 1,968 | 984 | 984 | 164 |
| 19 | `[2,1,6,132]` | 1,584 | 792 | 792 | 132 |
| 23 | `[2,1,6,121]` | 1,452 | 726 | 726 | 121 |
| 25 | `[2,1,6,86]` | 1,032 | 516 | 516 | 86 |
| 28 | `[2,1,6,178]` | 2,136 | 1,068 | 1,068 | 178 |
| 31 | `[2,1,6,27]` | 324 | 162 | 162 | 27 |
| 合计 | — | **8,496** | **4,248** | **4,248** | **708** |

这里的active slice固定由production lower-only policy导出为`source[0,0]`，不是根据tensor数值猜测。
`source[1,0]`是copy-through preserved direction。

六个β value均为`[domain, history_slot]`；当前formal只有native 31的`[6,1]`非空，其余五个为`[6,0]`。
通用schema不写死“31”“6×1”或“恰好一个active site”；这些只属于formal fixture acceptance。

## 3. 唯一新增模块与对象

建议新增单个模块：

```text
boundflow/runtime/asplos27_s4_mutable_state_admission.py
```

### 3.1 `S4MutableStateAdmissionError`

异常至少携带：

```text
detail_code                    # S4精确稳定码
verification_reason            # VerificationRejectionReason
slot_ordinal_or_none
semantic_path_or_none
```

异常必须在任何GPU allocation/launch、dense materialization或live mutation之前抛出。

### 3.2 `S4MutableSlotV1`

每个slot是frozen dataclass，只含可canonical JSON化的metadata：

```text
slot_ordinal
native_preactivation
provider_activation
provider_preactivation
provider_start_node

alpha_semantic_path
alpha_source_axes
alpha_source_shape
alpha_source_dtype
alpha_source_alias_group
alpha_source_hash
alpha_live_object_group
alpha_live_storage_group
alpha_live_version
alpha_live_shape / dtype / device
alpha_live_stride / storage_offset / contiguous
alpha_live_requires_grad / is_leaf
alpha_live_content_hash
alpha_mutable_slice              # exact [0,0]
alpha_active_shape               # [domain,width]
alpha_active_hash
alpha_preserved_slice            # exact [1,0]
alpha_preserved_shape
alpha_preserved_hash

feature_shape
alpha_flat_indices
alpha_layout_hash

beta_semantic_path
beta_source_axes
beta_source_shape
beta_source_dtype
beta_source_alias_group
beta_source_hash
beta_live_object_group
beta_live_storage_group
beta_live_version
beta_live_shape / dtype / device
beta_live_stride / storage_offset / contiguous
beta_live_requires_grad / is_leaf
beta_live_content_hash
beta_location_path/hash
beta_sign_path/hash
beta_active
beta_history_hash
```

对象不得持有`torch.Tensor`、DLPack capsule、device pointer、module handle、callback或live provider object。

### 3.3 `S4MutableStateAdmissionV1`

顶层receipt字段固定为：

```text
snapshot_hash
production_plan_hash
plan_binding_projection_hash
oracle_mapping_provenance_hash
topology_hash
optimizer_policy_hash
slots                         # canonical plan order
mutable_path_set_hash

alpha_source_count
alpha_stored_element_count
alpha_active_element_count
alpha_preserved_element_count
beta_slot_count
active_beta_slot_count
active_beta_element_count

gpu_execution_observed=false
dense_materialization_observed=false
timing_recorded=false
performance_claimed=false
schema_version=boundflow.asplos27-s4-mutable-state-admission/v1
admission_hash
```

`validate()`必须重算全部计数/hash，禁止信任构造方提供的汇总数字。

raw Python object id、data pointer和storage handle只允许在compile函数内用于分组，禁止进入canonical JSON。object/storage
group按plan/path首次出现顺序稳定编号。empty tensor的`data_ptr=0`不构成alias；除非是同一Tensor object，否则每个empty
path独立group。

### 3.4 `S4LiveMutableLeaseV1`与prepared wrapper

同一runtime模块内再定义非IR、非artifact对象：

```text
S4LiveMutableLeaseV1
  private strong refs + raw object/storage/version/layout/content tokens
  single-transfer / current-provider-revalidate / close

PreparedS4MutableStateAdmissionV1
  receipt: S4MutableStateAdmissionV1
  private lease: S4LiveMutableLeaseV1
```

lease/wrapper不得提供canonical serialization；pickle/deepcopy/artifact walker必须fail closed。canonical receipt仍保持
完全tensor-free。lease从S4-0活到S4-3 commit/abort，S4-1A不得pack后丢弃它。

## 4. 编译入口

冻结函数职责：

```text
prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: Mapping[str, torch.Tensor],
) -> PreparedS4MutableStateAdmissionV1
```

当前函数可接受R31 plan作为formal输入，但新增dataclass本身不得包含ResNet2B、native id、固定shape或P-anchor
常数。未来general plan只需满足同一metadata contract，不修改receipt schema。`live_mutable_sources`是瞬时普通mapping，
不是IR；函数返回前不得把Tensor/provider object保存到receipt、registry或closure，但必须把原Tensor强引用保存在私有
ephemeral lease，直到S4-3事务结束。lease不得进入artifact或跨query cache。

RVIR adapter在core入口按topology/plan从live `node.alpha[start]`和`node.sparse_betas[0].val`构造12-path普通`dict`；
必须传原Tensor引用，禁止用snapshot CPU clone、`.to()`副本或dense initializer结果冒充live source。location/sign/history
仍由snapshot read-only owner提供，不进入mutable mapping。helper不得引入新IR、global registry或延迟provider callback。

## 5. 确定性编译算法

严格顺序如下：

1. 冻结claim flags为false，验证input type；把live mapping复制为普通`dict`并拒绝callable/lazy provider view；
2. 调用`snapshot.validate()`与`production_plan.validate()`；
3. 校验lower-only optimizer policy：`bound_lower=true/bound_upper=false/fix_intermediate_bounds=true`；
4. 验证topology key唯一，以`production_plan.relu_layouts`顺序canonicalize并计算topology hash；输入tuple置换不得改hash；
5. 对每个layout唯一解析topology link、α path和β value/location/sign path；
6. 从plan layout/tensor spec与snapshot逐path重建`plan_binding_projection_hash`；不得把
   `production_plan.source_state_hash`误当`snapshot_hash`，前者只保存为dense oracle provenance；
7. 验证snapshot中mutable-copy-out path集合恰等于全部layout的α+β value path；
8. 验证α role/ownership/axes/dtype/finite/source hash与plan tensor spec一致；
9. 验证α leading axes为`[alpha_polarity,start_spec,domain,...]`，lower-only active slice为`[0,0]`；
10. 从CPU snapshot source分别计算active/preserved slice hash；不得先dense scatter再project；
11. 验证feature shape、flat indices、spec lookup与plan layout逐项一致；
12. 验证β value/location/sign shape、dtype、role与plan layout一致，并要求每domain β width与history长度exact；
13. 验证live mapping path集合与mutable path集合exact；逐path核对live shape/dtype/device/content与snapshot；
14. 观察live object/storage identity、stride、offset、contiguity、requires-grad/leaf和`_version`；按plan顺序生成稳定group；
15. 拒绝重复object、nonempty shared storage、非法view/offset；empty zero-pointer不得互相归为alias；
16. 汇总stored/active/preserved与β计数，从slot重算mutable path set hash；
17. 构造receipt，四个claim/execution flag全部false，且递归对象图tensor/provider/pointer-free；
18. 调用receipt `validate()`重算全部projection/hash；
19. 以同一次admission hash构造strong-ref lease和prepared wrapper；
20. 对receipt做tensor-free检查，对lease做serialization-forbidden检查后返回wrapper。

S4-0不得调用：

- `initialize_rvir_v4_native_pre_state`的dense构造路径；
- `bind_r31_runtime_inputs_v1`的GPU binding；
- TVM compile/launch；
- provider `compute_bounds/update_bounds`；
- Adam optimizer；
- wall-clock或CUDA event timing。

S4-0记录的live `_version`只是本query lease baseline。S4-1A buffer bind和S4-3 commit前必须从current provider mapping
再次验证同一path、同一Python object、同一raw storage/version/content；不能只重算稳定group。S4-0不能把一次admission
升级成可跨mutation或跨query永久复用的许可。

## 6. reason映射

S4-0至少冻结以下detail code，并映射到已有GC0 reason：

| S4 detail code | GC0/verification reason |
|---|---|
| `SNAPSHOT_SCHEMA_VERSION_MISMATCH` | `STATE_VERSION_MISMATCH` |
| `LIVE_TENSOR_VERSION_MISMATCH` | `STATE_VERSION_MISMATCH` |
| `BOUND_POLARITY_MISMATCH` | `BOUND_POLARITY_MISMATCH` |
| `TOPOLOGY_IDENTITY_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `MUTABLE_STATE_COVERAGE_INCOMPLETE` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `ACTIVE_BETA_COVERAGE_INCOMPLETE` | `BETA_ACTIVE_EMPTY_MISMATCH` |
| `ALPHA_MUTABLE_DIRECTION_MISMATCH` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `ALPHA_PRESERVED_DIRECTION_DRIFT` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `ALPHA_LAYOUT_IDENTITY_MISMATCH` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `BETA_LOCATION_SIGN_HISTORY_MISMATCH` | `BETA_LOCATION_SIGN_HISTORY_MISMATCH` |
| `MUTABLE_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_COVERAGE_MISMATCH` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `LIVE_SOURCE_CONTENT_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_SOURCE_OBJECT_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_STORAGE_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_STRIDE_OFFSET_MISMATCH` | `LAYOUT_NOT_NORMALIZABLE` |
| `EMPTY_TENSOR_FALSE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `PLAN_SNAPSHOT_PROJECTION_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `PLAN_ORACLE_PROVENANCE_UNVERIFIABLE` | `RECEIPT_IDENTITY_MISMATCH` |
| `BETA_HISTORY_WIDTH_MISMATCH` | `BETA_LOCATION_SIGN_HISTORY_MISMATCH` |
| `MUTABLE_SLOT_ORDER_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `DTYPE_OR_DEVICE_MISMATCH` | `DTYPE_OR_DEVICE_MISMATCH` |
| `NONFINITE_MUTABLE_STATE` | `DTYPE_OR_DEVICE_MISMATCH` |
| `CLAIM_FLAG_TRUE_BEFORE_FORMAL` | `RECEIPT_IDENTITY_MISMATCH` |
| `S4_0_EXECUTION_FORBIDDEN` | `RUNTIME_FALLBACK_REQUIRED` |
| `LIVE_SOURCE_OBJECT_REPLACED` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_SOURCE_STORAGE_REPLACED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_LEASE_ADMISSION_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_LEASE_ALREADY_TRANSFERRED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_LEASE_ALREADY_CLOSED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_LEASE_SERIALIZATION_FORBIDDEN` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_LEASE_PROVIDER_REBIND` | `RECEIPT_IDENTITY_MISMATCH` |

映射属于runtime binding的精化，不升级GC0 legality claim，也不要求修改`VerificationRejectionReason`。

## 7. 测试矩阵

建议测试文件：

```text
tests/test_asplos27_s4_mutable_state_admission.py
```

### 7.1 positive/structural

1. formal capture得到六slot、12 mutable path；
2. 重算`stored/active/preserved=8496/4248/4248`；
3. P为`1032/516/516`，但schema metadata不含`P`或模型名；
4. active β为1 slot/6元素，五empty β保留为empty slot；
5. slot顺序等于plan order，snapshot tensor顺序打乱后hash不变；
6. canonical JSON与stable hash跨两个fresh process一致；topology tuple置换不改变hash；
7. dataclass/object graph中不存在tensor、raw pointer、module、provider object或callback；
8. monkeypatch dense initializer、CUDA allocation、TVM compile和provider entry为必抛，positive admission仍通过；
9. receipt validate独立重算计数与hash；
10. formal binding与native pre-state oracle的12个semantic path及mapped slice hash一致。

### 7.2 minimum 38 negative/tamper

1. snapshot schema/version变更；
2. `bound_upper=true`或`bound_lower=false`；
3. topology少一个slot；
4. topology重复native/provider key；
5. 删除一个α mutable path；
6. 删除active β value；
7. 把active β伪装成empty；
8. α direction axes调换；
9. α active slice改为upper direction；
10. preserved slice content/hash漂移并全重签外层receipt；
11. feature index重复、越界或乱序；
12. β location越界/重复；
13. β sign与history coefficient不一致；
14. 两个mutable source共享alias group；
15. slot重排并全重签外层receipt；
16. dtype/device metadata漂移；
17. NaN/Inf mutable state；
18. `gpu_execution_observed/timing_recorded/performance_claimed`任一翻true；
19. admission过程试图调用dense initializer/TVM/provider；
20. admission hash、plan hash、snapshot hash任一篡改。
21. 两个distinct nonempty view共享storage；
22. 同一live Tensor object绑定两个mutable path；
23. 五个empty β都返回zero pointer但被错误归为alias；
24. live source `_version`漂移；
25. live source path缺失/多余；
26. live shape/dtype/device/content与snapshot漂移；
27. shape/dtype相同但stride/storage offset改变；
28. β width大于history长度但已验证前缀一致；
29. `plan.source_state_hash`被替换成snapshot hash；
30. live mapping是callable/lazy provider view或receipt泄漏raw pointer。
31. 全量same-content clone替换，stable projection/hash相同但object lease拒绝；
32. same-storage view替换；
33. empty β同shape clone替换；
34. receipt/lease来自不同admission；
35. lease重复transfer；
36. lease close后revalidate；
37. lease被pickle/deepcopy/artifact walker序列化；
38. S4-1A pack后current provider mapping rebind，S4-3 precommit拒绝。

每个负向测试必须断言exact `detail_code`和对应`VerificationRejectionReason`，不能只断言“抛异常”。

## 8. 文件和提交边界

S3外审批准后，S4-0只允许以下第一批代码：

```text
boundflow/runtime/asplos27_s4_mutable_state_admission.py
tests/test_asplos27_s4_mutable_state_admission.py
gemini_doc/BOUNDFLOW_ASPLOS27_S4_CHANGE_LOG_2026_08_28.md
必要的README/claims/status状态同步
```

禁止在同一批加入：

- evaluator/TIR实现；
- persistent GPU parameter buffer；
- dense terminal bridge；
- sealed optimizer policy driver；
- S4-1/S4-2 correctness；
- timing或性能headline。

建议提交切片：

1. `feat(runtime): add S4 mutable-state admission receipt and live lease`；
2. `test(runtime): close S4-0 admission and tamper gates`；
3. `docs: close S4-0 and preregister S4-1A`。

如果执行纪律要求一个logical commit，可把1—2合并，但文件/语义范围不扩大。

## 9. S4-0关闭门槛

全部成立才允许状态变为`VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`：

- formal inventory与本稿数字逐项一致；
- 12 mutable path、六slot、active/preserved方向完全闭合；
- snapshot/plan/topology/live source四方binding闭合，live object/storage/version guard成立；
- plan binding projection可从snapshot/layout/spec独立重算，不调用dense initializer；
- β width与history exact，不接受只匹配前缀；
- receipt tensor-free/pointer-free且canonical hash跨fresh process一致；lease不可序列化且强引用原始live targets；
- S4-1A与S4-3从current provider mapping复核同一object/storage/version，clone/rebind不能冒充；
- minimum 38类negative全部exact fail-closed；
- dense materialization/GPU execution/provider fallback/timing/performance flag全为0/false；
- targeted/full/static/DocOps通过；
- S3 external audit已经approved并正式close。

通过S4-0只开放S4-1A all-state buffer/ordered evaluator ABI；不开放single-evaluation TIR、10/9 trajectory、
same-solver timing或complete-query claim。

## 10. 当前状态

本文完成了S4-0的精确实现路线，但没有改变当前门禁：

```text
S3 exchange = ready_for_audit
S4-0 implementation = closed
S4 GPU execution/timing = closed
performance_claimed = false
```

下一外部状态变化仍应是S3 audit结果；在此之前只允许审阅/修正文档，不允许把本蓝图变成代码。
