---
status: diagnostic-complete-corrected-v3-code-closed
date: 2026-08-28
type: implementation-preflight-correction
topic: boundflow
slug: asplos27-s4-0-admission-preflight-correction
stage: s04
execution-authority: false
code-change-open: false
gpu-execution-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-0 admission开工前源码审计与V2修正

## 0. 直接结论

原S4-0方向——“只新增runtime admission receipt，不新增solver/graph/execution IR”——仍然正确，但原函数签名

```text
snapshot + topology + R31 plan → tensor-free admission
```

不能证明它声称的全部live ownership条件。原因是`ProductionStateSnapshotV4`保存的是CPU contiguous clone和来源描述，
不是当前solver内将被commit的live tensor：

- snapshot无法证明两个不同Tensor view是否共享同一CUDA storage；
- snapshot不保存live tensor `_version`；
- snapshot的`alias_group`按Python object `id`分组，不是storage alias分组；
- snapshot只保存`source_device`字符串，无法证明当前live source仍在该device；
- `R31FullRegionPlanV1.source_state_hash`绑定的是dense native mapping hash，不是snapshot hash；若S4-0禁止调用dense
  initializer，就不能把该字段当作可独立重算的snapshot binding。

因此S4-0必须同时形成两个不同生命周期的结果：一个tensor-free canonical receipt，以及一个持有原始Tensor强引用和
raw object/storage/version token的ephemeral live lease。mapping是runtime函数参数，不是新IR；lease也不是IR或artifact，
但必须一直活到S4-1A pack并随prepared runtime转移到S4-3 precommit。只返回receipt会允许same-content clone用同样的稳定
group编号冒充原object。完整反例与V3冻结接口见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_LIVE_LEASE_IMPLEMENTATION_READINESS_2026_08_28.md`。

## 1. 独立源码证据

### 1.1 snapshot是语义证据副本，不是live binding

`boundflow/runtime/rvir_v4_production_state.py`的事实：

1. `OwnedProductionTensorV4.own()`执行`detach().cpu().contiguous().clone()`；
2. `source_device`只保存构造时的字符串；
3. `ProductionStateBuilderV4`用`id(value)`分配`alias_group`；
4. metadata没有source object identity、storage identity、storage offset、stride或`_version`。

只读反例探针构造同一base tensor的两个不同view：

```text
live_objects_distinct=true
live_storage_shared=true
snapshot_alias_groups=[alias:000000, alias:000001]
snapshot_storage_shared=false
has_source_version=false
has_source_storage_identity=false
```

所以snapshot的alias字段可保留作capture object provenance，但不得用于S4 live mutation安全证明。直接修改旧snapshot schema
还会使既有RVIR artifact/hash大面积失效，S4-0不应顺手升级它。

### 1.2 current formal snapshot事实

从`artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt`独立恢复：

```text
snapshot_hash = 2a775b66559c20ddfc0bec97ec026898ba5eccfc984e02b217fcb7472d03a256
mutable paths = 12
roles = 6 alpha + 6 beta_value
source_device metadata = cuda:0
snapshot alias groups = 12 unique
alpha leading axes = (alpha_polarity, start_spec, domain)
```

这能证明语义path、content、shape、dtype和capture时device，但“12 unique snapshot alias groups”不能升级为“12个live
CUDA storage互不alias”。后者必须在真实live boundary重新观察。

### 1.3 R31 plan的identity边界

同一fixture现场重建：

```text
snapshot hash = 2a775b...a256
dense mapping hash = cfcebf...f8df
R31 plan hash = 39d617...910f
plan.source_state_hash = cfcebf...f8df
```

`plan.source_state_hash == mapping.stable_hash()`，但不等于`snapshot.stable_hash()`。同时R31 plan validator当前明确
写死六layout、domain=6、spec=1、start node和P-anchor shape。结论：

- R31 plan是当前ResNet2B formal specialization，可作为S4-0 v1 adapter输入；
- S4 receipt schema仍必须model/shape/site中立；
- `source_state_hash`只作为历史oracle provenance披露，不作为S4 snapshot binding的唯一证据；
- S4必须从plan layout/tensor spec对snapshot逐path重建`plan_binding_projection_hash`；
- 不能为了“通用”去修改已验证的R31 plan，也不能宣称R31 validator本身是generic production plan。

### 1.4 β/history验证仍需收紧

现有`validate_beta_history_consistency()`只比较每个history entry对应的location/sign前缀；条件允许
`len(history) < beta_width`。S4 mutable admission必须要求每domain的history长度与对应β width **exact**，并对empty β明确
要求width=0。否则攻击者可在已验证前缀后追加未拥有的β slot。

## 2. 修正后的唯一owner边界

S4-0仍不新增IR层。对象职责如下：

```text
ProductionStateSnapshotV4       semantic content/history/policy truth
ProductionReluTopologyV4        provider/native topology links
R31FullRegionPlanV1             current formal static compiled-plan specialization
Mapping[path, live Tensor]      current-provider source view
              │
              ▼
PreparedS4MutableStateAdmissionV1
  ├─ S4MutableStateAdmissionV1  tensor-free canonical runtime receipt
  └─ S4LiveMutableLeaseV1       ephemeral strong-ref runtime owner
```

mapping不能成为provider回调或延迟读取owner。receipt不保存Tensor/raw identity；lease则只在本进程当前exact-call中保存
强引用和raw token，禁止序列化、跨query缓存或重复transfer。S4-1A必须消费prepared admission并转移lease；S4-3 commit前
必须重新枚举current provider targets并与lease逐对象复核。S4-0不提前拥有GPU prepared buffer。

## 3. 修正函数签名

```text
prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: dict[str, torch.Tensor],
) -> PreparedS4MutableStateAdmissionV1
```

函数必须拒绝lazy mapping、callable、provider object和可在迭代中改变的自定义mapping；入口先复制为普通`dict`，验证
key/type后只读使用。

### 3.1 provider adapter如何构造mapping

mapping必须在RVIR exact-call进入core且任何candidate mutation前，从当前provider node直接取引用：

```text
alpha/{activation}/{start}      → node.alpha[start]
beta/{preactivation}/0/value    → node.sparse_betas[0].val
```

路径编码复用`ProductionStateSnapshotV4`现有规则。adapter只枚举topology/plan要求的12条mutable path，并在返回前验证
没有missing/extra；不得用snapshot `.value`、`.clone()`、`.to()`或dense initializer结果代替live source。location/sign/
history仍从snapshot作为read-only semantic truth校验，不加入mutable mapping。

建议helper保持在同一S4-0 runtime模块内或existing RVIR adapter边界，返回立即消费的普通`dict`；不新增
`LiveMutableStateIR`、全局registry或延迟provider callback。

## 4. live metadata投影

每个slot新增tensor-free字段：

```text
alpha_live_object_group
alpha_live_storage_group
alpha_live_version
alpha_live_shape / dtype / device
alpha_live_stride / storage_offset / contiguous
alpha_live_requires_grad / is_leaf
alpha_live_content_hash

beta_live_object_group
beta_live_storage_group
beta_live_version
beta_live_shape / dtype / device
beta_live_stride / storage_offset / contiguous
beta_live_requires_grad / is_leaf
beta_live_content_hash
```

### 4.1 稳定alias编号

- raw `id(tensor)`、`data_ptr()`、storage `_cdata`只用于本进程比较，不进入canonical JSON；
- 先按plan slot/path顺序观察，再把首次出现的object/storage identity编号为`object:000000`、`storage:000000`；
- nonempty tensor storage identity至少绑定device + storage object identity；data pointer只作双重检查；
- empty tensor的`data_ptr=0`不得形成跨path alias。除非是同一Tensor object，否则每个empty source独立编号；
- receipt保存group、shape、stride、offset等可重放投影，不保存原pointer值。

当前formal门禁要求12个live mutable path object group唯一；nonempty storage group也唯一。五个empty β不因零pointer
互相alias。后续若真实provider出现合法view alias，必须另预注册commit owner，不在S4-0自动接受。

### 4.2 ephemeral version/object lease

S4-0记录入口`_version`并由ephemeral lease强引用原Tensor。S4-1A bind和S4-3 commit前都必须从current provider
mapping验证：

```text
same semantic path
same Python object and raw storage identity
same shape/dtype/device/stride/offset
same _version
same content hash
```

任一变化在launch/mutation前拒绝。receipt中的稳定group只是可重放投影；真正的跨阶段identity由不可序列化lease保证，
不是可跨query复用的全局state version。

## 5. plan/snapshot绑定算法

不得用`plan.source_state_hash == snapshot_hash`，因为两者语义不同。正确算法：

1. `snapshot.validate()`和`plan.validate()`；
2. 以`plan.relu_layouts`顺序重排topology；topology输入tuple顺序不进入hash；
3. 从每个layout解析唯一α path和β value/location/sign path；
4. 在`plan.tensor_specs`中按`relu/{native}/alpha|beta`解析对应spec；
5. 对snapshot item逐项比较path、role、shape、dtype、content hash；
6. 比较layout的feature indices/location/split与snapshot layout/history；
7. 由这些可独立重算字段构造`plan_binding_projection_hash`；
8. `plan.source_state_hash`另存为`oracle_mapping_provenance_hash`，只校验SHA256格式，不冒充snapshot equivalence；
9. topology hash使用plan order的canonical link列表；输入tuple任意置换必须得到同一hash；
10. receipt同时绑定`snapshot_hash + plan_hash + plan_binding_projection_hash + topology_hash`。

这保留existing dense mapping作为oracle证据，却不让candidate admission调用dense initializer。

## 6. mutable coverage与β exact门禁

snapshot中`ownership=MUTABLE_COPY_OUT`的集合必须恰等于：

```text
{layout.alpha_path for layout in plan}
∪ {layout.beta_path for layout in plan}
```

每个β slot还必须：

- value/location/sign shape exact相等；
- domain count与plan exact；
- 每domain `history.locations/coefficient`长度恰等于β width；
- location/sign逐元素exact，不只比较前缀；
- width=0时value/location/sign都为空，且live tensor仍需path/object/version身份；
- active β width>0时value finite/nonnegative、location unique/in-range、sign属于冻结集合；
- plan `beta_locations`、snapshot location和history三方exact。

## 7. fail-closed顺序

为保证stable reason，验证顺序冻结为：

1. input type/schema/claim flags；
2. snapshot与plan自验证；
3. optimizer polarity；
4. canonical topology coverage/order；
5. plan/snapshot projection identity；
6. mutable path exact coverage；
7. α axes/slices/layout；
8. β exact width/location/sign/history；
9. live mapping key/type coverage；
10. live shape/dtype/device/content；
11. live object/storage alias、stride/offset；
12. live version；
13. aggregate/hash重算；
14. receipt self-validation；
15. 构造strong-ref lease并验证receipt/lease shared admission hash；
16. 返回不可序列化prepared wrapper。

任何失败发生在GPU allocation、dense materialization、TVM/provider调用和live mutation之前。

## 8. 新增/修正stable detail code

| detail code | verification reason |
|---|---|
| `SNAPSHOT_SCHEMA_VERSION_MISMATCH` | `STATE_VERSION_MISMATCH` |
| `LIVE_TENSOR_VERSION_MISMATCH` | `STATE_VERSION_MISMATCH` |
| `LIVE_SOURCE_COVERAGE_MISMATCH` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `LIVE_SOURCE_CONTENT_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_SOURCE_STORAGE_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_OBJECT_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_STRIDE_OFFSET_MISMATCH` | `LAYOUT_NOT_NORMALIZABLE` |
| `PLAN_SNAPSHOT_PROJECTION_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `PLAN_ORACLE_PROVENANCE_UNVERIFIABLE` | `RECEIPT_IDENTITY_MISMATCH` |
| `BETA_HISTORY_WIDTH_MISMATCH` | `BETA_LOCATION_SIGN_HISTORY_MISMATCH` |
| `EMPTY_TENSOR_FALSE_ALIAS` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_SOURCE_OBJECT_REPLACED` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_SOURCE_STORAGE_REPLACED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_LEASE_ADMISSION_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `LIVE_LEASE_ALREADY_TRANSFERRED` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `LIVE_LEASE_SERIALIZATION_FORBIDDEN` | `RECEIPT_IDENTITY_MISMATCH` |

旧蓝图的`STATE_VERSION_MISMATCH`必须拆开snapshot schema version与live Tensor `_version`，避免同一reason掩盖两个
不同owner边界。

## 9. 修正后的测试最低集

原20类negative保留并扩为至少44类；除原V2用例外新增关键用例：

1. 两个distinct view共享nonempty storage，拒绝；
2. 同一Tensor object绑定两个mutable path，拒绝；
3. 五个empty β均`data_ptr=0`但不误判alias；
4. live source `_version`在admission前变化，拒绝；
5. receipt完成后、S4-1A bind前version变化，后级必须拒绝；
6. live source device与snapshot `source_device`不一致；
7. shape/dtype相同但stride/storage offset不同；
8. content变更并同步伪造外层receipt，仍由snapshot/live重算拒绝；
9. topology tuple置换，canonical receipt/hash不变；
10. plan layout置换并全重签，按plan validator/projection拒绝；
11. `plan.source_state_hash`替换为snapshot hash，拒绝oracle provenance漂移；
12. β width大于history长度但前缀相同，拒绝；
13. empty β附加一个未进history的slot，拒绝；
14. live mapping使用callable/lazy mapping，拒绝；
15. object/pointer/storage raw identity没有进入canonical JSON或artifact；
16. 全量same-content clone替换时canonical receipt可相同，但lease以object replaced拒绝；
17. same-storage view替换、empty clone替换分别拒绝；
18. lease不能pickle/deepcopy/artifact遍历；
19. lease只能transfer一次，close后不可再用；
20. S4-1A pack后provider rebind，S4-3 precommit拒绝；
21. 外部mapping引用删除并GC后，lease强引用保持原Tensor直到commit/close。
22. `.data`和DLPack alias写入在`_version`不变时仍由content hash拒绝；
23. same-object `set_`按storage replaced拒绝；
24. lease/wrapper均为非dataclass `__slots__` class，copy/deepcopy/pickle拒绝；
25. input只接受exact built-in dict，dict subclass/custom Mapping拒绝。

positive还必须证明：

- canonical receipt递归对象图无Tensor/module/callback/provider object/raw pointer；prepared wrapper的Tensor只能存在于
  私有ephemeral lease，且serialization guard必须拒绝；
- monkeypatch dense initializer、CUDA allocation、TVM和provider entry为必抛，positive仍通过；
- 两个fresh process的canonical receipt/hash exact；
- formal计数仍为六slot、12 mutable path、`8496/4248/4248` α和1 active β/6元素；
- `snapshot/plan/mapping`三个hash按各自语义保留，不互相冒充。

## 10. 开工门禁与下一步

该修正不开放代码。当前仍为：

```text
S3 exchange = ready_for_audit
S4-0 code = closed
S4 GPU/timing/performance = closed
```

S3批准后第一笔S4-0代码必须以本修正后的四输入签名为准。若实现仍只有`snapshot + topology + plan`，则只能声称
“offline semantic admission”，不能关闭live mutable-state admission，也不能进入S4-1A。
