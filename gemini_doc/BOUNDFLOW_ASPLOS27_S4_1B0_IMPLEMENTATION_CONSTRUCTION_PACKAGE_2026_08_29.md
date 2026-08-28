---
status: diagnostic-complete-construction-ready-code-closed
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-1b0-implementation-construction-package
stage: s04
depends-on: validated-s4-1a-ordered-buffer-prepare
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4-1B0：三元input endpoint TIR逐文件施工包

## 0. 直接结论

S4-1B0的数学方向和位级CUDA机制成立，但原readiness文档仍把**隔离backend module**、**S4-1A base buffer**和
**S4-1B production arena**混成了一个物理口径。本轮源码审计与独立GPU诊断冻结以下施工边界：

1. S4-1B0只实现独立ternary pack/select backend、compile cache、immutable module receipt和隔离correctness
   micro-owner；不接S4 evaluator，不修改S4-1A owner；
2. 不新增`boundflow/ir`对象。这里是固定语义的backend lowering，用backend-local frozen dataclass足够；
3. pack按float32原始bits的exponent mask识别NaN/Inf，不能用浮点`x==x`、`isfinite`猜测或epsilon；
4. select显式接受`+1/-1/0`，非法值输出canonical qNaN bits=`0x7fc00000`；
5. midpoint固定`(lower+upper)*float32(0.5)`，不能重结合；GPU现场有两组逐位反例；
6. 隔离module需要5个DLPack view，但它们不是S4-1A的16个base view；
7. 隔离module真实输出storage为selector `18,432 B`加selected output `73,728 B`，合计`92,160 B`；
8. 最终production可以把selected output复用一块coefficient arena，但这个alias只在S4-1B phase/lifetime证明后成立；
9. cache lookup identity、compiled module identity、mutable hit/miss observation和formal tensor observation必须拆开；
10. warm production路径不得为了class count或content hash新增D2H/sync；这些只属于formal sidecar。

S3 DocOps exchange仍为`ready_for_audit`，S4-0和S4-1A也尚未实现/关闭。因此本文只把S4-1B0压到可逐文件施工
的合同，不开放production code、formal artifact或timing。

## 1. 为什么不再造一层IR

S4-1B0表达的是一个已经冻结的backend lowering：

```text
float32 coefficient
  -> int8 {-1,0,+1,-128} selector
  -> lower / upper / derived midpoint / poison
```

它没有新的solver控制流、state transition、effect schedule或跨operator plan选择。现有Bound/Plan/Task/Schedule IR
负责“为什么选择这个region、何时launch、storage由谁拥有”；本模块只负责“这个已选task如何降到两个TIR symbol”。
因此施工位置固定为：

```text
boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py
tests/test_asplos27_s4_ternary_endpoint.py
```

backend-local对象只承担：

```text
TernaryEndpointBuildSpecV1
TernaryEndpointScheduleSpecV1
CompiledTernaryEndpointV1
TernaryEndpointModuleReceiptV1
TernaryEndpointCacheObservationV1
TernaryEndpointModuleCacheV1
PreparedTernaryEndpointProbeV1       # isolated correctness only
```

禁止新增`TernaryEndpointIR`、per-site Plan或第二套runtime graph。后续S4-1B由现有Plan/Schedule/Prepared owner消费
compiled module与receipt。

## 2. 当前源码事实与复用边界

### 2.1 必须保留不变的历史模块

`boundflow/backends/tvm/r3_p_alpha_vjp.py`当前：

```text
boundflow_r31b2_pack_ainput_sign
source >= 0 -> int8(1), else int8(0)
```

`boundflow/backends/tvm/asplos27_s2_selected_value.py`当前input select：

```text
sign != 0 -> lower, else upper
```

两者都是已冻结binary v1资产，不能原地升级含义。S4使用新schema和新symbol：

```text
schema = boundflow.asplos27-s4-ternary-endpoint/v1
pack   = boundflow_s4_pack_ainput_endpoint_ternary
select = boundflow_s4_select_input_endpoint_ternary
```

实现与测试必须证明旧symbol、旧schema、旧module/source hash没有变化。

### 2.2 provider midpoint owner

pinned αβ-CROWN `auto_LiRPA/perturbations.py`的box center源码为：

```text
self.x0 = (x_U + x_L) / 2
center = (x_U + x_L) / 2.0
```

本合同规范化为lower-first的等价固定顺序：

```text
(lower + upper) * float32(0.5)
```

由于float32加法可溢出、乘法可下溢，禁止编译器或手工模板把它改写为两个half后相加。formal oracle也必须用相同
operation order，不得用float64平均后cast冒充provider逐位oracle。

### 2.3 现有cache机制只能参考

`R31B1ModuleCacheV1`只用compute capability索引；`CompiledS2SelectedValueV1.validate()`主要验证hash格式和静态字段。
S4-1B0不能复制这两个弱边界。可复用的是：

- `tvm.ir.save_json(module)`的canonical TIR hash；
- compiled imports的device source提取；
- zero-copy `tvm.runtime.from_dlpack(tensor)`与roundtrip pointer核对；
- first miss / exact hit的显式cache事件。

## 3. 精确数学与位级合同

### 3.1 pack

对每个float32 coefficient `a`：

```text
bits       = reinterpret<uint32>(a)
nonfinite  = (bits & 0x7f800000) == 0x7f800000

selector = nonfinite ? int8(-128)
         : a > +0.0f ? int8(+1)
         : a < +0.0f ? int8(-1)
         :             int8(0)
```

结果：

- `+0.0/-0.0`都为0；
- 正负subnormal保留符号，不得epsilon归零；
- 所有NaN payload和`±Inf`均为`-128`；
- generic builder接受`numel>0`，formal production spec固定`18,432`。

### 3.2 select

```text
+1 -> lower
-1 -> upper
 0 -> (lower + upper) * float32(0.5)
其他 -> reinterpret<float32>(uint32(0x7fc00000))
```

禁止：

- `selector != 0 ? lower : upper`；
- `else midpoint`吞掉非法selector；
- 用平台默认`NaN`常量而不冻结payload；
- `lower*0.5 + upper*0.5`；
- 新建physical center tensor。

### 3.3 independent GPU证据

在当前RTX 4060 Laptop / SM89、当前TVM环境中内存编译两个独立TIR symbol，16项边界输入得到：

```text
selector = [-128,-1,-1,-1,-1,-1,0,0,1,1,1,1,1,-128,-128,-128]
signed_zero_both_center = true
positive_negative_subnormal_preserved = true
nonfinite_sentinel_count = 4
invalid_selected_bits = 0x7fc00000
dlpack_pointer_exact = 5/5
```

本次diagnostic identity（不是未来production identity）：

```text
unscheduled_tir_hash = 1bbd8ee75e53ae2141dea04432fca504e4d44ad0599228fffdb794d671f4c394
scheduled_tir_hash   = 19a06888e1608d64687ad5ba2455e9ae15cf96ded291a1c090958346ac3ec5b4
device_source_hash   = b94c7f55eeeeea54d980156fd796df5b426e42f0d3e56ecbd8cda4eaf42fc2eb
```

hash来自16-element probe，正式generic implementation必须重新生成，禁止复制这些数字。

### 3.4 midpoint GPU反例

同一SM89 TIR同时执行两种表达式：

| input lower=upper | add-then-mul bits | mul-then-add bits |
|---|---:|---:|
| max finite | `0x7f800000` | `0x7f7fffff` |
| min subnormal | `0x00000001` | `0x00000000` |

```text
midpoint_reassociation_counterexample_count = 2
```

所以midpoint policy必须进入build spec、cache key、module receipt和formal raw。

## 4. 逐文件实现合同

### 4.1 backend常量与spec

新模块只import标准库和TVM lazy imports。建议常量：

```text
S4_TERNARY_ENDPOINT_SCHEMA
S4_TERNARY_ENDPOINT_PACK_SYMBOL
S4_TERNARY_ENDPOINT_SELECT_SYMBOL
S4_TERNARY_ENDPOINT_DEFAULT_THREADS = 256
S4_TERNARY_ENDPOINT_QNAN_BITS = 0x7fc00000
S4_TERNARY_ENDPOINT_NONFINITE_MASK = 0x7f800000
```

`TernaryEndpointBuildSpecV1`字段：

```text
schema_version
numel
value_dtype = float32
selector_dtype = int8
pack_symbol
select_symbol
endpoint_policy = ternary-box-endpoint-v1
midpoint_policy = add-then-mul-f32-half-v1
nonfinite_policy = ieee754-f32-exponent-bits-sentinel-minus-128-v1
invalid_output_policy = canonical-qnan-0x7fc00000-v1
target
compute_capability
```

`TernaryEndpointScheduleSpecV1`字段：

```text
threads_per_block = 256
pack_block = endpoint_selector
select_block = selected_endpoint
```

两者是backend build metadata，不是compiler IR。`validate()`拒绝未知schema/policy、非positive numel、dtype漂移、
symbol collision、target/CC不一致和threads不合法。

### 4.2 TIR builders

私有函数：

```text
_build_pack_primfunc(spec)
_build_select_primfunc(spec)
_schedule_elementwise(module, symbol, block, schedule)
build_ternary_endpoint_modules_v1(spec, schedule)
```

最后一个函数返回unscheduled和scheduled `IRModule`，使cache lookup在真正compile前即可重算两个TIR hash。
两个PrimFunc都写schema、policy和numel attrs，避免只有Python receipt知道语义。

### 4.3 compile result

`CompiledTernaryEndpointV1`至少保存：

```text
executable
unscheduled_tir_json
scheduled_tir_json
device_source
unscheduled_tir_hash
scheduled_tir_hash
device_source_hash
exported_symbols
global_workspace_bytes = 0
tvm_version
```

构造时必须：

1. 从实际IR JSON和device source重算hash；
2. 确认两个新symbol都出现在compiled source；
3. 确认旧binary symbol不在exported symbols；
4. 保存真实source，使cache hit可重哈希，不只保存64-char字符串；
5. 不记录timing或performance字段为true。

### 4.4 module receipt

`TernaryEndpointModuleReceiptV1`是immutable compiled identity：

```text
build_spec_hash
schedule_spec_hash
unscheduled_tir_hash
scheduled_tir_hash
device_source_hash
cache_key
target / compute_capability
tvm_version / tvm_commit / tvm_ffi_commit / torch_version
exported_symbols
global_workspace_bytes
performance_claimed = false
```

`validate_against(spec,schedule,compiled)`必须重算：

- build/schedule canonical hash；
- precompile TIR blueprint hash；
- cached compile result三份content hash；
- expected cache key；
- exported symbol exact tuple；
- version/target/CC/policy关系。

只检查hash格式不算通过。external stdlib replay不能重编TVM，但会重算receipt/root hash；外审现场再从source重编并比较
TIR/source hash。

### 4.5 cache lookup与observation分层

lookup key只能绑定compile前可知事实：

```text
module receipt schema
build_spec_hash
schedule_spec_hash
unscheduled_tir_hash
scheduled_tir_hash
target / compute_capability
tvm_commit / tvm_ffi_commit
```

`device_source_hash`是compile输出，不能循环地成为首次lookup key输入；它进入module receipt和cache entry value。

`TernaryEndpointModuleCacheV1.get(spec,schedule)`流程：

1. build/schedule cheap blueprint；
2. 重算TIR hash与expected key；
3. hit时对cached compiled source和module receipt完整重验；
4. miss时compile一次、生成receipt、完整重验后原子插入；
5. compile/validate失败不插入半条目，不fallback到binary/eager；
6. cache实例只保存compiled code，不保存Tensor、DLPack view或evaluation ordinal。

mutable计数单独返回`TernaryEndpointCacheObservationV1`：

```text
event = miss | hit
compile_count
miss_count
hit_count
entry_count
module_receipt_hash
```

这些计数不得进入module receipt，否则相同compiled module会因访问次数改变身份。

## 5. 隔离micro-owner，不越界接production runtime

`PreparedTernaryEndpointProbeV1`只为B0 correctness测试服务，构造参数固定为caller-owned：

```text
compiled + module receipt
coefficient[numel] float32 CUDA contiguous
lower[numel] float32 CUDA contiguous
upper[numel] float32 CUDA contiguous
selector[numel] int8 CUDA contiguous
selected[numel] float32 CUDA contiguous
observed current device/stream
```

它在prepare阶段：

- 验证shape/dtype/device/stride/storage offset/alias；
- 建5个DLPack view并roundtrip核对pointer、shape、dtype、stride；
- 禁止caller指定伪造device/stream ID；
- 保存exact module/cache identity；
- 不创建center tensor、status buffer或额外output；
- 不持有S4-1A ticket/lease，也不声称selector generation来自production coefficient pass。

`run_once()`：

```text
observe torch current stream == tvm_ffi raw stream
launch pack exactly once
launch select exactly once
return tensor-free launch counters
```

它不返回新的Python Tensor；caller仍拥有预分配`selector/selected`。non-default stream必须现场通过。异常后device/stream
恢复，fallback/eager/native shadow均0。

## 6. 三层物理账，禁止混算

### 6.1 S4-1A base（不属于B0）

```text
16 storage / 16 DLPack view / 34,080 logical B
```

它们是7 parameter、7 gradient、lower、upstream。没有coefficient、input bounds、selector或selected endpoint。

### 6.2 B0 isolated module

| 项 | 数值 |
|---|---:|
| unique tensor / DLPack view | 5 / 5 |
| argument occurrence | 6 |
| launch | 2 |
| selector output | 18,432 B |
| selected output | 73,728 B |
| output logical合计 | 92,160 B |
| center tensor/view | 0 / 0 |
| global workspace | 0 B |

fresh CUDA allocator probe：

```text
selector allocated delta = 18,432 B
selected allocated delta = 73,728 B
combined allocated delta = 92,160 B
combined reserved delta = 2,097,152 B
distinct storage = true
```

reserved是allocator slab，不得写成logical storage或peak memory claim。

### 6.3 future S4-1B production

最终ledger已有：

```text
six selectors = 55,296 B
two coefficient arenas = 147,456 B
V/lA arena = 149,856 B
```

要避免额外`73,728 B selected endpoint`，S4-1B必须冻结phase：

```text
pass A completes selector capture
  -> one coefficient arena has no live coefficient reader
  -> reinterpret as selected-endpoint scratch for pass B E0
  -> Conv0 consumes before overwrite
  -> pass B completes
  -> pass C recomputes coefficient before gradient emitter consumption
```

必须证明storage token相同、logical descriptor不同、generation变化明确、same stream或event依赖完整、没有result lease和
旧DLPack descriptor误用。未证明则分配独立73,728 B，并把2026-08-29修正后的`389,574 B` ledger改为
`463,302 B`。

这个alias门禁属于S4-1B，不属于B0。B0只证明两个TIR对caller-owned distinct storage正确。

## 7. warm receipt与formal observation分离

### 7.1 warm-safe launch receipt

只允许O(1) host事实：

```text
module_receipt_hash
cache_event
device_ordinal / stream_identity
evaluation_ordinal / parameter_state_version / selector_generation
prepared_descriptor_hashes
pack_launch_count = 1
select_launch_count = 1
argument_occurrence_count = 6
warm_dlpack_view_creation_count = 0
fallback/eager/native_shadow = 0/0/0
timing_recorded = false
performance_claimed = false
```

不得每次warm run `.cpu()` coefficient/selector，也不得同步统计positive/negative/zero/invalid。

### 7.2 correctness/formal sidecar

formal worker在显式synchronize之后可保存raw并重算：

```text
coefficient/lower/upper/selector/selected raw
bit-preserving content hashes
positive/negative/zero/invalid counts
canonical qNaN payload counts
old-binary zero misclassification count
independent selected output
descriptor/pointer/storage inventory
allocator diagnostic
```

formal ResNet2B fixture冻结：

```text
positive / negative / zero / invalid = 8689 / 9137 / 606 / 0
old_binary_zero_misclassified = 606
selected_bitwise_exact = true
```

这些是fixture expectation，不写进generic validator。

## 8. stable failure reasons与negative矩阵

B0 backend/micro-owner至少有下列20类稳定reason，顺序按spec→compile→prepare→launch→observation：

1. `TERNARY_ENDPOINT_SCHEMA_MISMATCH`；
2. `TERNARY_ENDPOINT_POLICY_MISMATCH`；
3. `TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH`；
4. `TERNARY_ENDPOINT_NONFINITE_POLICY_MISMATCH`；
5. `TERNARY_ENDPOINT_SYMBOL_COLLISION`；
6. `TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH`；
7. `TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH`；
8. `TERNARY_ENDPOINT_CACHE_KEY_MISMATCH`；
9. `TERNARY_ENDPOINT_CACHE_ENTRY_POISONED`；
10. `TERNARY_ENDPOINT_LEGACY_MODULE_COLLISION`；
11. `TERNARY_ENDPOINT_SHAPE_MISMATCH`；
12. `TERNARY_ENDPOINT_DTYPE_MISMATCH`；
13. `TERNARY_ENDPOINT_DEVICE_MISMATCH`；
14. `TERNARY_ENDPOINT_LAYOUT_MISMATCH`；
15. `TERNARY_ENDPOINT_ALIAS_MISMATCH`；
16. `TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH`；
17. `TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH`；
18. `TERNARY_ENDPOINT_LAUNCH_COUNT_MISMATCH`；
19. `TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED`；
20. `TERNARY_ENDPOINT_CLAIM_FLAG_MISMATCH`。

tests还必须通过具体counterexample证明：

- float classifier把NaN归zero会失败；
- `-128`经`selector!=0`错误取lower会失败；
- `+0/-0`分叉会失败；
- epsilon吞subnormal会失败；
- midpoint重结合会失败；
- canonical NaN payload变化会失败；
- 删除任一cache policy字段、改threads/target/TIR必须miss或拒绝；
- hit后篡改cached device source必须拒绝；
- binary v1 module不能命中新cache；
- `performance_claimed=true`必须拒绝。

这些是B0 unit/fault categories，不改变S4-4冻结的71类高层fully re-signed tamper编号。

## 9. 测试文件精确布局

`tests/test_asplos27_s4_ternary_endpoint.py`建议按以下组组织：

```text
test_pack_cpu_bit_oracle_covers_signed_zero_subnormal_nonfinite
test_select_cpu_bit_oracle_covers_midpoint_and_canonical_nan
test_midpoint_operation_order_has_two_frozen_counterexamples
test_build_spec_and_schedule_are_generic_and_canonical
test_tir_module_exports_exact_new_symbols
test_cache_first_miss_then_exact_hit_compiles_once
test_cache_key_rejects_policy_schedule_target_and_legacy_collisions
test_cache_hit_rehashes_cached_device_source
test_cuda_boundary_pack_select_matches_bit_oracle
test_cuda_non_default_stream_and_five_dlpack_pointers_are_exact
test_cuda_invalid_selector_produces_canonical_nan
test_isolated_outputs_account_for_92160_logical_bytes
test_formal_ainput_inventory_is_8689_9137_606_0
test_old_binary_misclassifies_exactly_606_zero_entries
test_legacy_r31b2_and_s2_identities_are_unchanged
test_all_twenty_negative_reasons_are_stable
```

CPU-only环境跳过CUDA positive，不得把skip写成PASS；spec/cache纯Python negative仍应运行。

## 10. future formal artifact拓扑

依赖门禁全部关闭后，B0 formal建议：

```text
5 fresh production-fixture workers
  - rebuild/load exact ternary module
  - reconstruct real Ainput/lower/upper
  - run pack/select once
  - save all five raw tensors and receipts

1 fresh cache worker
  - miss -> exact hit
  - compile_count remains 1

5 isolated fault workers
  - classifier/policy
  - cache/source tamper
  - descriptor/DLPack
  - stream/launch
  - invalid selector/claim flag
```

raw-first：worker先写raw和success/failure envelope，parent只在全部完整后生成summary。partial/resume拒绝。replayer用
stdlib重算bits、counts、hash、midpoint、selected、receipt和root；外审现场重编TIR并核对source hash。

internal状态只能是：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0
FORMAL-NO-GO-S4-1B0-TERNARY-ENDPOINT
```

不得在外审前写`VALIDATED-S4-1B0`。

## 11. 实施提交顺序

只有S3 approved+closed、S4-0 validated、S4-1A validated后：

1. `docs: activate S4-1B0 construction contract`；
2. `test(math): freeze ternary endpoint IEEE bit semantics`；
3. `feat(tvm): add isolated S4 ternary endpoint module`；
4. `test(tvm): close cache receipt and CUDA boundary gates`；
5. `artifact: generate S4-1B0 five-fresh evidence`；
6. `docs: deliver S4-1B0 external audit`；
7. `docs: close S4-1B0 or formal no-go`；
8. `docs: activate S4-1B production arena and phase alias proof`。

`feat(runtime): bind selector generation to S4 evaluator`从B0移到S4-1B/1D，不得越级塞进backend closure。

## 12. 可重算construction model

canonicalization：UTF-8、JSON `sort_keys=True`、`separators=(',', ':')`、原生boolean。model字段：

```json
{"backend_file":"boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py","cache":{"device_source_in_lookup_key":false,"hit_rehashes_cached_source":true,"mutable_counts_in_module_receipt":false,"precompile_tir_hashes_in_lookup_key":true},"claims":{"implementation":false,"performance":false,"production_alias":false,"production_correctness":false},"formal":{"cache_workers":1,"fault_workers":5,"positive_workers":5,"status_requires_external_audit":true},"math":{"invalid_output_bits":"0x7fc00000","midpoint_policy":"add-then-mul-f32-half-v1","nonfinite_mask":"0x7f800000","selector_values":[-128,-1,0,1]},"negative_reason_count":20,"production":{"selected_output_alias_requires_s4_1b_phase_proof":true,"warm_content_hash":false,"warm_count_sync":false},"scope":{"backend_compile":true,"evaluator_binding":false,"new_ir":false,"prepared_probe":true,"timing":false},"storage":{"isolated_dlpack_views":5,"isolated_output_allocated_bytes":92160,"selected_output_bytes":73728,"selector_bytes":18432,"s4_1a_base_view_overlap":0},"symbols":["boundflow_s4_pack_ainput_endpoint_ternary","boundflow_s4_select_input_endpoint_ternary"],"test_file":"tests/test_asplos27_s4_ternary_endpoint.py","threads":256}
```

SHA256：

```text
5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a
```

不得由实现硬编码该hash为PASS；实现必须从代码对象重建model并比较。

## 13. GO / STOP

### GO

- 前置四级门禁均已外审关闭；
- 新backend file不修改旧S2/R31B2；
- bit classifier、canonical poison、midpoint order逐位正确；
- cache key/module receipt/cache observation/formal observation四层分离；
- 5 isolated view和92,160 B输出账诚实；
- 20 negative、fresh cache与non-default stream通过；
- five-fresh真实inventory与raw replay通过；
- external audit批准。

### STOP

- 为了省事改旧binary symbol；
- 新增一层endpoint IR或per-site runtime旁路；
- 浮点classifier仍可能把NaN当zero；
- cache hit只检查64-char hash格式；
- 把mutable cache counts放进module identity；
- 把formal content hash/count塞进warm路径；
- 把5 isolated views写成S4-1A base views；
- 未证明phase alias却从ledger删除73,728 B；
- 用本轮diagnostic hash冒充未来production build；
- 外审前升级correctness/performance claim。

## 14. 当前状态

```text
S4-1B0 math/IEEE mechanism = diagnostic PASS
S4-1B0 source/cache/storage contract = construction-ready
S4-1B0 production implementation = closed
S4-1B0 formal artifact = closed
S4-1B production alias proof = closed
S4 timing/performance = closed
```
