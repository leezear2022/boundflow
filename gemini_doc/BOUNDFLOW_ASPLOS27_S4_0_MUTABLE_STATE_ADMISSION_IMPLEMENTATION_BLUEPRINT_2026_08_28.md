---
status: draft-implementation-blueprint
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

S4-0不需要新增solver IR、execution IR或另一套verification graph。它只需要一个**tensor-free typed runtime
binding receipt**，把已有四类权威对象闭合：

```text
ProductionStateSnapshotV4
  + ProductionReluTopologyV4
  + R31FullRegionPlanV1
  + VerificationRejectionReason
  → S4MutableStateAdmissionV1
```

该receipt只证明“六个compressed lower-α slot、六个sparse β slot及其layout/history/ownership可以被后续
compiled evaluator完整且确定性地绑定”。它不创建dense α/β、不分配GPU buffer、不执行TIR、不计时，也不改变
live solver state。

当前S3 external audit仍未返回，因此本文只有implementation blueprint地位；不能据此直接开始S4代码。

## 1. 已有事实与可复用owner

### 1.1 production snapshot

`boundflow/runtime/rvir_v4_production_state.py`已经提供：

- `ProductionStateSnapshotV4`：snapshot、history、optimizer policy及canonical stable hash；
- `OwnedProductionTensorV4`：semantic path、role、axes、dtype/shape、ownership、alias group、content hash；
- `ProductionTensorRole`与`ProductionTensorOwnership`；
- α layout、β location/sign/history一致性校验；
- finite、重复path、重复history key与content hash fail-closed。

因此S4-0不得重新定义tensor metadata schema，也不得复制snapshot validator。

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

S4-0只增加“production mutable slice如何绑定到这些layout”的runtime contract，不创建第二份Plan IR。

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

## 4. 编译入口

冻结函数职责：

```text
compile_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
) -> S4MutableStateAdmissionV1
```

当前函数可接受R31 plan作为formal输入，但新增dataclass本身不得包含ResNet2B、native id、固定shape或P-anchor
常数。未来general plan只需满足同一metadata contract，不修改receipt schema。

## 5. 确定性编译算法

严格顺序如下：

1. 调用`snapshot.validate()`与`production_plan.validate()`；
2. 校验lower-only optimizer policy：`bound_lower=true/bound_upper=false/fix_intermediate_bounds=true`；
3. 验证topology非空、key唯一，并计算canonical topology hash；
4. 以`production_plan.relu_layouts`顺序建立slot，不按snapshot物理排列或dict insertion order；
5. 对每个layout唯一解析topology link、α path和β value/location/sign path；
6. 验证snapshot中mutable-copy-out path集合恰等于六α+六β value；
7. 验证α role/ownership/axes/dtype/finite/source hash与plan tensor spec一致；
8. 验证α leading axes为`[direction,start_spec,domain,...]`，lower-only active slice为`[0,0]`；
9. 从source分别计算active/preserved slice hash；不得先dense scatter再project；
10. 验证feature shape、flat indices、spec lookup与plan layout逐项一致；
11. 验证β value/location/sign shape、dtype、role、history lineage与plan layout一致；
12. 验证全部mutable alias group唯一，且不与read-only layout/history对象共享写owner；
13. 汇总stored/active/preserved与β计数，从slot重算mutable path set hash；
14. 构造receipt，四个claim/execution flag全部false；
15. 调用receipt `validate()`并返回。

S4-0不得调用：

- `initialize_rvir_v4_native_pre_state`的dense构造路径；
- `bind_r31_runtime_inputs_v1`的GPU binding；
- TVM compile/launch；
- provider `compute_bounds/update_bounds`；
- Adam optimizer；
- wall-clock或CUDA event timing。

## 6. reason映射

S4-0至少冻结以下detail code，并映射到已有GC0 reason：

| S4 detail code | GC0/verification reason |
|---|---|
| `STATE_VERSION_MISMATCH` | `STATE_VERSION_MISMATCH` |
| `BOUND_POLARITY_MISMATCH` | `BOUND_POLARITY_MISMATCH` |
| `TOPOLOGY_IDENTITY_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `MUTABLE_STATE_COVERAGE_INCOMPLETE` | `VJP_OWNER_OR_SAVED_STATE_MISMATCH` |
| `ACTIVE_BETA_COVERAGE_INCOMPLETE` | `BETA_ACTIVE_EMPTY_MISMATCH` |
| `ALPHA_MUTABLE_DIRECTION_MISMATCH` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `ALPHA_PRESERVED_DIRECTION_DRIFT` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `ALPHA_LAYOUT_IDENTITY_MISMATCH` | `ALPHA_INDEX_OR_DIRECTION_MISMATCH` |
| `BETA_LOCATION_SIGN_HISTORY_MISMATCH` | `BETA_LOCATION_SIGN_HISTORY_MISMATCH` |
| `MUTABLE_ALIAS_CONFLICT` | `UNSAFE_ALIAS_OR_LIFETIME` |
| `MUTABLE_SLOT_ORDER_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` |
| `DTYPE_OR_DEVICE_MISMATCH` | `DTYPE_OR_DEVICE_MISMATCH` |
| `NONFINITE_MUTABLE_STATE` | `DTYPE_OR_DEVICE_MISMATCH` |
| `CLAIM_FLAG_TRUE_BEFORE_FORMAL` | `RECEIPT_IDENTITY_MISMATCH` |
| `S4_0_EXECUTION_FORBIDDEN` | `RUNTIME_FALLBACK_REQUIRED` |

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
6. canonical JSON与stable hash跨两个fresh process一致；
7. dataclass/object graph中不存在tensor、pointer、module或callback；
8. monkeypatch CUDA allocation、TVM compile和provider entry为必抛，positive admission仍通过；
9. receipt validate独立重算计数与hash；
10. formal binding与native pre-state oracle的12个semantic path及mapped slice hash一致。

### 7.2 minimum 15 negative/tamper

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

1. `feat(runtime): add S4 mutable-state admission receipt`；
2. `test(runtime): close S4-0 admission and tamper gates`；
3. `docs: close S4-0 and preregister S4-1A`。

如果执行纪律要求一个logical commit，可把1—2合并，但文件/语义范围不扩大。

## 9. S4-0关闭门槛

全部成立才允许状态变为`VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`：

- formal inventory与本稿数字逐项一致；
- 12 mutable path、六slot、active/preserved方向完全闭合；
- receipt tensor-free且canonical hash跨fresh process一致；
- minimum 15类negative全部exact fail-closed；
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
