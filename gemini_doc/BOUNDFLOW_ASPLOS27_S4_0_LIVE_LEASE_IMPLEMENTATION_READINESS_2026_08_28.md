---
status: diagnostic-complete-corrected-v3-code-closed
date: 2026-08-28
type: implementation-readiness-correction
topic: boundflow
slug: asplos27-s4-0-live-lease-readiness
stage: s04
execution-authority: false
code-change-open: false
gpu-execution-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-0：跨阶段live ownership与ephemeral lease实施就绪结论

## 0. 直接结论

S4-0 V2把输入从offline snapshot扩成了`snapshot + topology + plan + transient live mapping`，但其返回值仍只有
tensor-free receipt。这个设计可以证明“admission发生时看到的live Tensor与snapshot一致”，却不能证明S4-1A稍后
重新取得的Tensor仍是同一个Python object和同一个storage。

根因不是receipt缺少更多canonical字段，而是**稳定序列化投影不能表达跨时间的进程内对象身份**。因此冻结V3边界：

```text
current provider mapping
  → S4-0 validate/admit
  ├─ S4MutableStateAdmissionV1       canonical、tensor-free、可replay
  └─ S4LiveMutableLeaseV1            ephemeral、强引用、不可序列化
       → S4-1A pack/transfer
       → S4-2 evaluate/mutate
       → S4-3 current-provider rebind + precommit revalidate
       → commit/abort
       → close
```

这仍然不是新IR。receipt是审计证据；lease是prepared runtime transaction owner。lease不得进入canonical JSON、artifact、
hash、planner或跨query cache。

当前S3外审仍未关闭，所以本稿只纠正实施合同，不开放S4-0代码、GPU执行或计时。

## 1. 为什么tensor-free receipt不够

### 1.1 全量clone替换反例

构造一组live mutable source，生成V2计划中的canonical投影；随后把每个source替换为shape/dtype/device/content/
stride/offset和`_version`都相同的新clone，再按相同path顺序重新编号object/storage group。结果：

```text
TENSOR_FREE_RECEIPT_REACQUIRE_COUNTEREXAMPLE_PASS
canonical_projection_equal=True
canonical_hash_equal=True
75d3252e54c5383376e8b3286faae1030381de7b4f998c77746c8e518c353c9f
raw_object_identity_equal=False
nonempty_storage_identity_equal=False
empty_object_identity_equal=False
versions=[('alpha/a',0),('beta/b',0)]
```

这不是hash碰撞。两个mapping按同样顺序都会得到`object:000000...`和`storage:000000...`；canonical group只表达
mapping内部alias拓扑，不表达“第二次取得的object是否就是第一次那个object”。把raw `id`或pointer写进receipt也不合格：

- 跨进程不稳定，破坏artifact确定性；
- allocator可能重用地址；
- 暴露本机运行时身份，仍不能替代强引用生命周期；
- pointer不能证明Python object owner和PyTorch version counter相同。

所以必须同时保留tensor-free receipt和进程内强引用lease，不能二选一。

### 1.2 β未拥有后缀反例

把formal active β `/input-28`从width 1改成width 2，在value/location/sign末尾追加一项并同步更新全部content hash，
但history仍保持width 1。当前`ProductionStateSnapshotV4.validate()`接受了该snapshot：

```text
BETA_PREFIX_ONLY_VALIDATOR_COUNTEREXAMPLE_PASS
active_beta_layer=/input-28
original_width=1 mutated_width=2 history_width=1
snapshot_validate_accepted_unowned_suffix=true
```

因此S4-0的`beta_width == history_width`不是重复校验，而是阻止compiled optimizer取得history未拥有β slot的必要门禁。

## 2. 复用现有边界，不再造adapter

### 2.1 live source枚举

`boundflow/runtime/rvir_v4_live_return.py`已有公开helper：

```text
live_targets_from_pre_result_v4(...)
```

它已经从current provider `pre_result`取得production六α与六β value原始Tensor引用。S4-0必须复用该helper或其公开
协议，不再实现第二套node/path adapter，不从snapshot clone、dense initializer或terminal return重建source。

### 2.2 snapshot、plan与commit各自拥有的事实

- `ProductionStateSnapshotV4`：semantic content/history/policy truth；CPU clone，不是live owner；
- `R31FullRegionPlanV1`：当前formal static compiled specialization；不拥有live Tensor；
- `S4MutableStateAdmissionV1`：canonical projection、coverage、计数、hash与claim flags；
- `S4LiveMutableLeaseV1`：本次exact-call原始Tensor object/storage/version的时间性owner；
- existing FSG4 atomic commit：可复用version/storage检查和copy-out数学，但不能替代S4 lease，因为它没有证明
  S4-0到S4-3期间provider mapping没有整体rebind。

## 3. 冻结对象设计

### 3.1 `S4MutableStateAdmissionV1`

保持V2中tensor-free canonical receipt的字段和稳定hash职责，不保存：

- `torch.Tensor`强/弱引用；
- raw object id、storage handle、data pointer；
- provider object、callback、closure；
- lease token的本机随机值。

它可以跨进程replay并证明admission时的稳定投影，但文档和代码不得再称它本身关闭了跨阶段object identity。

### 3.2 `S4LiveMutableLeaseV1`

该对象只属于runtime，私有保存每条semantic path的：

```text
strong_tensor_reference
raw_object_identity
raw_storage_identity_or_empty_object_identity
entry_tensor_version
shape / dtype / device
stride / storage_offset
content_hash
receipt_object_group / receipt_storage_group
```

最低方法：

```text
revalidate_current_mapping(current_sources, phase) -> None
transfer_to_prepared_runtime(expected_admission_hash) -> None
mark_commit_started() -> None
mark_committed_or_aborted(outcome) -> None
close() -> None
```

`revalidate_current_mapping`必须同时证明：

1. current mapping path exact；
2. `current_tensor is strong_tensor_reference`；
3. nonempty storage identity exact；empty tensor以object identity为owner，不因zero pointer互相alias；
4. `_version`、shape、dtype、device、stride、offset exact；
5. 在应保持未变的phase，content hash exact；
6. receipt admission hash和lease admission hash exact。

若object已替换，即使内容完全相同也拒绝；若原object被in-place修改，即使后来把值改回，也以version漂移拒绝。

### 3.3 serialization与生命周期门禁

lease不得提供`to_dict()`或stable hash；`__getstate__`/`__reduce__`必须稳定抛出
`LIVE_LEASE_SERIALIZATION_FORBIDDEN`对应异常。不得仅依赖“调用方不会序列化”的约定。

冻结状态机：

```text
OPEN
  → TRANSFERRED_TO_PREPARED_RUNTIME
  → COMMITTING
  → COMMITTED | ABORTED_CLEAN | POISONED_NO_RETRY
  → CLOSED
```

- lease只能transfer一次；
- S4-1A不得丢弃lease后只保留canonical receipt；
- S4-2只修改candidate buffers，不修改lease source；
- S4-3必须从**当前provider mapping**重新枚举12条target并与lease逐对象比较；
- commit/abort后清空强引用并close；
- close后任何read/revalidate/transfer都拒绝；
- 不能跨query、跨core call、provider fallback或retry复用。

## 4. 冻结入口与返回类型

为避免调用方把receipt和lease拆散，推荐返回一个非序列化prepared admission wrapper：

```text
prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: Mapping[str, torch.Tensor],
) -> PreparedS4MutableStateAdmissionV1
```

其中：

```text
PreparedS4MutableStateAdmissionV1:
    receipt: S4MutableStateAdmissionV1
    _live_lease: S4LiveMutableLeaseV1

    transfer_to_buffer_prepare(...) -> transferred lease + receipt
    close()
```

wrapper、lease均不得canonicalize；只有`.receipt`允许进入artifact。若实现语言/typing原因采用
`tuple[receipt, lease]`，必须额外以shared admission hash和single-transfer guard防止错配；优先采用wrapper。

入口仍必须先把mapping materialize成普通`dict`并拒绝lazy/callable provider view，但生成lease时应保留dict中原始Tensor的
强引用，不能在函数返回前全部释放。

## 5. S4-1A与S4-3的修正接口

### 5.1 S4-1A

旧接口：

```text
prepare_s4_mutable_buffers_v1(admission, newly_reacquired_mapping, ...)
```

修正为：

```text
prepare_s4_mutable_buffers_v1(
    prepared_admission: PreparedS4MutableStateAdmissionV1,
    current_live_sources: Mapping[str, torch.Tensor],
    device,
    stream_identity,
) -> PreparedS4MutableBuffersV1
```

该函数先调用lease revalidate，再pack candidate buffer，并把lease ownership transfer进prepared runtime；prepared
runtime不能保留provider lookup callback，但必须保留强引用lease直到S4-3结束。

### 5.2 S4-3

在第一次live copy前，除原V2 precommit checks外必须：

1. 从current `pre_result`调用existing public helper重新枚举12条provider target；
2. 对lease执行object/storage/version/layout revalidation；
3. 证明commit target正是lease中的原始Tensor，而不是prepared candidate、clone或same-storage view；
4. 成功/clean abort/poisoned后按状态关闭lease，禁止fallback/retry复用。

如果provider在candidate运行期间把`node.alpha`/`sparse_betas.val`整体替换为新Tensor，即使数值相同，也必须在commit前
clean abort；不能向lease保存的旧对象写入后声称solver state已提交。

## 6. stable detail code增量

在V2 detail code之上至少增加：

| detail code | verification reason | 含义 |
|---|---|---|
| `LIVE_SOURCE_OBJECT_REPLACED` | `RECEIPT_IDENTITY_MISMATCH` | current provider path不再指向原Tensor object |
| `LIVE_SOURCE_STORAGE_REPLACED` | `UNSAFE_ALIAS_OR_LIFETIME` | 原object合同下storage identity漂移 |
| `LIVE_LEASE_ADMISSION_MISMATCH` | `RECEIPT_IDENTITY_MISMATCH` | receipt与lease不属于同一次admission |
| `LIVE_LEASE_ALREADY_TRANSFERRED` | `UNSAFE_ALIAS_OR_LIFETIME` | lease被第二个prepared runtime复用 |
| `LIVE_LEASE_ALREADY_CLOSED` | `UNSAFE_ALIAS_OR_LIFETIME` | close后继续使用 |
| `LIVE_LEASE_SERIALIZATION_FORBIDDEN` | `RECEIPT_IDENTITY_MISMATCH` | lease/wrapper试图进入artifact |
| `LIVE_LEASE_PROVIDER_REBIND` | `RECEIPT_IDENTITY_MISMATCH` | S4-3 current provider mapping整体/局部rebind |

reason仍映射到已有`VerificationRejectionReason`，不扩Verification IR vocabulary。

## 7. 修正后的测试最低集

S4-0 minimum negative从30扩为至少38类。除V2测试外新增：

1. 全部source替换为same-content clone，canonical projection/hash相同，但lease拒绝；
2. source替换为same-storage view，object guard拒绝；
3. 原object in-place变更，version guard拒绝；
4. empty β替换为同shape empty clone，object guard拒绝；
5. receipt来自admission A、lease来自admission B，拒绝；
6. 同一lease transfer两次，第二次拒绝；
7. close后revalidate/transfer，拒绝；
8. pickle/deepcopy/`dataclasses.asdict`或artifact walker试图遍历lease，稳定拒绝；
9. S4-1A pack后provider mapping rebind，S4-3 precommit拒绝；
10. lease保持强引用：外部mapping/局部变量删除并GC后，原Tensor仍活到commit/close；
11. commit/abort/poisoned每条路径均清空强引用且不可复用；
12. artifact递归对象图只含receipt，不含lease/Tensor/raw identity。

还必须保留β width/history exploit专项测试；只运行`snapshot.validate()`不算关闭。

只读lease guard原型结果：

```text
EPHEMERAL_LIVE_LEASE_GUARD_PROBE_PASS
same_content_clone=LIVE_SOURCE_OBJECT_REPLACED
same_storage_view=LIVE_SOURCE_OBJECT_REPLACED
inplace_mutation=LIVE_TENSOR_VERSION_MISMATCH
empty_clone=LIVE_SOURCE_OBJECT_REPLACED
strong_reference_required=true
canonical_receipt_alone_sufficient=false
```

## 8. 文件边界与实施顺序

S3批准后，S4-0仍只新增原计划的单个runtime模块和单个test模块；receipt、lease、prepared wrapper可以共处同一模块，
不新增IR文件、registry或provider adapter：

```text
boundflow/runtime/asplos27_s4_mutable_state_admission.py
tests/test_asplos27_s4_mutable_state_admission.py
```

建议代码顺序：

1. canonical slot/receipt和现有V2 projection validation；
2. live token捕获、strong-ref lease和serialization guard；
3. prepared wrapper与single-transfer lifecycle；
4. β width/history exact gate；
5. 38+ negative与fresh-process receipt determinism；
6. 只在S4-0关闭后修改S4-1A prepared buffer实现。

## 9. 关闭门槛

S4-0只有同时满足以下条件才能关闭：

- tensor-free receipt跨fresh process canonical exact；
- ephemeral lease证明S4-0→S4-1A→S4-3是同一批live object/storage/version；
- receipt与lease职责分离，artifact中lease/Tensor/raw identity为0；
- current provider mapping整体/局部rebind均在mutation/commit前拒绝；
- β width/history exact，未拥有后缀被拒绝；
- lease single-transfer/single-query/close/serialization门禁全部机械测试；
- minimum 38类negative exact reason；
- dense/GPU/TIR/provider callback/timing/performance仍为0/false；
- S3 external audit已经approved并close。

本稿不改变当前状态：

```text
S3 exchange = ready_for_audit
S4-0 implementation = closed
S4 GPU execution/timing/performance = closed
```
