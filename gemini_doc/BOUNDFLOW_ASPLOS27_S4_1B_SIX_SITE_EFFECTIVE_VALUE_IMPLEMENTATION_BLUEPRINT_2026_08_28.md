---
status: draft-requires-s4-1b0-dag-adjoint-closure
date: 2026-08-28
type: implementation-plan
topic: boundflow
slug: asplos27-s4-1b-six-site-effective-value
stage: s04
depends-on: validated-s4-1a-ordered-buffer-abi
execution-authority: false-pending-s3-external-audit-s4-0-s4-1a
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B：六site coefficient-adjoint graph实施蓝图

## 0. 直接结论

S4-1B不是六个独立BoundOp wrapper，也不是保存六份dense coefficient A。原稿曾把它定义为按coefficient sign
运行原始primal DAG的单向selected-primal graph；2026-08-28只读预检已证明这个等价在site19不成立：

```text
production compressed α19 max abs diff = 0.0011564247542992234
gradient sign mismatch = 9
```

因此本蓝图的规范owner改为**CROWN coefficient schedule的精确adjoint replay**。以下原始primal图只保留为
candidate lowering直觉，不再是correctness规范：

```text
selected input
  → pre17 → selected ReLU17
  → pre19 → selected ReLU19
  → residual6 add = pre23 → selected ReLU23
  → pre25 → selected ReLU25
  → residual11 add = pre28 → selected ReLU28
  → flatten/Gemm14 = pre31
```

一次execution写入一个149,856-byte persistent arena的六个slot。slot保存
`V_i = d lower / d T_i`，其中`T_i`是ReLU transform后的coefficient state；它不保存反向coefficient。pass C
重算coefficient并消费这些V生成六dα与active dβ。

实现前必须先关闭
`BOUNDFLOW_ASPLOS27_S4_1BC_DAG_ADJOINT_PREFLIGHT_CORRECTION_2026_08_28.md`定义的S4-1B0门禁。S2 selected
Relax/cuDNN与R31B2 selected-ReLU TIR只能作为五个已通过site和局部anchor的复用资产，不能直接充当六site规范；
仍禁止另写per-site Python executor。

## 1. 数学语义

规范定义不是“`V_i`必然等于普通primal `pre_i`”，而是：

```text
A_i = ReLU transform前incoming coefficient
T_i = A_i * selected_slope_i + beta_add_i
V_i = d lower / d T_i
```

只有逐site证明`V_i == selected pre_i`后，下面的selected-primal公式才可作为该site的优化lowering。

对每个ReLU site `i`，令production fixed bound为`l_i/u_i`，lower-α为`α_i`，pass A产生的incoming lower
coefficient为`A_i`。selected relaxation：

```text
ambiguous = (l_i < 0 < u_i)
lower_slope = ambiguous ? clamp(α_i,0,1) : (l_i >= 0 ? 1 : 0)
upper_slope = l_i >= 0 ? 1 : (u_i <= 0 ? 0 : u_i / max(u_i-l_i, eps))

slope = A_i >= 0 ? lower_slope : upper_slope
intercept = (A_i < 0 and ambiguous) ? -l_i * upper_slope : 0
selected_relu(pre_i) = slope * pre_i + intercept
```

input endpoint：

```text
selected_input = A_input >= 0 ? input_lower : input_upper
```

原稿建议随后只执行原始primal Conv/Add/Flatten/Gemm得到六个`pre_i`。该方式现在降为candidate：active β先在
coefficient pass中改变`A`，但DAG fanout、residual accumulation和coefficient injection的精确VJP必须由
coefficient-adjoint schedule负责。site19反例关闭前不得生成production module。

## 2. 六个输出slot

按production plan order冻结flat arena：

| slot | native | logical shape | flat shape | offset | elements | bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | `[6,8,16,16]` | `[12288]` | 0 | 12,288 | 49,152 |
| 1 | 19 | `[6,16,8,8]` | `[6144]` | 12,288 | 6,144 | 24,576 |
| 2 | 23 | `[6,16,8,8]` | `[6144]` | 18,432 | 6,144 | 24,576 |
| 3 | 25 | `[6,16,8,8]` | `[6144]` | 24,576 | 6,144 | 24,576 |
| 4 | 28 | `[6,16,8,8]` | `[6144]` | 30,720 | 6,144 | 24,576 |
| 5 | 31 | `[6,100]` | `[600]` | 36,864 | 600 | 2,400 |
| 合计 | — | — | `[37464]` | — | **37,464** | **149,856** |

offset/shape来自plan/value graph，通用schema不写死这些formal数字。prepared runtime只分配一个physical arena，
六个view在prepare阶段构造；warm path不创建view/tensor/dict。

## 3. sign bitmap inventory

coefficient-adjoint replay需要六张int8 bitmap来重放Pass A的离散branch choice：

| bitmap | 选择对象 | elements/bytes | 当前状态 |
|---|---|---:|---|
| `sign_ainput` | input lower/upper | 18,432 | 已有 |
| `sign_a18` | ReLU17 | 12,288 | 已有 |
| `sign_a20` | ReLU19 | 6,144 | 已有 |
| `sign_a24` | ReLU23 | 6,144 | 已有 |
| `sign_a26` | ReLU25 | 6,144 | **新增** |
| `sign_a29` | ReLU28 | 6,144 | **新增** |
| 合计 | — | **55,296** | 当前43,008 + 新增12,288 |

不需要`sign_a32`来生成pre31，因为pre31位于ReLU31之前；site31的incoming coefficient由pass C在gradient
emitter处直接使用。

bitmap取值冻结为`1 iff coefficient >= 0 else 0`，与现有R31B2一致。bitmap receipt必须绑定evaluation ordinal、
parameter state version、β/split/history identity、coefficient module hash与pointer generation。

## 4. 现有资产与真正缺口

### 4.1 已有TIR/Relax资产

- `R31B2_EFFECTIVE_PRE17_SYMBOL`：selected input→Conv0→pre17；
- `R31B2_EFFECTIVE_PRE23_SYMBOL`：从pre17内联selected17/pre19/selected19与residual6，输出pre23；
- `R31B2_EFFECTIVE_PRE25_SYMBOL`：selected23→Conv8→pre25；
- `S2_SELECTED_VALUE_FUNCTION`：Relax/cuDNN路径显式计算pre17、pre19、pre23、pre25；
- S2 persistent output copy与safe VM invocation；
- R31B1/D1C/D2B coefficient arena、metadata maps和prepare-time DLPack cache。

### 4.2 缺口

1. pre19目前只在Relax graph或fused pre23 kernel内部存在，未持久导出；
2. pre28/pre31尚未进入selected-value graph；
3. sign A26/A29尚未持久导出；
4. current α ABI仍是`[2,1,D,W]`，S4必须改为S4-1A active `[D,W]`；
5. current S2只copy/export pre25，S4需六slot persistent handoff；
6. 当前receipt没有all-six value inventory/version/lifetime。
7. 普通selected-primal图在site19 production compressed projection上已失败`1.156e-3/9 sign mismatch`；
8. 尚无对residual fanout/accumulate、bias与box concretization逐action绑定的adjoint schedule。

### 4.3 为什么不直接扩展`effective_pre23`大kernel

该kernel为每个pre23输出元素局部重算pre19；它没有一个唯一可安全写出的pre19 producer。强行增加pre19 output会
产生重复写/race或需要额外全图同步。all-state本来就必须保存pre19，因此应把selected graph拆成具有明确stage
边界的单向图，而不是继续把早期中间值藏在巨型kernel内部。

## 5. 推荐实现：从coefficient schedule派生adjoint replay

### 5.1 原因

S2 selected graph已经验证Conv/Residual局部语义、cuDNN+TIR混合、current stream、persistent output和S3修复后的
safe VM invocation；它继续作为kernel/graph复用资产。规范实现则必须从Pass A的typed coefficient action sequence
逐action派生VJP，显式编码seed、Linear/Conv right、ReLU transform、β add、residual duplicate/accumulate、bias与
box concretization。不得用S2 graph本身跳过S4-1B0。

### 5.2 logical adjoint stages

```text
E0 input_select + Conv0                     → pre17 slot
E1 selected_relu17 + Conv2                  → pre19 slot
E2 selected_relu19 + Conv4
   selected_relu17 + shortcut Conv5 + Add   → pre23 slot
E3 selected_relu23 + Conv8                  → pre25 slot
E4 selected_relu25 + Conv10
   + selected_relu23 residual Add11         → pre28 slot
E5 selected_relu28 + Flatten + Gemm14       → pre31 slot
```

上图仅描述期望的物理dataflow。每个stage还必须绑定其来源coefficient action与fanout provenance；site19在
S4-1B0关闭前不得标记为等价。logical stage不等于CUDA kernel；actual cuDNN/TIR/copy kernel count由compiled
module receipt披露。

### 5.3 persistent outputs

第一版correctness实现允许每个stage末尾用`call_tir_inplace`写入预分配arena slot，预计persistent copy count=6。
这不形成性能claim；S4-P若证明copy/launch显著，再让producer直接写slot或做horizontal fusion。不得为提前优化而让
S4-1B输出owner不可审计。

Relax函数返回六个arena view组成的fixed tuple/token，不允许VM动态创建payload tensor。wrapper只返回一个typed
lease，hot path不把六个Python tensor装dict。

### 5.4 active α ABI

selected-ReLU TIR统一接收：

```text
pre[D,F]
sign[D,F]
lower[D,F]
upper[D,F]
active_alpha[D,W]
alpha_map[F]
→ selected[D,F]
```

不得把active buffer回填成full `[2,1,D,W]`以兼容旧kernel。旧P/full-source接口保留为oracle/历史实现，S4 module
使用新schema version。

## 6. pass A补齐A26/A29

### 6.1 A29

Linear14反向产生site28 incoming coefficient后、应用ReLU28 transform前，直接pack `sign_a29`。这是existing explicit
ReLU边界，不增加coefficient materialization。

### 6.2 A26

residual11内部site25 coefficient必须复用D1C/D2B staged boundary：

```text
residual11 stage1 → A26 in existing arena scratch
  → pack sign_a26
  → residual11 stage2 → A24
```

不得在完整pass后额外执行`_recompute_a26`只为bitmap，也不得调用native Conv/CROWN。若existing staged boundary无法
在不改语义下导出A26，S4-1B STOP并回到D1C owner修复。

### 6.3 version纪律

六bitmap必须来自同一次coefficient pass与同一parameter state version。任何一张来自旧ordinal、旧α/β或不同stream
generation时，在effective graph launch前拒绝。

## 7. runtime对象

建议新增：

```text
boundflow/backends/tvm/asplos27_s4_coefficient_adjoints.py
boundflow/runtime/asplos27_s4_coefficient_adjoint_graph.py
```

`S4CoefficientAdjointLayoutV1`保存plan-derived slot/shape/offset/coefficient-action/fanout/sign dependency；
`PreparedS4CoefficientAdjointGraphV1`持有compiled module、one V arena、six views、six sign buffers和S4-1A
parameters；`S4CoefficientAdjointResultLeaseV1`只暴露ordinal/version/generation/ordered views/receipt。

## 8. structural receipt

至少记录：

```text
admission_hash / ordered_buffer_abi_hash / production_plan_hash
module/source/schedule hashes
evaluation_ordinal / parameter_state_version / sign_generation / value_generation

sign_bitmap_count=6
sign_bitmap_elements=55296
sign_bitmap_logical_bytes=55296
coefficient_adjoint_arena_count=1
coefficient_adjoint_slot_count=6
coefficient_adjoint_elements=37464
coefficient_adjoint_logical_bytes=149856
coefficient_action_sequence_hash / adjoint_action_sequence_hash
fanout_accumulation_provenance_hash
saved_dense_coefficient_count=0
full_alpha_repack_count=0
logical_stage_count=6
actual_kernel_launch_count / persistent_output_copy_count
prepare_dlpack_view_count / warm_dlpack_view_count=0
warm_python_tensor_construction_count=0
dynamic_output_allocation_count=0
fallback/eager/native_shadow=0
timing_recorded=false / performance_claimed=false
```

coefficient-adjoint arena必须与dense coefficient A分别命名和计数，禁止用“arena”总称隐藏二者。

## 9. correctness oracle

S4-1B0与S4-1B至少四方：

1. full provider-independent PyTorch CROWN autograd：六dense α/β求梯度后投影production ownership；
2. coefficient-action adjoint oracle：不调用candidate module，逐action重放VJP；
3. float64 no-autograd gradient formula，近零A位置直接比较gradient、禁止以除法构造V；
4. existing S2/R31B2交集：对已证明site比较局部值与最终gradient，证明复用未改旧语义。

每个site/fresh run记录shape/dtype/device、max abs/rel diff、finite和content hash。冻结门槛：float32及float64
comparison均`atol=rtol=2e-4`；但最终compressed gradient必须`max abs/rel <=2e-5`且sign exact。site19必须
显式关闭`0.0011564247542992234/9`反例，不能用value tolerance或其余五site通过替代。

至少five fresh process；input/state/sign/module identity逐run绑定。S4-1B不计时。

## 10. negative gates

至少覆盖：

1. sign bitmap少/多/slot错配；2. A26/A29来自旧version；3. bitmap非int8或值非0/1；
4. arena offset重叠/空洞/越界；5. slot顺序漂移；6. active α被full-source repack；
7. empty β错误传入；8. bound/alpha-map identity漂移；9. residual branch/Add顺序漂移；
10. pre28漏skip或pre23漏shortcut；11. Flatten/Gemm14 layout漂移；12. warm DLPack/Python view构造；
13. VM动态输出allocation；14. saved float32 coefficient跨stage；15. native/provider fallback；
16. lease未释放即重写arena；17. 全重签后修改bytes/copy/kernel/claim；18. timing/performance flag提前为true；
19. ordinary primal替换coefficient adjoint；20. residual fanout/accumulation VJP provenance漂移；
21. site19反例未关闭却标admitted；22. coefficient/adjoint action sequence hash错配。

## 11. 与S4-1C gradient的接口

S4-1C pass C在incoming coefficient仍在两个arena之一时立即发射：

```text
dalpha_i[d,k]
  = upstream[d] * incoming_A_i[d,feature(k)] * coefficient_adjoint_V_i[d,feature(k)]
```

并应用lower-direction/ambiguous/`A>=0`/feature ownership门禁。

active β的局部B4-B2公式为`-adjoint_relu * split_sign`；在full composition中`adjoint_relu`由对应
coefficient-program adjoint `V_i`承担，因此site31 emitter必须用B4-B2 sparse Linear、full PyTorch autograd与
coefficient-action adjoint三方确认，不能只凭普通primal类比实现。

S4-1B只交付values，不交付gradient；任何gradient数字仍标未验证。

## 12. 当前状态与提交顺序

只有S3 approved+closed、S4-0 validated、S4-1A validated且S4-1B0关闭site19反例后才能实现：

1. `test(math): close six-site coefficient-schedule adjoint reduction`；
2. `feat(compiler): add six-site coefficient-adjoint layout and sign manifest`；
3. `feat(tvm): lower coefficient adjoint replay with persistent outputs`；
4. `test(tvm): close S4-1B five-fresh coefficient-adjoint correctness`；
5. `docs: close S4-1B and open S4-1C emitters`。

当前保持：

```text
S3 exchange = ready_for_audit
S4-0/S4-1A/S4-1B0/S4-1B implementation = closed
S4 timing/performance = closed
```
