---
status: draft-requires-s4-1b0-ternary-endpoint-closure
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

# ASPLOS'27 S4-1B：六site selected-primal / coefficient-adjoint实施蓝图

## 0. 直接结论

S4-1B不是六个独立BoundOp wrapper，也不是保存六份dense coefficient A。第一轮只读预检曾发现旧二元
selected-primal在site19失败：

```text
production compressed α19 max abs diff = 0.0011564247542992234
gradient sign mismatch = 9
```

后续逐层tap已定位：失败来自input affine在`A==0`时错误选择lower，而provider的`abs(A)`零点次梯度对应
input center。改用三元`positive→lower / negative→upper / zero→center`后，site19误差降为
`4.2375177145e-08`且六site/active β全部sign exact。

因此规范owner仍定义为**CROWN coefficient schedule的精确VJP**，而下列selected-primal graph恢复为已经逐site
验证的优化lowering；无需另造一套全新DAG-adjoint runtime：

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
`BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_BOX_ENDPOINT_SUBGRADIENT_CLOSURE_2026_08_28.md`定义的S4-1B0门禁。
S2 selected Relax/cuDNN与R31B2 selected-ReLU TIR继续作为六site lowering的主要复用资产；S4以新symbol升级input
endpoint pack/select schema，zero分支从lower/upper派生center，不新增center tensor，仍禁止另写per-site Python executor。

## 1. 数学语义

规范定义不是“`V_i`必然等于普通primal `pre_i`”，而是：

```text
A_i = ReLU transform前incoming coefficient
T_i = A_i * selected_slope_i + beta_add_i
V_i = d lower / d T_i
```

只有逐site证明`V_i == selected pre_i`后，下面的selected-primal公式才可作为该site的优化lowering；三元endpoint
只读复核已经给出六site设计证据，production closure仍由S4-1B0 five-fresh artifact完成。

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

input endpoint必须表达`abs(A)`在零点的次梯度：

```text
selected_input = A_input > 0 ? input_lower
               : A_input < 0 ? input_upper
               : input_center
```

随后执行原始primal Conv/Add/Flatten/Gemm得到六个`pre_i`。active β先在coefficient pass中改变`A`，因此该图仍
消费同一ordinal/version的coefficient branch selectors。DAG fanout、residual accumulation与VJP provenance必须
进入receipt/oracle，但当前证据不再要求用一套独立coefficient-adjoint runtime替换该图。

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

selected-primal lowering需要一张三元endpoint selector和五张二元coefficient branch bitmap：

| bitmap | 选择对象 | elements/bytes | 当前状态 |
|---|---|---:|---|
| `endpoint_ainput_v2` | input lower/upper/center | 18,432 | **语义升级** |
| `sign_a18` | ReLU17 | 12,288 | 已有 |
| `sign_a20` | ReLU19 | 6,144 | 已有 |
| `sign_a24` | ReLU23 | 6,144 | 已有 |
| `sign_a26` | ReLU25 | 6,144 | **新增** |
| `sign_a29` | ReLU28 | 6,144 | **新增** |
| 合计 | — | **55,296** | 当前43,008 + 新增12,288 |

不需要`sign_a32`来生成pre31，因为pre31位于ReLU31之前；site31的incoming coefficient由pass C在gradient
emitter处直接使用。

`endpoint_ainput_v2`取值冻结为`+1 iff A>0`、`-1 iff A<0`、`0 iff A==0`；formal inventory为
`8689/9137/606`。其余五张bitmap仍为`1 iff coefficient >= 0 else 0`，因为ReLU显式`where`把zero归入lower
branch。selector/bitmap receipt必须绑定evaluation ordinal、parameter state version、β/split/history identity、
coefficient module hash、lower/upper与derived-center formula hash、pointer generation。总物理bytes不变。

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
7. 旧二元input endpoint在site19失败`1.156e-3/9 sign mismatch`；三元规则已只读闭合，但未形成formal artifact；
8. current `R31B2_PACK_AINPUT_SYMBOL`只能编码二元`>=0`，S4缺独立ternary pack/select和derived-center schema；
9. receipt尚未逐action绑定residual fanout/accumulate、bias与box concretization provenance。

### 4.3 为什么不直接扩展`effective_pre23`大kernel

该kernel为每个pre23输出元素局部重算pre19；它没有一个唯一可安全写出的pre19 producer。强行增加pre19 output会
产生重复写/race或需要额外全图同步。all-state本来就必须保存pre19，因此应把selected graph拆成具有明确stage
边界的单向图，而不是继续把早期中间值藏在巨型kernel内部。

## 5. 推荐实现：复用selected-primal graph，规范对照coefficient VJP

### 5.1 原因

S2 selected graph已经验证Conv/Residual局部语义、cuDNN+TIR混合、current stream、persistent output和S3修复后的
safe VM invocation；它继续作为主kernel/graph复用资产。S4-1B0补充`endpoint_ainput_v2`pack/select与center输入后，
扩展同一图到六slot。correctness oracle仍从Pass A typed coefficient action sequence定义VJP，显式核对seed、
Linear/Conv right、ReLU transform、β add、residual duplicate/accumulate、bias与box concretization。不得只做图自比。

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

上图描述期望物理dataflow。每个stage还必须绑定来源coefficient action与fanout provenance；E0必须绑定三元
endpoint schema、lower/upper identity、derived-center formula及zero inventory。logical stage不等于CUDA kernel；actual cuDNN/TIR/copy kernel
count由compiled module receipt披露。

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

input_endpoint_selector_schema=ternary-box-endpoint-v2
input_endpoint_positive/negative/zero_count=8689/9137/606
input_lower_hash / input_upper_hash / derived_center_formula_hash
selector_and_sign_buffer_count=6
selector_and_sign_elements=55296
selector_and_sign_logical_bytes=55296
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
2. coefficient-action VJP oracle：不调用candidate module，逐action重放或用independent taps闭合；
3. float64 no-autograd gradient formula，近零A位置直接比较gradient、禁止以除法构造V；
4. existing S2/R31B2交集：对已证明site比较局部值与最终gradient，证明复用未改旧语义。

每个site/fresh run记录shape/dtype/device、max abs/rel diff、finite和content hash。冻结门槛：float32及float64
comparison均`atol=rtol=2e-4`；但最终compressed gradient必须`max abs/rel <=2e-5`且sign exact。site19必须
显式证明二元规则复现`0.0011564247542992234/9`反例、三元规则关闭至`4.2375177145e-08/0`，不能用其余
五site通过替代。

至少five fresh process；input/state/sign/module identity逐run绑定。S4-1B不计时。

## 10. negative gates

至少覆盖：

1. selector/sign buffer少/多/slot错配；2. A26/A29来自旧version；3. endpoint selector非int8或值非-1/0/+1；
4. arena offset重叠/空洞/越界；5. slot顺序漂移；6. active α被full-source repack；
7. empty β错误传入；8. bound/alpha-map identity漂移；9. residual branch/Add顺序漂移；
10. pre28漏skip或pre23漏shortcut；11. Flatten/Gemm14 layout漂移；12. warm DLPack/Python view构造；
13. VM动态输出allocation；14. saved float32 coefficient跨stage；15. native/provider fallback；
16. lease未释放即重写arena；17. 全重签后修改bytes/copy/kernel/claim；18. timing/performance flag提前为true；
19. binary endpoint替换ternary endpoint；20. residual fanout/accumulation VJP provenance漂移；
21. zero count或derived-center formula漂移；22. site19 zero-subgradient反例未关闭却标admitted；
23. coefficient/VJP action sequence hash错配。

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

只有S3 approved+closed、S4-0 validated、S4-1A validated且S4-1B0关闭ternary endpoint后才能实现：

1. `test(math): close ternary box endpoint zero-subgradient`；
2. `feat(tvm): add ternary input endpoint pack and select`；
3. `feat(compiler): extend selected-primal graph to six persistent slots`；
4. `test(tvm): close S4-1B five-fresh selected/VJP correctness`；
5. `docs: close S4-1B and open S4-1C emitters`。

当前保持：

```text
S3 exchange = ready_for_audit
S4-0/S4-1A/S4-1B0/S4-1B implementation = closed
S4 timing/performance = closed
```
