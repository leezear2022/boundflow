---
status: implementation-construction-ready-gate-closed-v1
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-1c-compressed-gradient-terminal-la-construction
stage: s04
depends-on: s3-external-approval-and-s4-0-s4-1a-s4-1b0-s4-1b-closure
execution-authority: false
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1C compressed gradient 与 terminal lA 实施施工包

## 0. 直接结论

本施工包把此前“六个 dα、一个 dβ、六个 terminal lA copy”的公式蓝图收束成可逐文件实现、可机械拒绝
错误顺序的 production 合同。本轮没有修改 production 代码、没有做性能计时，也没有升级 correctness claim。

冻结结论如下：

1. Pass C 的 coefficient recompute 只需走到 A18；不执行 ReLU17、Conv0-right 或 box concretize；
2. 非 terminal evaluation 固定为 **17 个 logical launch/action**：10 个 coefficient 动作、6 个 dα emitter、
   1 个 active dβ emitter；
3. terminal ordinal 9 固定为 **23 个 logical launch/action**：在上述17项中插入6个 pre-transform lA copy；
4. site31 是唯一双 emitter site，严格顺序为
   `emit dα31 → emit dβ31 → copy A31 to V31/lA31 → ReLU31 transform`；
5. 其余site严格为`emit dα → terminal copy（仅ordinal 9）→ transform或arena reuse`；
6. 六个 terminal lA 不新增physical storage：它们在emitter消费V后原地覆盖37,464-element V arena；
7. 六个V slot是同一storage的互不重叠、无空洞interval；GPU lifecycle probe确认warm dynamic
   allocated/reserved delta=`0/0`；
8. S4-1A/B/C argument DLPack descriptor总数仍为110；terminal copy复用现有A/V descriptor，不再加DLPack；
9. result-facing普通Torch view相对argument DLPack额外增加**6个**：五个Conv-shaped terminal view加一个
   `[D,1]` lower view。site31 emitter view本身已是`[D,1,100]`，可直接作为terminal shaped view，不新增第七个；
10. runtime O(1) receipt与formal raw sidecar必须分开：production warm path不得为了content hash引入D2H、同步或
    per-emitter Python检查；
11. 以上均为待实现合同，不是已实现性能或正确性结论；S3外审与S4前序门禁关闭前，S4-1C code仍关闭。

## 1. 为什么现在需要施工包

旧蓝图已经把数学公式、metadata、七个gradient symbol和V/lA arena方向写清，但仍存在五个实现歧义：

- “7 launches”只描述emitter局部，无法回答完整Pass C到底发射多少次；
- terminal copy的插入点没有逐ordinal展开，site31可能被错误地在dβ前覆盖；
- terminal view、DLPack descriptor与physical storage三种口径容易混算；
- native handoff当前clone六个独立tensor，而S4目标是一个arena上的one-shot lease；
- runtime receipt若照搬formal content hash，会把同步和D2H重新塞回热路径。

因此本施工包不新增solver IR，也不另造execution framework；它把现有
`coefficient pass + selected-value V + compressed emitter + one-shot handoff`资产组合成一个严格phase machine。

## 2. 复用资产与明确不复活的路线

### 2.1 直接复用

| 资产 | 本阶段用途 | 不拥有的职责 |
|---|---|---|
| S4-1A prepared parameter/gradient buffers | dα/dβ caller-owned output、upstream、state identity | 不执行Pass C |
| S4-1B coefficient pass | 重算A31/A29/A26/A24/A20/A18 | 不拥有gradient或terminal lease |
| S4-1B selected graph | 生成六个coefficient-program adjoint V | 不覆盖V为terminal lA |
| B4-B2 sparse Linear | site31 α/β数学oracle | 不作为六site production wrapper |
| B4-B2 dense/sparse Conv | Conv site局部VJP oracle | 不串联per-site autograd |
| B4-A terminal handoff | shape/topology/one-shot消费合同 | 不继续clone六份dense tensor |
| R31B1/B2 bounded coefficient arena | A与residual scratch物理owner | 不保存跨层dense A |
| RVIR-v4 state/trajectory | evaluation ordinal与state version | 不拥有compiler内部schedule |

### 2.2 禁止复活

- B4-C2跨层保存dense A/autograd history；
- 六个B4-B2 Python wrapper串联；
- terminal ordinal额外执行第11次CROWN；
- 把empty β变成五个physical zero tensor；
- 每个site重新建立DLPack；
- 在warm path逐tensor做content hash或CPU读取；
- 先覆盖V再尝试恢复gradient；
- 以CUDA Graph掩盖基本phase或allocation错误。

## 3. 数学与方向所有权

### 3.1 dα

对site `i`、domain `d`、compressed ordinal `k`映射feature `f=index_i[k]`：

```text
dalpha_i[d,k]
  = sum_s upstream[d,s] * A_i[d,s,f] * V_i[d,s,f]
```

仅在以下条件全部成立时输出上述值：

- lower direction；
- `lower_i[d,f] < 0 < upper_i[d,f]`；
- `A_i[d,s,f] >= 0`，即lower slope被当前coefficient选择；
- active alpha在闭区间`[0,1]`；
- compressed index合法、唯一、严格升序；
- A/V/bounds/alpha/upstream全部finite且lower<=upper。

合法但非ambiguous或A<0时输出+0.0。非法输入输出canonical qNaN bits=`0x7fc00000`；全局index
duplicate/unsorted必须在launch前拒绝，kernel的safe-read只是最后一道防OOB屏障。

### 3.2 dβ

当前唯一active β位于site31：

```text
location = [17,17,31,17,17,31]
sign     = [ 1, 1, 1,-1,-1,-1]
Q        = 1 per domain
```

公式为：

```text
dbeta31[d,q]
  = sum_s upstream[d,s] * (-V31[d,s,location[d,q]] * sign[d,q])
```

β location用runtime int32 metadata；sign保持int8，到TIR内部cast。它们来自split/history state，因此不能像旧
B4-B2 micro template那样无条件固化为编译常量。module可跨合法state复用，PlanInstance/receipt必须绑定metadata
content identity和effective version。

### 3.3 terminal lA

terminal lA是每个ReLU transform之前的incoming coefficient A，不是transform后的T，也不是V：

```text
lA_i[d,s,*feature] = A_i[d,s,*feature] before ReLU transform
```

当前`S=1`，但result ABI必须保留spec轴。六site元素数：

```text
site17 = 12,288
site19 =  6,144
site23 =  6,144
site25 =  6,144
site28 =  6,144
site31 =    600
total  = 37,464 float32 = 149,856 B
```

## 4. Pass C的精确动作序列

### 4.1 为什么coefficient部分只有10项

Pass C只需在每个split ReLU入口获得A并发射compressed gradient。最后一个site17发射完成后，没有后续site需要
更早层coefficient，因此不再执行ReLU17、Conv0-right、Ainput pack或concretize。

10个coefficient动作是：

```text
seed
Linear16-right
ReLU31 coefficient
Linear14-right
ReLU28 coefficient
residual11 stage1
residual11 stage2
ReLU23 coefficient
residual6 stage1
residual6 stage2
```

加6个dα与1个dβ，非terminal总计`10+6+1=17`。

### 4.2 非terminal 17-action表

| ordinal | action | produced/consumed state | 下一合法动作 |
|---:|---|---|---|
| 0 | seed | objective→A32 | Linear16 |
| 1 | Linear16-right | A32→A31 | emit dα31 |
| 2 | emit dα31 | read A31/V31 | emit dβ31 |
| 3 | emit dβ31 | read V31/location/sign | ReLU31 |
| 4 | ReLU31 coefficient | A31→T31 | Linear14 |
| 5 | Linear14-right | T31→A29 | emit dα28 |
| 6 | emit dα28 | read A29/V28 | ReLU28 |
| 7 | ReLU28 coefficient | A29→T28 | residual11 stage1 |
| 8 | residual11 stage1 | T28→A26 scratch | emit dα25 |
| 9 | emit dα25 | read A26/V25 | residual11 stage2 |
| 10 | residual11 stage2 | T28+A26→A24 | emit dα23 |
| 11 | emit dα23 | read A24/V23 | ReLU23 |
| 12 | ReLU23 coefficient | A24→T23 | residual6 stage1 |
| 13 | residual6 stage1 | T23→A20 scratch | emit dα19 |
| 14 | emit dα19 | read A20/V19 | residual6 stage2 |
| 15 | residual6 stage2 | T23+A20→A18 | emit dα17 |
| 16 | emit dα17 | read A18/V17 | close evaluation |

非terminal禁止任何lA copy，V arena继续保持`COEFFICIENT_ADJOINT`语义直至evaluation关闭。

### 4.3 terminal 23-action表

| ordinal | action | phase effect |
|---:|---|---|
| 0 | seed | objective→A32 |
| 1 | Linear16-right | A31 ready |
| 2 | emit dα31 | first V31 reader consumed |
| 3 | emit dβ31 | last V31 reader consumed |
| 4 | copy terminal lA31 | V31 slot becomes TERMINAL_LA31 |
| 5 | ReLU31 coefficient | transform A31 |
| 6 | Linear14-right | A29 ready |
| 7 | emit dα28 | V28 reader consumed |
| 8 | copy terminal lA28 | V28 slot becomes TERMINAL_LA28 |
| 9 | ReLU28 coefficient | transform A29 |
| 10 | residual11 stage1 | A26 scratch ready |
| 11 | emit dα25 | V25 reader consumed |
| 12 | copy terminal lA25 | V25 slot becomes TERMINAL_LA25 |
| 13 | residual11 stage2 | A24 ready |
| 14 | emit dα23 | V23 reader consumed |
| 15 | copy terminal lA23 | V23 slot becomes TERMINAL_LA23 |
| 16 | ReLU23 coefficient | transform A24 |
| 17 | residual6 stage1 | A20 scratch ready |
| 18 | emit dα19 | V19 reader consumed |
| 19 | copy terminal lA19 | V19 slot becomes TERMINAL_LA19 |
| 20 | residual6 stage2 | A18 ready |
| 21 | emit dα17 | V17 reader consumed |
| 22 | copy terminal lA17 | V17 slot becomes TERMINAL_LA17; close |

site25和site19位于residual stage1/stage2边界，没有独立ReLU transform，但仍必须先emit、再copy、最后允许
scratch被stage2复用。

### 4.4 mechanical state validator

实现必须把上述表编码为typed action inventory，而不是只靠注释。validator至少维护：

```text
coefficient_ready_sites
dalpha_emitted_sites
dbeta_emitted_sites
terminal_copy_sites
current_action_ordinal
evaluation_mode
stream_id
generation
```

正例要求：

```text
nonterminal actions = 17, errors = 0
terminal actions    = 23, errors = 0
dalpha sites        = {17,19,23,25,28,31}
dbeta sites         = {31}
terminal copies     = empty or all six, never a subset
```

至少三个结构篡改必须稳定拒绝：copy31与dβ31交换、ReLU28与copy28交换、删除dβ31。

## 5. TIR module与导出symbol

### 5.1 gradient module

建议文件：

```text
boundflow/backends/tvm/asplos27_s4_compressed_gradient.py
boundflow/runtime/asplos27_s4_gradient_emitters.py
tests/test_asplos27_s4_compressed_gradient.py
tests/test_asplos27_s4_gradient_phase.py
```

gradient module固定7个导出symbol：

```text
boundflow_s4_emit_dalpha_site17
boundflow_s4_emit_dalpha_site19
boundflow_s4_emit_dalpha_site23
boundflow_s4_emit_dalpha_site25
boundflow_s4_emit_dalpha_site28
boundflow_s4_emit_dalpha_site31
boundflow_s4_emit_dbeta_site31
```

六个dα来自一个layout-parameterized semantic template的六个实例；site31 dβ是第二个semantic template。第一版允许
7个launch，correctness closure前不融合。

### 5.2 terminal copy module

terminal copy可放在同一TVM module，但receipt必须作为独立semantic family：

```text
boundflow_s4_copy_terminal_la_site17
boundflow_s4_copy_terminal_la_site19
boundflow_s4_copy_terminal_la_site23
boundflow_s4_copy_terminal_la_site25
boundflow_s4_copy_terminal_la_site28
boundflow_s4_copy_terminal_la_site31
```

所以完整module可有13个导出symbol：7 gradient + 6 copy。copy symbol只是typed device copy，不增加workspace、
output storage或argument descriptor。后续若profile证明值得与emitter融合，必须产生新template/schedule/module receipt，
不得静默改变首版合同。

### 5.3 module cache key

至少绑定：

```text
schema_version
site_id
kind = dalpha | dbeta | terminal_copy
D/S/F/W/Q
dtype/device/compute_capability
schedule threads/block
template hash
scheduled TIR hash
device source hash
TVM/tvm-ffi revision
canonical qNaN policy
clamp endpoint policy
safe-index policy
```

runtime metadata content不进入module cache key，但进入PreparedInstance identity；否则每次β location或alpha index变化都
重新编译，违背state/compile分层。

## 6. ABI与descriptor总账

### 6.1 dα ABI

每site 8次argument occurrence：

```text
A[D,S,F]
V[D,S,F]
lower[D,F]
upper[D,F]
active_alpha[D,W]
alpha_indices[W] int32
upstream[D,S]
compressed_dalpha[D,W]
```

六site共48次occurrence。

### 6.2 dβ ABI

```text
V31[D,S,F]
beta_location[D,Q] int32
beta_sign[D,Q] int8
upstream[D,S]
compressed_dbeta[D,Q]
```

共5次occurrence；因此gradient emitter总occurrence=`48+5=53`。

### 6.3 unique emitter views

```text
A 6 + V 6 + bounds 12 + alpha 6 + alpha_indices 6
+ upstream 1 + dα 6 + beta_location/sign/dβ 3 = 46
```

与S4-1A base 16重叠14；与S4-1B flattened bound另重叠12。因此：

```text
S4-1B prepared descriptor union = 90
S4-1C new over S4-1B          = 46 - 14 - 12 = 20
S4-1A/B/C full union          = 110
```

terminal copy的A/V就是上述descriptor，新增0。不得把13个symbol、53次argument occurrence、46个emitter
unique view、110个完整prepared descriptor混成同一计数。

### 6.4 result-facing普通Torch views

V emitter view形状为`[D,1,F]`。terminal result形状为：

```text
17: [6,1,8,16,16]
19: [6,1,16,8,8]
23: [6,1,16,8,8]
25: [6,1,16,8,8]
28: [6,1,16,8,8]
31: [6,1,100]
```

site31与emitter view形状一致，可复用；前五个需要prepare-time普通Torch reshape view。lower physical buffer为`[6]`，
result lease另需要`[6,1]`普通view。因此相对argument DLPack额外普通view=`5+1=6`，不是旧文档模糊写法的7。

这些view均新增0 storage、0 warm allocation、0 warm DLPack。

## 7. V/lA arena与phase-safe alias

### 7.1 固定布局

| site | element interval | emitter shape | terminal shape |
|---:|---:|---|---|
| 17 | `[0,12288)` | `[6,1,2048]` | `[6,1,8,16,16]` |
| 19 | `[12288,18432)` | `[6,1,1024]` | `[6,1,16,8,8]` |
| 23 | `[18432,24576)` | `[6,1,1024]` | `[6,1,16,8,8]` |
| 25 | `[24576,30720)` | `[6,1,1024]` | `[6,1,16,8,8]` |
| 28 | `[30720,36864)` | `[6,1,1024]` | `[6,1,16,8,8]` |
| 31 | `[36864,37464)` | `[6,1,100]` | `[6,1,100]` |

六interval互不重叠、无空洞，完整覆盖`[0,37464)`。所有view的storage token相同、data pointer按offset不同。

### 7.2 phase transition

每个slot：

```text
VALUE_EMPTY
→ VALUE_READY(generation, selected_graph_hash)
→ GRADIENT_READ_ENQUEUED(stream, emitter_receipt)
→ GRADIENT_READ_COMPLETE_BY_SAME_STREAM_ORDER
→ TERMINAL_LA_WRITE_ENQUEUED（仅terminal）
→ TERMINAL_LA_READY(generation, topology, spec_axis)
→ LEASED_ONCE
→ RELEASED
```

nonterminal在gradient read完成后直接进入evaluation close，不允许把slot伪标成terminal lA。

### 7.3 site31覆盖反例

合成GPU probe把V31填为逐元素不同值、A31填为常数。正确copy前dβ为：

```text
[-1017,-1117,-1231,+1317,+1417,+1531]
```

若先copy A31再发射dβ，错误结果变成：

```text
[-6,-6,-6,+6,+6,+6]
```

因此“dα后即可copy”的一般规则在site31不成立；必须等待dα和dβ两个reader都消费。

## 8. runtime receipt与formal sidecar分层

### 8.1 warm runtime O(1) receipt

每次evaluation只允许记录：

```text
prepared_id / module_id
evaluation_generation / state_version
mode = nonterminal | terminal
action_count = 17 | 23
dalpha_launch_count = 6
dbeta_launch_count = 1
terminal_copy_count = 0 | 6
stream_id / device ordinal
fallback/eager/native_shadow = 0
warm_dlpack_construction = 0
dynamic_output_allocation = 0
phase completion bitmask
```

静态shape、pointer、module hash、metadata hash在prepare/admission绑定，不在每个launch重算。

### 8.2 formal raw sidecar

formal five-fresh需要能同时审计V与terminal lA，但terminal run最终已用lA覆盖V。因此formal runner可在Pass C前、计时
区间外抓取或hash V raw，再执行Pass C并抓取lA。必须明确：

- sidecar不进入production warm timing；
- V hash发生在覆盖前；
- lA hash发生在copy完成后；
- raw payload绑定state/module/schedule/evaluation identity；
- replay从raw重算公式、inventory、phase和summary；
- runtime receipt不得谎称自己在覆盖后仍content-hash了V。

## 9. native handoff迁移

现有B4-A handoff验证了六site topology、shape与one-shot消费，但producer会clone六个独立coefficient tensor。S4迁移为：

```text
Prepared V/lA arena
  → terminal Pass C six in-place copies
  → NativeTerminalLowerAdjointLeaseS4V1
  → KFSB consumer
  → release all six slot capabilities
```

冻结规则：

1. output顺序使用plan site order`17,19,23,25,28,31`，不是pass C reverse execution order；
2. lineage仍绑定native preactivation、provider activation/preactivation、producer ordinal/output、feature shape；
3. terminal shaped view必须保留spec axis；
4. lease只能消费一次；
5. KFSB结束立即release，不能把lA活性延伸到queue；
6. assembly不得clone payload；如外部consumer强制需要owned tensor，必须显式计入copy/bytes并重新过memory gate；
7. lower result与six lA来自同一terminal evaluation generation。

## 10. metadata与memory ledger

S4-1C额外prepare metadata：

```text
alpha indices: 708 * int32 = 2,832 B
beta location:   6 * int32 =    24 B
beta sign:       6 * int8  =     6 B
total                         2,862 B
```

gradient outputs与metadata已包含在S4-1D corrected subtotal中。terminal lA复用V arena，因此additional physical
storage=`0 B`。两个residual stage scratch是两个coefficient storage的offset slice，additional physical
storage也=`0 B`。

当前已修正known logical subtotal：

```text
S4-1D = 389,574 B
S4-2  = 491,774 B
S4-3  = 559,838 B
```

若selected-input alias条件失败，S4-1D增加73,728 B至463,302 B。上述是设计账，不是CUDA peak实测claim。

## 11. CUDA lifecycle诊断结果

只读construction probe在RTX 4060 Laptop GPU上得到：

```text
V/lA physical storage count = 1
slot count                  = 6
elements                    = 37,464
interval non-overlap        = true
full cover                  = true
DLPack pointer roundtrip     = 6/6 exact
non-default stream          = true
inside tvm_ffi.use_torch_stream:
  torch stream == TVM FFI stream
context exit restores prior FFI stream
warm lifecycle allocated delta = 0 B
warm lifecycle reserved delta  = 0 B
```

该probe只证明arena/view/stream/lifecycle合同可行，不是S4 production correctness或speedup证据。

现有native handoff fixture另独立确认：

```text
lower shape = [6,1]
terminal order = 17,19,23,25,28,31
terminal total = 37,464
all contiguous = true
```

native当前六个lA是六份独立storage；这正是S4必须迁移到单arena lease的差异，而不是可以提前声称已完成的结果。

## 12. fail-closed reason设计

建议新增或复用以下稳定reason：

```text
S4_GRADIENT_MODULE_IDENTITY_MISMATCH
S4_GRADIENT_TEMPLATE_HASH_MISMATCH
S4_GRADIENT_SCHEDULE_HASH_MISMATCH
S4_GRADIENT_DEVICE_SOURCE_HASH_MISMATCH
S4_GRADIENT_STATE_VERSION_MISMATCH
S4_COEFFICIENT_SITE_NOT_READY
S4_VALUE_SITE_NOT_READY
S4_DALPHA_DUPLICATE_EMIT
S4_DBETA_SITE_OR_INVENTORY_MISMATCH
S4_EMPTY_BETA_PHYSICAL_LAUNCH
S4_ALPHA_INDEX_INVALID
S4_BETA_LOCATION_INVALID
S4_BETA_SIGN_INVALID
S4_GRADIENT_NONFINITE_INPUT
S4_GRADIENT_OUTPUT_POINTER_DRIFT
S4_GRADIENT_CROSS_STREAM_USE
S4_TERMINAL_COPY_IN_NONTERMINAL
S4_TERMINAL_COPY_BEFORE_ALL_READERS
S4_TERMINAL_COPY_DUPLICATE
S4_TERMINAL_COPY_POST_TRANSFORM
S4_TERMINAL_LA_SPEC_AXIS_MISMATCH
S4_TERMINAL_LA_INVENTORY_INCOMPLETE
S4_TERMINAL_LA_LEASE_REUSED
S4_PASS_C_ACTION_COUNT_MISMATCH
S4_PASS_C_ACTION_ORDER_MISMATCH
S4_WARM_DLPACK_CONSTRUCTION_FORBIDDEN
S4_DYNAMIC_GRADIENT_ALLOCATION_FORBIDDEN
S4_DENSE_GRADIENT_ESCAPE
S4_NATIVE_SHADOW_OR_FALLBACK
S4_CLAIM_FLAG_TRUE_BEFORE_FORMAL
```

## 13. negative test matrix

至少覆盖：

1. A/V来自不同evaluation generation；
2. A/V site错配；
3. A/V shape或spec axis错配；
4. alpha index越界；
5. alpha index重复；
6. alpha index乱序；
7. active alpha越界；
8. alpha clamp endpoint 0/1被错误拒绝；
9. stable ReLU产生非零dα；
10. A<0产生lower dα；
11. A NaN被gate静默成0；
12. V Inf被gate静默成0；
13. lower>upper；
14. invalid index在predicate前OOB读取；
15. beta location越界；
16. beta sign不是±1；
17. beta location/sign state hash漂移；
18. empty β触发physical launch；
19. site31缺dβ；
20. site31重复dβ；
21. site31在dβ前copy；
22. site31在dα前copy；
23. site28在dα前copy；
24. site25在stage1 producer前emit；
25. site19在stage1 producer前emit；
26. ReLU31在copy前执行；
27. ReLU28在copy前执行；
28. ReLU23在copy前执行；
29. residual stage2在copy前复用scratch；
30. nonterminal出现任意copy；
31. terminal缺任一copy；
32. terminal重复任一copy；
33. terminal使用post-transform coefficient；
34. terminal output丢spec axis；
35. terminal result顺序使用reverse pass order；
36. lease消费两次；
37. KFSB后lease未release；
38. V/lA interval overlap；
39. V/lA interval有空洞；
40. V/lA出现多于一个physical storage；
41. terminal copy新增output storage；
42. terminal copy新增DLPack descriptor；
43. warm建立DLPack；
44. warm动态分配gradient；
45. cross-stream reader/copy；
46. TVM FFI stream未绑定当前Torch stream；
47. context退出后FFI stream未恢复；
48. runtime receipt做D2H content hash；
49. formal V hash发生在覆盖后；
50. full re-sign后篡改17/23 action count；
51. full re-sign后篡改module/device source；
52. full re-sign后篡改metadata content；
53. 13 symbols伪称one kernel；
54. 53 occurrence伪称46 descriptor；
55. 46 emitter views伪称110 full union；
56. result普通view计入argument DLPack；
57. result普通view计入physical storage；
58. metadata写旧2,880 B；
59. terminal lA再跑第11次CROWN；
60. provider/native shadow或fallback；
61. 保存跨层dense A或dense gradient；
62. correctness closure前performance flag=true。

## 14. positive closure设计

S4-1C未来开放后，建议拆成两个formal mode，各五个fresh process：

### 14.1 nonterminal mode

- 17-action exact；
- six dα + one active dβ；
- terminal copy=0；
- 与production native autograd、full PyTorch autograd、coefficient adjoint replay和float64公式比较；
- max abs/rel diff `<=2e-5`、sign exact；
- empty β token exact；
- warm allocation/DLPack/fallback为0。

### 14.2 terminal mode

- 23-action exact；
- gradient与nonterminal同语义；
- six terminal lA key/order/shape/value exact；
- terminal total37,464；
- one-shot handoff count=1、rerun=0；
- V raw在覆盖前形成formal sidecar；
- lower与lA绑定同一terminal generation；
- KFSB消费后lease release。

### 14.3 artifact

raw-first、manifest先写、partial拒绝resume；replay不得importproduction summary。至少重算：

```text
module/template/schedule/device-source identity
17/23 action sequence
six dα / one dβ numerical formula
index/location/sign legality
V pre-overwrite inventory
terminal lA post-copy inventory
110 descriptor projection
storage intervals
warm allocation/view counters
claim flags
```

全重签tamper必须覆盖结构、数值、phase、storage、receipt与claim。

## 15. 逐文件提交计划

门禁开放后只允许以下顺序：

```text
feat(tvm): add seven-symbol compressed gradient module
test(tvm): close isolated gradient math and poison gates
feat(runtime): add typed 17-action pass-c driver
test(runtime): close nonterminal pass-c phase and descriptor gates
feat(tvm): add six terminal-lA copy symbols
feat(runtime): add typed 23-action terminal phase and one-shot arena lease
test(runtime): close terminal ordering alias and handoff gates
test(formal): run five-fresh nonterminal and terminal closures
docs: close S4-1C and only then open S4-1D
```

每个提交一个logical change。S4-1C关闭前禁止Adam、10/9 trajectory、same-solver timing和性能claim。

## 16. construction manifest

```json
{
  "schema": "boundflow.asplos27-s4-1c-construction/v1",
  "scope": {
    "execution_authority": false,
    "code_change_open": false,
    "timing_open": false,
    "performance_claimed": false
  },
  "pass_c": {
    "coefficient_action_count": 10,
    "dalpha_launch_count": 6,
    "dbeta_launch_count": 1,
    "nonterminal_action_count": 17,
    "terminal_copy_count": 6,
    "terminal_action_count": 23,
    "reverse_sites": [31, 28, 25, 23, 19, 17]
  },
  "module": {
    "gradient_symbol_count": 7,
    "terminal_copy_symbol_count": 6,
    "total_symbol_count": 13
  },
  "descriptor": {
    "gradient_argument_occurrence": 53,
    "gradient_unique_view": 46,
    "s4_1abc_argument_union": 110,
    "terminal_copy_new_argument_view": 0,
    "result_extra_ordinary_torch_view": 6
  },
  "arena": {
    "storage_count": 1,
    "slot_count": 6,
    "element_count": 37464,
    "byte_count": 149856,
    "terminal_additional_physical_bytes": 0
  },
  "metadata_bytes": 2862,
  "memory_known_logical_bytes": {
    "s4_1d": 389574,
    "s4_2": 491774,
    "s4_3": 559838
  }
}
```

canonical JSON SHA256：

```text
ad8ea91c39419cbfef0cf3eaa8db7fc339e54798daecf67ca6d97254a9755b93
```

## 17. 当前门禁

```text
S3 exchange = ready_for_audit
S4 production code = closed
S4-1C implementation = closed
S4 timing/performance = closed
next executable repository action = external audit S3
next design-only action = refresh S4 design audit handoff with this package
```

本施工包只减少未来实现歧义；它不绕过当前DocOps next，也不把S4设计诊断误写成已验证功能。
