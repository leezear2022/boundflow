---
status: implementation-ready-gate-closed-v2
date: 2026-08-28
type: selector-gradient-tir-arena-readiness
topic: boundflow
slug: asplos27-s4-1bc-selector-gradient-tir-implementation-readiness
stage: s04
depends-on: asplos27-s4-1b0-ternary-tir-abi-implementation-readiness
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B/1C：六selector、七gradient TIR与arena phase实施就绪合同

## 0. 结论

S4-1B/1C 已从“六site公式蓝图”压到可以逐文件实现的后端合同。本轮没有写production代码、没有计时，也没有
升级correctness claim。只读源码、真实production layout和三组CUDA探针得到以下修正：

1. nonfinite fail-closed不能只覆盖三元Ainput；另外五张二元ReLU branch selector也必须保留`-128` invalid
   sentinel，否则NaN会因`A>=0`为false而静默选择upper branch；
2. 所有runtime feature index/location必须先clamp到安全读地址，再以合法性谓词决定输出canonical NaN，不能让
   tamper在validator失误时触发越界CUDA读；
3. 六dα+一dβ第一版应明确披露7个symbol/launch、53次参数出现、emitter所见46个unique view；与S4-1A
   base 16重叠14项，因此prepared owner新增32个view，总计48；
4. β sign从provider split/history得到，物理dtype应保持int8并在TIR内cast；不需要旧蓝图的float32 sign copy，
   metadata总量由2,880 B修正为2,862 B；
5. V/terminal-lA的37,464-element arena六slot互不重叠且只占一个storage；reverse pass C可以在同一stream按
   `emitter read A/V → terminal copy pre-transform A → transform/reuse A`安全复用slot；
6. 真实A18/A20/A24/A26/A29/Ainput的positive/negative/zero inventory已从existing staged pass提取，证明A26/A20
   可直接来自两个persistent residual scratch，不需要新增dense A；
7. 隔离七symbol CUDA/TIR对独立PyTorch reference逐位exact，float64最大差`2.3576e-7`，并能把A/V/bound/
   α/upstream/index异常poison为NaN。

这仍只是implementation-readiness evidence。production closure必须在S3外审批准后按S4-0→1A→1B0→1B→1C
逐级执行five-fresh、raw/replay/tamper。

## 1. 当前真实layout

冻结plan hash：

```text
39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f
```

| site | F | W | active α elements `[6,W]` | alpha index hash |
|---:|---:|---:|---:|---|
| 17 | 2,048 | 164 | 984 | `cbf2e55f...a2610` |
| 19 | 1,024 | 132 | 792 | `d6427ba7...4b891` |
| 23 | 1,024 | 121 | 726 | `61931ead...647b2` |
| 25 | 1,024 | 86 | 516 | `f9314465...34f19` |
| 28 | 1,024 | 178 | 1,068 | `c2354a49...d8fbd` |
| 31 | 100 | 27 | 162 | `d9d56525...54642` |
| 合计 | — | 708 indices | **4,248** | — |

所有index严格升序、唯一、在`[0,F)`内。formal α全在`[0,1]`且真实存在0/1 endpoint，因此emitter不能把
clamp endpoint误写成开区间。

唯一active β为site31：

```text
location = [17,17,31,17,17,31]
sign     = [ 1, 1, 1,-1,-1,-1]
shape    = [6,1]
```

其余五site是typed empty token，不能建立physical β tensor、location/sign或launch。

## 2. 六张selector的统一fail-closed规则

### 2.1 合法值域

| selector | legal | invalid | zero语义 |
|---|---|---:|---|
| `endpoint_ainput_v2` | `-1/0/+1` | `-128` | center |
| `sign_a18/a20/a24/a26/a29` | `0/1` | `-128` | `A>=0`的lower branch |

三元endpoint schema保持：

```text
boundflow.asplos27-s4-ternary-endpoint/v1
```

`endpoint_ainput_v2`只是buffer semantic name，不是另一个schema version。禁止在receipt中写一个并不存在的
`ternary-box-endpoint-v2`后让审计方误以为有两套schema。

### 2.2 binary pack

五个binary pack都必须使用float32 exponent位检查：

```text
bits = reinterpret_uint32(A)
nonfinite = (bits & 0x7f800000) == 0x7f800000
selector = nonfinite ? int8(-128) : (A >= +0.0f ? int8(1) : int8(0))
```

selected-ReLU读取时必须显式：

```text
selector == 1 -> lower slope
selector == 0 -> upper slope
otherwise     -> canonical qNaN bits=0x7fc00000
```

不能用`selector != 0`，因为它会把`-128`当成lower branch。

### 2.3 storage账

```text
endpoint Ainput  = 18,432 B
binary A18      = 12,288 B
binary A20/A24/A26/A29 = 4 × 6,144 B
S4 total        = 55,296 B
```

相对current R31B2的43,008 B，S4因A26/A29新增**12,288 B**。只有derived center新增0 B；不能把
“没有center tensor”误写成“selector总bytes不变”。

## 3. 真实coefficient selector inventory

用existing D2B staged coefficient-sign pass，只在transform前clone diagnostic tap；A26/A20直接读取两个stage1
scratch。该探针只执行coefficient-sign scope，因此没有错误调用要求完整backward的D2B receipt。

| value | elements | positive | negative | zero | nonfinite | `A>=0` one |
|---|---:|---:|---:|---:|---:|---:|
| A18 | 12,288 | 6,070 | 5,746 | 472 | 0 | 6,542 |
| A20 | 6,144 | 3,037 | 3,107 | 0 | 0 | 3,037 |
| A24 | 6,144 | 3,049 | 3,095 | 0 | 0 | 3,049 |
| A26 | 6,144 | 2,977 | 3,005 | 162 | 0 | 3,139 |
| A29 | 6,144 | 2,876 | 3,268 | 0 | 0 | 2,876 |
| Ainput | 18,432 | 8,689 | 9,137 | 606 | 0 | 9,295 |

existing A18/A20/A24/Ainput v1 bitmap与`A>=0`逐位一致。S4不修改旧bitmap；新S4 symbol加入invalid sentinel，
formal正常数据仍必须产生上表相同legal分类。

捕获点：

```text
A29    Linear14-right之后、ReLU28 transform之前
A26    residual11 stage1 persistent scratch
A24    residual11 stage2之后、ReLU23 transform之前
A20    residual6 stage1 persistent scratch
A18    residual6 stage2之后、ReLU17 transform之前
Ainput Conv0-right之后的final coefficient arena
```

scope计数为4 staged residual launch、8个其余B1 launch、2 scratch storage、persistent dense A=false。

## 4. 七个gradient TIR symbol

### 4.1 symbols

```text
boundflow_s4_emit_dalpha_site17
boundflow_s4_emit_dalpha_site19
boundflow_s4_emit_dalpha_site23
boundflow_s4_emit_dalpha_site25
boundflow_s4_emit_dalpha_site28
boundflow_s4_emit_dalpha_site31
boundflow_s4_emit_dbeta_site31
```

schema：

```text
boundflow.asplos27-s4-compressed-gradient-emitter/v1
```

第一版一个module含7个symbol；不是七个Python wrapper，也不是一个伪称的单kernel。

### 4.2 dα ABI

```text
incoming_A[D,S,F] float32
adjoint_V[D,S,F] float32
lower[D,F] float32
upper[D,F] float32
active_alpha[D,W] float32
alpha_indices[W] int32
upstream[D,S] float32
compressed_dalpha[D,W] float32 caller-owned
```

每个runtime index先执行`safe_f=clamp(raw_f,0,F-1)`，然后读取safe_f并建立：

```text
valid = raw_f in [0,F)
     && all inputs finite by IEEE exponent bits
     && lower <= upper
     && alpha in [0,1]
```

若`valid=false`，output为canonical qNaN；若valid但非ambiguous或`A<0`，output为float32 zero；否则：

```text
dalpha[d,k] = sum_s upstream[d,s] * A[d,s,f] * V[d,s,f]
```

safe-read不是接受坏index；它只避免在fail-closed输出之前发生越界CUDA访问。duplicate/unsorted index仍由
prepare/admission validator在launch前拒绝，kernel无法局部判断全局duplicate。

### 4.3 dβ ABI

```text
adjoint_V[D,S,F] float32
beta_location[D,Q] int32
beta_sign[D,Q] int8
upstream[D,S] float32
compressed_dbeta[D,Q] float32 caller-owned
```

location同样safe-read后poison invalid；sign只接受`-1/+1`，TIR内cast到float32：

```text
dbeta[d,q] = sum_s -upstream[d,s] * V[d,s,location[d,q]] * float(sign[d,q])
```

β parameter值不参与数学公式，不为了“绑定身份”加入`beta*0`。parameter generation/content由prepared owner和launch
receipt绑定；TIR输入只包含真实数据依赖。

### 4.4 nonfinite poison的必要性

如果仍沿用`if A>=0 and ...: product else 0`，则`A=NaN`会走else并返回有限0，最终result finite gate无法发现。
新module先检查A/V/lower/upper/α/upstream，任何非有限依赖都输出qNaN；这样S4-1D finite gate才能真正fail closed。

## 5. CUDA probe结果

### 5.1 reference错误必须保留

第一次diagnostic reference把`upstream[D,1]`直接与`A[D,1,W]`相乘，被PyTorch广播为跨domain的
`[D,D,W]`，产生最大约32的错误；TIR六路sign仍exact。该FAIL来自reference shape bug，不是candidate通过证据。

修正为显式`upstream[D,1,1]`后重新运行同一module，才得到可采信结果。

### 5.2 corrected result

```text
status = PASS
seven symbols / seven launches
overall candidate-vs-PyTorch max diff = 0
all gradient signs exact = true
overall candidate-vs-float64 max diff = 2.3575648810947314e-07
alpha nonzero count = 2,167
beta max diff = 0
global workspace = 0
scheduled TIR alloc_buffer occurrences = 0
```

六dα output hash：

| site | hash |
|---:|---|
| 17 | `3c94184b...3ccf` |
| 19 | `bb8daa06...b34f` |
| 23 | `943b1d79...8fba` |
| 25 | `33f27940...b9ee` |
| 28 | `15ac664e...21c2` |
| 31 | `77e96971...ee8c` |

dβ hash=`40ea1047...3153`。这些来自synthetic A/V + real layout/bounds/α的design probe，不是production
gradient artifact，正式实现必须重建hash。

### 5.3 poison probe

| mutation | poisoned output count |
|---|---:|
| A=NaN | 1 |
| V=Inf | 1 |
| lower>upper | 1 |
| α=1.1 | 1 |
| upstream=NaN | 27 |
| index=F+7 | 6 |

每项至少一个NaN，证明异常不会静默变0。正式negative还必须覆盖duplicate/unsorted、wrong site、wrong
generation、β bad location/sign和fully re-signed receipt。

## 6. module/cache identity

本轮隔离diagnostic：

```text
unscheduled_tir_hash = 054d27afb86f647405953ed44fff42594caf7963cccb32f76625e3cf6989c2d6
scheduled_tir_hash   = 248fe51d047f5df85e359eea00a40a753634adf9a9023b16f2e86c5ee880d194
device_source_hash   = 31e6d444db509f8b32ac1ff6722aa318bf7bb6aaf138528d7ae633931e8ae49f
cache_key_hash       = d12841020ecbc8695e160cd9496e374d028bc973b3647347193cdbed908f4503
threads              = 128
```

正式cache key至少绑定schema、七symbol、plan-derived`(site,F,W)`、index hashes、β location/sign hashes、
unscheduled/scheduled TIR、target/CC、dtype、threads、finite/index policy。diagnostic hash不进入production receipt。

## 7. 精确view与参数账

六α每个8 arguments，β为5 arguments：

```text
argument occurrence = 6×8 + 5 = 53
launch = 6 + 1 = 7
```

emitter所见unique DLPack view：

```text
A views 6 + V views 6 + lower/upper 12 + active alpha 6 + alpha index 6
+ upstream 1 + dalpha output 6 + beta location/sign/dbeta 3 = 46
```

46/46 pointer exact在现场成立。S4-1A已有16 base views，emitter复用六alpha parameter、六alpha gradient、
一beta gradient和upstream，共14项：

```text
base_view_count = 16
additional_tir_view_count = 46 - 14 = 32
total_prepared_view_count = 48
```

active β parameter和final lower result仍属于base owner，但不作为gradient emitter实参。receipt必须同时列base/
additional/total，不能把emitter46与prepared48混为同一个口径。

static metadata：

```text
alpha indices 708 × int32 = 2,832 B
beta locations 6 × int32 = 24 B
beta signs     6 × int8  = 6 B
total                         2,862 B
```

旧稿把sign临时转为float32得到2,880 B；该copy无语义必要，现由int8 ABI取代。

## 8. V arena与terminal-lA phase

一个physical float32 arena：

| site | offset interval | elements |
|---:|---|---:|
| 17 | `[0,12288)` | 12,288 |
| 19 | `[12288,18432)` | 6,144 |
| 23 | `[18432,24576)` | 6,144 |
| 25 | `[24576,30720)` | 6,144 |
| 28 | `[30720,36864)` | 6,144 |
| 31 | `[36864,37464)` | 600 |

总计37,464 elements/149,856 B，六data pointer不同但storage token相同，interval无重叠。V arena与两个main
coefficient arena、residual11 scratch、residual6 scratch共四个coefficient physical storage完全不相交。

reverse pass C固定`31→28→25→23→19→17`。terminal ordinal每site同stream：

```text
V_READY + A_PRETRANSFORM_READY
  -> EMITTER_LAUNCHED
  -> TERMINAL_COPY_A_TO_V_SLOT_ENQUEUED
  -> A_TRANSFORM_OR_REUSE_ENQUEUED
  -> slot phase TERMINAL_LA after stream completion
```

现场六slot simulation得到gradient read-before-copy exact、terminal slot==pretransform A exact、dynamic allocation=0。
由于每个V slot只由自己的emitter读取，copy后不再被其他site读取；同stream ordering足够，不需要每siteevent/sync。

禁止cross-stream copy、copy post-transform A、emitter未入队先覆盖V、result lease未消费即下一evaluation重写arena。

## 9. receipt最小字段

selector/value receipt至少包含六slot schema/value set/sentinel/count/generation/capture point/action/fanout hash，以及：

```text
selector_buffer_count=6 / elements=55296 / bytes=55296
V_arena_count=1 / slot_count=6 / elements=37464 / bytes=149856
```

gradient receipt至少包含：

```text
seven exported symbols and module identities
plan/layout/index/location/sign hashes
finite_policy / safe_index_policy / canonical_nan_bits
launch_count=7 / argument_occurrence_count=53
emitter_unique_view_count=46 / pointer_exact=46
base_view_count=16 / additional_tir_view_count=32 / total_prepared_view_count=48
alpha_output_count=6 / beta_output_count=1 / empty_beta_launch_count=0
alpha_index_bytes=2832 / beta_location_bytes=24 / beta_sign_bytes=6
saved_dense_A=0 / dense_gradient_output=0 / global_workspace=0
warm_view/allocation/python_dispatch=0
fallback/eager/native_shadow=0
timing_recorded=false / performance_claimed=false
```

terminal receipt另绑定reverse order、每slot read/copy/transform sequence、pretransform coefficient hash、spec-axis view、
one-shot lease和arena generation。

## 10. 正式negative矩阵补充

在已有矩阵上至少新增：

1. 任一binary selector pack删除IEEE nonfinite检查；
2. `-128`被`selector!=0`当成lower branch；
3. ternary schema名和buffer semantic版本混用；
4. selector总bytes仍伪写成43,008或“不变”；
5. A/V/α/upstream NaN因gate false静默输出0；
6. lower>upper或α越界静默输出0；
7. raw index/location直接访问后才验证，允许OOB；
8. safe clamp被误写为接受invalid index；
9. β sign转float32后缺少±1身份；
10. beta parameter通过`*0`伪装为TIR数据依赖；
11. emitter46、base16、total48三种view口径混淆；
12. metadata bytes写成旧2,880；
13. diagnostic广播错误结果被当candidate failure或production evidence；
14. coefficient-only probe错误调用完整D2B receipt；
15. V slot overlap/空洞/越界；
16. V arena与A/scratch alias；
17. terminal copy早于emitter或来自post-transform A；
18. cross-stream复用无event；
19. seven launch伪写成one kernel；
20. diagnostic module hash复制进production receipt。

## 11. 获批后的实施顺序

```text
test(math): freeze six selector finite/sentinel semantics
feat(tvm): add isolated S4 binary selector pack symbols
test(tvm): close real six-selector inventory and sentinel tamper
feat(tvm): add seven-symbol compressed gradient module
test(tvm): close safe-index/nonfinite/shape/layout CUDA gates
feat(runtime): bind 48 prepared views and reverse phase state
test(runtime): close terminal read-copy-transform ordering
test(formal): run five-fresh six dα + active dβ + terminal lA
```

前置条件仍为S3 external approved、S4-0、S4-1A、S4-1B0、S4-1B依次关闭。S4-1C通过前不得接Adam或计时。

## 12. 当前门禁

```text
S3 exchange = ready_for_audit/r001
S4 production implementation = closed
S4-1B/1C selector/gradient/arena ABI = implementation-ready design
S4-1B/1C production correctness/formal = closed
S4-1D/S4-2/S4-3/S4-4/timing/performance = closed
```

本合同只减少获批后的实现歧义，不构成S4 correctness、same-solver或性能证据。
