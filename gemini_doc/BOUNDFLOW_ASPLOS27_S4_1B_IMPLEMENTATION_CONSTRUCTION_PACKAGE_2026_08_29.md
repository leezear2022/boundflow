---
status: construction-ready-code-closed
date: 2026-08-29
type: implementation-construction-package
topic: boundflow
slug: asplos27-s4-1b-six-site-value-construction
stage: s04
depends-on:
  - external-approved-s3-optimizer-runtime
  - validated-s4-0-admission
  - validated-s4-1a-buffer-owner
  - validated-s4-1b0-ternary-endpoint
execution-authority: false
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
construction-model-hash: a9b1d90df3cd122eb43491d327432ded52f957928d77e1dbcf2e7286bc4a317d
---

# ASPLOS'27 S4-1B：六站点 selected-value 实现施工包

## 0. 直接结论

S4-1B 的数学方向和复用路线不变：先执行一次 coefficient pass 捕获六张 selector，再用一张
Relax/TIR/cuDNN 图计算六个 selected-primal value，最后由 S4-1C 重算 coefficient 并即时发射 compressed gradient。
本轮把旧蓝图继续压到逐文件、逐参数、逐 phase 可施工，同时纠正两个此前会直接误导实现的物理账：

1. **完整 S4-1B 不是 48 个 DLPack view，而是 90 个 prepare-time argument descriptor。**`48`只等于
   S4-1A base 与七个 gradient emitter 的局部并集，没有覆盖 pass A、selected graph、六个 V output 与
   selected-input scratch；
2. **完整 S4-1A/B/C 是 110 个 argument descriptor。**S4-1C isolated emitter 仍是46个，其中14个与base重叠、
   12个flattened bounds已由S4-1B拥有，故只再增加20个；
3. **D1C residual11/residual6 scratch不是额外physical storage。**源码将它们分别定义成
   `scratch_1[6144:12288]`与`scratch_0[12288:18432]`，只新增两个带offset的view，不新增`49,152 B`；
4. 因此selected-input成功复用coefficient arena时，S4-1D/S4-2/S4-3 known logical subtotal应为
   **`389,574 / 491,774 / 559,838 B`**，而不是`438,726 / 540,926 / 608,990 B`。若phase alias失败，
   S4-1D必须另加`73,728 B`，成为`463,302 B`。

这些结论来自现有production源码、机械descriptor集合重算和GPU storage identity probe；它们是实现合同与设计账
纠正，不是S4 production correctness、peak memory或performance claim。S3 exchange仍为`ready_for_audit`，所以本包
不开放任何S4 production代码。

## 1. 本阶段到底实现什么

### 1.1 唯一输出

S4-1B一次evaluation只产生：

```text
six selectors:
  endpoint_ainput_v2, sign_a18, sign_a20, sign_a24, sign_a26, sign_a29

one V arena:
  V17, V19, V23, V25, V28, V31

one structural receipt:
  pass-A generation + pass-B generation + module/cache/stream/arena identity
```

其中`V_i = d lower / d T_i`，`T_i`是对应ReLU transform之后的coefficient state。S4-1B不写dα/dβ，
不运行Adam，不发布terminal lA，不计时。

### 1.2 不新增一套顶层IR

本阶段继续使用现有：

- production plan/trace决定site、shape、alpha map与coefficient action order；
- R31B1/D1C的TIR coefficient kernels与两个bounded arena；
- S4-1B0的ternary endpoint TIR语义；
- S2的Relax→cuDNN/TIR→VM编译和current-stream执行机制；
- S4-1A的ordered parameter/gradient/lower/upstream owner。

只新增backend-local Relax/TIR lowering、prepared runtime owner和receipt，不引入新的Primal/Bound/Plan/Task/
Schedule顶层对象，也不复制solver控制流。

### 1.3 明确不做

- 不复活full-source α `[2,1,D,W]`；
- 不保存任何跨phase dense A；
- 不增加per-site Python executor；
- 不在warm path创建DLPack view、Torch tensor或result dict；
- 不用CUDA Graph掩盖基本VM/wrapper成本；
- 不把logical stage数写成CUDA kernel数；
- 不形成speedup、peak-memory、complete-query或ASPLOS-ready claim。

## 2. 源码事实与复用边界

### 2.1 S2 selected graph已经提供什么

`boundflow/backends/tvm/asplos27_s2_selected_value.py`当前拥有：

- 28个只读参数加1个caller-owned persistent output；
- input select、ReLU17/19/23 selected TIR；
- Conv0/2/4/shortcut5/8五次cuDNN Conv；
- pre17→pre19→residual6 pre23→pre25的Relax dataflow；
- `call_tir_inplace`把pre25写到caller-owned output；
- source/partitioned/lowered Relax hash与device source hash。

`PreparedS2SelectedValueGraphV1`实际创建29个argument DLPack view，并额外用一次
`torch.from_dlpack(initial_result)`检查返回值指针；receipt里历史`prepare_dlpack_view_count=30`混合了这两个口径。
S4必须拆成：

```text
prepare_argument_dlpack_descriptor_count
prepare_result_roundtrip_wrapper_count
warm_argument_dlpack_descriptor_count
warm_result_roundtrip_wrapper_count
```

### 2.2 S4相对S2的必改项

1. input select升级为ternary `positive→lower / negative→upper / zero→center`；
2. ReLU active α从旧`[2,1,D,W]`改为S4-1A `[D,W]`；
3. 增加selected ReLU25、Conv10、residual11 add、selected ReLU28、Flatten/Gemm14；
4. 从只导出pre25改为导出pre17/pre19/pre23/pre25/pre28/pre31六个caller-owned slot；
5. selected input写入caller-owned coefficient-arena alias，而不是VM动态output；
6. module cache从只按compute capability改为完整immutable compile key；
7. runtime identity从`pointer+shape+dtype`升级为S4-1A exact view key与generation guard；
8. warm result只保留VM owner token，不再建立Torch roundtrip wrapper。

### 2.3 staged residual已经提供真正的插入点

`PreparedR3D2BStagedBackwardCandidateV1`把两个monolithic residual拆为：

```text
residual11 stage1: A29 --Conv10-right--> A26 scratch
residual11 stage2: A29 + transform(A26) --Conv8-right--> A24

residual6 stage1:  A24 --Conv4-right--> A20 scratch
residual6 stage2:  A24 + transform(A20) --Conv2/shortcut5-right--> A18
```

但当前dispatch会连续启动stage1/stage2。S4 pass A不能直接复用该整段dispatch；它必须在两段之间插入
`pack_a26`与`pack_a20`。这只是scheduler/orchestrator重排，不需要重新推导residual数学。

### 2.4 residual scratch物理owner纠正

当前源码精确为：

```text
residual11_scratch = coefficient_arena_1[6144:12288]
residual6_scratch  = coefficient_arena_0[12288:18432]
```

GPU probe复核：

```text
residual11.storage_cdata == arena1.storage_cdata
residual11.storage_offset == 6144
residual6.storage_cdata  == arena0.storage_cdata
residual6.storage_offset == 12288
physical storage count(arena0,arena1,residual6,residual11) == 2
additional residual scratch bytes == 0
```

所以receipt仍要记录两个scratch descriptor、offset与non-overlap，但memory ledger不得再创建两个独立storage或
追加49,152 B。

## 3. 建议文件与职责

审计批准并逐级开放后，新增文件固定为：

```text
boundflow/backends/tvm/asplos27_s4_six_site_value.py
boundflow/runtime/asplos27_s4_coefficient_selector_pass.py
boundflow/runtime/asplos27_s4_six_site_value.py
tests/test_asplos27_s4_coefficient_selector_pass.py
tests/test_asplos27_s4_six_site_value.py
```

### 3.1 backend文件

只拥有：

- ternary selected-input consumer；
- active-α selected-ReLU TIR template；
- six persistent-copy TIR template；
- six-site Relax graph；
- cuDNN partition、default TIR schedule和编译；
- immutable `CompiledS4SixSiteValueV1`。

禁止import solver、optimizer、provider、artifact writer或DocOps。

### 3.2 selector pass runtime

只拥有：

- pass A 19-action顺序；
- staged residual中间插入点；
- six selector buffers；
- coefficient arena generation转换前置条件；
- `S4SelectorPassReceiptV1`。

禁止拥有selected graph VM、V arena或gradient emitter。

### 3.3 selected-value runtime

只拥有：

- immutable compiled module引用；
- plan-instance 49个selected graph argument descriptor；
- one V arena与six slot views；
- coefficient arena的selected-input shaped descriptor；
- current-stream VM调用与result token；
- `S4SixSiteValueReceiptV1`。

S4-1D随后组合selector pass、selected-value runtime和S4-1C emitter，S4-1B自身不成为另一个solver入口。

## 4. 精确phase状态机

### 4.1 状态

```text
PREPARED
  → PASS_A_RUNNING
  → SELECTORS_READY
  → ARENA_REBOUND_FOR_SELECTED_INPUT
  → PASS_B_RUNNING
  → VALUES_READY
  → COEFFICIENT_RECOMPUTE_READY

任一post-begin failure → POISONED_NO_RETRY
```

`PREPARED`前的admission rejection不消耗evaluation generation。进入`PASS_A_RUNNING`后，任何异常都烧毁generation，
不能reset、fallback、重跑native或继续queue。

### 4.2 三类generation

receipt分别记录：

```text
evaluation_generation
coefficient_arena_generation
selector_generation
value_arena_generation
```

六selector必须来自同一个evaluation/parameter state/coefficient generation。V arena六slot必须来自同一个
selected graph invocation。不得仅比较ordinal字符串而忽略storage generation。

### 4.3 pass边界

```text
Pass A owns coefficient arena writes and selector writes.
Pass B owns selected-input alias writes and V arena writes.
Pass C owns coefficient arena recompute and gradient writes.
```

Pass B不能读旧coefficient descriptor；Pass C不能在VM仍有selected-input live reader时重写alias storage。

## 5. Pass A：19个逻辑launch的冻结顺序

第一版construction schedule固定为：

| ordinal | action | input/output边界 | selector |
|---:|---|---|---|
| 0 | seed | objective→A32 | — |
| 1 | Linear16-right | A32→A31 | — |
| 2 | ReLU31 coefficient | A31→T31 | — |
| 3 | Linear14-right | T31→A29 | — |
| 4 | pack A29 | read A29 | `sign_a29` |
| 5 | ReLU28 coefficient | A29→T28 | — |
| 6 | residual11 stage1 | T28→A26 scratch | — |
| 7 | pack A26 | read A26 scratch | `sign_a26` |
| 8 | residual11 stage2 | T28+A26→A24 | — |
| 9 | pack A24 | read A24 | `sign_a24` |
| 10 | ReLU23 coefficient | A24→T23 | — |
| 11 | residual6 stage1 | T23→A20 scratch | — |
| 12 | pack A20 | read A20 scratch | `sign_a20` |
| 13 | residual6 stage2 | T23+A20→A18 | — |
| 14 | pack A18 | read A18 | `sign_a18` |
| 15 | ReLU17 coefficient | A18→T17 | — |
| 16 | Conv0-right | T17→Ainput | — |
| 17 | pack Ainput | read Ainput | `endpoint_ainput_v2` |
| 18 | box concretize | read Ainput+bias→lower | — |

`19`是当前construction logical launch envelope，不是未来编译产物的性能数字。若实现把pack与producer合法融合，
必须先由新schedule/module receipt证明语义等价，再更新合同；不得事后只改计数。

### 5.1 selector合法值

```text
endpoint_ainput_v2: {-128,-1,0,+1}
other five signs:  {-128,0,1}
-128: nonfinite invalid sentinel
```

formal有效执行的invalid count必须为0。consumer先验证合法谓词，再用safe index读取；`selector != 0`不得把`-128`
解释为lower branch。

### 5.2 Ainput pack与concretize的顺序

两者都读取最终Ainput，因此顺序必须是：

```text
Conv0-right full-write Ainput
→ pack ternary endpoint selector
→ concretize lower
→ revoke coefficient-generation read capability
```

如果先把arena改写为selected input再执行pack/concretize，语义已经损坏。合成GPU负例中，错误顺序
`select→pack`产生`12,253`个selector mismatch；正确`pack→select`则selector与selected value均bitwise exact。

## 6. coefficient arena与selected-input phase alias

### 6.1 为什么物理上可行

Ainput与selected input都有18,432个float32元素：

```text
Ainput flat descriptor          [18432]
selected-input shaped descriptor [6,3,32,32]
bytes                            73,728
```

当前schedule在Conv0-right后由arena1持有Ainput。pack与concretize完成后该值无后续reader；Pass C会从seed重新计算
coefficient。因此同一个physical storage可以在Pass B改作selected-input scratch。

### 6.2 prepare-time双descriptor，不在warm重建view

prepare同时建立：

```text
(storage_token, offset=0, shape=[18432], stride=[1], dtype=f32, role=Ainput)
(storage_token, offset=0, shape=[6,3,32,32], stride=contiguous, dtype=f32,
 role=selected_input)
```

两个descriptor pointer exact但shape不同，必须分别进入90-view inventory。generation guard控制谁可被dispatch；不能销毁
旧view后在warm path重新`from_dlpack`。

### 6.3 alias transition的六个前置条件

1. action18 concretize完成；
2. current stream与TVM FFI stream exact；
3. Ainput live-reader count为0；
4. selector receipt已sealed且独立validate；
5. coefficient descriptor generation被逻辑撤销；
6. selected-input descriptor generation被原子激活。

Pass B完成后，反向transition还要求VM invocation完成、selected-input live-reader为0、六V slot receipt已sealed。失败时
poison整个prepared object，不允许临时分配73,728 B悄悄fallback。

### 6.4 alias失败时的合法替代

若外审或实现证明任一live-reader/stream/generation条件不成立，唯一合法替代是prepare时新增独立
`selected_input[18432]` storage：

```text
S4-1D known logical subtotal = 389,574 + 73,728 = 463,302 B
selected-input physical storage count = 1
performance_claimed = false
```

不得保留`389,574 B`账同时暗中分配fallback output。

## 7. Pass B：六站点Relax graph

### 7.1 dataflow

```text
selected input
  → Conv0+bias → pre17 → copy V17
  → selected ReLU17
  → Conv2+bias → pre19 → copy V19
  → selected ReLU19
  → Conv4+bias ───────────────┐
    selected ReLU17→shortcut5 ├→ add → pre23 → copy V23
                              ┘
  → selected ReLU23
  → Conv8+bias → pre25 → copy V25
  → selected ReLU25
  → Conv10+bias ──────────────┐
    selected ReLU23───────────┤→ add → pre28 → copy V28
                              ┘
  → selected ReLU28
  → flatten → Gemm14+bias → pre31 → copy V31
```

所有residual branch必须来自同一upstream selected value；不能把main branch结果误作shortcut输入。

### 7.2 42个只读输入

| 组 | descriptor数 | 内容 |
|---|---:|---|
| input/Conv0 | 5 | lower、upper、endpoint selector、weight、bias |
| site17/Conv2 | 7 | lower、upper、active α、alpha map、sign、weight、bias |
| site19/Conv4+shortcut5 | 9 | five site inputs + four parameters |
| site23/Conv8 | 7 | five site inputs + two parameters |
| site25/Conv10 | 7 | five site inputs + two parameters |
| site28/Gemm14 | 7 | five site inputs + two parameters |
| **合计** | **42** | — |

site31没有selected ReLU输入，因为V31就是pre31；因此Pass B不消费α31、map31或sign A32。

### 7.3 七个caller-owned write target

```text
selected_input scratch = 1
V arena slots          = 6
total write arguments  = 7
```

所以Relax函数argument descriptor总数=`42+7=49`。六V output全部用`call_tir_inplace`写入caller-owned slot；
Relax返回fixed tuple/token只作VM ownership，不创建payload storage。

### 7.4 expected physical operation envelope

第一版源图预计包含：

```text
selected TIR            6  # input + ReLU17/19/23/25/28
persistent copy TIR     6
convolution calls       6  # Conv0/2/4/shortcut5/8/10
Gemm/Linear14 calls     1
```

这不是冻结CUDA kernel count。cuDNN、cuBLAS、TIR schedule和VM可能各自展开多个device kernel；compile receipt必须从
partitioned/lowered module与device source独立重算actual function/call/kernel identity。

### 7.5 active α ABI

每个selected-ReLU TIR固定：

```text
pre[D,F]
sign[D,F] int8
lower[D,F]
upper[D,F]
active_alpha[D,W]
alpha_map[F] int32
output[D,F]
```

禁止为了复用S2旧kernel，把active α扩回`[2,1,D,W]`或保存preserved polarity。

## 8. V arena与result view

物理arena仍为37,464 float32=`149,856 B`：

| site | shape | offset elements | elements |
|---:|---|---:|---:|
| 17 | `[6,8,16,16]` | 0 | 12,288 |
| 19 | `[6,16,8,8]` | 12,288 | 6,144 |
| 23 | `[6,16,8,8]` | 18,432 | 6,144 |
| 25 | `[6,16,8,8]` | 24,576 | 6,144 |
| 28 | `[6,16,8,8]` | 30,720 | 6,144 |
| 31 | `[6,100]` | 36,864 | 600 |

六slot必须无重叠、无空洞、完整覆盖`[0,37464)`。S4-1C emitter会用`[D,1,F]`形状读取同一storage，因此那些是
额外DLPack descriptor，不是额外storage。

terminal lA lease需要六个普通Torch shaped view（增加spec轴），lower lease需要一个`[D,1]`普通Torch view；它们应在
prepare建立并单独计数，不进入argument DLPack 110口径，也不新增physical storage。

## 9. 90/110 DLPack descriptor完整重算

### 9.1 S4-1A base：16

```text
6 active α parameter
1 active β parameter
6 dα output
1 dβ output
1 lower physical [D]
1 upstream [D,1]
= 16
```

lower physical buffer冻结为`[D]`；result lease只暴露prepare-time普通Torch view `[D,1]`。

### 9.2 selected graph：49，与base重叠5

49来自42 read inputs与7 caller-owned write targets。selected graph消费active α17/19/23/25/28，正好与base重叠5；
不消费α31或active β。

### 9.3 Pass A新增：30

| 类别 | 新descriptor数 | 说明 |
|---|---:|---|
| coefficient arena slice/full | 7 | arena0 `[60]/[6144]/[12288]/[18432]`；arena1 `[600]/[6144]/[18432]` |
| residual scratch slice | 2 | same storage、nonzero offset，不新增bytes |
| bias accumulator | 1 | `[D]` |
| objective | 1 | `[D,10]`/production plan shape |
| Linear16 weight/bias | 2 | 其余14个model parameter已在selected graph |
| alpha map31 | 1 | 其余5张map已在selected graph |
| dense β map/split map | 2 | 当前均为`[D,100]`，dtype i32/i8 |
| flattened input/ReLU bounds | 14 | input与六ReLU，各lower/upper |
| **合计** | **30** | — |

因此：

```text
S4-1B union = base16 + selected49 - overlap5 + passA30 = 90
```

### 9.4 S4-1C再增加20

isolated emitter unique descriptor仍为46：

```text
overlap base                         14
overlap S4-1B flattened ReLU bounds 12
new V [D,1,F] views                   6
new A [D,1,F] views                   6
new alpha indices                     6
new compressed beta location/sign     2
= 46
```

注意compressed beta sign `[D,1]`不等于Pass A dense split map `[D,100]`，两者不能合并。因此：

```text
S4-1C new over S4-1B = 46 - 14 - 12 = 20
S4-1A/B/C full union = 90 + 20 = 110
```

### 9.5 descriptor key

每项唯一性基于：

```text
(storage_token, storage_offset, shape, stride, dtype, device, descriptor_role,
 descriptor_generation)
```

pointer相同但shape/stride/offset/role不同仍是不同descriptor。global cache与artifact不得保存raw pointer/storage token；
runtime receipt保存不可序列化sidecar identity，正式artifact保存role/shape/stride/dtype/offset与indexed raw tensor hash。

## 10. compile cache与module receipt

### 10.1 immutable cache key

至少绑定：

```text
schema version
production plan / trace / selected graph layout hash
ternary endpoint and nonfinite policy hash
active-alpha ABI hash
V arena layout hash
source/partitioned/lowered Relax hashes
TIR template/schedule hashes
target, compute capability, TVM/tvm-ffi/CUDA/cuDNN identities
dtype/layout/static shape signature
```

不得只按compute capability命中，也不得把instance tensor、pointer、state version或evaluation ordinal放进global key。

### 10.2 immutable compiled receipt与mutable cache observation分离

`CompiledS4SixSiteValueV1`只保存可复用编译事实；每次prepare另建：

```text
cache_lookup_key_hash
cache_event = miss | warm_hit | disk_hit
cache_entry_generation
prepare_ordinal
```

不能因第二次lookup命中而修改共享compiled object上的`cache_event`。

### 10.3 module validate

validate必须独立重算或绑定：

- source/partitioned/lowered IR hash；
- device source hash集合；
- exported symbol集合；
- cuDNN/cuBLAS partition function与call inventory；
- six selected TIR与six persistent-copy symbol；
- global workspace/device function metadata；
- `performance_claimed=false`。

只检查64字符格式不够；B2-5/S4 formal replay必须从冻结source identity重新编译或独立解析module receipt。

## 11. prepared runtime与VM结果owner

### 11.1 prepare允许

- compile/cache lookup；
- 分配V arena和six selector；
- 建立90个argument descriptor；
- 建立普通result/terminal shaped view；
- 创建VM与PackedFunc；
- 在独立capture/preparation stream做固定次数qualification invoke；
- 建立六个result roundtrip wrapper核对caller-owned V slot pointer，然后立即释放wrapper。

### 11.2 warm禁止

```text
from_dlpack creation = 0
torch.from_dlpack wrapper creation = 0
Torch tensor/view/dict creation = 0
dynamic output allocation = 0
module compile/cache mutation = 0
content hash / class-count D2H sync = 0
fallback/eager/native shadow = 0
```

VM每次返回的NDArray/tuple owner token可由一个固定字段替换持有，不得像当前S2一样无界append到`result_owners` list。

### 11.3 current-stream

每次launch前同时核对：

```text
torch.cuda.current_stream(device).cuda_stream
tvm_ffi.get_raw_stream(cuda device)
prepared expected stream identity
```

三者不一致在首个launch前拒绝。第一版只准non-default single stream。

## 12. 修正后的logical memory ledger

排除model parameters、fixed bounds、compiled module storage、VM/cuDNN workspace与allocator metadata：

| 类别 | bytes | physical owner |
|---|---:|---|
| active α/β parameters | 17,016 | 7 storages |
| dα/dβ outputs | 17,016 | 7 storages |
| six selectors | 55,296 | 6 int8 storages |
| V/terminal-lA arena | 149,856 | 1 storage |
| two coefficient arenas | 147,456 | 2 storages，已含residual scratch slices |
| lower/upstream/bias | 72 | 3 storages |
| compressed static metadata | 2,862 | 8 storages |
| **S4-1D合计** | **389,574** | **34 physical storages** |

下游加法：

```text
S4-1D                              389,574 B
+ S4-2 policy/optimizer additions 102,200 B
= S4-2                            491,774 B
+ S4-3 candidate/rollback          68,016 B
+ persistent upper/depths              48 B
= S4-3                            559,838 B
```

cross-device拆分保持原CPU账：

```text
S4-2 CUDA/CPU/total = 491,718 / 56 / 491,774 B
S4-3 CUDA/CPU/total = 559,758 / 80 / 559,838 B
```

### 12.1 为什么旧448,000 B probe不能继续作为验证

旧probe按36个独立buffer实例化，把两个residual scratch真的分配成独立storage，因此只能证明旧的过度分配设计可实例化，
不能证明production owner与源码reuse一致。新implementation probe必须从真实prepared object枚举storage `_cdata`去重，
并验证scratch storage identity/offset，不得手工照ledger重新分配一遍。

### 12.2 VM/cuDNN workspace必须双口径测量

当前S2 selected graph在已prepare D2B owner之后的诊断为：

```text
graph compile torch allocated delta       0
graph prepare torch allocated delta  24,576 B
graph prepare cuda mem_get_info delta -25,165,824 B
warm torch allocated delta                0
warm cuda mem_get_info delta              0
```

这说明Torch allocator `warm=0`不能证明TVM VM/cuDNN没有prepared footprint。该数只属于本机S2诊断，不可直接升级为
S4 formal数字；S4 artifact必须同时记录Torch allocated/reserved与CUDA `mem_get_info`，并披露测量顺序、同步点和噪声。

## 13. receipt字段

`S4SelectorPassReceiptV1`至少：

```text
evaluation/parameter/coefficient/selector generation
19 action ordinals + action sequence hash
six selector schemas/shapes/dtypes/pointers(sidecar only)
selector counts and invalid counts
residual scratch storage-token equality + offsets
concretize-before-rebind proof
stream/ffi stream identity
launch count and no-fallback counters
```

`S4SixSiteValueReceiptV1`至少：

```text
compile/cache/module identities
selected graph layout and six V slot hashes
argument descriptor count=49
S4-1B union descriptor count=90
base overlap count=5
prepare result wrapper count=6
warm descriptor/wrapper/tensor creation=0
selected-input alias storage/generation transition
VM invocation count=1
logical stage count=6
actual partition/call/kernel inventory
persistent output copy count=6
dynamic output allocation=0
fallback/eager/native shadow=0
timing_recorded=false
performance_claimed=false
```

receipt JSON不含raw pointer、Tensor、NDArray、PackedFunc或VM object；这些只在不可序列化runtime sidecar中验证。

## 14. fail-closed reason与负向测试

至少覆盖下列类别，每类固定stable reason：

1. S3/S4前置门禁未关闭；
2. plan/trace/ordered-buffer hash错；
3. parameter state version错；
4. pass A action漏项、重复、换序；
5. pack A29不在ReLU28 transform前；
6. pack A26不在residual11两stage之间；
7. pack A20不在residual6两stage之间；
8. pack Ainput不在concretize前；
9. endpoint selector dtype/value/sentinel错；
10. binary selector dtype/value/sentinel错；
11. nonfinite selector被静默映射；
12. six selector generation不一致；
13. residual scratch被分配成新storage；
14. residual scratch storage owner或offset错；
15. residual scratch与live output overlap；
16. selected-input alias过早rebind；
17. coefficient live reader不为0；
18. old descriptor在new generation dispatch；
19. VM完成前恢复coefficient generation；
20. selected-input独立fallback storage未入ledger；
21. 42 read input少/多/换序；
22. 7 write target少/多/换序；
23. active α被扩回full source；
24. empty β进入Pass B；
25. site31 α/map/sign错误进入Pass B；
26. residual6/11 branch输入错误；
27. pre17/19/23/25/28/31 slot换序；
28. V arena重叠、空洞、越界；
29. V arena由六个storage伪装；
30. argument descriptor不是49；
31. S4-1B union不是90；
32. S4-1A/B/C union不是110；
33. pointer相同shape不同被错误去重；
34. view key缺stride/offset/dtype/device/generation；
35. warm创建DLPack descriptor；
36. warm创建Torch roundtrip wrapper；
37. warm创建payload tensor/dict；
38. VM动态output allocation；
39. result owner token无界累积；
40. current stream/FFI stream不一致；
41. default stream被接受；
42. module cache含instance pointer/tensor；
43. mutablecache event写回immutable compiled receipt；
44. source/partition/lowered/module hash只做格式检查；
45. partition/call inventory漂移；
46. six persistent copy漏项；
47. result wrapper pointer不等于caller-owned slot；
48. lower physical `[D]`被改成额外storage `[D,1]`；
49. terminal Torch view被错计为DLPack/storage；
50. residual scratch 49,152 B被重复计入；
51. VM/cuDNN footprint只用Torch allocator披露；
52. raw pointer进入artifact；
53. full re-sign后改view/byte/phase/claim；
54. timing/performance flag翻转；
55. post-begin failure reset/retry/fallback。

## 15. correctness与formal closure

S4-1B实现后必须至少五个fresh process，每个只执行一次evaluation。四方oracle：

1. provider-independent full PyTorch CROWN autograd；
2. coefficient-action VJP oracle；
3. float64 no-autograd公式；
4. existing S2/R31B2交集。

每run冻结：

- six selector full raw payload/hash/count；
- six V full raw payload/hash；
- lower、compressed α/β source identity；
- all descriptor role/shape/stride/dtype/offset；
- module/cache/stream/hardware identities；
- arena storage identity、generation与phase transition；
- Torch allocator与CUDA driver memory observation；
- exact launch/call/copy inventory；
- fail-closed counters。

数值门槛沿用上游冻结值：selected V对float32/float64均`atol=rtol=2e-4`；最终compressed gradient的
`2e-5/sign exact`属于S4-1C，不可在S4-1B提前宣称。site19必须显式重放binary endpoint反例与ternary修复。

artifact必须raw-first、manifest先写、partial result拒绝resume；replay从raw重算summary与110-view下游projection；tamper至少
覆盖本节55类并包含改raw后全重签外层digest。

## 16. 实现提交顺序

只有S3 external approved且S4-0、S4-1A、S4-1B0依次validated后，才允许：

```text
feat(runtime): add S4 coefficient selector pass with staged insertions
test(runtime): close selector generation, residual slice and order negatives
feat(tvm): add active-alpha six-site selected Relax/TIR graph
feat(runtime): bind 49 graph arguments and 90 S4-1B descriptors
test(tvm): close alias, six-slot, stream, cache and VM ownership
artifact: close five-fresh S4-1B raw/replay/tamper
docs: close S4-1B and only then open S4-1C implementation
```

每个提交只做一个logical change。S4-1B正确性关闭前，S4-1C implementation、S4-1D assembly和所有timing保持关闭。

## 17. 当前门禁

```text
S3 exchange status        = ready_for_audit
S4-0 implementation       = closed
S4-1A implementation      = closed
S4-1B0 implementation     = closed
S4-1B implementation      = closed
S4 timing/performance     = closed
```

本施工包的`construction-ready`只表示接口、phase、物理owner和测试矩阵已冻结，不表示可绕过上述门禁。

## 18. canonical construction model

```json
{"claims":{"correctness":false,"memory_peak":false,"performance":false},"counts":{"base_descriptor":16,"full_s4_1abc_descriptor":110,"pass_a_new_descriptor":30,"pass_a_logical_launch":19,"selected_graph_argument_descriptor":49,"selected_graph_base_overlap":5,"selected_graph_read_argument":42,"selected_graph_write_argument":7,"s4_1b_descriptor":90,"s4_1c_new_over_s4_1b":20},"files":["boundflow/backends/tvm/asplos27_s4_six_site_value.py","boundflow/runtime/asplos27_s4_coefficient_selector_pass.py","boundflow/runtime/asplos27_s4_six_site_value.py","tests/test_asplos27_s4_coefficient_selector_pass.py","tests/test_asplos27_s4_six_site_value.py"],"memory":{"residual_scratch_additional_bytes":0,"selected_alias_failure_additional_bytes":73728,"s4_1d":389574,"s4_2":491774,"s4_3":559838},"phase_order":["pass_a_coefficient_and_selectors","seal_selectors","rebind_coefficient_arena_to_selected_input","pass_b_six_site_values","seal_values","rebind_for_pass_c_recompute"],"scope":{"code_open":false,"external_s3_required":true,"timing":false},"storage":{"coefficient_arena":2,"residual_scratch_physical":0,"selector":6,"value_arena":1},"view_key":["storage_token","storage_offset","shape","stride","dtype","device","role","generation"]}
```

canonical JSON使用UTF-8、sorted keys、compact separators、`allow_nan=false`。hash见front matter；后续任何字段改变都必须
更新hash、changelog与外审handoff。

## 19. 本轮设计验证

已执行：

- 静态源码核对S2 28+1参数、D2B staged insertion、D1C residual slice owner；
- 机械set重算`16/49/5/30/90/46/14/12/20/110`；
- GPU storage identity probe，确认两个residual scratch只对应两个coefficient storage；
- ledger算术重算`389,574/491,774/559,838`；
- selected-input同storage双shape descriptor与pack-before-select正/负顺序诊断；
- S2 selected graph Torch allocator与CUDA driver prepare/warm双口径诊断。

待实现后验证：production five-fresh correctness、compiled exact partition/kernel receipt、new prepared owner真实memory、
raw replay、full re-sign tamper与全量回归。
