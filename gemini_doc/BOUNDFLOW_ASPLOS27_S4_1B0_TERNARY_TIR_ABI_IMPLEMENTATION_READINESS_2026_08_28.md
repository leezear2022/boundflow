---
status: implementation-ready-gate-closed-v2
date: 2026-08-28
type: tir-abi-cache-receipt-readiness
topic: boundflow
slug: asplos27-s4-1b0-ternary-tir-abi-implementation-readiness
stage: s04
depends-on: asplos27-s4-1b0-ternary-endpoint-implementation-readiness
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B0：三元 Endpoint TIR、缓存与回执实施就绪合同

## 0. 结论

S4-1B0 的数学合同已经进一步压到可直接实现、可拒绝错误缓存命中、可由外部模型独立审计的 TIR ABI。
本轮没有写 S4 production 代码，也没有计时。现场 CUDA 探针得到两个重要结论：

1. `A > 0 / A < 0 / else zero`本身可正确覆盖`+0.0/-0.0`和正负 subnormal；
2. 使用浮点`x == x`识别 NaN 在当前 TVM/CUDA 编译链中**不安全**：第一次探针中 NaN 被错误编成 zero。

因此正式 pack ABI 不再依赖浮点 NaN 比较，而冻结为 IEEE-754 float32 exponent 位检查。修正后的探针把
NaN、`+Inf`、`-Inf`全部编码为 reserved `-128`，并在 select 输出 canonical quiet NaN；真实 production
Ainput 的`8,689/9,137/606` inventory、selected output 和历史模块隔离全部通过。

这份文档新增的不是一层 IR，而是一个后端模块的具体 ABI、cache key、receipt 和 fail-closed 责任。

## 1. 现场发现的编译器反例

### 1.1 失败设计

最初设计以类似下式表达 finite：

```text
finite(x) = (x == x) && (abs(x) != +Inf)
```

在`torch 2.12.1+cu132 / TVM 0.23.dev0 / sm_89`现场编译后，小型 CUDA 探针得到：

```text
input NaN selector = 0       # 错误，应为 -128
input +Inf selector = -128
input -Inf selector = -128
status = FAIL
```

也就是说，`x == x`形式的 NaN 检查在当前 lowering/代码生成优化下不能作为 verification fail-closed
边界。不能因为 PyTorch eager 中该表达式成立，就假设生成的 CUDA 仍保留相同异常语义。

### 1.2 修正设计

正式 float32 classifier 冻结为：

```text
bits = reinterpret_uint32(coefficient)
exponent = bits & 0x7f800000
nonfinite = exponent == 0x7f800000

selector = nonfinite ? int8(-128)
         : coefficient > +0.0f ? int8(+1)
         : coefficient < +0.0f ? int8(-1)
         : int8(0)
```

该检查同时覆盖 NaN 与正负 infinity，不依赖 fast-math 下可被化简的 NaN 浮点谓词。mantissa 不参与
finite 判定；所有 exponent=`0xff`的 float32 一律先拒绝，不能进入`A==0`语义。

正式实现必须给 PrimFunc 绑定：

```text
boundflow.nonfinite_classifier = ieee754-f32-exponent-bits-v1
```

若未来支持 float16/bfloat16/float64，必须使用独立 schema/version 和对应 exponent mask，不得复用本合同。

## 2. 冻结 TIR ABI

### 2.1 schema 与 symbol

```text
schema = boundflow.asplos27-s4-ternary-endpoint/v1
pack   = boundflow_s4_pack_ainput_endpoint_ternary
select = boundflow_s4_select_input_endpoint_ternary
```

旧资产保持：

```text
boundflow_r31b2_pack_ainput_sign
boundflow_s2_select_input_tir
r3-1b2-p-alpha-vjp/v1
boundflow.asplos27-s2-selected-value/v1
```

禁止修改旧 symbol 的含义后继续沿用旧名字。

### 2.2 pack ABI

```text
input : coefficient[18_432] float32 contiguous CUDA
output: endpoint_selector[18_432] int8 contiguous CUDA
```

合法 output 仅为`-1/0/+1`；`-128`是 invalid sentinel，只能流向失败传播，不能作为第四个 endpoint。
shape、dtype、device、contiguity、storage offset、generation/evaluation ordinal 必须在 launch 前由 prepared
owner 验证。

### 2.3 select ABI

```text
input_lower[18_432] float32
input_upper[18_432] float32
endpoint_selector[18_432] int8
selected_endpoint[18_432] float32
```

逐元素语义：

```text
+1   -> lower
-1   -> upper
 0   -> (lower + upper) * float32(0.5)
-128 -> canonical quiet NaN, bits=0x7fc00000
other-> canonical quiet NaN, bits=0x7fc00000
```

select 不能简单写成`selector != 0 ? lower : upper`，也不能用 final `else midpoint`吞掉非法值。实现必须
显式比较`+1/-1/0`，其余值统一生成 canonical NaN，最后由 S4-1D result finite gate 在 lease/commit 前拒绝。

### 2.4 midpoint operation order

provider owner 的 operation order 冻结为：

```text
(lower + upper) * float32(0.5)
```

不能改写为：

```text
lower * float32(0.5) + upper * float32(0.5)
```

确定性 float32 反例：

| lower/upper | provider bits | reassociated bits | 差异 |
|---|---:|---:|---|
| max-finite / max-finite | `0x7f800000`（+Inf） | `0x7f7fffff` | overflow 顺序不同 |
| min-subnormal / min-subnormal | `0x00000001` | `0x00000000` | underflow 顺序不同 |

现场计数为`midpoint_reassociation_counterexample_count=2`。因此 module identity 必须绑定
`midpoint_policy=add-then-mul-f32-half-v1`，不能只绑定“数学上等价的 midpoint”。

## 3. 精确物理账

两个独立 TIR symbol 的最小账为：

| 项 | 数值 | 说明 |
|---|---:|---|
| launch | 2 | pack 1 + select 1；是否随后融合由 S4-1B profile 决定 |
| unique tensor | 5 | coefficient、lower、upper、selector、selected output |
| argument occurrence | 6 | pack 2 + select 4；selector跨两个symbol复用 |
| prepare DLPack view | 5 | 5/5 pointer exact |
| warm DLPack view | 0 | prepared owner复用既有view |
| selector storage | 18,432 B | existing int8 slot |
| center tensor | 0 | midpoint在select内部派生 |
| center DLPack view | 0 | 不存在physical center input |
| global workspace | 0 | 纯elementwise TIR |

这里的`launch=2`是 S4-1B0 独立模块的设计账，不是最终 whole-region headline。未来若将 pack/select 合并进
相邻 kernel，必须形成新 module/schema/hash，并重新执行 correctness closure；不能事后把当前两 launch receipt
改写为一 launch。

## 4. 编译缓存合同

### 4.1 cache key

S4 ternary endpoint cache key 至少绑定：

```text
schema
pack_symbol
select_symbol
unscheduled_tir_hash
scheduled_tir_hash
target
compute_capability
dtype=float32
selector_dtype=int8
numel=18432
threads_per_block=256
endpoint_policy=ternary-box-endpoint-v1
midpoint_policy=add-then-mul-f32-half-v1
nonfinite_policy=ieee754-f32-exponent-bits-sentinel-minus-128-v1
```

不能像历史`R31B2ModuleCacheV1`那样只按 compute capability 取条目；S4 cache 类型本身也必须独立，避免把旧
binary模块从另一个registry命中后冒充ternary模块。

### 4.2 现场 identity

小型16元素诊断模块：

```text
unscheduled_tir_hash = 23007e40ea02ef385f5ec6b9a478f1ed5ca0d3036a0243b59b90944fad024fd8
scheduled_tir_hash   = 1077db593b8d115e628409a387e0dd247aaf72e15e6071c723d7e00ed9a65d41
device_source_hash   = 16ced1039a4d6936fe2732601fb720ac70def475b068e942e5591324a57a0913
new_cache_key_hash   = 7c8ab6cdbddf910d53db499b8ba5f0162b292dac2e99ec36932aa77af3969ed4
old_cache_key_hash   = 870f59499f422256a24fc4a67ae882c2e02066bc89a728fbc6ed599f541e3fc4
cache_key_isolated   = true
```

真实18,432元素 scheduled module hash：

```text
c25b7590796f2cc19f679e23af89e792a0fd9ac9cce4c18528581007b363e6fb
```

这些是未提交production实现的 design-time diagnostic identity。正式实现的 source/scheduled/device hash
必须重新冻结，不得把上面hash复制进production receipt冒充build结果。

### 4.3 miss/hit 责任

S4-1B0正式测试必须建立fresh cache：

1. 第一次同key调用：`compile_count=1/cache_miss=1/cache_hit=0`；
2. 第二次同key调用：同一module receipt，`compile_count=1/cache_miss=1/cache_hit=1`；
3. endpoint policy、midpoint policy、nonfinite policy、threads、target、scheduled TIR任一变化必须miss；
4. 旧R31B2 binary key查询必须miss或稳定拒绝，不能返回S4 ternary executable；
5. cache hit后仍验证module receipt，不以key命中替代内容身份。

## 5. 模块与执行回执

### 5.1 module receipt

至少包含：

```text
schema_version
pack_symbol / select_symbol
unscheduled_tir_hash / scheduled_tir_hash / device_source_hash
target / compute_capability / tvm_version
threads_per_block
endpoint_policy / midpoint_policy / nonfinite_policy
input_numel / selector_bytes / global_workspace_bytes
exported_symbols
compile_count / cache_miss_count / cache_hit_count
performance_claimed=false
```

validate 必须重算 canonical receipt hash，并与编译返回的真实 module/sources 比较；不能只检查“64位hex格式”。

### 5.2 launch receipt

至少包含：

```text
evaluation_ordinal / parameter_state_version / selector_generation
coefficient_hash / lower_hash / upper_hash / selector_hash
positive_count / negative_count / zero_count / invalid_count
pack_launch_count=1 / select_launch_count=1
argument_occurrence_count=6
prepare_dlpack_view_count=5 / pointer_exact_count=5
warm_dlpack_view_count=0
extra_center_tensor_count=0 / extra_center_dlpack_view_count=0
fallback_count=0 / eager_candidate_count=0 / native_shadow_count=0
timing_recorded=false / performance_claimed=false
```

formal fixture才要求`8689/9137/606/0`。generic runtime不能把这些模型特定数量写进schema validator；它只验证
总数等于numel、invalid=0和raw重算一致。formal protocol另行绑定fixture expectation。

## 6. 两组现场 CUDA 结果

### 6.1 边界/异常探针

输入覆盖negative、`-0.0`、`+0.0`、positive、正负min-subnormal、NaN、`+Inf`、`-Inf`：

```text
selector = [-1,0,0,1,-1,1,-128,-128,-128,-1,1,0,0,1,-1,1]
signed_zero_both_center = true
subnormal_sign_preserved = true
nonfinite_sentinel_count = 3
invalid_outputs_nan = 3
launch_count = 2
dlpack_pointer_exact = 5/5
workspace = 0
status = PASS
```

### 6.2 真实 production Ainput

在冻结 ResNet2B property-0 pre-state上，复用existing coefficient pass产生真实Ainput，然后只用隔离新TIR
pack/select：

```text
positive / negative / zero / invalid = 8689 / 9137 / 606 / 0
old_binary_zero_misclassified = 606
selected_bitwise_exact = true
selected_hash = 7e95e07580002f81a9212250d080b85974286dca1bd67bdea9649a044239b652
derived_center_hash = 2a3b69e1bb4b8c8d768e4a7f50b2ec2964ea9704cfd6f35d8b2083a79cb5f003
coefficient_hash = e33066f5a5760ff870a1fed06a34993332627d2e61f655554df31c0211bfa799
lower_hash = e65509e640a045a550a1d92bcabd452c1be42ff887d401fdcacf9ee94f8f7c4d
upper_hash = 120a624bc5bf49b0e1704920ab7cabcb9a8e98df5b182172609ef00f40497a0f
legacy R31B2 module hash before/after =
  3871bf0e42ec9ce129d32bb408a5e9320d51026da6998aa81ebf0415822be575
legacy_module_unchanged = true
status = PASS
```

`derived_center_hash`与旧 readiness 文档中的短写`d616...`不同，原因是旧数字来自另一份diagnostic payload
投影；正式语义owner应绑定本次从实际 lower/upper 逐元素按provider order派生的完整tensor hash。外审不得把
两个不同scope的hash当作矛盾，也不得把任一design-time hash当作production artifact。

## 7. 实施时的负向矩阵

S4-1B0正式实现至少拒绝：

1. 浮点`x==x` NaN classifier替换位级classifier；
2. exponent mask、dtype或reinterpret方向错误；
3. NaN/Inf被归zero；
4. `-128`或其他非法selector被当center；
5. `+0.0/-0.0`分类不一致；
6. epsilon把subnormal吞成zero；
7. binary v1 module/symbol/cache entry冒充ternary；
8. cache key删除schema、policy、scheduled hash、target或threads；
9. cache hit跳过module receipt复核；
10. midpoint被重结合为half+half；
11. canonical NaN payload变化但receipt未变；
12. lower/upper/coefficient ordinal、generation或content漂移；
13. selector shape/dtype/stride/offset漂移；
14. 新增center tensor、pointer、view、allocation或workspace；
15. launch count、argument occurrence或view count漂移；
16. formal inventory不再是`8689/9137/606/0`；
17. 旧binary误编码606不再能由raw重算；
18. S2/R31B2旧module/source/symbol hash发生变化；
19. fallback/eager/native shadow非零；
20. timing或performance flag翻转。

这些加入S4-1B0自身negative后，不改变S4-4总共68类攻击的冻结数量；S4-4中endpoint分区仍按更高层语义
攻击聚合，不把每个底层unit negative重复算成新的formal攻击编号。

## 8. 获批后的最小实现顺序

前置条件保持：S3 external audit approved并关闭，然后S4-0、S4-1A逐级关闭。

```text
test(math): freeze IEEE classifier and midpoint bit semantics
feat(tvm): add isolated S4 ternary pack/select module
test(tvm): close cache key, module receipt and boundary CUDA probes
test(tvm): close real 18432 Ainput inventory and legacy isolation
feat(runtime): bind prepared views, generation and result finite gate
test(runtime): close fully re-signed endpoint tamper
```

在这些步骤完成前，S4-1B six-site production lowering、S4-1C gradient emitter、S4-1D evaluator formal、
timing与performance仍关闭。

## 9. 当前门禁

```text
S3 exchange = ready_for_audit/r001
S4 production code = closed
S4-1B0 math/source/TIR ABI/cache/receipt = implementation-ready design
S4-1B0 production implementation/formal = closed
S4-1B/1C/1D/timing/performance = closed
```

本文只建立实现前可证伪合同和CUDA design evidence，不升级任何production correctness或performance claim。
