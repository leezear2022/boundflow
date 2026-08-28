---
status: implementation-ready-gate-closed
date: 2026-08-28
type: source-map-and-patch-blueprint
topic: boundflow
slug: asplos27-s4-1b0-ternary-endpoint-implementation-readiness
stage: s04
depends-on: asplos27-s4-1b0-ternary-box-endpoint-subgradient-closure
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B0：三元 Endpoint 源码映射与逐文件实施蓝图

## 0. 结论

S4-1B0已经从数学合同压到可直接编码的最小补丁，但S3外审前仍不写production代码。最终实施选择为：

1. 保持S2/R31B2所有v1 symbol与source/module hash不变；
2. S4新增独立schema和两个新symbol：ternary pack、ternary input select；
3. 不新增`input_center`tensor；zero分支直接从existing lower/upper派生`(lower+upper)*0.5`；
4. existing 18,432-byte int8 Ainput buffer原地升级为`-1/0/+1`selector；
5. coefficient VJP继续作为规范oracle，S4 selected-primal graph消费新selector；
6. plan tensor count、prepared DLPack input count、warm allocation和workspace均不因center增加。

这比“增加一个center tensor”更忠实于provider源码，也避免污染已冻结历史artifact。

## 1. 当前真实源码映射

| 责任 | 当前文件/符号 | 当前语义 | S4动作 |
|---|---|---|---|
| Ainput pack | `backends/tvm/r3_p_alpha_vjp.py::_pack_sign_primfunc` | `A>=0→1 else0` | 不修改；新增ternary pack |
| v1导出 | `R31B2_PACK_AINPUT_SYMBOL` | `boundflow_r31b2_pack_ainput_sign` | 历史冻结，禁止改义 |
| direct pre17 select | `_effective_pre17_primfunc` | nonzero→lower, zero→upper | S4新pre17/select lowering |
| S2 input select | `asplos27_s2_selected_value.py::_input_select_primfunc` | nonzero→lower, zero→upper | v1冻结；S4建v2 |
| S2 Relax ABI | `build_s2_selected_value_relax_module_v1` | 28 inputs，第三项sign | v1冻结；S4 selector仍占同一逻辑slot |
| runtime pack点 | `r3_compiled_p_alpha_vjp.py::_coefficient_sign_pass` | Conv0-right后pack Ainput | S4调用新pack symbol |
| runtime value点 | `_effective_value_pass` | lower/upper/sign→pre17 | S4调用新select/pre17 |
| static input truth | `R31FullRegionPlanV1.tensor_specs` | lower、upper、objective + params + 6×4 | 不新增center spec |
| provider truth | `auto_LiRPA/perturbations.py` | `center=(x_U+x_L)/2` | 作为pinned数学owner |

当前`R31FullRegionPlanV1.validate()`机械要求tensor数为`3 + len(parameter_names) + 4*6`。加入center会改变plan
identity、runtime bind长度、DLPack views、所有历史plan/module receipt；而provider没有独立第三份center source truth，
所以不应这样做。

## 2. 新S4 backend合同

建议新增：

```text
boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py
```

冻结schema与symbol：

```text
schema = boundflow.asplos27-s4-ternary-endpoint/v1
pack   = boundflow_s4_pack_ainput_endpoint_ternary
select = boundflow_s4_select_input_endpoint_ternary
```

不得复用旧symbol名称后改变实现；否则S2/S3 frozen module hash会被静默重定义。

### 2.1 pack TIR

```text
coefficient[18432] float32
  → endpoint_selector[18432] int8

selector[i] = !isfinite(coefficient[i]) ? -128
            : coefficient[i] > 0 ? +1
            : coefficient[i] < 0 ? -1
            : 0
```

这会把`+0.0`与`-0.0`都映射为0；任何非零subnormal仍按其真实符号映射，禁止epsilon分类。`-128`是
reserved invalid sentinel，不是第四种合法endpoint。

### 2.2 select TIR

```text
selected[i] = selector[i] == +1 ? lower[i]
            : selector[i] == -1 ? upper[i]
            : selector[i] ==  0 ? (lower[i] + upper[i]) * float32(0.5)
            : canonical_nan
```

midpoint的operation order必须与pinned provider保持`add→multiply by 0.5`；不得改成`lower*0.5+upper*0.5`后
假设bitwise identity。S4 receipt绑定derivation schema/hash，而不是绑定一个不存在的center input pointer。

### 2.3 schedule/物理账

- 两个elementwise TIR均使用当前256 threads schedule helper；
- global workspace=`0`；
- selector buffer=`18,432 int8 / 18,432 bytes`；
- extra center tensor/view/allocation=`0`；
- select output应写入existing selected graph/pre17 producer路径，不新增Python-visible tensor；
- formal S4整图最终是否单独保留select kernel由S4-1B profile决定，S4-1B0不计时。

## 3. 为什么不能直接修改v1

以下资产已经以源码/IR/module hash进入S2/S3 artifact：

- `S2_SELECTED_VALUE_SCHEMA=v1`；
- source/partitioned/lowered Relax IR hash；
- selected device source hashes；
- R31B2 module/device source hash与exact exported symbols；
- S2 prepared argument count=`28`、prepare DLPack view count=`30`；
- S3 optimizer/runtime以S2 prepared program为内部executor。

直接修改`_input_select_primfunc`或`R31B2_PACK_AINPUT_SYMBOL`会使历史replay变成“同名不同义”，并可能让外审
误以为S2/S3原始性能来自三元实现。正确策略是：旧artifact继续按v1 replay；S4 v2以新symbol/hash独立形成证据。

## 4. runtime与receipt补丁映射

S4-1B0不应直接修改`PreparedS2CrownProgramV1`。在S4 evaluator owner中加入：

```text
endpoint_selector_schema
endpoint_pack_module_hash
endpoint_select_module_hash
endpoint_positive_count
endpoint_negative_count
endpoint_zero_count
derived_center_formula_schema/hash
input_lower_hash / input_upper_hash
selector_generation / parameter_state_version / evaluation_ordinal
extra_center_tensor_count=0
extra_center_dlpack_view_count=0
workspace_bytes=0
performance_claimed=false
```

formal fixture预期`positive/negative/zero=8689/9137/606`。计数只在correctness/formal路径冻结；后续timing路径不得
为每次warm call增加host同步统计。production runtime只需生成selector并将其generation绑定到本次coefficient pass。

nonfinite coefficient不应被两个comparison都false后静默当zero。pack写`-128`，select传播canonical NaN；
S4-1D existing final-finite gate在result lease/commit前拒绝。失败慢路径再扫描selector sentinel并给出稳定
`NONFINITE_AINPUT_COEFFICIENT`，正常路径不新增status buffer、计数kernel或额外host同步。

## 5. 逐提交实施顺序

S3 approved+closed且S4-0/1A依次关闭后：

1. `test(math): freeze ternary endpoint and signed-zero semantics`
   - CPU/PyTorch独立公式；
   - asymmetric lower/upper；
   - positive、negative、`+0.0`、`-0.0`、subnormal；
   - nonfinite必须变为`-128`并在result lease前拒绝。
2. `feat(tvm): add isolated S4 ternary endpoint module`
   - exact two symbols；
   - module/source hash；
   - zero workspace；
   - 不改任何v1 symbol。
3. `test(tvm): run formal Ainput pack/select CUDA probe`
   - exact`8689/9137/606`；
   - selected output逐位等于independent formula；
   - old binary明确误编码606；
   - no center tensor/view/allocation。
4. `feat(runtime): bind selector generation to S4 evaluator`
   - same ordinal/version/stream；
   - lower/upper/derivation identity；
   - fail before result lease/commit。
5. `test(runtime): close fully re-signed endpoint tamper set`
   - 然后才开放S4-1B six-site graph。

## 6. 冻结测试矩阵

### 6.1 正向

1. `A>0→lower`；
2. `A<0→upper`；
3. `+0.0→midpoint`；
4. `-0.0→midpoint`；
5. positive/negative subnormal不能被归零；
6. asymmetric bounds midpoint exact；
7. formal 18,432 inventory exact；
8. new TIR与independent PyTorch逐位相等；
9. new module workspace=0；
10. old S2/R31B2 hashes、symbols、tests、artifact replay不变。

### 6.2 负向/fail-closed

1. selector出现`-2/+2/-128`却仍发布result；
2. zero改为positive；
3. binary v1 symbol冒充ternary；
4. midpoint公式改写/operation order漂移；
5. lower/upper identity或generation漂移；
6. selector来自旧ordinal/version；
7. selector dtype改为bool/int32；
8. selector shape/stride/offset漂移；
9. `+0.0/-0.0`分类不一致；
10. epsilon吞掉subnormal；
11. NaN被当zero；
12. Inf进入result lease；
13. 新增center tensor或warm DLPack view；
14. 修改v1 source/module hash；
15. 全重签后删除zero count或derivation hash；
16. `performance_claimed=true`。

## 7. 本轮独立CUDA探针

### 7.1 小型signed-zero探针

内存中编译两个新TIR，输入含negative、`-0.0`、`+0.0`、positive和正负subnormal：

```text
status=PASS
selector=[-1,0,0,1,-1,1,-1,1]
workspace_bytes=0
diagnostic_module_hash=8ecfca40...b630c0
```

### 7.2 nonfinite sentinel探针

独立CUDA/TIR验证NaN、`+Inf`、`-Inf`不会误归zero：

```text
status=PASS
selector=[-128,-128,-128,0,1,-1]
invalid_outputs_nan=3
extra_status_buffer=0
```

### 7.3 formal production Ainput探针

从existing compiled coefficient pass读取真实Ainput，再由独立新TIR执行pack/select：

```text
status=PASS
positive/negative/zero=8689/9137/606
old_binary_zero_misclassified=606
selected_hash=7e95e075...39b652
derived_center_hash=d6164a06...f5b003
diagnostic_tir_module_hash=eb3e7ec6...250fb5
extra_center_tensor_count=0
selector_bytes=18432
```

探针是design-time evidence，不是production artifact或correctness closure；正式实现仍需five-fresh、raw/replay和
fully re-signed tamper。

## 8. 当前门禁

```text
S3 exchange = ready_for_audit/r001
S4 production code = closed
S4-1B0 source map/ABI/probe = implementation-ready
S4-1B0 formal implementation = closed
S4-1B/1C/1D/timing/performance = closed
```

唯一合法下一外部动作仍是S3审计；本文只把获批后的第一刀变成确定、最小且不破坏历史hash的补丁。
