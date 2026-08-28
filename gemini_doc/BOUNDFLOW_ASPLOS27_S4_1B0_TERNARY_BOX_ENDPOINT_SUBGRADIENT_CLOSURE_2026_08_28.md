---
status: design-corrected-implementation-closed
date: 2026-08-28
type: diagnosis-and-implementation-contract
topic: boundflow
slug: asplos27-s4-1b0-ternary-box-endpoint-subgradient-closure
stage: s04
supersedes: asplos27-s4-1bc-dag-adjoint-preflight-correction-v1-diagnosis
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B0：三元 Box Endpoint / Zero-Subgradient 语义闭合

## 0. 结论

S4 六 site selected-primal lowering 并没有被 residual DAG 或 fanout 证伪。此前 site19 的
`1.156e-3 / 9 sign mismatch`反例来自更窄、也更可修复的语义错误：输入 box concretization 使用了二元
endpoint 选择

```text
A_input >= 0 ? input_lower : input_upper
```

但 provider 的实际 lower concretization 是：

```text
lower(A) = A * center - abs(A) * radius + bias
```

PyTorch 在`A == 0`处采用`d abs(A) / dA = 0`，因此其关于`A`的精确 VJP endpoint 是三元的：

```text
A > 0  → input_lower
A < 0  → input_upper
A == 0 → input_center
```

用该三元规则重新执行完整六 site projection 后，六 dα 与 active dβ 全部通过。故 S4-1B0 的正确任务不是
重写一套全新的 DAG-adjoint runtime，而是冻结并实现**三元输入端点/零次梯度 ABI**。coefficient-program VJP
仍是规范定义；修正后的 selected-primal graph 是已经逐 site 验证的优化 lowering。

## 1. 独立只读根因链

探针固定仓库冻结的 ResNet2B property-0 production pre-state、同一 CUDA device、objective、α/β、split 与
六个 topology site，不修改 production 源码、不记录 timing。

### 1.1 逐层 tap

- exact post-ReLU19 coefficient adjoint到手写 pre19：max diff `0.05061209201812744`，9个sign mismatch；
- exact Conv2-right A18 branch adjoint到旧 selected17：max diff `0.04563498497009277`，sign mismatch `0`；
- 用 exact V18 重建 pre19：max diff `0`，sign exact；
- Conv0-right input-affine adjoint到旧二元 endpoint：max diff `0.03221011161804199`，sign exact。

不一致只出现在`A_input == 0`链路；它不是 residual 累加、Conv transpose、coefficient arena、β injection 或
site19 专属 ownership 错误。

### 1.2 冻结输入 coefficient inventory

formal production Ainput 共18,432个元素：

| 类别 | 数量 | selector编码 | endpoint |
|---|---:|---:|---|
| positive | 8,689 | `+1` | lower |
| negative | 9,137 | `-1` | upper |
| exact zero | 606 | `0` | center |
| 合计 | **18,432** | int8 ternary | — |

这606个exact-zero不是可忽略的数值噪声；把它们错误映射为lower会沿 Conv/Residual 图传播，并最终在site19
compressed ownership上形成此前反例。

## 2. 修正后的六 site 闭合结果

candidate只把 input endpoint 从二元规则改为三元规则；其余 selected-ReLU、Conv、Add、Flatten、Gemm、
compressed ownership 与 control 不变。

| site | owned width/domain | dα max abs diff | sign mismatch | 判定 |
|---:|---:|---:|---:|---|
| 17 | 164 | `7.3341652751e-09` | 0 | PASS |
| 19 | 132 | `4.2375177145e-08` | 0 | PASS |
| 23 | 121 | `4.06289473176e-08` | 0 | PASS |
| 25 | 86 | `4.47034835815e-08` | 0 | PASS |
| 28 | 178 | `8.19563865662e-08` | 0 | PASS |
| 31 | 27 | `1.63912773132e-07` | 0 | PASS |

active β31：

```text
max abs diff = 1.1920928955078125e-07
sign mismatch = 0
```

这些数字是 design-time read-only evidence，不升级 production correctness claim。S4-1B0 实现后仍须按冻结
five-fresh、artifact、replay与tamper协议正式关闭。

## 3. 规范与优化 lowering 的关系

规范数学 owner 保持：

```text
A_i = ReLU transform前incoming coefficient
T_i = A_i * selected_slope_i + beta_add_i
V_i = d lower / d T_i
dα_i = upstream * A_i * V_i
dβ_i = upstream * (-V_i * split_sign_i)
```

实现可用 selected-primal graph 生成`V_i`，但其等价依赖以下已冻结条件：

1. input affine的zero-subgradient按center表达；
2. residual/fanout topology与coefficient action provenance一致；
3. ReLU coefficient branch仍按provider的`A >= 0`二元规则，不误改为三元；
4. active β、split/history、bias与terminal lA ordering保持既有合同；
5. 六 site最终compressed gradient逐项对规范 oracle闭合。

因此“coefficient VJP规范”与“selected-primal优化 lowering”不是二选一：前者定义正确性，后者是在这些前提下
更便宜的实现。

## 4. ABI与存储设计

### 4.1 只修改 Ainput selector 语义

现有`sign_ainput`物理存储继续使用18,432-byte int8 buffer，但语义升级为`endpoint_ainput_v2`：

```text
+1 = A > 0 = lower
-1 = A < 0 = upper
 0 = A == 0 = center
```

总 sign/selector 存储仍为55,296 bytes，不新增 bitmap，也不增加动态分配。

### 4.2 其他五张 coefficient branch bitmap保持二元

`sign_a18/sign_a20/sign_a24/sign_a26/sign_a29`仍编码provider ReLU branch：

```text
1 iff A >= 0 else 0
```

原因是这些位置执行显式`where(A >= 0, lower_slope, upper_slope)`；zero明确属于lower branch，不涉及
`abs`在零点的次梯度。禁止为了接口统一而把它们改成三元语义。

### 4.3 derived center身份

R31 plan只有input lower/upper，provider也以`(x_U+x_L)/2`派生center；因此不得新增physical `input_center`
tensor、pointer或DLPack view。select TIR在zero分支按冻结operation order`(lower+upper)*float32(0.5)`派生center。
prepared program绑定lower/upper identity、derived-center formula schema/hash和selector schema hash。

现有`R31B2_PACK_AINPUT_SYMBOL`的“`>=0`打包为1，否则0”只能作为历史二元实现；S4 all-site模块不得原样复用。
需以新S4 schema和新symbol新增ternary pack/select lowering，避免同一buffer名称掩盖不同语义版本；S2/R31B2 v1
symbol与hash必须保持不变。

## 5. S4-1B0实现门禁

S3外审批准并关闭前仍不得写production S4代码。开放后，S4-1B0必须先于S4-1B完成：

1. 冻结`endpoint_ainput_v2`schema、三元pack/select与derived-center TIR合同；
2. exact-zero必须按逐位`A == 0`分类，不得用epsilon把非零值并入zero；
3. 绑定606/8,689/9,137 inventory及其content hash；
4. full PyTorch autograd、stdlib/float64公式、selected-primal candidate三方比较；
5. 六dα与active dβ最终gradient均`max abs/rel <=2e-5`且sign exact；
6. five fresh process绑定state/action/selector/lower/upper/derivation/module/layout hash；
7. fallback/eager/native shadow、dynamic allocation、timing与performance flag均为0/false；
8. replay必须从raw重算三元分类、六site gradient和summary，而非只验外层digest。

## 6. fail-closed与formal tamper

至少新增/改写以下稳定拒绝：

1. `BINARY_INPUT_ENDPOINT_SUBSTITUTED_FOR_TERNARY`；
2. `INPUT_ENDPOINT_ZERO_CLASS_COUNT_MISMATCH`；
3. `INPUT_CENTER_DERIVATION_SCHEMA_MISMATCH`；
4. `INPUT_ENDPOINT_SELECTOR_VALUE_OUT_OF_RANGE`；
5. `INPUT_ENDPOINT_SELECTOR_SCHEMA_MISMATCH`；
6. `SITE19_ZERO_SUBGRADIENT_COUNTEREXAMPLE_NOT_CLOSED`。

本阶段S4-4保留68类fully outer-resigned攻击总数；2026-08-29 S4-3 readiness另增3类，当前总数为71类。
当时后四类语义修正为：

- 65：把三元Ainput selector替换成旧二元`A>=0→lower`；
- 66：篡改zero class/count、lower/upper identity或derived-center formula；
- 67：将606个zero重写为positive/lower并全重签，复现旧site19错值；
- 68：terminal lA post-transform copy或spec-axis identity篡改。

## 7. 保留与被取代的结论

继续成立：两个coefficient arena、D1C residual scratch、37,464-element V/lA arena、compressed α/β ABI、
terminal lA incoming-A语义、phase-safe alias、无跨层dense A保存。

被取代：

- “ordinary selected-primal在真实DAG上原则性失效”；
- “site19要求另造完整coefficient-action adjoint runtime”；
- “输入endpoint bitmap只需二元`A>=0`”。

历史文件名中的`DAG_ADJOINT_PREFLIGHT_CORRECTION`为兼容引用保留，但其v1根因诊断以本文为准。

## 8. 当前门禁

```text
S3 exchange = ready_for_audit
S4-0/S4-1A implementation = closed pending S3
S4-1B0 ternary endpoint = design-corrected, implementation closed
S4-1B selected-primal lowering = reopened in design, implementation closed
S4-1C/S4-1D/timing/performance = closed
```

本文只纠正设计和正式门禁，不声明production correctness或performance。
