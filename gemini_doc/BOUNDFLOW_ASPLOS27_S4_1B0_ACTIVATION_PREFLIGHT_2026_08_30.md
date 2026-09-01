---
status: preflight-pass-waiting-s4-1a-external-audit
date: 2026-08-30
type: activation-preflight
topic: boundflow
slug: asplos27-s4-1b0-activation-preflight
stage: s04
depends-on: external-approved-s4-1a-ordered-buffer-prepare
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B0 激活预检：三元 endpoint TVM/TIR 边界

## 0. 结论

在 S4-1A 外审尚未返回期间，对 S4-1B0 做了一次不修改 production 的激活预检。结果为
`PREFLIGHT-PASS-WAITING-S4-1A-EXTERNAL-AUDIT`：

- 施工合同的 canonical construction model 可独立重算，SHA256 与冻结值逐位一致；
- 当前 TVM/CUDA 环境可构建并执行独立 ternary pack/select TIR 原型；
- IEEE-754 nonfinite、signed zero、subnormal、canonical qNaN 和 midpoint operation order 均按合同通过；
- 新 backend/test 路径、新 schema 和两个 symbol 没有 production 命名冲突；
- 旧 R31B2/S2 binary-v1 路径仍保持原语义，没有被本轮改写；
- S4-1B0 代码、formal、timing 和 performance 仍全部关闭。

本报告是 activation readiness evidence，不是 `VALIDATED-S4-1B0`，也不替代 S4-1A 外审。

## 1. 权威边界

预检基线：

```text
branch = feat/rvir-v4-production-state-ownership-v1
HEAD = dc0abe9
S4-1A exchange = ready_for_audit / round 1
next gate = external-approved-s4-1a-ordered-buffer-prepare
```

S4-1B0 唯一施工合同仍为：

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md
```

本轮没有修改 claims map、execution memo 或 current status，因为这些权威文档正确地保持：S4-1A 待外审，
S4-1B0 及后继代码关闭。

## 2. 当前源码与命名空间盘点

### 2.1 legacy 路径保持不变

当前 production 资产：

```text
boundflow/backends/tvm/r3_p_alpha_vjp.py
  git blob = a99d771c6140d59f5cf895fea1dd5a68c046703d
  legacy symbol = boundflow_r31b2_pack_ainput_sign
  legacy pack = coefficient >= 0 -> 1, else 0

boundflow/backends/tvm/asplos27_s2_selected_value.py
  git blob = 7ce4c2548d9ffb54ff39d72f0fd4a7d46ca7906a
  legacy select = sign != 0 -> lower, else upper
```

两者是 binary-v1 历史资产。S4-1B0 必须新增独立 schema/symbol，不能原地改变它们。

### 2.2 proposed 路径无冲突

下列文件在预检基线均不存在：

```text
boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py
tests/test_asplos27_s4_ternary_endpoint.py
```

在排除 `gemini_doc/` 与 `artifacts/` 后，仓库没有出现下列待实现 identity：

```text
boundflow.asplos27-s4-ternary-endpoint/v1
boundflow_s4_pack_ainput_endpoint_ternary
boundflow_s4_select_input_endpoint_ternary
```

因此开工时可以用独立 backend module，不需要迁移旧 symbol，也不需要新增顶层 IR。

## 3. 当前物理环境

预检实际环境：

```json
{"cc":[8,9],"cuda_available":true,"device":"NVIDIA GeForce RTX 4060 Laptop GPU","torch":"2.12.1+cu132","tvm":"0.23.dev0","tvm_ffi":"0.1.3.dev11+gae346ec92"}
```

这里只证明当前机器能执行下一阶段 correctness；不形成跨设备可移植性或性能 claim。

## 4. construction model 机械复算

从施工合同第 12 节唯一 JSON block 读取对象，以 UTF-8、`sort_keys=True`、
`separators=(',', ':')` canonicalize 后重算：

```text
computed = 5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a
expected = 5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a
match = true
top-level fields = 12
```

这证明本轮读取的 scope、math、cache、storage、formal topology 与施工合同冻结对象一致；实现仍必须从
实际代码对象重建 identity，禁止硬编码该 hash 为 PASS。

## 5. 仓库外 TVM/CUDA 位级原型

### 5.1 原型范围

在一次性 Python 进程中用 TVM TE/TIR 构建两个临时 symbol：

```text
s4_preflight_pack
s4_preflight_select
```

它们只存在于进程内，没有写入 `boundflow/`、`tests/` 或 artifact。pack 使用 float32 raw bits exponent mask；
select 使用 `+1/-1/0/other` 四路与 canonical qNaN `0x7fc00000`。

### 5.2 实际结果

```json
{
  "add_then_mul_bits": ["0x7f800000", "0x1"],
  "canonical_qnan_exact": true,
  "dlpack_pointer_exact": 5,
  "invalid_output_bits": ["0x7fc00000"],
  "midpoint_counterexample_count": 2,
  "mul_then_add_bits": ["0x7f7fffff", "0x0"],
  "selector": [-128, -128, -1, -1, -1, 0, 0, 1, 1, 1, -128, -128, 1, -1, 1, -1],
  "selector_exact": true
}
```

因此当前工具链现场确认：

1. NaN 与 `+/-Inf` 都进入 `-128` sentinel；
2. `+0.0/-0.0` 都进入 midpoint 分支；
3. 正负最小 subnormal 保留符号；
4. 非法 selector 统一输出 bits=`0x7fc00000`；
5. `(lower+upper)*0.5` 与 `lower*0.5+upper*0.5` 在 max-finite 和 min-subnormal 上确有两个逐位反例；
6. 五个 caller-owned DLPack view pointer 都 exact。

### 5.3 首次原型失败的真实原因

第一次原型把固定 `N=16` module 的 midpoint 反例输入错误地缩成长度 2。TVM runtime shape guard 在 kernel
launch 前拒绝：

```text
Argument ... shape[0] has an unsatisfied constraint: 16 == shape[0]
```

检查 PrimFunc 后确认参数顺序仍为 `lower, upper, selector, output`，不存在 TE 参数重排问题。修正方法是保持
ABI shape 为 16，只在前两个元素放置反例。修正后全部断言通过。

这条失败说明正式实现必须：

- 在 module receipt 中绑定 `numel`；
- 在 cache key 中绑定 build spec/TIR hash；
- 依赖 runtime shape guard，并在 Prepared owner 中提前给出稳定 `SHAPE_MISMATCH` reason；
- formal 反例不得通过改变 production ABI shape 构造。

## 6. 外审批准后的精确开工顺序

S4-1A 获批并由 executor 正式 close 后，只允许：

1. 激活施工合同，更新 status/claims 文档但保持 timing=false；
2. 先提交 CPU/bit oracle 和 construction-model tests；
3. 再新增 isolated TVM module、cache、receipt 与 prepared probe；
4. 完成 20 类 stable negative、non-default stream 与 cache miss/hit；
5. 最后生成 `5 positive + 1 cache + 5 fault` fresh-process artifact；
6. external audit 批准前不得进入 S4-1B production phase/arena alias。

不允许在第一刀中接 evaluator、S4-1A ticket、optimizer、same-solver 或 timing。

## 7. 当前门禁

```text
S4-1A external audit                         = pending
S4-1B0 construction model/source preflight  = PASS
S4-1B0 implementation authority              = false
S4-1B0 production code                       = closed
S4-1B0 formal                                = closed
S4-1B/timing/performance                     = closed
```
