---
status: implementation-open-no-external-audit-this-round
date: 2026-08-28
type: plan
topic: boundflow
slug: asplos27-s2-coarse-crown-custom-vjp
execution-authority: true
external-audit: deferred-by-user
performance-claimed: false
---

# ASPLOS’27 S2 coarse CROWN + custom VJP canonical pipeline计划

## 1. 目标和边界

S2把已有R3/B4-B2 correctness资产迁入S1已经建立的canonical compiled path，不新造solver IR：

```text
compressed α/β + bounds + weights + objective
  → verification-aware coefficient reverse wavefront
  → selected effective-value Conv chain
  → compact lower
  ↔ direct custom VJP
  → compressed dα（P-anchor）
```

本轮用户明确推迟中间外审，因此按短提交连续完成实现、correctness、timing、artifact和closure，最后一次性交
外审。same-solver、optimizer 10/9、BaB、总体10×仍关闭。

## 2. 开工前只读归因

在RTX 4060 Laptop、P-anchor、6 domains上，30次warm诊断得到：

| scope | median |
|---|---:|
| native eager forward+VJP | `9.00198 ms` |
| existing D2B direct forward+VJP | `6.48346 ms` |
| D2B forward | `1.74483 ms` |
| D2B coefficient/sign | `0.86938 ms` |
| D2B effective-value | `3.69254 ms` |
| D2B recompute-A26 | `0.07373 ms` |

lower max diff=`3.09944e-6`，compressed dα max diff=`6.14673e-8`，sign exact。该表是路线诊断，
不是formal性能claim。

根因是旧`effective_pre23`把
`selected-ReLU17→Conv2→selected-ReLU19→Conv4 + shortcut Conv5`内联到每个输出线程；每线程串行重算
前层Conv，约六千万级标量工作，破坏了卷积复用。旧D1C residual schedule也仍标记
`serial-reference/vector_width=1`。所以S2不得只把旧D2B包进新API。

## 3. S2-A：canonical effective-value Relax/cuDNN chain

建立一个standard Relax function：

```text
select(input L/U, sign_Ainput)
  → cudnn Conv0(stride2,pad1)+bias
  → select ReLU17(lower/upper/α/sign_A18)
  → cudnn Conv2(stride2,pad1)+bias
  → select ReLU19(lower/upper/α/sign_A20)
  → cudnn Conv4(pad1)+bias
  + cudnn shortcut Conv5(stride2)+bias
  → select ReLU23(lower/upper/α/sign_A24)
  → cudnn Conv8(pad1)+bias
  → pre25
```

要求：

- 不是PyTorch eager/torch.compile fallback；5个Conv（Conv0、Conv2、Conv4、shortcut
  Conv5、Conv8）必须进入TVM cuDNN codegen；
- elementwise select/slope/intercept由Relax/TIR执行；
- module、parameters、metadata和DLPack view只在prepare构造；
- warm path一个prepared VM/CUDA-Graph submission；
- pre17/pre19/pre23不逃逸，不进入Python，不跨evaluation保存；
- 与旧D2B effective path逐张量allclose/sign通过；
- cuDNN不可用时fail closed，不静默退回serial TIR形成formal candidate。

## 4. S2-B：coefficient wavefront与custom VJP owner

复用`PreparedR3D2BStagedBackwardCandidateV1`的数学和两slot arena，但替换其effective-value pass；然后按profile
决定是否继续优化coefficient wavefront：

1. forward coefficient reverse wavefront；
2. backward coefficient/sign wavefront；
3. S2-A effective-value chain；
4. recompute-A26 + compressed-gradient kernel。

直接API为：

```text
PreparedS2CrownProgram.run_vjp(dynamic_alpha, upstream)
  → compact_lower + compressed_dalpha + ExecutionReceipt
```

它是custom VJP boundary，不依赖PyTorch autograd保存dense A；PyTorch `autograd.Function`只保留为兼容adapter。
P-anchor schema/plan不得硬编码scratch slot数；实际two-slot由已有lifetime trace证明并进入compile receipt。

## 5. Correctness/ownership门禁

- native、旧direct D2B、S2 canonical三方lower/dα；
- lower `atol=rtol=2e-4`，gradient `atol=rtol=2e-4`，sign exact；
- active β必须真实执行（不是伪零tensor），β/state version不变；
- saved dense A=`0`，Python-visible intermediate A=`0`；
- warm allocation=`0`、warm DLPack construction=`0`；
- module cache不能持dynamic tensor；
- source state、plan、lifetime trace、Relax、lowered module、device source、cuDNN partitions逐层hash绑定；
- fallback/eager/native shadow=`0`；
- input/α/β/metadata pointer、shape、dtype、version漂移均reject-before-launch；
- 至少10类fully outer-resigned tamper由semantic replay拒绝。

## 6. Timing协议与kill gate

先做单evaluation同scope三方：

```text
N = native eager forward+VJP
D = existing direct D2B forward+VJP
P = S2 canonical prepared forward+VJP
```

6个fresh process、`NDP/NPD/DNP/DPN/PND/PDN`六全排列，每worker 30 groups；device boundary sync、相同
non-default stream、相同输入与upstream，compile/cold单报。

冻结门槛：

- P/D geomean `>=0.90x`；
- P/N geomean `>=4.00x`、worst `>=3.50x`为S2研究目标；
- 若correctness/structure通过但只有`>=1.20x`，关闭`VALIDATED-S2-4X`，保留
  `VALIDATED-REDUCED-S2-CANONICAL-CROWN`并重新归因；
- 若P/N worst `<0.98x`，S2性能NO-GO，不进入optimizer；
- 不得用old B4-B2 local 4.898×、D2B region 53.9×或S1 IBP 2.50×替代本协议。

## 7. 提交顺序

1. `docs: freeze ASPLOS27 S2 coarse CROWN canonical protocol`
2. `build(tvm): enable and verify conda cuDNN support`
3. `feat(compiler): add selected-value CROWN Relax cuDNN chain`
4. `feat(runtime): add prepared S2 coarse CROWN direct VJP owner`
5. `test(runtime): close S2 correctness ownership and tamper gates`
6. `bench(runtime): close S2 six-fresh formal`
7. `docs(docops): deliver S1+S2 combined next-round audit`

每步都记录DocOps；本轮不在步骤间等待外审。

## 8. Claim边界

即使达到4×，也只允许说P-anchor单evaluation coarse CROWN/custom VJP同scope结果；不能说optimizer、exact-call、
same-solver、complete-query、跨模型或总体10×。S3只有在S2 worst `>=0.98x`且trajectory资产可复用时开放。
