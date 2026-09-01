---
status: implemented-isolated-correctness
updated: 2026-08-26T19:15:00+08:00
type: changelog
topic: boundflow
slug: mr5-generalized-multi-conv-tir
stage: s01
---

# MR5 Generalized Multi-Conv TIR 修改记录

## 1. 实现范围

- 新增shape/stride-keyed `MR5GeneralizedConvSignatureV1`，分别冻结C0/C1/C2的Cin/Cout、输入/输出
  spatial、stride/padding/output-padding与schedule thread extent；
- 新增真正支持stride 1/2的lower-CROWN ReLU slope/intercept + ConvTranspose forward TIR；
- 新增与forward索引严格转置的custom backward TIR，返回incoming-A与full-α VJP；
- 新增signature-keyed module cache、TIR/module/source/workspace receipt、当前stream和DLPack零拷贝门禁；
- 新增独立PyTorch `conv_transpose2d`表达式，不复用TIR作为oracle；
- 保持`performance_claimed=false`语义：本切片无timer、无production替换、无speedup claim。

## 2. 与旧P-anchor实现的实质差异

旧实现只支持`16→16,8×8,stride=1`。本实现实际编译并运行：

| Site | Conv primal | CROWN incoming A | CROWN result A | stride/output-padding |
|---|---|---|---|---|
| C0 | `3×32×32→8×16×16` | `[6,1,8,16,16]` | `[6,1,3,32,32]` | `2/1` |
| C1 | `8×16×16→16×8×8` | `[6,1,16,8,8]` | `[6,1,8,16,16]` | `2/1` |
| C2 | `16×8×8→16×8×8` | `[6,1,16,8,8]` | `[6,1,16,8,8]` | `1/0` |

stride-2 forward使用整除与边界门禁恢复ConvTranspose gather；backward直接用
`ih=oh*stride-padding+kh*dilation`构造精确adjoint，未沿用C2的stride-1索引。

## 3. 验证

- 三site unscheduled/scheduled TIR结构与不同signature identity：PASS；
- 三site真实CUDA forward、bias、incoming-A VJP、full-α VJP对独立PyTorch oracle：PASS；
- 每sitelaunch=`1/1`、fallback/eager=`0/0`、DLPack pointer exact、module receipt绑定：PASS；
- stride alias和nonfinite输入fail closed：PASS；
- focused=`8 passed`；
- mypy三文件clean；
- pylint=`10.00/10`；
- Black与`git diff --check`：PASS。

## 4. 尚未完成

该提交只关闭MR5第8节第2步“generalized TIR + isolated math tests”。三site production route、5 pair
formal、optimizer trajectory、atomic rollback、replay/tamper和尚未运行的full regression仍pending，不能
开放timing。
