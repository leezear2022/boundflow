---
status: implemented-pending-clean-source-formal
updated: 2026-08-25T04:45:00+08:00
type: changelog
topic: boundflow
slug: r3-1b2-compiled-p-alpha-vjp-implementation
stage: s01
---

# R3-1b2 Compiled P-α VJP 实现记录

## 1. 本轮完成

在不修改 R3-1b1 formal code chain 的前提下，新增了独立的 b2 CUDA/TIR module 与 custom
Function：

- 10 个 b2 TIR symbols：forward arena clear、4 个 coefficient sign bitmap、A20 sign
  checkpoint、3 个 effective-value replay、Conv10-right A26 重算与 compressed gradient；
- 复用 b1 的两个 coefficient arena 和已冻结 reverse kernels，不创建第三 coefficient buffer；
- custom forward 调用一次 b1 compiled full-lower，随后 compiled clear 两个 arena，避免将 forward
  dense A 留到 backward；
- custom backward 重放 lower coefficient，只保留 4 个 byte sign bitmap（共 43,008 B）和一个
  pre25 value workspace；A26 在输出 gradient 前重算，不跨阶段保存；
- gradient 直接写 production `[2,1,6,86]` layout，direction 1 为零，P beta absent；
- candidate backward 不调用 `torch.autograd.grad`、native oracle、CROWN eager evaluator或
  `_evaluate_full_region`；外层 `torch.autograd.grad` 只负责触发已注册 custom VJP。

## 2. 单 worker CUDA 结果

相对独立 native oracle：

- final lower max abs diff=`3.933906555175781e-06`，sign exact；
- compressed dα max abs diff=`6.146728992462158e-08`，sign exact；
- dα nonzero=`281/281`；
- custom forward/backward=`1/1`；
- b1 forward/backward launches=`15/15`，b2 launches=`10`；
- coefficient scratch=`2`，sign bitmap=`4` / `43,008 B`；
- saved dense A=`0`，Python-visible intermediate coefficient=`0`；
- warm dynamic CUDA allocated bytes=`0`；
- static DLPack pointers=`79/79`，upstream pointer=`1/1`；
- b2 module/device hashes=`3871bf0e42ec9ce129d32bb408a5e9320d51026da6998aa81ebf0415822be575` /
  `842cb3f28c66ec013a9a78aded3741ed63f36935f0183454e967fdf606413fd8`。

## 3. 验证

```text
pytest -q tests/test_r3_compiled_p_alpha_vjp.py tests/test_r3_p_alpha_vjp_oracle.py
```

结果=`5 passed`。覆盖 exact symbols/global-workspace、native parity、receipt/ownership、default-stream
fail closed 和 candidate source escape gate。mypy clean；pylint在限定动态API豁免后为`10.00/10`。

## 4. 当前边界与下一动作

### 4.1 Formal protocol implementation

已新增 fresh worker、raw-first artifact generator/replay、12类fully re-signed tamper probe与artifact
tests。协议固定 source/code revision、capture/model、trace/plan、b1/b2 module、完整 lower/dα raw、
float32 tensor hash与所有ownership receipt；replay从raw独立重算数值差、sign、nonzero和tensor hash，
不采信summary。当前仍需先提交这些脚本为clean source，再运行正式artifact。

当前状态仅为 `IMPLEMENTED-R3-1B2-PENDING-CLEAN-SOURCE-FORMAL`。单次同进程 smoke 不能关闭
R3-1b2，也不能证明 five-fresh memory gate。`timing_recorded=false`、
`performance_claimed=false`、`r3_1_admitted=false`。

下一步提交 clean source，再从该固定 revision 生成 single-worker raw-first artifact、独立 semantic
replay与 fully re-signed tamper。该门禁通过后才决定是否进入 b3 five-fresh；不接 optimizer、不计时。
