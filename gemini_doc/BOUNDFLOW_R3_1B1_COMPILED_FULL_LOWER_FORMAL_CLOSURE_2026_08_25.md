---
status: validated-r3-1b1-compiled-full-lower
updated: 2026-08-25T10:55:00+08:00
type: closure
topic: boundflow
slug: r3-1b1-compiled-full-lower-formal-closure
stage: s01
---

# BoundFlow R3-1b1 compiled full-lower 正式关闭

## 1. Verdict

R3-1b1 关闭为 `VALIDATED-R3-1B1-COMPILED-FULL-LOWER`，只开放 R3-1b2 compiled
P-alpha VJP。R3-1 整体仍未 admit；five-fresh、optimizer mutation与 timing继续关闭。

正式 artifact：`artifacts/r3-structured-owner/r3-1b1-full-lower-v1`。source commit =
`bdfa53dc4d92bc6a3bc2dab3c31996e49209e4e3`。

## 2. Semantic result

独立 fresh subprocess 同时运行 eager native oracle 与 compiled full-lower candidate：

- native lower：`[-0.37089300, -0.42217249, -0.47373629, -0.36606002,
  -0.44085169, -0.49036360]`；
- candidate lower：`[-0.37089586, -0.42217457, -0.47374010, -0.36605799,
  -0.44085133, -0.49036396]`；
- max abs diff=`3.814697265625e-06 <= 2e-4`；finite/sign exact；
- artifact replay从 raw重新计算并逐字节复现 summary hash
  `d7a6702c713c5606b31408e9e597e49dda805e901d949c34559ab33094d96f2e`。

## 3. Compilation and ownership result

- 15 个 CUDA TIR symbol 覆盖 objective seed 到 input concretize 的完整 lower recurrence；
- module hash=`003f38c0cccee27cd210014fadda9c7fa8f9b2fae2e93b853be0a3c3101649ba`；
- device source hash=`c4112b4055636259cde16514be7f58145bcfa316256121bdae4f1c60778e1ddf`；
- coefficient scratch恰为2块，每块`18,432` float32=`73,728 B`，指针不同；
- global workspace=`0`，Python-visible intermediate coefficient=`0`；
- DLPack pointer=`70/70` exact；non-default PyTorch stream与TVM FFI stream exact；
- warm dynamic allocated bytes=`0`；fallback/eager/native shadow=`0/0/0`；
- module cache tensor-free；production tensor、lookup metadata和scratch均归 plan instance所有。

两个 residual region 的跨层系数只在 TIR local scalar 中重算，没有保存第三份 dense A，也没有调用
`LinearOperator.to_dense()`。

## 4. Fail-closed evidence

10/10 fully re-signed attacks被 semantic replay拒绝：candidate/native lower、module/device source hash、
launch count、scratch alias、warm allocation、stream mismatch、compiled-region 与 DLPack exact count。

专项测试=`5 passed`；全量=`1585 passed, 3 skipped, 6 warnings`；3 个 skip仍为既有 TVM
重复编译与冻结 VNN-COMP checkout环境边界。mypy clean、pylint=`10.00/10`、black/diff/DocOps
lint通过。

## 5. Claim boundary and next action

本关闭仅证明 no-grad full-lower compiled correctness/ownership：

- `compiled_region=true`；
- `custom_vjp=false`；
- `r3_1_admitted=false`；
- `timing_recorded=false`；
- `performance_claimed=false`。

下一唯一动作是 R3-1b2：实现 compiled P-alpha VJP，输出 compressed dα `[2,1,6,86]`，禁止
`torch.autograd.grad`、native shadow或保存 dense A。b2 单 worker通过前不得启动 five-fresh b3。
