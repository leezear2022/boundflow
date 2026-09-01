---
status: validated-formal-closure
updated: 2026-08-25T10:55:00+08:00
type: changelog
topic: boundflow
slug: r3-1b1-compiled-full-lower-implementation
stage: s01
---

# BoundFlow R3-1b1 compiled full-lower implementation

> **Formal closure**：source=`bdfa53d`的artifact/replay与10/10全重签tamper已通过；本实现记录的
> pending状态由`BOUNDFLOW_R3_1B1_COMPILED_FULL_LOWER_FORMAL_CLOSURE_2026_08_25.md`取代。
> closure后的全量回归为`1585 passed, 3 skipped`。

## 1. Scope

本轮只实现预注册 DAG 的 R3-1b1：冻结 ResNet2B property 0、start node `25/Conv_8`、
domain/spec `6/1` 的 lower-only compiled forward。没有实现 custom VJP、optimizer mutation、five-fresh
或 timing，也没有打开 R3-2A。

## 2. Implementation

- 新增 `boundflow/backends/tvm/r3_full_lower_forward.py`：15 个 CUDA TIR 导出符号覆盖 objective
  seed、Gemm16、ReLU31、Gemm14、ReLU28、Add11 residual、ReLU23、Add6 residual、
  ReLU17、Conv0 与 input concretize；
- 两个 residual region 使用跨层局部标量重算，不生成第三个 dense coefficient workspace；
- 新增 `boundflow/runtime/r3_full_lower_forward_tir.py`：plan-instance owns 两个
  `18,432`-element float32 scratch、一个 `[6]` bias accumulator和一个 `[6]` output；
- module cache 只保存 compiled code；production tensors、compressed alpha/beta、lookup metadata与
  DLPack views均由 plan instance持有；
- P-anchor beta仍为空；只为真实 active beta 的 ReLU31建立一份 compressed beta map；
- runtime强制 non-default current stream、DLPack exact pointer、source/trace/plan/tensor/metadata identity
  与 single-use launch；无 fallback、eager或native shadow路径。

## 3. Preliminary GPU smoke

在 RTX 4060 Laptop / sm_89、独立 non-default stream 上：

- native lower：`[-0.37089324, -0.42217249, -0.47373629, -0.36606026,
  -0.44085169, -0.49036312]`；
- compiled lower：`[-0.37089586, -0.42217457, -0.47374010, -0.36605799,
  -0.44085133, -0.49036396]`；
- max abs diff：`3.814697265625e-06`，小于冻结的 `2e-4`；finite/sign exact；
- launch `15`，coefficient scratch `2`，每块 `73,728 B`；
- DLPack pointer `70/70` exact；current stream与 TVM FFI stream exact；
- warm dynamic allocated bytes `0`；Python-visible intermediate coefficient `0`；
- module hash `003f38c0cccee27cd210014fadda9c7fa8f9b2fae2e93b853be0a3c3101649ba`；
- device source hash `c4112b4055636259cde16514be7f58145bcfa316256121bdae4f1c60778e1ddf`。

专项测试 `3 passed`，覆盖静态完整符号/global-workspace门禁、真实 GPU parity/ownership和 default
stream、trace、tensor、metadata篡改拒绝。

## 4. Claim boundary

当前状态仅为 `IMPLEMENTED-R3-1B1-SMOKE-PASSED-ARTIFACT-PENDING`。上述单次 smoke 不是正式
artifact closure；`timing_recorded=false`、`performance_claimed=false`、`r3_1_admitted=false`。

下一步生成独立 raw-first R3-1b1 artifact、semantic replay与 fully re-signed tamper；通过后才可关闭 b1并
开放 b2 compiled P-alpha VJP。

## 5. Artifact tooling prepared

实现 smoke 后继续新增：

- fresh-process worker：独立构造 native oracle 与 compiled candidate；
- raw-first artifact generator/replay：绑定 source commit、7 个 code blob、模型/capture digest、
  module/device-source/trace/plan receipt；
- 10 类 fully re-signed tamper：candidate/native lower、module/device source hash、launch count、scratch
  alias、warm allocation、stream mismatch、compiled-region 和 DLPack exact count；
- artifact tests当前在 artifact 尚未生成时显式 skip，不构成 closure。

tooling 的 mypy、pylint 10.00/10与现有专项测试已通过。下一提交先冻结 tooling source，再从该 clean
source生成正式 artifact。
