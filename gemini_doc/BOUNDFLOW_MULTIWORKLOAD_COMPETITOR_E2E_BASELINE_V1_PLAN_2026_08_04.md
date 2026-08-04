---
status: completed
updated: 2026-08-04T05:05:00Z
type: plan
topic: boundflow
slug: multiworkload-competitor-e2e-baseline-v1
stage: s01
---

# Multiworkload Competitor E2E Baseline v1 Plan

## Goal

- 把当前单一 CIFAR10 ResNet2B/property-0 证据扩展为至少三种真实拓扑的 VNN-COMP 2021
  CPU 可执行矩阵，并以固定提交的 αβ-CROWN 建立同机 competitor E2E 诊断基线。
- 首先把 VNNLIB box 与 unsafe-output DNF 编译为 typed Query IR，再由显式 workload
  Plan/Task/Schedule 驱动 ONNX import、native verification、competitor execution 与 result emit；
  不允许 runner 中用隐式路径字符串代替 IR 所有权。
- 本机无可用 CUDA driver，本轮不得声明 GPU、speedup、Pareto 或 ASPLOS-ready；产出的 CPU
  时间只用于发现阶段瓶颈和验证协议可执行性。

## Scope

- 固定三项首批 workload：
  - `mnistfc/mnist-net_256x2 + prop_0_0.03`：MLP，784 输入、10 输出；
  - `cifar10_resnet/resnet_2b + prop_0_eps_0.008`：residual CNN，3072 输入、10 输出；
  - `oval21/cifar_base_kw + first CSV property`：sequential CNN，3072 输入、10 输出。
- selection 必须来自固定 VNN-COMP commit 与 CSV ordinal；model/VNNLIB/CSV SHA256、ONNX op
  inventory、input/output shape、solver policy、timeout、device 和 thread count 全进入 canonical hash。
- native v1 只接受完整 box input constraints，以及每个 unsafe DNF disjunct 恰含一个线性 output
  inequality 的 VNNLIB 子集；任何缺界、重复界、非连续变量、非线性项或多不等式 disjunct 必须
  fail closed。
- αβ-CROWN 固定为 commit `e5c7e17bf0488843acb77b7519f59876717a49f4`，CPU、attack skip、
  activation BaB、显式 timeout；BoundFlow 与 competitor 分别报告 status、phase time、node/domain
  count和限制，不以不同算法配置计算 speedup。

## Tasks

1. [x] 实现 VNNLIB box/property Query IR、稳定 hash、parser 与 αβ-CROWN parser parity tests。
2. [x] 实现三 workload suite manifest 以及 acquire/import/plan/execute/emit Task/Schedule IR。
3. [x] 实现 BoundFlow CPU native runner 和固定 αβ-CROWN subprocess adapter，隔离全局状态并保存原始
   exit/status/timing 记录。
4. [x] 冻结 generate/replay artifact；加入 model/property/result/schedule/digest tamper tests。
5. [x] 执行三 workload 矩阵、focused/full regression、Black/Mypy/Pylint 与 DocOps closure。
6. [x] 依据 verdict/tightness/phase breakdown 决定下一步是补 native Conv/BaB 能力、扩展 workload，
   还是迁移到可用 CUDA 主机执行同一冻结协议。

## Validation

- 对三份真实 VNNLIB，native parser 的 input lower/upper、C、rhs 与固定 αβ-CROWN parser
  bitwise/数值一致；parser 篡改与 unsupported formula 均拒绝。
- replay 在不执行 competitor 的情况下重新编译 Query/Workload/Task/Schedule IR 并逐 hash 对比；
  competitor fresh rerun 作为独立可选门禁，不允许 replay 空转。
- CPU 结果必须保持 `performance_claimed=false`；只有相同 device/method/steps/timeout 且至少 5 次
  fresh-process 重复后，才允许另立性能 claim。

## Rollback

- 新 parser、suite IR 和 runner 均为 additive；旧 ResNet frozen artifacts 与默认 runtime 行为不变。
- 若任一真实模型超出 importer/runtime capability，记录 typed `unsupported` 原因并保留其 selection，
  不用更容易的 toy 网络静默替换。

## Links

- changelog: `gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
