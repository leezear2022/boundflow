# BoundFlow 真实 Verifier IR 路线：外部审计交接

> 日期：2026-08-03
> 用途：交给独立大模型或人工审计
> 仓库：`leezear2022/boundflow`
> 分支：`feat/real-verifier-ir-integration-v1`
> Draft PR：<https://github.com/leezear2022/boundflow/pull/4>
> PR base：`feat/compiler-ir-stack-v1`（stacked on PR #3）

## 1. 请求审计的最终结论

本路线已按约定范围关闭为：

> **VALIDATED-REDUCED（CPU correctness/integration）**

成立的不是 verifier acceleration，而是两项集成 correctness：

1. VNN-COMP ResNet-2B prop0 的 initial plain-CROWN 在显式保留 external intermediate
   bounds 与 adaptive ReLU lower-slope policy 后恢复等价；
2. activation-BaB 的 provider-owned αβ-CROWN exact call 能经过 Bound IR → PlanTemplate →
   PlanInstance → Task IR → Schedule IR 后调用原 external method，并保留 query identity、顺序、
   requested output 与 parent lineage。

请重点检查本文第 8 节的限制是否与代码、artifact、claims map 一致。任何把本结果写成 CUDA、
性能、BoundFlow αβ kernel 或完整 VNN-COMP E2E claim 的解释都应判为错误。

## 2. 起点与问题来源

IR-first 主线已在 `feat/compiler-ir-stack-v1` 完成 IR-1—4 validated-reduced，但 IR-5
adaptive system-performance final 因 Global p90 regret `1.26160× > 1.20×`、gray 无
compiler Pareto、无 multi-budget switch，以 VALIDATED-NO-GO 关闭。基线 commit/tag：

- commit `d457b22`；
- tag `ir5-final-validated-nogo`；
- draft PR #3。

在此之前，PR-14 的真实 αβ-CROWN 审计发现：

- 3 个 workload 共 540 个 `compute_bounds` 调用；
- `activation_bab_bound` 394 个，对原 fused capability eligibility 为 `0/394`；
- initial replay 的 simple MLP lower 等价；
- VNN-COMP ResNet-2B prop0 local whole-query lower 对 external max diff `796.765`，sign
  仅 `3/9`。

因此不允许直接进入 PR-14C E2E 或继续性能优化，必须先修 correctness。

## 3. 本路线提交链

被审计的 implementation/closure/DocOps baseline 相对 `d457b22` 共 6 个提交；本交接文档
将在其后的独立 docs commit 中提交，不计入下表的实现范围：

| Commit | 内容 |
|---|---|
| `1406d4b` | 冻结 RVIR 所有权、两类合法路径和 RVIR-1—4 门禁 |
| `05669ad` | external intermediate-bound semantics + adaptive ReLU policy |
| `3e7460e` | activation external exact-call typed Bound/Plan/Task/Schedule 路径 |
| `e03b3d2` | 冻结自包含 CPU correctness artifact |
| `5a5a8a4` | 关闭路线并更新权威状态/claims；核心 closure tag 指向此提交 |
| `6428665` | 默认启用 DocOps，补录 change/validation 事件 |

发布状态：

- annotated tag `rvir-cpu-correctness-validated-reduced` → `5a5a8a4`；
- 审计准备时 PR #4 的代码/DocOps baseline → `6428665`；PR live head 还会包含本交接文档的
  后续 docs commit；
- tag 刻意指向核心代码/工件/closure，而不包含后续 DocOps 流程提交。

## 4. RVIR-1：ResNet 根因与修复

### 4.1 根因

本地 whole-query 路径使用逐层 IBP trace 作为后续 ReLU pre-activation bounds，而 external
αβ-CROWN initial path 使用递归 CROWN intermediate bounds。误差从第二组 ReLU 开始累积；最后
一组本地 IBP width max 约 `394.717`，external 约 `2.141`。

只把 lower slope 从 zero 改为 adaptive 不足以修复：max diff 仍约 `810.805`。注入 6 组
external pre-activation bounds，并使用 adaptive lower slope 后，手工复核 max diff 约
`2.15e-6`；正式 runner 为 `3.0994415283203125e-6`。

### 4.2 代码责任

- `boundflow/ir/bound.py`
  - `IntermediateBoundSource.LOCAL_FORWARD/EXTERNAL_VERIFIER`；
  - `ReluLowerSlopePolicy.ZERO/ADAPTIVE`；
  - 两字段进入 canonical JSON 与 stable hash。
- `boundflow/runtime/abcrown_adapter.py`
  - 捕获逐 ReLU external lower/upper、ordinal、node identity 和 aggregate SHA256；
  - count/order/shape 失配 fail closed。
- `boundflow/runtime/bound_ir_interpreter.py`
  - 普通与 fused 路径都按 IR policy 执行 slope。
- `scripts/replay_pr14_abcrown_initial_crown.py`
  - replay 使用 external intermediate bounds，不再用 local IBP 伪装 external semantics。

### 4.3 冻结结果

- external intermediate count：6；
- aggregate hash：`d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`；
- lower allclose：true；
- lower max diff：`3.0994415283203125e-6`；
- sign agreement：`9/9`；
- nonnegative：`6/6`；
- device：CPU。

## 5. RVIR-2/3：typed external exact call

### 5.1 IR 结构

- Bound IR schema 升为 `boundflow.bound_ir/v1.1`；
- `BoundOpKind.EXTERNAL_VERIFIER_CALL` 与 `ExternalVerifierCallAttrs` 显式拥有 provider、phase、
  effective method、requested bounds、input/objective hash、α/β/split/cuts version；
- Plan 使用 `RegionKind.EXTERNAL_VERIFIER`、`BackendKind.EXTERNAL_ABCROWN`；
- Task 使用 `TaskIRKind.EXTERNAL_VERIFIER_CALL` 和 external-state dependency；
- backend implementation 固定为 `external_abcrown_exact_call/v1`；
- Schedule 必须恰有一次 launch 和一次 emit；undeclared backend 拒绝执行。

`boundflow/runtime/verifier_ir_integration.py` 将每个 query 编译为：

```text
ExternalVerifierCallSpec
  -> BFBoundModule
  -> PlanTemplate
  -> PlanInstance
  -> TaskIRModule
  -> ScheduleModule
  -> original αβ-CROWN exact_call exactly once
```

external verifier 继续拥有 α/β/split 算法、domain management 与 termination；BoundFlow 只拥有
typed admission、identity、调度边界和结果证据。

### 5.2 真实 CPU observer on/off

冻结 αβ-CROWN commit：`e5c7e17bf0488843acb77b7519f59876717a49f4`。工作负载为 upstream
`simple_mlp.onnx + robustness_mlp.vnnlib`，CPU、BaB、30 秒 timeout、skip attack。

| 项目 | baseline | typed observer |
|---|---:|---:|
| solver status | unknown | unknown |
| visited domains | 380 | 380 |
| final lower | -0.18902308 | -0.18902308 |
| captured query | N/A | 377 |
| compiled/dispatched/completed | N/A | 377/377/377 |

附加 identity 证据：

- activation calls：343，effective method 全为 αβ-CROWN；
- requested outputs：377 个均显式 lower-only；
- root queries：30；parent links：347；
- query IDs 与 typed execution record IDs 数量、顺序完全一致；
- `semantics_owner=external_verifier`；
- `performance_claimed=false`。

## 6. RVIR-4 artifact

路径：`artifacts/rvir/rvir-cpu-correctness-v1-20260803/`

| 文件 | 用途 |
|---|---|
| `activation_calls.jsonl` | 394 个历史 activation query 完整 identity + 五层 IR hash |
| `online_execution.json` | 当前 adapter v2 的 377-call CPU 对照与 lineage |
| `resnet_semantics.json` | ResNet external-semantics correctness |
| `manifest.json` | 文件/source digest、coverage、环境与 claim 边界 |

历史 394 个调用分布：

- official simple MLP：343；
- VNN-COMP ResNet-2B prop0：51；
- effective method αβ-CROWN：394/394；
- typed admission：394/394。

Artifact 内嵌完整历史 query，不依赖本机 ignored PR-14A source artifact。replay 会逐行重新生成
Bound/PlanTemplate/PlanInstance/Task/Schedule 五层 hash，并比较整行内容。

## 7. 已执行验证

```bash
conda run -n boundflow pytest -q tests
# 452 passed, 37 skipped

conda run -n boundflow python scripts/run_real_verifier_ir_artifact.py replay \
  --artifact-dir artifacts/rvir/rvir-cpu-correctness-v1-20260803
# {"activation_call_count":394,"performance_claimed":false,"status":"replayed"}

conda run -n boundflow python -m mypy \
  boundflow/runtime/verifier_ir_integration.py \
  boundflow/runtime/abcrown_adapter.py boundflow/ir/bound.py \
  boundflow/ir/plan.py boundflow/ir/task_v1.py \
  boundflow/planner/plan_ir_builder.py
# Success: no issues found in 6 source files

conda run -n boundflow python -m pylint \
  scripts/run_real_verifier_ir_artifact.py tests/test_real_verifier_ir_artifact.py
# 10.00/10

git diff --check
# PASS
```

DocOps：

- `.docops/s.md`：topic `boundflow`、stage `s01`、PR `[4]`；
- RVIR change：`ev000009`；
- RVIR validation：`ev000010` / pass；
- `dol validate`：PASS；
- `dol lint --soft`：PASS。

## 8. 必须保留的限制

1. `0/394` 仍是 fused BoundFlow kernel replacement coverage；不能用 394/394 typed admission
   覆盖或删除该历史结论。
2. 历史 adapter v1 的 394 行全部缺：
   - split tensor values（只有 unresolved token）；
   - 精确 requested polarity（artifact 保守 assumed both）；
   - parent lineage。
3. 当前 adapter v2 的 377-call 实时证据补齐 requested lower-only 与 parent lineage，但只覆盖
   upstream simple MLP CPU。
4. ResNet 结果只证明 external-semantics initial plain-CROWN；不证明 local IBP-intermediate
   whole-query 路径等价，也不证明 complete verifier E2E。
5. 本机 NVIDIA driver 不可通信；没有本轮 fresh CUDA evidence。
6. typed compile/hash/validation 明显增加运行开销；没有 performance claim。
7. external 请求 lower-only；旧 BoundFlow benchmark 同时计算 lower+upper，公平性能合同未建立。
8. IR-5 仍为 VALIDATED-NO-GO，IR-6 不启动，ASPLOS-ready 仍为 NO。

## 9. 建议独立审计顺序

1. 核对 branch、6 个 commit、tag 与 PR base/head；
2. 运行 artifact replay，确认 394 行自包含重算；
3. 从 `activation_calls.jsonl` 独立统计 workload、effective method 和三项 legacy limitation；
4. 核对 `online_execution.json` 的 377/377、343 activation、347 parent links、380-domain
   observer equivalence；
5. 核对 `resnet_semantics.json` 的 6 bounds、hash、max diff 和 sign；
6. 抽查 external-call attrs 在五层 IR 的 typed linkage 和 fail-closed tests；
7. 运行 full/focused tests，并检查跳过项是否均为 CUDA/环境边界；
8. 对照 `asplos_claims_map.md`、`current_status_after_pr13.md` 和 closure 文档，确认没有性能
   或 CUDA claim 漂移。

## 10. 请审计方输出

请按“成立 / 不成立 / 不可现场审计”逐项给出：

- 提交、tag、PR、工作树和文件范围；
- RVIR-1 数值与 external intermediate-bound ownership；
- RVIR-2 五层 IR 类型、hash、state identity、backend 限制；
- RVIR-3 query/result/parent accounting 与 observer equivalence；
- RVIR-4 artifact integrity、self-contained replay 与 394 coverage；
- 全量测试、mypy、pylint、DocOps 状态；
- 第 8 节所有 claim limitation 是否被权威文档一致保留；
- blocker/major/minor findings，以及是否同意 VALIDATED-REDUCED 关闭。
