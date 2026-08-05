# Audit NRIR45 prepared intermediate refinement closure

- task: nrir45-20260805
- doc: nrir45-20260805/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: b6eb6974c58289585dfa07a4daa7e7383c9bd7df
- created: 2026-08-05T02:35:05Z

## Original request

---
status: ready-for-audit
updated: 2026-08-05T02:30:00Z
type: audit-handoff
topic: boundflow
slug: nrir45-prepared-intermediate-refinement
stage: s01
pr: 56
---

# NRIR45 Prepared Intermediate Refinement 外部审计交接

## 1. 审计目标

请独立审计 PR #56 的 NRIR45 prepare-once intermediate-refinement capsule，不采信执行方给出的
测试数量、计时、hash 或结论。审计目标是判断本轮能否以 fixed ResNet2B property 0 CPU8
internal production admission 的 `VALIDATED-REDUCED` 关闭。

本轮不得升级为公平竞品性能、GPU、多 workload、多 property、完整 verifier E2E、property
closure 或 ASPLOS-ready 结论。最终验证结果仍是 9/9 unknown，两个正式 artifact 都必须保持
`performance_claimed=false`。

## 2. 固定范围

- repository: `leezear2022/boundflow`
- PR: `#56`
- base: `main@b6eb6974c58289585dfa07a4daa7e7383c9bd7df`
- feature closure: `8b8766e118e038d76d2bd363ab42d45833f3031f`
- publication head: `af1031eee0740b68d258c5780aa898c30b2b6fe2`
- branch: `feat/prepared-intermediate-refinement-capsule-v1`

请先独立运行：

```bash
git diff --stat b6eb6974c58289585dfa07a4daa7e7383c9bd7df..af1031eee0740b68d258c5780aa898c30b2b6fe2
git log --oneline b6eb6974c58289585dfa07a4daa7e7383c9bd7df..af1031eee0740b68d258c5780aa898c30b2b6fe2
git diff --check b6eb6974c58289585dfa07a4daa7e7383c9bd7df..af1031eee0740b68d258c5780aa898c30b2b6fe2
```

重点代码与证据：

- `boundflow/ir/prepared_intermediate_refinement.py`
- `boundflow/runtime/native_prepared_intermediate_refinement.py`
- `boundflow/runtime/native_prepared_per_child_refinement.py`
- `boundflow/runtime/native_prepared_shared_parametric_ancestral.py`
- `boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py`
- `boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py`
- `scripts/run_prepared_intermediate_refinement_formal.py`
- `scripts/run_prepared_intermediate_refinement_global_formal.py`
- `tests/test_native_prepared_intermediate_refinement.py`
- `tests/test_native_objective_branch_scorer_ownership.py`
- `artifacts/prepared-intermediate-refinement/`

## 3. Acceptance criteria

### AC1：typed capsule 与执行所有权成立

- capsule 必须有明确的 prepare-once 边界，不能只是给旧函数改名；
- Task/Schedule 必须覆盖 `ADMIT_EXACT -> CONSUME_PLAN_TARGETS ->
  EXECUTE_SELECTED_CROWN -> COMMIT_RESULT -> EMIT_RECEIPT` 五阶段；
- schedule 顺序、依赖、单次执行和 receipt ownership 必须 fail closed；
- prepared 路径只能作为 additive composition，NRIR42/NRIR44 frozen 路径不得被静默改写；
- exact selected clauses、node ancestry、node count、worst lower 和最终结果必须与 control 一致。

### AC2：prepare-once 不是跳过正确性验证

- 首次 `prepare` 必须执行历史完整 Program/Plan/Task/Schedule 准入与 hash；
- hot path 可以复用已准入 digest，但不得接受错误 capsule、错误 runtime input 或错误 owner；
- 容器成员替换、目标变异、Tensor 原地变异和跨 capsule 混用必须被拒绝；
- 显式 `validate_full` 必须绕过 fast path，重新执行完整 validator/hash；
- artifact replay 必须逐 capsule 进行 full validation，而不是只信缓存 token；
- 请增加或手工运行至少一条“语义相同但对象 identity 不同”和一条 Tensor mutation
  探针，确认拒绝发生在 selected-CROWN 执行之前。

### AC3：Phase A 局部机制证据可独立复核

请从三个 shard 原始 JSON 独立重算，而不是只读取 `formal.json` 汇总：

- clause 2/3 每组 control 与 prepared 的语义输出完全相等；
- 每条 prepared queue 是 30 capsules，且 30/30 full replay；
- target-selection 调用数 `246 -> 98`；
- full Program validation 调用数 `186 -> 38`；
- full Program hash 调用数 `217 -> 39`；
- clause 2 median `12.981239077 -> 9.444102937 s`，ratio
  `0.7275193747670009`；
- clause 3 median `13.122778401 -> 9.666283004 s`，ratio
  `0.7366033860072952`；
- 改善大于预注册 pooled MAD 门禁；
- formal hash 必须是
  `be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`。

### AC4：Phase B 全局证据可独立复核

请从三个 Phase-B shard 独立重算：

- floor timing 为 `8.625021900 / 8.583826475 / 8.628564671 s`；
- whole trace 为 `31.262520857 / 31.319771924 / 31.470077702 s`；
- measured wall 为 `36.396630538 / 36.513682717 / 36.611708798 s`；
- 相对 frozen NRIR44 的 trace median ratio 为 `0.7102675919765453`；
- measured-wall median ratio 为 `0.6157384929388071`；
- 两项改善都大于 pooled MAD；
- 三轮 selected clauses 均为 `[2, 3]`，nodes 均为 `[31, 31]`；
- 每轮 60/60 capsules full replay；
- worst lower 精确保持
  `[-35.53092575073242, -30.258447647094727]`；
- formal payload hash 必须是
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`。

### AC5：artifact replay 与篡改门禁真实有效

- 两个 manifest 的所有文件 SHA256 必须先验重算；
- replay 必须从 typed payload 重建 IR/receipt 并逐字段比较，不得只比较外层 digest；
- Phase A 内置 synchronized outer-rehash tamper 必须失败；
- 请复制 artifact 到临时目录，修改一个 source target 或 schedule 字段，并同步修复 payload/manifest
  外层 digest；语义重建仍必须拒绝；
- benchmark model/property digest 必须分别为
  `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d` 和
  `89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff`；
- `performance_claimed=false` 不得被删除或改写。

### AC6：回归与 claim boundary

- targeted tests、全量测试、Black、mypy、Pylint 必须独立通过；
- 37 个 skip 必须核对原因，不能当作 CUDA/GPU 已验证；
- claims map、status、memo、master plan 和 changelog 必须保持同一结论；
- NRIR45 只能声称 fixed internal workload 上减少重复 validation/selection/hash ownership 成本；
- 最终 9/9 unknown，因此没有 property closure；
- 不得出现公平竞品、10x、GPU、多 workload 或 ASPLOS-ready claim 漂移。

## 4. 建议独立复核命令

先激活环境：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
```

设置一个包含冻结 VNN-COMP 2021 文件的根目录；不要依赖执行方计时进程：

```bash
export NRIR45_BENCHMARK_ROOT=/path/to/vnncomp2021
sha256sum \
  "$NRIR45_BENCHMARK_ROOT/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx" \
  "$NRIR45_BENCHMARK_ROOT/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
```

正式 replay：

```bash
python scripts/run_prepared_intermediate_refinement_formal.py replay \
  --benchmark-root "$NRIR45_BENCHMARK_ROOT" \
  --artifact-dir artifacts/prepared-intermediate-refinement/vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1

python scripts/run_prepared_intermediate_refinement_global_formal.py replay \
  --benchmark-root "$NRIR45_BENCHMARK_ROOT" \
  --artifact-dir artifacts/prepared-intermediate-refinement/vnncomp21-resnet2b-property0-three-repeat-cpu-phase-b-v1
```

测试与静态门禁：

```bash
python -m pytest -q \
  tests/test_native_prepared_intermediate_refinement.py \
  tests/test_native_objective_branch_scorer_ownership.py
python -m pytest -q tests

python -m black --check \
  boundflow/ir/prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_per_child_refinement.py \
  boundflow/runtime/native_prepared_shared_parametric_ancestral.py \
  boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py \
  boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py \
  scripts/run_prepared_intermediate_refinement_formal.py \
  scripts/run_prepared_intermediate_refinement_global_formal.py \
  tests/test_native_prepared_intermediate_refinement.py

python -m mypy --follow-imports=skip \
  boundflow/ir/prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_per_child_refinement.py \
  boundflow/runtime/native_prepared_shared_parametric_ancestral.py \
  boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py \
  boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py \
  scripts/run_prepared_intermediate_refinement_formal.py \
  scripts/run_prepared_intermediate_refinement_global_formal.py

python -m pylint \
  boundflow/ir/prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_intermediate_refinement.py \
  boundflow/runtime/native_prepared_per_child_refinement.py \
  boundflow/runtime/native_prepared_shared_parametric_ancestral.py \
  boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py \
  boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py \
  scripts/run_prepared_intermediate_refinement_formal.py \
  scripts/run_prepared_intermediate_refinement_global_formal.py \
  tests/test_native_prepared_intermediate_refinement.py
```

DocOps：

```bash
dol exchange validate nrir45-20260805
dol validate
dol lint --soft
```

## 5. 执行方结果（仅供定位，必须独立复核）

- targeted：10 passed；
- full suite：984 passed，37 skipped，7 warnings；
- Black：pass；mypy：clean；Pylint：10.00/10；
- Phase A/B replay：exit 0；
- PR #56：base/main 正确、draft、mergeable，仓库当前没有配置 GitHub status checks。

## 6. 已知限制与高风险审计点

1. 结果只来自一个固定 ResNet2B property 0、CPU8 workload；
2. 计时是 internal production admis...(truncated)

## Scope

PR #56 typed prepare-once capsule, Phase A/B formal evidence, replay/tamper, regression, and claim boundary

## Acceptance criteria

- AC1 typed capsule and five-stage execution ownership are fail closed and additive
- AC2 prepare-once fast path preserves full admission, mutation, identity, and explicit full-replay gates
- AC3 Phase A raw shards independently reproduce exact semantics, call-count reductions, timing ratios, MAD gates, and formal hash
- AC4 Phase B raw shards independently reproduce exact semantics, 60/60 full replay, timing ratios, MAD gates, and payload hash
- AC5 artifact digests, typed semantic reconstruction, and synchronized outer-rehash tampering fail closed
- AC6 regression gates pass and claims remain fixed-workload CPU internal VALIDATED-REDUCED
