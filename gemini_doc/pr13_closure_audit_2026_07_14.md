# PR-13 Closure Audit（2026-07-14）

## 最终判定

> **PR-13：VALIDATED-REDUCED**
> 代码提交：`fda5b82`；closure tag：`pr13-validated-reduced`。
> 关闭范围：state-versioned query contract、动态兼容 batching、same-solver adapter、reduced
> RTX 4060 fixed/E2E evidence。
> 不成立的主张：VNN-COMP complete-verification 提升、PR-12 Planner 在 αβ/split 中生效、
> runtime 超越普通 batching、完整 C3 论文贡献已验证。

## 五切片审计

| 切片 | 判定 | 证据 |
|---|---|---|
| PR-13A Query/State Contract | PASS | deterministic query/parent/split-lineage；8/8 fixed replay |
| PR-13B BatchManager | PASS foundation | exact buckets、budget/deadline、OOM bisection、0 loss/order restore |
| PR-13C Same-Solver Adapter | PASS foundation | original/runtime 7/7 query/state/search 对齐；capability guard |
| PR-13D Fixed + E2E | VALIDATED-REDUCED | RTX 4060、5 repeats、16-node chain-CNN、safe/unsafe/unknown |
| PR-13E Closure | PASS | 本审计、Artifact Appendix、Claims Map、状态与 tag |

## Correctness Gate

- fixed stream：16/16 query，0 bounds/branch/state-value mismatch，0 NaN/Inf，lower≤upper；
- same solver：hard/safe/unsafe 的 status 与 nodes evaluated 在三种 variant 间一致；
- query accounting：0 loss、0 duplicate、0 invalid result；
- reuse：四级 validity policy；父 α 仅 warm start，父 β/final bounds 不作 child exact reuse；
- capability：forged `plain_crown_fused` 在 αβ physical executor 调用 0 次时拒绝；
- αβ batch optimizer：per-query 参数使用 loss sum，避免 Adam epsilon 因 1/B gradient scaling
  改变单 query 轨迹；
- explicit counterexample：当前 solver 不返回 counterexample，故 primal replay 为 N/A，不能声称
  已通过该子项。

## Runtime Gate

- dynamic batching、partial/deadline flush、memory first-fit：PASS；
- OOM 自动拆批：PASS（deterministic fault path），真实 GPU OOM：NOT DONE；
- queue wait、fill、execution p50/p90/p99：PASS；
- dispatch-plan cache hit/miss：PASS；compiled-plan cache：N/A；
- default/custom Torch CUDA stream：PASS，custom test 未使用全局同步；
- query identity 的 model/input/spec 版本在 solver 外层预计算，split 使用 host lineage hash；
  GPU state content hash 不在 runtime 热路径逐节点执行：PASS；
- PR-12 multi-backend Planner：0 dispatch，因 αβ/split capability 不兼容而安全拒绝。

## 研究价值 Gate

- fixed runtime / per-node：96.52×，达到 ≥1.3×；
- hard E2E runtime / per-node：9.93×，达到 ≥1.2×；
- hard runtime / batched original：0.980×，即约 2.0% 开销；
- easy 单节点：runtime 相对 per-node 为 0.955×/0.859×，固定开销为负收益；
- non-toy workload：FAIL；
- 结论：物理 batching 的价值成立，runtime abstraction 在 hard reduced stream 中能保留收益，
  但“query-aware runtime 超越普通 batching”的研究假设尚未成立。

## 为什么不是 VALIDATED

- 没有 VNN-COMP ONNX/VNNLIB 或 multi-block ResNet；
- hard workload 只有 16 nodes，未得到完整 time-to-verify/timeout-sensitive solved count；
- 没有真实 GPU OOM、GPU-active/branch/prune 分解；
- αβ/split 只能 dense，PR-12 Planner/compiled cache 尚未进入实际 solver 查询；
- 没有证明 runtime 相比 batched original 有稳定净收益。

## 为什么不降为 MECHANISM-ONLY

- adapter 进入真实 host solver，而非合成 query list；
- RTX 4060 上有 5 次重复的 fixed 和 true E2E，最终 solver 状态/节点数对齐；
- safe、unsafe、unknown 三种终态均覆盖；
- runtime overhead、negative easy cases、cache/queue/peak memory 都有结构化原始记录。

## 权威工件

- 代码：`fda5b82`（四组 manifest 均记录完整 SHA）；
- A：`artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714/`；
- B：`artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714/`；
- C：`artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714/`；
- D：`artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714/`；
- Artifact Appendix：`gemini_doc/pr13_artifact_appendix_2026_07_14.md`。

## 收尾验证

```text
PR-13 focused（CPU sandbox）: 14 passed, 1 skipped
custom CUDA stream（RTX 4060）: 1 passed
全量（boundflow Conda env）: 326 passed, 30 skipped
Mypy（12 个 PR-13 source/script/test）: success
Pylint（12 个 PR-13 source/script/test）: 10.00/10
git diff --check: PASS
changed-file FPGA-CSP/HDL/toolchain scan: 0 match
third-party submodule modifications: 0
```

## 下一研究门禁

下一阶段不得把 96× 直接写成 runtime 新颖性。要升级 C3，必须加入 VNN-COMP/多 block
ResNet、真实长搜索流，并让 query-aware scheduling/cache/multi-backend 相对公平 batched original
产生可归因收益；否则论文 C3 应降级为执行基础设施而不是核心贡献。
