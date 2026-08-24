---
status: ready-for-combined-external-audit
updated: 2026-08-25T06:20:00+08:00
type: handoff
topic: boundflow
slug: r3-1b2-to-r3-2b-combined-external-audit
stage: s01
---

# BoundFlow R3-1b2 → R3-2B 合并外部审计交接

## 1. 审计请求

请独立审计PR #60中R3 structured-owner/custom-VJP路线从compiled P-α VJP到single-site timing kill gate
的完整证据链。不要采信closure数字；请从raw、源码和replay独立重算。

- GitHub仓库：<https://github.com/leezear2022/boundflow>
- PR：<https://github.com/leezear2022/boundflow/pull/60>
- 分支：`feat/rvir-v4-production-state-ownership-v1`
- 关闭点：`d4c5b4c`（R3-2B closure）
- 审计范围起点：`2fa8624`（R3-1b2数学归约）
- 排除：未跟踪用户文件`docs/CIBC_for_DAC.pdf`、历史pre-hardening `/tmp`副本。

## 2. 预期总判定

应同时判断两件事，不能互相覆盖：

1. R3-2A correctness/memory是否可采信为`VALIDATED-R3-2A-P-TRAJECTORY`；
2. R3-2B timing是否必须关闭为`VALIDATED-NO-GO-R3-2B-DISPATCH-GRANULARITY`。

预期R3-3保持关闭，`performance_claimed=false`。

## 3. AC1：提交顺序与范围

独立核对：预注册均早于实现/formal；R3-1b2/b3冻结代码在后续R3-2A/B中未被修改；artifact manifest的
source revision和code blob匹配；所有production默认路径、第三方submodule和用户PDF未改。

关键提交包括：`2fa8624`、`8a2575c`、`3b60d4a`、`1441689`、`c69fa1f`、`e7ae590`、
`30f8d6e`、`342205a`、`f43eb76`、`d4c5b4c`。

## 4. AC2：R3-1b2 compiled custom VJP

从源码证明candidate不调用native oracle/eager CROWN，custom forward/backward恰1/1，两个coefficient arena、
saved dense A=0、warm allocation=0、非默认stream与DLPack pointer exact。独立重跑targeted并核对
`artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1/`。预期one-evaluation lower/dα max diff
约`3.8147e-6/6.1467e-8`，12/12 tamper拒绝。

## 5. AC3：R3-1b3 five-fresh memory

独立解析`artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1/`。核对5 pair/10 fresh、order、
lower/dα/sign、α/β version、compiled receipts和absolute peak。预期worst allocated/reserved=
`0.0641686053/0.1666666667`，但不得外推optimizer/query latency。

## 6. AC4：R3-2A 10/9 trajectory

独立解析并逐step重算`artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1/`：

- 5 pair、10 fresh、每worker 10 evaluation/9 Adam+9 scheduler；
- 每步lower/dα/α-before/after/Adam moments与native配对；
- dynamic rebind只改变P-α identity，immutable α/β/split/history不漂移；
- candidate 10/10 custom forward/backward、saved dense A=0、scratch=2、warm alloc=0；
- 正式最大diff=`8.58306884765625e-06 / 8.288770914077759e-08 /
  2.384185791015625e-07 / 4.190951585769653e-08 / 1.0459189070388675e-11`；
- memory worst=`0.05869108004453792 / 0.16666666666666666`；
- replay与12/12 fully re-signed tamper。

## 7. AC5：R3-2B公平timing与NO-GO

独立解析`artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1/`全部300个raw latency sample。亲读
timed path确认host wall+device boundary sync、3 warmup+30 samples、完整optimizer/10/9 wrapper计入、
compile/prepared/capture在外，且无SHA/CPU copy/profiler/memory reset/native shadow。

独立重算预期geomean/worst=`0.1339893740788718/0.13037077164706176`，必须NO-GO；correctness
保持，memory worst=`0.05845666485175469/0.15384615384615385`。核对10/10 tamper。

## 8. AC6：测试与claim边界

复现或抽查：

```bash
conda run -n boundflow pytest -q tests/test_r3*.py
conda run -n boundflow pytest -q tests
conda run -n boundflow mypy boundflow/runtime/r3_optimizer_trajectory_timing.py \
  scripts/run_r3_optimizer_trajectory_timing_worker.py \
  scripts/run_r3_optimizer_trajectory_timing_artifact.py
conda run -n boundflow pylint boundflow/runtime/r3_optimizer_trajectory_timing.py \
  scripts/run_r3_optimizer_trajectory_timing_worker.py \
  scripts/run_r3_optimizer_trajectory_timing_artifact.py
```

预期R3 targeted=`54 passed`；全量=`1606 passed,3 skipped,6 warnings`；pylint=`10.00/10`。权威
文档必须一致保留R3-2A memory/correctness与R3-2B performance NO-GO，不能出现R3-3已开放或ASPLOS-ready。

## 9. AC7：下一路线是否越权

审计`BOUNDFLOW_R3_D0_MICROPHYSICS_ATTRIBUTION_PLAN_2026_08_25.md`只预注册只读归因：不得已经实现
CUDA Graph、schedule tuning或multi-site；Amdahl route必须使用R3-2B真实N/C budget，允许整体被证伪。

## 10. 输出格式

请给出verdict、findings分级、AC1–AC7逐项PASS/FAIL与独立命令/数字、无法现场复核项，并明确是否同意：

1. 同时采信R3-2A VALIDATED与R3-2B NO-GO；
2. 只开放R3-D0 attribution，继续关闭R3-3。

