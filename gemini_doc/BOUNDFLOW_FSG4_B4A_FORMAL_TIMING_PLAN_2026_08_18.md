---
status: implemented-pending-clean-source-formal-run
updated: 2026-08-18T09:30:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_FSG4_B4A_FORMAL_TIMING
stage: s01
---

# FSG4/B4-A 独立正式计时计划

## 0. 准入与边界

B4-A five-fresh correctness 已在 source=`43d4117` 关闭为
`INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`。本阶段只测 B3 control 与 B4-A terminal
lower/lA handoff candidate；不得使用 correctness artifact 中的 latency，不得混入 B4-B/TIR、JIT、
runtime batching/stream 或 allocator 改动。

## 1. 冻结问题

在相同 αβ-CROWN host、solver flags、模型/property、GPU、torch/CUDA、prepared core、KFSB、device
commit 与 post/queue 条件下，消除一次 terminal export parent CROWN 是否满足：

- B3/B4-A core wall geomean `>=1.03x`；
- 六个 control pair 中每个 query wall ratio `B3/B4-A >=0.98x`。

未同时通过时，B4-A correctness/mechanism 保留，但不得累计为 B4 performance candidate。

## 2. Fresh 进程矩阵

固定 6 个 block，每个 block 包含 B3/B4-A 的 control/profile 各一，共 24 个独立进程。顺序冻结为：

```text
0: B3-control, B3-profile, B4A-control, B4A-profile
1: B4A-profile, B3-control, B3-profile, B4A-control
2: B4A-control, B4A-profile, B3-control, B3-profile
3: B3-profile, B4A-control, B4A-profile, B3-control
4: B3-control, B4A-profile, B4A-control, B3-profile
5: B4A-profile, B3-profile, B3-control, B4A-control
```

control 才进入 headline ratio；profile 只报告 optimizer、terminal export assembly、KFSB 等分层 wall/GPU
归因和 closure，不与 control 混算。每个 worker 启动前执行与 B3 正式实验相同的 AC power、GPU process、
thermal/power stability preflight。v2环境拒绝后的加固要求：最终sample还必须GPU `<=45°C`且
`sw_thermal_slowdown=Not Active`；active信号即使与power counter精确耦合也不准入。

## 3. Raw-first 与环境

- one subprocess per position，seed/solver protocol 与 B3 相同；
- raw worker、stdout/stderr、outer metadata 先落盘，再派生 pair/summary；
- complete source-bound worker 可 resume；partial worker 必须拒绝；
- venv Python 保留 symlink，不可 `resolve()` 成裸解释器；
- logs 使用 `$BOUNDFLOW_ROOT/$ABCROWN_ROOT/$VNNCOMP_ROOT/$PYTHON` 别名并拒绝本机路径；
- protocol 绑定 source/code blobs、B4-A five-fresh manifest/file hash、模型/property与三个外部 commit。

## 4. Correctness 与 activation

六个 control pair 必须保持：

- final lower `allclose(atol=rtol=2e-4)`、sign exact，全部离散 solver/post/queue 语义 exact；
- 19 个 post-query terminal export raw tensor 同容差、sign exact；
- B3：handoff=0、terminal export CROWN rerun=1；
- B4-A：handoff=1、terminal export CROWN rerun=0、lineage=6；
- 两侧 provider/fallback=0；optimizer 10/9、forward=4、KFSB child=3、device commit=12保持；
- audit/content hash 时间排除在 query/core headline 之外。

## 5. Measurement 与分类

对六个 control pair 从 raw 重算：core/query wall、core/query GPU、peak allocated/reserved 的逐pair ratio、
geomean、min/max。必须报告全部 pair，不删除 outlier。

分类：

- `VALIDATED-B4-A-PERFORMANCE-CANDIDATE`：correctness/environment/measurement/replay/tamper全PASS，
  core wall geomean `>=1.03x` 且 query wall worst pair `>=0.98x`；
- `VALIDATED-NO-GO-B4-A-PERFORMANCE`：正确性成立但任一性能门禁未过；
- correctness/environment/measurement失败：blocker，不得形成性能分类。

artifact 在外审批准前仍写 `performance_claimed=false`；candidate admission 与正式外部批准分离。

## 6. Profile 与 CUDA 边界

profile worker必须 closure error `<=1%`、residual `<=3%`，披露 profiler perturbation，不参与headline。
本阶段报告高层 optimizer/terminal-export/KFSB wall与GPU span差异。CUDA kernel/launch差异若现有profile raw
不能独立重建，则明确记为 `DEFERRED-TO-B4-A-KERNEL-DELTA`，不得用B4-0单配置归因冒充差分证据，也不阻塞
本阶段预注册的core/query判定。

## 7. Replay 与 tamper

root replay必须从24个raw worker重建sequence、correctness、activation、environment、profile closure、
paired ratios和分类。outer-resigned tamper至少覆盖：source/code/five-fresh identity、worker swap/delete、
configuration/mode/order、raw latency、semantic tensor、handoff/rerun/lineage、provider/fallback、profile
closure、summary threshold/classification；全部必须拒绝。

## 8. 验证与下一步

固定 related tests 必须在 exchange 中逐文件列出：

- `tests/test_fsg4_b4a_terminal_lower_adjoint_handoff.py`
- `tests/test_fsg4_b4a_correctness_pairs.py`
- `tests/test_fsg4_b4a_correctness_pairs_artifact.py`
- `tests/test_fsg4_b4a_formal_timing.py`
- `tests/test_fsg4_b4a_formal_timing_tamper.py`
- `tests/test_fsg4_b3_explicit_counters.py`
- `tests/test_fsg4_b3_same_solver_timing.py`
- `tests/test_fsg4_b3_same_solver_worker.py`
- `tests/test_fsg4_b3_same_solver_artifact.py`

另跑 full pytest、Black、Mypy、Pylint、`git diff --check`、DocOps lint。

24-process formal runner、root replay与12类outer-resigned tamper probe已实现；下一唯一动作是提交clean
source后执行GPU实验。B4-B/TIR保持关闭。
