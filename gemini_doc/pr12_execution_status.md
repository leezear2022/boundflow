# PR-12 持续执行状态

> 本文件是跨 Codex 会话的唯一恢复入口；每个 PR-12 子阶段结束时必须更新。

## 当前快照

```text
branch:                     feat/pr12-fused-crown-task
PR-12G commit:              44f87ae
PR-12G tag:                 pr12g-validated-reduced (local annotated tag)
PR-12H commit:              abc2e2a
PR-12I commit:              9627a3c
current phase:              PR-12J complete；next PR-12K profiler
PR-12 overall:              IN PROGRESS
PR-13:                      BLOCKED
```

工作区开始 PR-12H 前为 clean，`44f87ae` 已与远程同分支同步；PR-12G tag 尚未 push。

## 已成立证据

- PR-12D：single-consumer fusion、fanout fallback、完整 step contract、DLPack alias、TVM-FFI
  custom stream correctness；
- PR-12E/F v1：正式 historical runtime/Pareto 与 frozen held-out，性能目标失败；
- PR-12G v2：84/84 candidate rows correct；5/5 budget feasible；0 unsafe；median/p90 regret
  1.000×/1.054×；eager/chunked/TIR 选择 1/2/2；
- PR-12H 收尾全量：321 passed、1 skipped。
- PR-12I：正式 v2 为 72 rows（54 ok、18 N/A、0 correctness failure）；structured 与
  TVM-unfused baseline 已闭合；`torch.compile(fullgraph=True)` 因 `ContextVar.set` 结构性失败；
  fused E2E geomean 0.546× eager、median peak ratio 0.512，负结果保留。
- PR-12J：正式 v4 为 3/3 correct、所有 restart 为真实 disk hit；Linear/Conv 不可摊销；
  mini-ResNet 对 eager 的 fresh/disk-first/process break-even 为 4668/1062/4450，均超过 Q=1024，
  且对 chunked 不可摊销。

authoritative PR-12G 工件：

```text
artifacts/phase7a-pr12/pr12g-multibackend-v2-freeze-20260713/
artifacts/phase7a-pr12/pr12g-multibackend-v2-calibration-canonical3-20260713/
artifacts/phase7a-pr12/pr12g-multibackend-v2-final-canonical3-20260713/
artifacts/phase7a-pr12/pr12g-multibackend-v2-planner-replay-canonical3-20260713/
artifacts/phase7a-pr12/pr12g-multibackend-v2-report-canonical3-20260713/
```

## PR-12H 当前工作

- [x] 创建 `pr12g-validated-reduced` annotated tag；
- [x] 定义三层机器可读 benchmark contract；
- [x] 将历史 kernel/Pareto runner 标记为 non-compliant legacy evidence；
- [x] contract tests、全量测试和静态门禁；
- [x] PR-12H change doc 与索引；
- [x] PR-12H 提交（本文件随该提交冻结）。

## PR-12I 当前工作

- [x] dense/structured/chunked/TVM-unfused/TVM-fused 统一合同 runner；
- [x] TVM-unfused Linear/Conv2d 显式 scaled-A workspace；
- [x] default/custom stream correctness；
- [x] 条件 `torch.compile` fullgraph probe 与结构化失败记录；
- [x] raw JSONL→CSV→Pareto/summary/manifest；
- [x] PR-12I 收尾：focused 9 passed；全量 327 passed/1 skipped；mypy success；pylint
  10.00/10；Black/diff check 通过。

## PR-12J 当前工作

- [x] 分离 Planner IR、TIR generation、schedule、compile、serialization、module load；
- [x] memory hit、独立进程 disk hit 与 library SHA validation；
- [x] Q=1..1024 fresh/disk/process/memory-cache total model；
- [x] v1 tuple/list bug 与 v2 warm SHA 污染均保留并修复；
- [x] authoritative v4：3/3 correctness、0 hidden recompile；
- [x] PR-12J 收尾：focused/integration 5 passed；全量 330 passed/1 skipped；mypy success；
  pylint 10.00/10；Black/diff check 通过。

## 下一步

PR-12J 提交后立即进入 PR-12K：先审计 Nsight Compute/CUPTI 权限与版本，再按冻结 workload/
section 采集 profiler；工具不可用时记录依赖并给出明确降级证据，不先修改 schedule。

## 恢复命令

```bash
conda activate boundflow
source env.sh
git status --short --branch
pytest -q tests/test_phase7a_pr12j_fused_cache.py \
  tests/test_phase7a_pr12j_amortization_runner.py
```

顶层计划：`gemini_doc/pr12_mid_long_term_completion_plan.md`。合同：
`docs/pr12_benchmark_contract.md`。
