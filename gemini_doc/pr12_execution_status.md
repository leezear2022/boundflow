# PR-12 持续执行状态

> 本文件是跨 Codex 会话的唯一恢复入口；每个 PR-12 子阶段结束时必须更新。

## 当前快照

```text
branch:                     feat/pr12-fused-crown-task
PR-12G commit:              44f87ae
PR-12G tag:                 pr12g-validated-reduced (local annotated tag)
current phase:              PR-12H complete；next PR-12I baseline
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

## 下一步

PR-12H 提交后立即进入 PR-12I：先实现统一 region-runtime harness，再加入 structured eager 与
TVM unfused baseline；不要开始 profiler 或新 TIR schedule。

## 恢复命令

```bash
conda activate boundflow
source env.sh
git status --short --branch
pytest -q tests/test_phase7a_pr12h_benchmark_contract.py
```

顶层计划：`gemini_doc/pr12_mid_long_term_completion_plan.md`。合同：
`docs/pr12_benchmark_contract.md`。
