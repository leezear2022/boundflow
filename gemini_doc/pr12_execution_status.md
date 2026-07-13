# PR-12 持续执行状态

> 本文件是跨 Codex 会话的唯一恢复入口；每个 PR-12 子阶段结束时必须更新。

## 当前快照

```text
branch:                     feat/pr12-fused-crown-task
PR-12G commit:              44f87ae
PR-12G tag:                 pr12g-validated-reduced (local annotated tag)
PR-12H commit:              abc2e2a
PR-12I commit:              9627a3c
PR-12J commit:              cd7bc6b
PR-12K commit:              0428c0e
PR-12L commit:              8f6f0d3
PR-12M commit:              fd3cbb4
current phase:              PR-12N closure complete
PR-12 overall:              VALIDATED-REDUCED
closure tag:                pr12-validated-reduced
PR-13:                      GO / READY；not started
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
- PR-12K：权威 raw v3/report v4 为 30/30 correct；硬件 counter 因 `ERR_NVGPUCTRPERM` 不可用；
  fusion 对 TVM-unfused 最大 launch 降幅仅 1.96%，按 5% activity 阈值为 3/6 退化、1/6 改善、
  2/6 中性；唯一选择分支为 `E_STOP_OPTIMIZING_TIR`。

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

## PR-12K 当前工作

- [x] 审计 ncu/CUPTI 版本、路径与权限；
- [x] 实测 `ERR_NVGPUCTRPERM`，明确禁止硬件 counter claim；
- [x] 6 workload×5 backend complete final-bound CUPTI activity profile；
- [x] raw trace→CSV/图/summary/manifest；
- [x] v1 range double-count、v2 count schema、v3/v4 权威工件完整披露；
- [x] PR-12L 唯一分支选择为 E：停止继续优化 TIR。
- [x] PR-12K 收尾：focused 2 passed；全量 332 passed/1 skipped；mypy success；pylint
  10.00/10；Black/diff check 通过。

## PR-12L 当前工作

- [x] 只选择 `E_STOP_OPTIMIZING_TIR`；
- [x] 拒绝同时扩 Linear tile、CUDA Graph、chunk family 与 Conv capability；
- [x] 保留 fused backend 为 compile-aware Planner 候选；
- [x] 冻结 PR-12M 的全新 split、多预算与 expected-reuse 约束；
- [x] 无 TIR/schedule/runtime 代码修改。

## PR-12M 当前工作

- [x] capability→budget→risk→amortized latency Planner；
- [x] expected reuse 与 memory/disk/fresh cache 概率进入决策 dump；
- [x] v3 split→25/25 calibration→model freeze→25/25 final，无 held-out 泄漏；
- [x] 16/32/64/128 MiB/unbounded × Q1/Q32/Q1024 共 75 decisions；
- [x] 72/72 feasible opportunities 选到可行 backend，0 unsafe；
- [x] feasible median/p90/max regret 1.000×/1.000×/1.016×；
- [x] 选择 eager/chunked/structured/fused 四类 backend，随 reuse/budget 改变计划；
- [x] 3 个 16 MiB capacity failure 单独保留，不纳入 feasible regret 门禁。
- [x] PR-12M 收尾：focused 9 passed；全量 340 passed/1 skipped；mypy success；pylint
  10.00/10；Black/diff check 通过。

## PR-12N 当前工作

- [x] H–M code/test/artifact/negative-result closure audit；
- [x] I/J/K/M primary hash 重算与 manifest 交叉核验；
- [x] third-party SHA 与无关来源污染审计；
- [x] reduced Artifact Appendix、Claims Map、README/总账；
- [x] closure 判定 `VALIDATED-REDUCED`；
- [x] PR-13 硬门禁 GO，但本阶段不启动 PR-13；
- [x] annotated closure tag `pr12-validated-reduced` 随 closure commit 创建。

## 下一步

PR-12 已关闭。下一次独立工作只能从 PR-13 real multi-domain/BaB adapter 开始，先读取
`gemini_doc/pr12_closure_audit_2026_07_14.md` 的 scope/禁止事项；不得继续回调 PR-12 TIR 或
把 reduced closure 扩写成论文级完整结论。

## 恢复命令

```bash
conda activate boundflow
source env.sh
git status --short --branch
pytest -q tests/test_phase7a_pr12k_cupti_profiler.py \
  tests/test_phase7a_pr12k_cupti_postprocess.py
```

顶层计划：`gemini_doc/pr12_mid_long_term_completion_plan.md`。合同：
`docs/pr12_benchmark_contract.md`。
