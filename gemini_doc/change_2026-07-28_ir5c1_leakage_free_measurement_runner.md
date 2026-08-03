# 变更记录：IR-5C1 leakage-free measurement runner

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`7b50275`（IR-5C0 typed measured workload foundation）
> 状态：runner/冻结协议完成；最终 CUDA artifact 仍 pending

## 改动

- 新增 `CandidateMeasurement`，逐候选保存：
  - Bound/PlanInstance/Task/Schedule hash；
  - cold latency、warm samples；
  - CUDA resident baseline、incremental peak、合计 peak；
  - final lower/upper hash、reference diff 与 allclose；
  - TVM cache 的 TIR generation/schedule/compile/serialization/load 原始事件。
- 新增 calibration-only backend model：
  - latency 只从 calibration workload 的 warm samples 拟合；
  - compile/setup 只读取 calibration TVM cache event；
  - API 对 held-out row 进入 calibration fit 明确 fail closed。
- 新增正式 artifact runner：
  - 冻结 calibration/held-out seeds 与 shapes；
  - fixed/local/global/oracle 共享同一 held-out observation；
  - 输出 split、raw JSONL、calibration model、outcomes、summary 与 SHA-256 manifest；
  - replay 检查文件摘要、split/context 漂移，并可重放 reference final-bound 语义。
- resource context 同样在最终测量前冻结：
  - high-memory 固定为 64 MiB；
  - `heldout-medium` low-memory 固定为 8,800,000 bytes；
  - `heldout-large` low-memory 固定为 9,400,000 bytes；
  - expected query count 与 warm-cache context 均写入 `split.json`。

## 方法学纠正

开发期 v1 pilot 曾用 held-out 实测最小 peak 生成 low-memory budget。该做法会让
held-out 结果反向塑造评测 context，因此已判定为不可作为最终证据，且不进入提交。
最终 runner 改为常量 resource contract；后续 v2 fresh run 不读取 held-out peak 来决定
任何 prediction 或 context。

## 验证与边界

- CPU 端到端 artifact generate/replay/semantic replay 通过；
- IR-5C1/IR-5C0/evaluator 定向：4 passed；
- Mypy：0 issues；
- Pylint：10.00/10。

本提交只冻结 runner 与方法学契约，不记录最终 CUDA 数字。正式 artifact 必须在本提交
成为 HEAD 后 fresh generate，使 manifest 的 `git_commit` 精确指向产生它的代码。

