# PR-10 外部评审意见整合记录

> 日期：2026-07-12
> 范围：只更新研究计划、claims 与门禁，不启动 PR-11 代码实现。
> 基线：`263ea81`（PR-10 complete, feature-gated）

## 1. 采纳的核心判断

PR-10 通过表示、正确性与研究机会门禁，但数据否定了“structured 应成为统一默认表示”的
假设。代表性 plain CROWN 大点中，structured 峰值显存降低约 29.8%，但 latency 增加约
9.17×；α/αβ structured 显存恶化，并在 6 个大点 OOM。默认 dense、structured opt-in 的
现状保持不变。

这组结果被正式用作 C2 的动机：表示选择必须感知 bound method、是否构建梯度图、优化阶段、
运行时 capability 和显存预算，而不能退化为固定 lazy/dense 策略。

## 2. 本次文档修改

- `boundflow_asplos_master_plan_2026_07_12.md`
  - PR-11 更名为 Method- and Autograd-Aware Materialization Planner；
  - 增加显式 context、三种 v1 action、capability filter 和字典序目标；
  - 增加 Method-Only、Memory-Threshold 与 per-case Oracle baseline；
  - 增加 workload-family held-out、regret、feasibility 和 OOM 门禁；
  - 将 differentiable fused path 的 saved-state/autograd contract 写入 PR-12。
- `asplos_claims_map.md`
  - 修复 PR-10 已完成但仍写“尚未修改路径”的陈旧状态；
  - 增加 C1-E2/E3、C1-L1/L2、C2-M1 与 C2-H1；
  - 明确 PR-10 profile 是 Planner 动机/校准数据，不是 Planner 有效性证据。
- `asplos_execution_memo_v1_0.md`
  - 更新基线到 `263ea81`；
  - 写入 PR-10 最终判定和 PR-11 最小执行边界；
  - 细化 8 月 5 日 Go/No-Go。
- `pr10_dense_structured_comparison_2026_07_12.md`
  - 增加正式研究解释；
  - 限制 α/αβ OOM 的因果表述，避免在 saved-tensor/allocator 分解前过度归因。

## 3. 明确未采纳或暂缓的内容

- 不把附件中的外部论文类比写成已证因果；当前只保留仓库实测支持的结论。
- 不在 PR-11 v1 加入 chunk、checkpoint、offload 或 fused structured autograd。
- 不重新打开 PR-10 优化 Python structured path。
- 不把内部 `median regret <= 20%` 目标预写成论文已取得结果。

## 4. 下一步

文档定稿后，唯一实现主线是 PR-11：先建立 context/capability/action/plan dump 和离线 Oracle，
再实现 feasibility-first 的自动策略，并在 held-out workload 与 unseen memory budget 上执行
Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy、Global
Planner、Oracle 的公平对照。
