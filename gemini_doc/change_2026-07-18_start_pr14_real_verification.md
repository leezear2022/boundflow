# 变更记录：启动 PR-14 Verification-Aware Execution on Real Verification Workloads

## 背景

本地 `main@263ea81` 只到 PR-10，早期审计遗漏了远端 research branches 与 annotated tags，导致
错误提出重复 PR-10B.2。完整 Git 审计确认真实冻结基线为 `57a854b` / tag
`pr13-validated-reduced`：PR-11、PR-12、PR-13 均已 `VALIDATED-REDUCED`，PR-13A 也已经完成
真实 BaB recorder/fixed replay。

## 本次修改

- 从 `pr13-validated-reduced^{}` 创建 `feat/pr14-real-verification`；
- 新增 `gemini_doc/current_status_after_pr13.md`，冻结 PR-13 后真实状态、证据边界和当前缺口；
- 新增 `gemini_doc/pr14_execution_plan.md`，将下一主线收敛为真实 verifier workload coverage
  与 execution；
- 更新 ASPLOS 执行 memo，使唯一顺序延伸到 PR-14；
- 更新 `gemini_doc/README.md` 与 `docs/change_log.md`。

## 决策

- 不回到 `bench/pr10b2-real-bab-fixed-domain-replay`；
- PR-14A 复用现有 `BoundQuery`、recorder、lineage 和 replay，不再创建重复 query schema；
- 不重写 host solver，不新增验证算法，不恢复孤立 TIR 调优；
- PR-14 先用 MLP/CNN/ResNet-block coverage matrix 回答真实 query distribution 与 backend
  eligibility，再决定 full E2E 和 C3 定位；
- 最终 baseline 必须是 same-solver original batched executor。

## 验证

本次仅修改 Markdown 文档和分支基线，验证项为：

- `git diff --check`；
- Markdown 相对路径存在性检查；
- 分支/HEAD/tag 一致性检查；
- 工作区不包含 third-party 修改；
- CUDA 13.3 环境 stash 保持不变。
- PR-13 focused 回归：15 passed。
