# 2026-07-09：分支盘点与 Linux TVM 算子交接文档

## 摘要

新增分支盘点文档，梳理当前 BoundFlow 仓库的本地/远程分支、每条已提交分支的职责，以及当前本机未提交的 Phase 7A/7B WIP 内容。

## 主要改动

- 新增：`gemini_doc/branch_inventory_2026-07-09_linux_tvm_handoff.md`
  - 记录 `origin/main`、`origin/codex/phase7a-structured-crown-docs`、`origin/feat/macos-arm64-dev-env` 的提交关系和用途。
  - 明确当前工作区存在 PR-15 到 PR-28 的未提交 WIP，不属于任何远程分支。
  - 给出切换到 Linux 后继续 TVM 算子开发的推荐基线：`origin/feat/macos-arm64-dev-env`。
  - 建议新开 `feat/linux-tvm-operators`，并先验证 `install_dev.sh`、`test_env.py` 和 Phase 4C/4D TVM 路径。

## 影响面

- 仅新增文档，不改 runtime、planner、backend、测试或环境脚本行为。
- 当前已有未提交 WIP 不纳入本次文档变更。

## 验证

- `git fetch --all --prune`
- `git branch -vv --all`
- `git log --oneline --reverse origin/main..origin/codex/phase7a-structured-crown-docs`
- `git log --oneline --reverse origin/codex/phase7a-structured-crown-docs..origin/feat/macos-arm64-dev-env`
- `git status --short`

