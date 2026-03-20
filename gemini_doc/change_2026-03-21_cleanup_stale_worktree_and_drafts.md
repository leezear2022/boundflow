# 变更记录：清理旧 worktree 与无效草稿文件

## 动机

在 PR-8 合回 `main` 之后，仓库里还残留了两类无效工件：

- 一个已经被主线覆盖的旧 feature worktree：
  - `phase7a-pr6-alpha-beta-crown-chain-cnn`
- 两份未跟踪的草稿/聊天记录：
  - `gemini_doc/Untitled-1.md`
  - `gemini_doc/chatlog.md`

这些内容都不是当前项目的有效源码或文档资产，继续保留只会污染工作区状态。

## 主要改动

### 1) 删除已过时的 PR-6 worktree 与分支

- 清理：
  - `.worktrees/phase7a-pr6-alpha-beta-crown-chain-cnn`
  - git branch `phase7a-pr6-alpha-beta-crown-chain-cnn`

清理前先确认了该分支没有主线之外的独有提交：

- `git rev-list --left-right --count main...phase7a-pr6-alpha-beta-crown-chain-cnn`
- 输出：`4  0`

含义是：

- `main` 比该分支多 4 个提交
- 该分支比 `main` 多 0 个提交

因此可以安全删除。

### 2) 删除两份无效草稿文件

- 删除：`gemini_doc/Untitled-1.md`
  - 内容是外部机器上的 shell 搜索日志，与 BoundFlow 仓库无关
- 删除：`gemini_doc/chatlog.md`
  - 内容是早期对话记录，不属于项目文档或实验工件

## 影响面

- 主仓库只保留当前有效的 `main` worktree
- 工作区不再因为无关草稿而处于脏状态
- `gemini_doc/` 中仅保留项目相关设计、阶段总结与变更记录

## 如何验证

```bash
git worktree list
git status --short
```

期望结果：

- `git worktree list` 只剩主仓库一项
- `git status --short` 不再出现：
  - `gemini_doc/Untitled-1.md`
  - `gemini_doc/chatlog.md`

## 备注

- 这次清理不涉及源码语义改动，因此不需要重新跑测试。
- 若后续还有临时分析草稿，建议放在仓库外，或在确认有长期价值后再整理成正式文档落到 `gemini_doc/`。
