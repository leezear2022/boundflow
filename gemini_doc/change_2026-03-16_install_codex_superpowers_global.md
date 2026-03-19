# 变更记录：为 Codex 全局安装 Superpowers，并补跨主机复用文档

**日期**: 2026-03-16
**类型**: 主机配置 / 文档
**范围**: `~/.codex/`、`gemini_doc/`、`docs/`

---

## 背景与动机

需要把 `superpowers` 装成主机级 Codex skill，而不是只对当前仓库生效。这样这台机器上的所有 Codex 工程都能复用同一套技能。

同时，还需要留下一份可跨主机复用的说明文档，让另一台机器上的 Codex 读完后能自动完成同样的安装，而不是机械照抄固定路径。

---

## 主要改动

### 1. 主机级安装

在当前主机上完成了以下配置：

- clone：`~/.codex/superpowers`
- 建立软链接：`~/.codex/skills/superpowers -> ~/.codex/superpowers/skills`
- 更新 Codex 配置：`~/.codex/config.toml`
  - 新增：

```toml
[features]
collab = true
```

这样 `dispatching-parallel-agents`、`subagent-driven-development` 等依赖协作能力的 skill 也可用。

### 2. 仓库内文档

新增长期文档：

- `gemini_doc/codex_superpowers_global_install.md`

文档内容覆盖：

- 为什么不能盲目假设 `~/.agents/skills`
- 如何自动探测当前主机实际生效的 skill 发现目录
- 如何执行主机级安装
- 如何验证、更新、卸载
- 一段可直接发给 Codex 的执行指令，便于其它主机复用

### 3. 记录与索引

- 新增：`gemini_doc/change_2026-03-16_install_codex_superpowers_global.md`
- 更新：`gemini_doc/README.md`
- 更新：`docs/change_log.md`

---

## 验证

已在当前主机上检查：

```bash
ls -la ~/.codex/skills/superpowers
git -C ~/.codex/superpowers rev-parse --short HEAD
rg -n '^\[features\]|^collab = true$' ~/.codex/config.toml
```

验证结果：

- `~/.codex/skills/superpowers` 已是有效软链接
- `~/.codex/superpowers` 已是有效 git 仓库
- `~/.codex/config.toml` 已包含 `collab = true`

本次未运行仓库测试，因为改动仅涉及主机级 Codex 配置与文档记录，不涉及 BoundFlow 代码语义。

---

## 影响范围

- 对当前主机：
  - 所有 Codex 工程在重启 Codex 后都可以发现 `superpowers`
- 对仓库：
  - 新增了一份可跨主机复用的安装文档
  - 保留了本次主机配置变更的可审计记录

---

## 后续建议

- 在其它主机复用时，优先让 Codex 读取 `gemini_doc/codex_superpowers_global_install.md`，并按其中的“目录探测 + 安装 + 验证”流程执行。
- 若未来 Codex 的 skill discovery 目录再次变化，只需更新该文档中的探测规则，不必改动 BoundFlow 代码。
