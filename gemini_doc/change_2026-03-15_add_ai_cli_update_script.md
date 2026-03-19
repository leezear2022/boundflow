# 变更记录：新增本机 AI CLI 更新脚本

**日期**: 2026-03-15
**类型**: 脚本 / 工具
**范围**: `scripts/`

---

## 背景与动机

本机已经安装了三套常用命令行工具：

- `gemini` -> `@google/gemini-cli`
- `claude` -> `@anthropic-ai/claude-code`
- `codex` -> `@openai/codex`

这三个工具当前都以 `npm` 全局包形式安装在用户的 `nvm` Node 环境中。手动逐个执行更新命令容易漏版本、也不方便先做检查，因此补一个统一脚本。

---

## 主要改动

新增脚本：`scripts/update_ai_clis.sh`

能力如下：

- 默认同时处理 `gemini`、`claude`、`codex` 三个目标。
- `--check` 只读检查：
  - 显示全局 npm prefix
  - 显示包名、PATH 中的二进制路径、当前版本、最新版本、状态
- 支持只更新指定目标：
  - `bash scripts/update_ai_clis.sh codex`
  - `bash scripts/update_ai_clis.sh gemini claude`
- 缺失包默认跳过，不会未经确认直接安装。
- `--install-missing` 可在缺失时执行安装。
- 更新后会再次打印最终状态，便于确认是否成功。

脚本内置的目标映射：

- `gemini` -> `@google/gemini-cli`
- `claude` -> `@anthropic-ai/claude-code`
- `codex` -> `@openai/codex`

---

## 使用方式

只检查版本状态：

```bash
bash scripts/update_ai_clis.sh --check
```

更新全部：

```bash
bash scripts/update_ai_clis.sh
```

只更新其中一个：

```bash
bash scripts/update_ai_clis.sh codex
```

缺失时也安装：

```bash
bash scripts/update_ai_clis.sh --install-missing gemini claude codex
```

---

## 验证方式

本次未直接执行实际更新，避免无意修改本机全局 CLI 版本；仅做了以下验证：

- `bash -n scripts/update_ai_clis.sh`
- `bash scripts/update_ai_clis.sh --check`

其中 `--check` 已确认本机当前识别到：

- `gemini`：`@google/gemini-cli`
- `claude`：`@anthropic-ai/claude-code`
- `codex`：`@openai/codex`

---

## 影响范围

- 新增：`scripts/update_ai_clis.sh`
- 新增：`gemini_doc/change_2026-03-15_add_ai_cli_update_script.md`
- 更新：`docs/change_log.md`

---

## 后续建议

- 如果后续改为 `brew` / `pipx` / `uv tool` 安装，可在此脚本中再增加安装器检测分支。
- 如果需要周期性自动更新，可以再包一层 crontab/systemd timer，但建议保留 `--check` 作为默认预览入口。
