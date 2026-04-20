# 变更记录：新增简洁版 AI CLI 更新脚本

## 动机

现有更新脚本 `scripts/update_ai_clis.sh` 功能完整，但输入参数较多。补一个更短、可直接执行的版本用于快速更新本机常用 CLI：

- `gemini` (`@google/gemini-cli`)
- `claude` (`@anthropic-ai/claude-code`)
- `codex` (`@openai/codex`)

## 主要改动

- 新增 `scripts/update_ai_clis_simple.sh`：
  - 只做三件事：检查 `npm` 存在、逐个执行 `npm install -g <pkg>@latest`、输出完成提示；
  - 无需参数，默认按三条工具顺序更新。

## 使用方式

```bash
bash scripts/update_ai_clis_simple.sh
```

## 影响范围

- 新增脚本文件：`scripts/update_ai_clis_simple.sh`
