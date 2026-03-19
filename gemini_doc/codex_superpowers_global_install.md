# Codex Superpowers 全局安装说明

这份文档的目标有两个：

- 在当前主机上，把 `superpowers` 装成主机级 Codex skill，使所有工程都可用。
- 在其它主机上，让 Codex 读完本文档后，可以按同一套步骤自动完成安装，而不是机械照抄某个固定路径。

---

## 1. 适用前提

- 已安装 `codex`
- 已安装 `git`
- 当前用户对 `~/.codex` 有写权限

`superpowers` 仓库地址：

- `https://github.com/obra/superpowers`

---

## 2. 关键原则

不要盲目假设 skill 发现目录一定是 `~/.agents/skills`。

不同 Codex 版本/分发方式可能使用不同的本机发现目录。执行安装前，先探测“当前这台机器上正在生效的目录”，再决定把 `superpowers` 链接到哪里。

推荐规则：

1. 如果 `~/.codex/skills` 已存在，优先使用它。
2. 如果 Codex 日志里出现 `~/.codex/skills`，优先使用它。
3. 否则回退到 `~/.agents/skills`。

在本仓库当前这台主机上，实际生效的是：

- `~/.codex/skills`

---

## 3. 给 Codex 的直接执行指令

如果你在另一台主机上，希望 Codex 读完本文档后直接替你安装，可以直接对 Codex 说：

```text
请阅读 gemini_doc/codex_superpowers_global_install.md，并严格执行“4. 自动安装步骤”。
要求：
1. 先自动判断当前机器正在使用的 Codex skill 发现目录，而不是假设固定路径。
2. 把 superpowers 安装成主机级配置，使这台机器上的所有工程都可用。
3. 若 ~/.codex/config.toml 中未启用 collab，则补上 [features] collab = true。
4. 完成后执行“5. 验证”里的检查。
5. 最后明确提醒我重启 Codex。
```

---

## 4. 自动安装步骤

下面这组命令是推荐的主机级安装流程。Codex 在执行时应先读懂，再运行。

```bash
set -euo pipefail

SKILL_ROOT="$HOME/.agents/skills"

if [ -d "$HOME/.codex/skills" ]; then
  SKILL_ROOT="$HOME/.codex/skills"
fi

if [ -f "$HOME/.codex/log/codex-tui.log" ] && grep -Eq "($HOME/.codex/skills|\\.codex/skills)" "$HOME/.codex/log/codex-tui.log" 2>/dev/null; then
  SKILL_ROOT="$HOME/.codex/skills"
fi

if [ -d "$HOME/.codex/superpowers/.git" ]; then
  git -C "$HOME/.codex/superpowers" pull --ff-only
else
  git clone https://github.com/obra/superpowers.git "$HOME/.codex/superpowers"
fi

mkdir -p "$SKILL_ROOT"

if [ -e "$SKILL_ROOT/superpowers" ] && [ ! -L "$SKILL_ROOT/superpowers" ]; then
  echo "目标路径已存在且不是软链接: $SKILL_ROOT/superpowers" >&2
  exit 1
fi

ln -sfn "$HOME/.codex/superpowers/skills" "$SKILL_ROOT/superpowers"
```

然后确保 `~/.codex/config.toml` 启用了：

```toml
[features]
collab = true
```

如果原文件里还没有 `[features]` 段，就追加一个；如果已经有 `[features]` 但没有 `collab = true`，就在该段补上。

---

## 5. 验证

安装后至少检查下面三项：

```bash
SKILL_ROOT="$HOME/.agents/skills"
[ -d "$HOME/.codex/skills" ] && SKILL_ROOT="$HOME/.codex/skills"

ls -la "$SKILL_ROOT/superpowers"
git -C "$HOME/.codex/superpowers" rev-parse --short HEAD
grep -En '^\[features\]|^collab = true$' "$HOME/.codex/config.toml"
```

预期结果：

- `superpowers` 是一个指向 `~/.codex/superpowers/skills` 的软链接
- `~/.codex/superpowers` 是有效 git 仓库
- `~/.codex/config.toml` 中有 `collab = true`

最后必须重启 Codex，因为 skill discovery 在启动时完成。

---

## 6. 更新与卸载

更新：

```bash
git -C "$HOME/.codex/superpowers" pull --ff-only
```

卸载：

```bash
SKILL_ROOT="$HOME/.agents/skills"
[ -d "$HOME/.codex/skills" ] && SKILL_ROOT="$HOME/.codex/skills"

rm -f "$SKILL_ROOT/superpowers"
rm -rf "$HOME/.codex/superpowers"
```

---

## 7. 本机已验证结果

截至 2026-03-16，本仓库所在主机已按本文档完成安装，实际生效状态为：

- clone 目录：`~/.codex/superpowers`
- skill 发现目录：`~/.codex/skills`
- 软链接：`~/.codex/skills/superpowers -> ~/.codex/superpowers/skills`
- 配置：`~/.codex/config.toml` 已启用 `[features] collab = true`

如果别的主机采用不同的 skill 发现目录，仍应先做“目录探测”，再决定软链接目标。
