# 变更记录：忽略 `artifacts/` 与 `out/` 产物目录

## 动机

`artifacts/` 与 `out/` 目录属于运行产物（JSONL/CSV/图/manifest 等），具有：

- 体积持续增长、内容随机器/时间变化；
- 不适合作为源码的一部分进入版本控制；
- 复现应通过 `scripts/run_phase5d_artifact.py` / bench + postprocess 重新生成。

因此将其加入 `.gitignore`，避免误提交。

## 本次改动

- 更新：`.gitignore`
  - 新增忽略：`artifacts/`、`out/`

## 如何验证

```bash
mkdir -p artifacts/phase5d out/phase5d
git status --porcelain
```

