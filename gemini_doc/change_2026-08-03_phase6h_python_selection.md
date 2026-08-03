# Phase6H runner Python 选择修复记录

日期：2026-08-03

## 背景

外部审计发现：即使全量测试由 Conda Python 的绝对路径启动，
`scripts/run_phase6h_artifact.sh` 仍会通过 `PATH` 重新解析裸 `python`。当它落到
`/usr/bin/python` 时，Phase6H smoke 会因缺少 PyTorch 失败。

## 修改

- runner 新增统一的 Python 选择顺序：`PHASE6H_PYTHON`、已激活环境的
  `${CONDA_PREFIX}/bin/python`、最后才是 `PATH` 中的 `python`；
- sweep、report、plot、Torch 元信息和 `pip freeze` 全部复用同一解释器；
- 显式解释器不存在时，在 benchmark 前以退出码 2 fail closed；
- smoke 测试把 `PATH` 限制为系统目录并显式传入当前测试解释器，覆盖审计中的复现场景；
- AE README 记录解释器覆盖方法和失败语义。

## 验证结果

- 修复前复跑审计场景（Conda pytest 绝对路径 + `PATH=/usr/bin`）：
  `1 failed`，内部 `/usr/bin/python` 报 `ModuleNotFoundError: torch`；
- 修复后同一受限 `PATH` 场景：`2 passed in 3.72s`；
- 全量回归：`457 passed, 37 skipped, 5 warnings in 55.73s`；
- `black --check`、`bash -n`、Pylint（10.00/10）与 `git diff --check` 通过；
- `dol lint --soft`：通过，无缺失规则。
