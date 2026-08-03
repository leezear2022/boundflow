# 变更记录：移除测试对 ignored PR-12 split artifacts 的依赖

## 问题

新环境执行完整 `pytest tests` 时有 6 个 PR-12 smoke/contract 失败。失败均来自测试直接读取
`.gitignore` 下的历史 `artifacts/phase7a-pr12/.../heldout_split.json`，而不是实现回归。干净 clone
没有这些本地实验工件，因此无法达到仓库文档要求的完整测试门禁。

## 修改

- 新增 `tests/pr12_split_fixtures.py`，直接调用已冻结在代码中的 v1 `_heldout_split()` 与 v2
  `build_split()`；
- 纯 contract 测试直接使用内存 split；
- 需要 controller/subprocess 共享路径的 runner 测试把确定性 v2 split 写入各自 `tmp_path`；
- 不复制或提交历史 raw artifact，不改变任何 split ID、case 或 benchmark 结果。

## 验证

- 原 6 个失败测试在无 PR-12 ignored artifacts 的新环境中通过；
- 完整 `pytest -q tests`：`372 passed, 1 skipped, 6 warnings`；
- 完整测试不再依赖当前机器的历史 artifact 目录。
