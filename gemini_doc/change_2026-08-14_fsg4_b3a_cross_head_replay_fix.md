# FSG4/B3-A 跨 HEAD Replay 修复记录

日期：2026-08-14

## 问题

B3-A artifact在source HEAD `c7851c8`生成和重放均通过；artifact随关闭提交进入后续HEAD后，replay的
historical-source分支仍固定遍历B2 `CODE_PATHS`，未包含B3-A新增的
`boundflow/runtime/fsg4_b3_prepared_core.py`。manifest正确包含该文件，因此字典比较fail closed。

## 修复

historical-source重算改为使用`_code_paths(configuration)`，与source-HEAD分支及manifest inventory检查
一致。没有改artifact、源revision、GPU输出、counter或hash；只修复跨HEAD verifier覆盖范围。

## 验证要求

- 从关闭提交后的HEAD replay B3-A artifact；
- 冻结B3-A artifact测试通过；
- B3-0历史artifact replay不回归；
- tamper code-revision攻击仍被拒绝。
