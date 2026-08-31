# BoundFlow ASPLOS'27 S4-1A closure note 恢复记录

status: validated-static-pending-clean-publication
date: 2026-08-31
scope: activation-gate evidence recovery
performance-claimed: false

## 1. 事件

S4-1A Round 2已获得条件批准，executor执行`dol exchange close`时遗漏`--note`，因此append-only
`closure.json`中的note为空。DocOps exchange本身合法关闭并通过validate，但S4-1B0 activation gate按冻结策略
拒绝空note，返回`closure-note-empty`。

已经提交的closure历史不可改写，也不重新伪造exchange状态。

## 2. 恢复方式

activation gate增加一个严格受限的替代证据路径：

- 首选仍是非空closure note；
- 只有note为空时，才读取已随关闭提交版本化的
  `BOUNDFLOW_ASPLOS27_S4_1A_EXTERNAL_AUDIT_CLOSURE_CHANGELOG_2026_08_31.md`；
- addendum路径进入critical-path clean检查；
- gate绑定其固定SHA256=`3718a3694d9bff4acea32c931053bd079fedeb45a0b49c062958ff638c2c6b21`；
- 同时要求状态、approved round、performance=false和VALIDATED claim四个语义标记；
- addendum缺失、内容篡改或语义缺失均fail closed。

该方案保留原closure与exchange append-only历史，不把空note静默改成非空。

## 3. 边界

本修复只恢复activation evidence，不改变外审结论或S4-1B0施工合同。timing、performance、same-solver、
complete-query和10x仍关闭。

## 4. 验证结果

- activation self-test：`18/18 PASS`，其中新增合法addendum与篡改拒绝两项；
- 两个checker的mypy：clean；
- activation checker Pylint：`10.00/10`；
- Black、diff、DocOps exchange validate/lint：PASS；
- 真实gate在本提交发布前按预期因critical path dirty而拒绝；提交、推送并达到clean publication后必须再以
  `versioned-closure-addendum`来源验证PROCEED。
