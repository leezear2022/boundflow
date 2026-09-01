---
status: implemented-pre-formal
updated: 2026-08-26T18:10:00+08:00
type: changelog
topic: boundflow
slug: mr4-production-conv-site-census-formal-gates
stage: s01
---

# MR4 Production Conv Site Census Formal Gates 修改记录

- 新增MR4机械validator，绑定worker source=`1fa4f0f`、5 fresh顺序、三site topology与150 rows；
- 对每row严格验证evaluation/site顺序、grad 9/1、β absent、handoff、shape与静态成本；
- 对run0→run1..4的outer/final α/module state执行冻结容差与sign exact比较；
- 机械派生C0/C1/C2 MAC ratio、materialization bytes、projected `30/27` launches与MR5 route；
- 新增raw-first、拒绝resume的5-process formal runner和历史code revision replay；
- 新增16类fully re-signed tamper probe；
- synthetic/negative=`17 passed`，Black/mypy/pylint 10/10；
- 本提交不含formal raw，不开放MR5或任何timing/performance claim。
