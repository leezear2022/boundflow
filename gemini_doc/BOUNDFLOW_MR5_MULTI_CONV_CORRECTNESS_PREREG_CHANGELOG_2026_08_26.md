---
status: active
updated: 2026-08-26T18:58:00+08:00
type: changelog
topic: boundflow
slug: mr5-multi-conv-correctness-prereg
stage: s01
---

# MR5 Multi-Conv Correctness 预注册修改记录

- 冻结C0/C1 stride-2与C2 stride-1三site typed Instance，不允许复制P硬编码ABI；
- 冻结三site顺序、30/27 launch、absent β与零pending dense A ownership；
- 冻结逐site/逐evaluation/optimizer/final semantic门禁；
- 冻结C0/C1独立PyTorch数学oracle与evaluation-5/C1 atomic rollback；
- MR4 full regression=`1764 passed, 3 skipped`，预注册已激活，candidate可按冻结顺序开工。
