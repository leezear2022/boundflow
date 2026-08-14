# FSG4/B3-B Terminal Optimizer Schedule 实现候选记录

日期：2026-08-14
状态：`IMPLEMENTED-PENDING-FRESH-GPU`

## 目标

B3-A已消除core内module move与重复scope，但production仍物化10份完整lower/α/β step snapshot，backward
仍重建optimizer已经拥有的同一父forward trace。B3-B只处理这两个重复点，不修改atomic commit、KFSB、
TIR/JIT/runtime或arena。

## 修改

- 新增`boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py`：
  - typed 10 evaluation/9 update Schedule IR，双学习率及decay进入稳定hash；
  - terminal-only executor不构造逐step对象，只保留terminal lower/α/β；
  - typed parent forward trace绑定scope/graph/split和完整interval/local-ReLU inventory。
- backward export支持exact typed forward trace；不提供时保持旧rebuild路径，提供时严格验证后复用。
- live executor只有在prepared core和terminal schedule同时显式提供时启用B3-B；B2/B3-A不变。
- diagnostic新增B3-B配置及固定counter gate：相对B3-A只允许snapshots `10→0`、forward `5→4`。

## 验证

- terminal Schedule为exact 10 evaluation/9 update；
- terminal lower/α/β与formal trace第10次evaluation逐元素相同；
- forward handoff前后backward lower、六个lA和12个intermediate tensor逐元素相同；
- learning-rate action或split identity篡改fail closed；
- targeted=`42 passed in 5.80s`；
- mypy touched runtime source clean；Pylint=`10.00/10`；`git diff --check` PASS。

## 下一步

先提交实现source，再运行fresh GPU B3-B counter/correctness artifact。未通过前不能把snapshots/forward
目标写成实测事实，也不能进入B3-C或性能计时。
