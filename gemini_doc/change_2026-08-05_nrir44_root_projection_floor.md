# NRIR-44 Root-Projection Floor Schedule v1 关闭记录

## 改动

- 新增 typed ranking-only consumer/liveness Plan、Instance、Task、Schedule 与 Trace；
- additive projected floor 仅把九条 objective child query 从 n31d4 改为 n1d0；
- Phase A 通过后才将 projected floor 接入冻结 NRIR-42 global production runtime；
- 新增两阶段 fresh-process formal、manifest、replay 与同步重哈希篡改测试。

## 结果

- Phase A old/projected floor median=`24.235039/9.876515 s`，ratio=`0.407530`，evaluations=`279→9`；
- baseline/refinement/root lower/upper/branch、rank 与 selected 三轮 exact；
- Phase B floor=`8.538814/8.622447/8.648849 s`，whole=
  `43.571040/44.144990/44.095736 s`，对 frozen NRIR-42 median ratio=`0.764254`；
- 每轮 selected `[2,3]`、production nodes `[31,31]`，worst-active lower=
  `-35.530926/-30.258448`；
- Phase A formal hash=`ecb553d8…ff0fe`；Phase B formal payload hash=`2f22d44f…7272d9`。
- targeted `11 passed`、全量 `979 passed, 37 skipped`，Black/mypy/Pylint `10.00/10`。

## 边界

- 状态为 fixed ResNet2B property 0 CPU8 `VALIDATED-REDUCED`，`performance_claimed=false`；
- final 仍为 9/9 unknown；没有 GPU、多 workload、公平竞品、property closure 或 ASPLOS-ready claim；
- 一般 complete verifier 未声明 ranking-only consumer contract 时继续使用完整 floor。
