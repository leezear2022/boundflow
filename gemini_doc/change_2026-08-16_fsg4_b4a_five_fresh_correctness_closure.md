# 2026-08-16 FSG4/B4-A five-fresh correctness 内部关闭

## 正式证据

- source=`43d41172a7ab810621782f4a51955c677526ed88`；
- artifact=`artifacts/fsg4-b4a-five-fresh/resnet2b-prop0-v1/`；
- manifest hash=`19c03abf9925bd10dc292be34d47f07567b2e6c46857847a21c8598e619f56c7`；
- report hash=`1ebc2d842d771fb355cb7a9c6cad5b1f7a4a7933848ec9732ce7668fd2c1d17f`；
- 冻结交替顺序运行10/10独立fresh worker、5/5 B3/B4-A direct pair；
- 每pair从post-query raw比较19个张量：terminal lower、六层lA、六组intermediate lower/upper；
- 五对max abs diff依次为`3.2186508e-06 / 4.4107437e-06 / 4.4703484e-06 /
  4.6491623e-06 / 6.1094761e-06`，全局最大`6.109476089477539e-06 <= 2e-4`；
- stdout/stderr使用确定性root alias，artifact扫描无`/home/`或`/tmp/`本机路径；
- 所有tensor sign exact，final solver discrete semantics、source/GPU/runtime environment exact；
- B4-A 5/5均为handoff=`1`、terminal export CROWN rerun=`0`、lineage=`6`、provider/fallback=`0`；
- root replay从raw重建同一report/manifest并PASS；`performance_claimed=false`。

## 边界

本轮关闭的是B4-A跨fresh-process数值、lineage、ownership与物理激活正确性，不使用correctness worker
的latency形成性能结论。状态=`INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`，只开放独立正式
B3/B4-A timing。B4-B/TIR、B4-C/D、B5—B7仍关闭。

## 下一步

实现并预注册独立B3/B4-A正式计时artifact，门禁保持：core geomean `>=1.03x`，每个query pair
`B3/B4-A >=0.98x`。未过core门禁时只保留correctness/mechanism，不得形成B4 performance candidate。
