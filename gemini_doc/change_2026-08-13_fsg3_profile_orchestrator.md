# 2026-08-13 — FSG3 Profile Spans 与 36-Process Orchestrator

## 改动

- profile worker实现B0/B1/B2有序、互斥的host/CUDA span；span内不做CUDA synchronize；
- B0记录provider core，B1分离typed pre-state与provider core，B2分离typed pre-state、optimizer、
  backward、KFSB、atomic commit；三路均独立记录official post/queue；
- runtime identity绑定Python executable/version、Torch、CUDA、cuDNN、driver与GPU total memory，跨run
  不一致由replay拒绝；
- 新增正式orchestrator：固定六种排列、36 fresh process，保存worker envelope、完整stdout/stderr、
  host/GPU环境、paired raw、profile spans、closure、summary、failure rows、README与digest manifest；
- formal `generate`要求相关代码路径clean；`smoke`只运行block 0并显式
  `formal_artifact=false/performance_claimed=false`；
- formal每个worker前冻结cool/idle preflight：`<=50°C`、thermal inactive、AC、无额外CUDA进程，
  5秒轮询/900秒上限；通过后只执行一次，worker内失败不得选择性重跑；
- static replay从36条worker raw重算顺序、counter、语义、环境、closure、profile perturbation、paired
  statistics与最终状态，不采信summary投影。

## 非正式 block-0 smoke

顺序执行`B0C,B0P,B1C,B1P,B2C,B2P`，结果：

- `run_count=6`、runtime identity count=`1`、semantic failures=`[]`；
- B0/B1/B2 profile core closure error分别约`0.000547 / 0.000924 / 0.002975`，全部低于1%；
- worker与orchestrator均退出0，stdout/stderr/envelope/metadata完整落盘；
- environment admitted=`0/6`：后台全量pytest进程占用GPU，且温度58→64°C时thermal counter增加；
  门禁正确拒绝该轮，它不进入正式统计。

## 验证边界

本切片证明profile和orchestrator物理路径，不形成speedup结论。正式36-run必须在代码提交后、无其它CUDA
compute process、GPU冷却且每个worker环境准入的条件下重新生成。

提交前验证：targeted `23 passed`；全量`1221 passed, 3 skipped`；四个改动模块mypy clean；pylint
`10.00/10`；black与`git diff --check`通过。
