---
status: implemented-awaiting-r3-2a-formal
updated: 2026-08-25T05:21:31+08:00
type: changelog
topic: boundflow
slug: r3-2a-optimizer-trajectory-implementation
stage: s01
---

# R3-2A Optimizer Trajectory 实现修改记录

## 实现

- 新增R3-2A dynamic rebind receipt：每步只重绑P-anchor compressed α content hash，immutable
  tensor、静态topology和既有compiled admission保持不变；
- 新增P-anchor 10 evaluation/9 Adam step + ExponentialLR + clamp轨迹runtime；
- native与candidate复用同一冻结schedule，但运行在独立mode，candidate每步调用R3-1b2 compiled
  custom VJP且无native shadow/fallback/eager；
- 保存逐步lower/dα/α before-after/Adam moments/rebind/compiled receipt，便于formal replay；
- 新增动态rebind、immutable drift、10/9逐步parity、source隔离和schedule tamper测试。

## 边界

当前只是实现与targeted correctness验证入口；未生成five-fresh formal artifact，未读取latency，
`performance_claimed=false`。R3-2B仍关闭。

实现前检查的一对非正式diagnostic通过：10步最大lower/dα/α差分别为`8.5831e-06`、
`8.2888e-08`、`2.3842e-07`，allocated/reserved ratio约`0.05869/0.16667`。这些值只用于确认
预注册门槛可执行，必须由clean-source five-fresh artifact重新生成后才能形成R3-2A判定。

formal clean-source检查只审计tracked文件；未跟踪的用户输入`docs/CIBC_for_DAC.pdf`不属于本轮源码，
不得因为它存在而阻塞artifact，也不得自动加入、移动或删除。

首次formal启动进一步暴露DocOps hook会在每条命令后追加`.docops/ev.jsonl`，使tracked全零变化条件
不可达。runner现在只豁免这一条append-only事件日志；任何其它tracked路径变化仍fail closed。
