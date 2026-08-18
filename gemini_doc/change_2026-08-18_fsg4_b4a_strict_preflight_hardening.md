---
status: implemented-pending-clean-source-formal-run-v3
updated: 2026-08-18T18:30:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-strict-preflight-hardening
stage: s01
---

# FSG4/B4-A Strict Preflight 加固

## v2拒绝事实

source=`ee73bc2`的v2从position 0运行并越过v1的B4-A profile计数失败点。worker 0—4完整通过；worker 5
`block-01-pos-01-B3-control`进程完成后，outer typed validator读到：provider/fallback均0、语义与序号正常，
但environment为`admitted=false`，其中hardware thermal=false、software thermal=true、software power
cap=true、两counter不再耦合，因此`independent_thermal_slowdown=true`。

v2 protocol SHA256=`5af684b7...fbca5`，worker 5=`10612daa...4eb8`，worker 0/4分别为
`6aa141da...b74d8`/`e8f8bffe...cec83`。v2本机目录保留诊断raw但按仓库策略不提交；不得resume、不得形成
paired ratio或性能分类。

## 门禁缺口

旧formal preflight继承B3条件：温度`<=50°C`且“独立”thermal inactive。worker 5启动前的sample为49°C，
software thermal与power counter暂时相等，因而被视作coupled并准入；执行结束时counter分离，正确触发
environment拒绝。连续短worker下，该条件缺少热余量。

## 加固

- B4-A formal专属最终preflight sample要求GPU `<=45°C`；
- `sw_thermal_slowdown`必须字面为`Not Active`，不接受active但与power counter耦合；
- runner在每个worker前轮询，直到严格条件成立或900秒fail closed；等待sample全部进入metadata；
- protocol显式绑定`software_thermal_signal_must_be_inactive=true`和45°C阈值；
- replay/resume使用同一严格validator，tamper温度攻击同步到新阈值。

## 验证与边界

新增单测覆盖45°C边界、46°C拒绝及coupled-active拒绝；formal targeted=`11 passed`，固定related=
`63 passed`，全量=`1353 passed, 3 skipped`，Black/Mypy/Pylint `10.00/10`及diff check通过。本次只
加固环境准入，不改变B3/B4-A代码、计时门槛、顺序或分类。下一步提交clean source，然后从position 0
生成v3；`performance_claimed=false`，B4-B/TIR关闭。
