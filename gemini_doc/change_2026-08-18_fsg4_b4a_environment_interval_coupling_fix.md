---
status: validated-pending-clean-source-v5
updated: 2026-08-18T11:20:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-environment-interval-coupling-fix
stage: s01
---

# FSG4/B4-A 环境计数区间耦合修复

## 1. v4 失败事实

source=`03043a3` 的正式计时 v4 从 position 0 完成 run 00—18，19/19 均
`environment.admitted=true`。第 20 个位置 run 19（`B3-profile`）也返回了完整 worker raw，但旧环境
投影给出 `admitted=false`，outer runner 因而 fail closed。v4 不续跑、不形成 ratio。

身份与边界：

- protocol SHA256：`99717cfa...672`；
- run 00/18/19 SHA256：`8932ad43...37c` / `48983660...f34` / `61e07553...2e9`；
- run 19 的 B3 activation 正常：forward=`4`、optimizer bound evaluation=`10`、optimizer
  trace/evaluation/update=`1/10/9`、handoff/rerun=`0/1`、provider/fallback=`0/0`。

## 2. 根因

run 19 的 worker 区间内：

- software thermal counter：`272851004 → 274913481`；
- software power-cap counter：`272796425 → 274858902`；
- 两者增量完全相同，均为 `2062477 µs`；
- before/after 两侧的即时 reason 均为 `Not Active/Not Active`；
- 两个累计计数在 worker 开始前已有固定 `54579 µs` 历史偏移。

旧 `_environment_gate()` 要求两个**累计绝对值**在 before 和 after 都相等。因此，一次发生在 worker
区间外的历史偏移会永久毒化后续 worker，即使待测区间内两个驱动别名增量完全相同。该判据与函数现有
“interval”语义不一致；这是环境证据投影错误，不是新的 B4-A 性能或正确性失败。

## 3. 修改

- 保留计数单调、software thermal/power signal同时存在、before/after reason成对一致等约束；
- 耦合判据改为 worker 区间增量严格相等：
  `thermal_after - thermal_before == power_after - power_before`；
- 不相等 1 µs 仍判为 independent thermal slowdown 并拒绝；
- B4-A formal replay 从 worker raw `environment_before/after` 重算四项 thermal/power projection，不再只
  信任 worker 派生布尔值；
- 新增 environment-counter-delta outer-resigned 攻击，tamper 清单由 13 类增至 14 类。

## 4. 结论与下一步

v4 保留为 fail-closed 诊断，不能在修改后的代码下补跑或升级。完成 fixed related/full/static/DocOps
验证并提交新的 clean source 后，只允许从 position 0 生成 v5。`performance_claimed=false`，B4-B/TIR
继续关闭。

## 5. 验证

- 固定 10 文件 related（含基础 worker 与历史 B3 artifact replay）：`70 passed`；
- 全量：`1353 passed, 3 skipped`（6 个既有 warning）；
- interval offset/delta 正负路径与 B4-A formal/tamper 专项：`18 passed`；
- Black：5 个触达文件 clean；
- Mypy：`--explicit-package-bases` 下 3 个脚本 clean；
- Pylint：3 个脚本 `10.00/10`；
- `git diff --check`：PASS。
