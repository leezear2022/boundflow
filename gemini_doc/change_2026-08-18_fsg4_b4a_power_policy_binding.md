---
status: validated-pending-clean-source-v4
updated: 2026-08-18T10:55:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-power-policy-binding
stage: s01
---

# FSG4/B4-A GPU 功耗策略绑定

## 1. 失败事实

source=`be2fa96` 的正式计时 v3 从 position 0 完成了 20 个 worker。worker 20
（block 5、`B4-A-profile`）本身返回完整 raw，correctness、activation 与 profile 计数均正常，但
environment 判定为 `admitted=false`：software thermal counter 在执行期间出现独立增长，不能作为稳定
性能样本。v3 因而整体 fail closed，不续跑、不形成任何 B3/B4-A ratio。

可审计身份：

- protocol SHA256：`5449a6ee...dd`；
- run 00/19/20 SHA256：`1a12f520...28` / `9536c4a5...a5` / `7cbf9122...ae`；
- worker 20：forward=`4`、optimizer bound evaluation=`10`、optimizer trace/evaluation/update=
  `1/10/9`、handoff/rerun=`1/0`、provider/fallback=`0/0`。

## 2. 根因边界

该机器的 `nvidia-powerd.service` 原为 active。现场观测其 Dynamic Boost 会使移动 GPU 的生效功耗
上限高于 55 W 默认值，并使动态 thermal limit 下的 software thermal signal 在长序列中反复触发。
仅约束每个 worker 启动温度 `<=45°C` 不足以约束 worker 执行期间的功耗策略，因此 v3 失败是实验环境
合同缺口，不是 B4-A correctness 或 activation 失败。

已临时停止 `nvidia-powerd.service`；当前 service=`inactive`，`enforced.power.limit=55.00 W`，空闲时
software thermal/power-cap signal 均 inactive。正式实验结束后恢复 service。

## 3. 修改

- formal protocol 绑定 `nvidia_powerd_state=inactive` 与 `gpu_power_limit_watts=55.0`；
- 每个 worker preflight 都现场读取 service 状态和 `enforced.power.limit`，不一致立即 fail closed；
- raw preflight 保存这两个字段，root replay 重验；
- 新增 power-policy outer-resigned 攻击，tamper 清单由 12 类增至 13 类；
- 保留原 `<=45°C`、software thermal inactive、AC power、无外部 compute process 等门禁。

## 4. 结论与下一步

本轮只修复实验协议，不产生性能 claim，`performance_claimed=false`。v1/v2/v3 都是诊断证据，不得
恢复或选样。完成固定 related/full/static/DocOps 验证并提交 clean source 后，只允许在新目录从
position 0 生成 v4；B4-B/TIR 继续关闭。

## 5. 验证

- 固定 9 文件 related：`63 passed`；
- 全量：`1353 passed, 3 skipped`（6 个既有 warning）；
- B4-A formal/tamper 专项：`11 passed`；
- Black：4 个触达文件 clean；
- Mypy：`--explicit-package-bases` 下 2 个脚本 clean；
- Pylint：2 个脚本 `10.00/10`；
- `git diff --check`：PASS。
