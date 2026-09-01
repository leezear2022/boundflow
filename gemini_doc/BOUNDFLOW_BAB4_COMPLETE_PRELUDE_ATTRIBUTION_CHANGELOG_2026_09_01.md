# BAB4 complete-verifier prelude 归因修改记录

status: tooling-implemented-profile-pending
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 动机

完全 warm-matched 五组 raw 表明，BAB4 exact-call core 平均节省约 `38.7 ms`，但 complete query 平均只
节省约 `24.6 ms`。按 pairwise 差值重算，pre-core 候选相对 control 中位慢 `12.26 ms`、均值慢
`14.03 ms`；此前用两组各自中位数相减得到的 `17.6 ms` 不再作为正式差值。

已有 host ledger 进一步显示 root incomplete 差异中位仅 `0.002 ms`，`general_bab` 基本完整保留 core
收益，损失集中在 `complete_verifier → general_bab` 的前置事务。

## 2. 新增归因边界

在既有 nested host observer 中按 opt-in 方式增加：

- `complete_verifier`；
- `cuda_empty_cache`；
- `gc_collect`；
- `complete_bab`；
- `prepare_for_act_bab`；
- `general_bab`。

默认正式 worker 不安装这些 wrapper。只有 `--attribute-complete-prelude` 诊断运行启用，结果写入
`diagnostics.complete_prelude_timings`，并同时记录 inclusive/exclusive 闭合时间。

新增 `run_bab4_complete_prelude_profile.py`，执行三对交替 fresh `B4-A-WARM ↔ BAB4-WARM`，保留每个
worker raw，并汇总候选减对照的逐 pair 差值。该工具只做路由，`profile_timing_claimed=false`、
`performance_claimed=false`。

首轮启动在 GPU 执行前 fail closed：对 venv Python 使用 `Path.resolve()` 会跟随解释器 symlink 到 uv
基础解释器，导致 venv site-packages 与 torch 消失。修正为与既有正式 runner 相同的 `absolute()`，既
保留绝对路径也保留 venv 解释器身份。失败目录仅含首个 worker 的 import stderr，不进入归因数字。

第二次启动完成首个 worker 后，诊断因 `complete_prelude_timings=None` fail closed。根因是参数已穿过
S4 与 B4-A adapter，但旧的 `B3 → FSG3` adapter 未透传。补齐该布尔诊断参数；此次 worker raw 只用于
发现 wiring 缺口，不进入归因数字。

第三次启动完成 4 个 admitted worker 后，第 5 个 worker 因 NVIDIA software thermal signal 被环境门禁
拒绝；没有代码/数值失败。诊断 runner 增加与正式 five-fresh 同类的有限环境重试，保留全部 attempt raw，
只选择 `environment.admitted=true` 的 worker，且不放宽任何温度、电源或外部进程门禁。

## 3. 三对 fresh 结果

成功目录：`/tmp/bab4-complete-prelude.eXdePz/profile`。三对均为 fully warm-matched，结果只用于
attribution：

| 候选减 control 的 pairwise 差值 | pair 0 | pair 1 | pair 2 | 中位 |
|---|---:|---:|---:|---:|
| pre-core | `+5.593 ms` | `+4.997 ms` | `+14.612 ms` | `+5.593 ms` |
| `gc.collect` | `+11.217 ms` | `+10.406 ms` | `+11.578 ms` | `+11.217 ms` |
| `cuda.empty_cache` | `+0.064 ms` | `-1.192 ms` | `+0.126 ms` | `+0.064 ms` |
| `prepare_for_act_bab` | `+0.003 ms` | `+0.006 ms` | `+0.002 ms` | `+0.003 ms` |
| `complete_bab` exclusive | `+0.032 ms` | `-0.008 ms` | `+0.022 ms` | `+0.022 ms` |

`general_bab` inclusive 候选分别快 `42.738/31.610/34.750 ms`，说明 core 收益确实传入 BaB；
`gc.collect` 的稳定额外扫描时间是 complete prelude 损失的主要可操作来源。profiled query/core
geomean 分别为 `1.04009x/1.17544x`，只作扰动披露，不替代正式五对数字。

下一动作是实现可撤销的 prepared-runtime GC isolation：query 前完整收集，把确定长期存活的 prepared
compiler/runtime 对象移出 query 内全代扫描，query 后恢复；必须验证语义、GC 状态恢复、peak memory 与
多次重复 query，无收益或内存恶化即撤销。
