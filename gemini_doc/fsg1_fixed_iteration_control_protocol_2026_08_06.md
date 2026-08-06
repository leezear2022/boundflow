# FSG1 固定迭代 Control/Profile 协议修正记录

## 起因

首次正式 ResNet2B B0 采集使用相同 60 秒 wall timeout。control/profile 分别访问约
`150022/150018` domains；这是 timeout 边界的自然漂移，不满足 FSG1 对相同求解工作量的
exact 语义门禁，因此执行方主动中止后续 worker，未生成正式证据或性能声明。

## 修正

- 使用 official αβ-CROWN 已有的 `bab/max_iterations`，默认固定为 16；
- `bab/timeout=60` 只保留为 fail-closed 保险丝，不再作为正常 ResNet 终止条件；
- 固定 `solver/batch_size=256` 并关闭 `solver/auto_enlarge_batch_size`，避免 observer 引起的
  allocator 状态改变搜索批量；固定 seed=100，并在 precompile 后重置 seed；
- control/profile 使用相同 model/property、cold isolated property、solver 配置、BaB 迭代预算；
- `max_iterations` 进入 raw worker protocol，并由 control/profile exact 比较和 semantic replay 约束；
- 不修改 αβ-CROWN、auto_LiRPA 或 VNN-COMP vendored source，也不自定义 solver 异常终止路径。

## Claim 边界

本协议测量固定 16 次 BaB iteration 的可比求解前缀。它用于 B0 全栈分母和 observer
扰动归因，不等同于 complete-query latency、TTV 或 BoundFlow speedup。若 workload 在预算前
自然 solved，则保留其真实终态；否则只主张固定前缀可比。

batch 256 是在任何正式候选或 speedup 测量前冻结的 measurement configuration：batch 64 的
single-pair observer ratio 为 `1.060620 > 1.05`，无法满足预注册扰动门禁；增大 batch 只用于让
被测 GPU work 相对 event 记录成本足够大，control/profile 两侧完全相同，不能解释为性能调优结果。

batch256 的下一次 diagnostic 仍为 `1.075754 > 1.05`，定位到 observer 每 call 使用
`inspect.stack()` 构造完整 frame metadata。正式 runner 改为只沿 `f_back` 读取最多20层的等价
phase判定；这只减少测量器开销，不改变、缓存或替换 solver 工作。

轻量phase observer的后续single-pair diagnostic得到control/profile scope=
`4.047353/4.178563 s`、ratio=`1.032419 <= 1.05`，两侧result exact且visited domains均为
`[6064]`，profile捕获234个调用。该结果只准入测量器，仍不作为正式五轮performance evidence。

## 验证要求

- ResNet control/profile 的 result（含 visited domains）必须 exact；
- 每 workload 五个 fresh、交替 AB/BA pair；
- profile median perturbation 必须 `<=1.05`；
- raw-first artifact 必须通过独立 semantic replay；
- 全程 `performance_claimed=false`。

协议代码准入验证：定向`10 passed`、全量`1089 passed, 3 skipped`、Black、mypy、Pylint
`10.00/10`与`git diff --check`全部通过。正式artifact必须从提交后的clean code revision生成。
