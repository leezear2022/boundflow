# 2026-08-13 — FSG3 Formal v1 Environment Abort

> 2026-08-13 后续诊断修订：本文准确记录 v1 当时按 schema v2 门禁被拒绝的事实，但“software
> thermal counter 增长即独立热降频”的归因已被 schema v3 取代。RTX 4060 Laptop + driver
> 610.57.04 会把 software power-cap 与 software thermal 的 reason/counter 严格镜像；后续正式轮
> 必须使用 v3 的原始双计数和独立 thermal 投影，不得用本文的旧单计数解释。

## 结论

首个formal v1整轮在7个连续fresh worker后以`ENVIRONMENT-NOT-AUDITABLE`整轮中止。该轮不形成
manifest、speedup或baseline claim，也不会与后续v2混合。

## 已执行顺序

完整执行block 0的`B0C,B0P,B1C,B1P,B2C,B2P`，随后执行block 1的首个`B0P`；7/7 worker
代码退出0、语义路径有效，但environment admitted均为false。

## 独立raw原因

- orchestrator preflight在`<=50°C`、thermal inactive时放行；
- fresh worker加载Python/Torch/solver并初始化CUDA后，其`environment_before`已经回到约52°C；
- query期间software thermal counter分别增长，因而worker gate正确拒绝；
- 无额外CUDA compute process、AC与device identity均正常。

按当时冻结的v2单计数合同，这证明v1的环境检查位置错误：它在fresh worker初始化之前，不能约束真正
计时前的设备状态。继续剩余29个位置不会改变该合同下的整轮不可审计结论，因此按整轮fail-fast停止；
没有删除、替换或重跑任何已执行位置。后续v3诊断没有追认v1 latency，也没有把7个旧值混入新轮。
本地失败证据保留于`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v1-aborted-post-init-thermal/`。

## v2修正

- 保留outer preflight用于排除并发进程；
- fresh worker完成import/CUDA初始化后、compile/query计时前，再等待到`<=45°C`、thermal inactive、AC、
  除自身与固定compositor外无CUDA进程；
- worker preflight raw samples进入envelope并由replay重算；
- orchestrator每完成一个worker立即刷新partial `worker_runs.jsonl/run_metadata.jsonl`，即使整轮被中断也
  不丢失preflight证据；
- v2从block 0 position 0重新执行完整36-run，禁止引用v1 latency。

阈值冻结前的单worker pilot显示`48→50°C`时thermal reason虽为inactive，但counter增加约`1.35s`并被
拒绝；历史独立B0 profile在`45→46°C`曾准入。因此正式v2阈值在任何36-run开始前最终冻结为45°C。
