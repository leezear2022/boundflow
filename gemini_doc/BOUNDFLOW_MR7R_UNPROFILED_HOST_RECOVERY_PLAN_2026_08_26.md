---
status: preregistered
updated: 2026-08-26T00:25:00+08:00
type: plan
topic: boundflow
slug: mr7r-unprofiled-host-recovery
stage: s01
---

# MR7-R Unprofiled Host Attribution Recovery 预注册

## 1. 问题

MR7 的 host ledger 本身来自 unprofiled control，但全局资格被一个`1.239399`的CUPTI profile/control
扰动样本否决。MR7-R只回答两个问题：

1. perf-counter category ledger相对原MR6 diagnostic是否足够低扰动；
2. 不启profiler时，FFI/layout/post-output boundary是否稳定跨过冻结的15%与15 ms门槛。

本轮不重新测device share，不改变TIR、schedule、allocator、solver、guard policy或production default。

## 2. 冻结协议

- 5个独立pair，共10个fresh process；
- 每pair包含：`baseline=MR6 diagnostic worker`与`ledger=MR7 control worker`；
- 顺序固定：`BL/LB/BL/LB/BL`；
- workload、seed、10/9、C2→C1→C0、module和diagnostic guard policy继承MR6/MR7；
- headline clock为host `perf_counter_ns`；CUDA event只做方向一致性，不形成category share；
- raw-first、禁止partial resume；独立replay与fully re-signed tamper。

## 3. Correctness与扰动门禁

- 5/5 allclose、sign exact；
- 5/5 30 forward / 27 backward、cache/module/stream/fallback exact；
- 5/5 host category closure error `<=2%`；
- ledger/baseline host ratio median必须在`[0.95,1.05]`，worst必须在`[0.90,1.10]`；
- host/event方向5/5一致；
- 若扰动门禁失败，状态=`INVALID_MR7R_LEDGER_PERTURBATION`，不得形成share。

## 4. Host opportunity门禁

仅在§3全过后计算：

- 每个ledger run定义`boundary=ffi_dlpack_stream + layout_materialization + post_output_guard`；
- 5-run median outer share `>=15%`且absolute median `>=15 ms`；
- 至少4/5个run同时分别满足`share>=15%`与`absolute>=15 ms`；
- 以真实median share代入parity目标`1.107412x`，required region speedup必须存在且`<=10x`。

全过状态=`VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY`；否则状态=
`VALIDATED_NO_GO_MR7R_HOST_BOUNDARY`。

## 5. 通过后只开放什么

只开放FCR-1/MR7-A的预注册与ABI/correctness实现：把layout、relaxation、bound op、epilogue、minimal
saved-state backward放入一个compiled region，使用persistent arena并显著减少57 launch/约540 DLPack
crossing。不得只实现更薄的逐算子wrapper；timing必须在correctness关闭后另行预注册。

## 6. 禁止项

- 不重跑或筛选MR7 CUPTI pair来追求通过；
- 不把MR7的8.69% device share变成正式claim；
- 不直接实现schedule sweep；
- 不形成query/queue/competitor/ASPLOS-ready claim。
