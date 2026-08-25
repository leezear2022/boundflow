---
status: validated-no-go-r3-3-s-isolated-physics
updated: 2026-08-26T05:30:00+08:00
type: changelog
topic: boundflow
slug: r3-3-active-beta-timing-implementation
stage: s01
---

# R3-3 Active-β Isolated Timing 实现变更记录

## 变更

- 新增 public-PyTorch CUDA active-β S-anchor baseline，dense α/β reconstruction、Linear/bias
  epilogue 与 `torch.autograd.grad` 全部在 call 内；
- 新增 cache-hit sparse Linear TIR custom forward/backward candidate wrapper；
- 新增 CUDA event wrapper timing、absolute/incremental memory observation、parity 和结构证据；
- 新增 6 fresh AB/BA worker、raw-first artifact/replay、bootstrap/worst/memory gate 派生与
  12 类重签 tamper 探针；
- formal summary 始终保持 `performance_claimed=false`、`r3_4_open=false`，最终 claim 只能由
  tamper 后 closure 授予。

## 当前验证

- baseline/candidate parity 和 timing/memory primitive：`2 passed`；
- 非正式 10-sample smoke 的 candidate/baseline 约为 `0.63x–0.72x`，只用于确认计时
  路径物理执行，不形成 claim；
- formal artifact 尚未生成，须先提交 clean source。

## 边界

本实现不修改 TIR/schedule，不做 tuning，不开放 R3-4/same-solver。若 formal 低于预注册
门禁，必须以 NO-GO 关闭当前 fixed schedule，不得改阈值或切 kernel-only latency。

## 最终结果

formal geomean/bootstrap lower/worst=`0.668275x/0.629157x/0.599089x`，12/12 tamper，
full=`1658 passed,3 skipped`。当前 fixed schedule 已 NO-GO，R3-4/same-solver 不开放。
