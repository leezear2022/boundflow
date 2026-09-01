---
status: fixed-b4b1-reference-execution-policy-pending-v2-artifact
updated: 2026-08-18T15:52:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 reference 执行策略修正

## 发现

首次全量回归为`1402 passed, 3 skipped, 1 failed`。唯一失败是v1 reference artifact exact
record replay；同一测试单独运行通过。独立扫`torch.set_num_threads(1/2/4/8/16)`确认v1在2/8/16
线程下逐字节一致，在1/4线程下S-anchor派生record变化。此前测试改变PyTorch全局线程数且未恢复，
暴露出reference protocol没有冻结CPU执行策略。

## 修正

- protocol新增`torch_num_threads=1`、deterministic algorithms=true、float32 matmul precision=
  highest、MKLDNN=false；
- root replay在可恢复context内应用上述策略，结束后逐项恢复调用方全局状态；
- 1/4/8线程入口重算得到同一records/summary，maximum difference降为
  `6.109476089477539e-07`；
- v1保留为历史失效证据并要求replay拒绝，下一步从本提交生成v2 formal artifact。

## 边界

这是证据可重放性修正，不是性能优化；`performance_claimed=false`、`tir_admitted=false`保持。
B4-B2继续关闭。
