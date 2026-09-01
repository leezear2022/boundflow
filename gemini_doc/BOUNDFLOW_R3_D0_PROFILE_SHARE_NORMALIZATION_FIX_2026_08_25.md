---
status: corrected-before-closure
updated: 2026-08-25T08:00:00+08:00
type: changelog
topic: boundflow
slug: r3-d0-profile-share-normalization-fix
stage: s01
---

# R3-D0 Profile Share 归一化修正

## 发现

首轮5-pair artifact全部校准通过，但replay把profile session中的`compiled_kernel_sum_ns`直接除以
unprofiled 30-sample median `C`。pair 1由此产生`compiled share=1.0321`，暴露两个不同measurement
scope被直接混用。

## 修正

- kernel/host/compiled-region share只在同一个`profiled_host_wall_ns`scope内计算；
- `profiled_host_residual = profiled_host_wall - kernel_union`；
- Amdahl使用的unprofiled host residual由profile比例投影：
  `H = C * profiled_host_residual / profiled_host_wall`；
- compiled region share同理使用`compiled_kernel_sum / profiled_host_wall`，再代入冻结的unprofiled
  `target_candidate / C`公式。

门槛、worker顺序、样本、region所有权和10x上限均未改变。首轮
`VALIDATED-NO-GO-R3-SO-CVJP-PERFORMANCE`因scope混用失效，不进入claim；必须从新source重新运行5
fresh formal。

## 验证

- synthetic ledger继续验证union、projected residual与route；
- 首轮raw显示5个candidate的同scope compiled share为`0.9967–0.9974`，而旧算法曾产生`>1`，修正
  消除了该不变量违反；
- 后续重新运行完整artifact/replay/tamper。
