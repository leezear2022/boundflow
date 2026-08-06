---
status: implementation-ready-formal-run-pending
updated: 2026-08-06T06:31:05Z
type: changelog
topic: boundflow
slug: nrir49a-g1-gpu-attribution-v1
stage: s01
---

# BoundFlow NRIR49A G1 GPU Attribution v1 Changelog

## Summary

- G0六项 CUDA门禁通过后启动 G1；
- 正式性能结果尚未运行，所有 opportunity/Amdahl/memory门槛已预注册；
- 本阶段只做 profiler与只读归因，不改 production TIR/kernel/default policy。

## Changes

- 将 VNN-COMP sparse checkout扩展到 frozen `cifar10_resnet`，commit未变；model/property digest与
  NRIR48历史 artifact exact；
- 验证 clauses 2/3 queue的 GPU direct-root入口；clause 2保持31 nodes、15 groups与历史 worst lower；
- 冻结5 fresh-process Latin chunk矩阵、paired profiler perturbation、CUDA event/wall/memory、Amdahl反解、
  physical-memory admission和9文件artifact/replay合同。
- worker cache绑定runner与production依赖源码SHA，源码变化后fail closed；仅豁免记录在案且不超过
  64 MiB的`kwin_wayland`桌面合成器，其他GPU compute PID全部拒绝。

## Validation

- G0 formal v2：六项 PASS，`ready_for_g1`；
- 非正式 clause 2 GPU admission probe：31 nodes、15 groups、max depth 4、worst lower
  `-35.53092575073242` exact；
- runner合同单测8项通过，Black/mypy/Pylint 10.00/10；重启后全量回归
  `1057 passed, 3 skipped`（407.58秒），正式 G1 timing/summary尚未运行。
- 两次前台worker被Codex自动续轮回收，均未落盘JSON；`nohup`也被执行器cgroup清理，三者均不构成
  benchmark run；后续应使用可持续前台session或user systemd unit承载长跑。
- 主机随后重启进入`dgpu_disable=1`：NVIDIA PCI设备、模块和`/dev/nvidia*`均不存在；已将
  `dgpu_disable=0`排队为delayed apply，`gpu_mux_mode=1`继续保持Optimus/Hybrid，等待重启。
- 14:29 CST再次重启后`dgpu_disable=0`已生效；RTX 4060 PCI+nvidia driver+device nodes、
  `nvidia-smi` 8188 MiB与Torch CUDA 13.2/SM89全部通过，仅有7 MiB `kwin_wayland`。

## Decisions

- 完整九子句 floor直接 GPU probe在deadline后触发cache-coverage校验，不作为 G1入口；
- 使用与 floor相同的shared→objective root数学构造后直接进入冻结31-node queue；
- chunk sweep仅为 harness override，禁止回写默认32；
- semantic-valid domain batch当前上限1，禁止复制输入伪造memory pressure。

## Follow-Ups

1. 重启后复核`dgpu_disable=0`、NVIDIA PCI/module/device、`nvidia-smi`与Torch CUDA；
2. 运行五轮formal矩阵并生成/replay artifact；
3. 按冻结门槛选择 G2、memory-only或 NO-GO。

## Links

- plan: [NRIR49A G1 plan](BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
