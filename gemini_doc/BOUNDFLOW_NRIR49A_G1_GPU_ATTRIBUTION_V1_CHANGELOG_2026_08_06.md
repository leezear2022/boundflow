---
status: validated-no-go
updated: 2026-08-06T07:59:16Z
type: changelog
topic: boundflow
slug: nrir49a-g1-gpu-attribution-v1
stage: s01
---

# BoundFlow NRIR49A G1 GPU Attribution v1 Changelog

## 后续范围纠正（2026-08-06）

本 changelog 与冻结 artifact 记录的数值和判定不变。这里的 `VALIDATED-NO-GO` 仅关闭
selected-CROWN-only incremental G2/G3，不关闭 BoundFlow operator→IR→JIT→runtime→memory 的
累计全栈路线；`1.076410x` 仅是把该实测单一区域降为零耗时的 deletion-only Amdahl 上限，不是
BoundFlow 全栈上限。原 `gpu-winner-reselection` 是历史机器输出，当前路线已由
[Full-Stack GPU Baseline and Attribution v1](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
取代。

## Summary

- G0六项 CUDA门禁通过后启动 G1；
- 五个 fresh GPU worker与9文件artifact/replay已完成；
- selected-CROWN GPU queue share中位仅`7.0986% <20%`，latency Amdahl目标不可达，physical-memory
  admission亦未成立，G1以`VALIDATED-NO-GO`关闭；
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
- runner合同单测增至10项，连同既有CUPTI测试共`11 passed`，Black/mypy/Pylint 10.00/10；重启后全量回归
  `1057 passed, 3 skipped`（407.58秒），正式 G1 timing/summary尚未运行。
- 两次前台worker被Codex自动续轮回收，均未落盘JSON；`nohup`也被执行器cgroup清理，三者均不构成
  benchmark run；后续应使用可持续前台session或user systemd unit承载长跑。
- 主机随后重启进入`dgpu_disable=1`：NVIDIA PCI设备、模块和`/dev/nvidia*`均不存在；已将
  `dgpu_disable=0`排队为delayed apply，`gpu_mux_mode=1`继续保持Optimus/Hybrid，等待重启。
- 14:29 CST再次重启后`dgpu_disable=0`已生效；RTX 4060 PCI+nvidia driver+device nodes、
  `nvidia-smi` 8188 MiB与Torch CUDA 13.2/SM89全部通过，仅有7 MiB `kwin_wayland`。
- 首次systemd formal尝试在第0轮末尾的整queue CUPTI profile达到26.5 GiB host peak并被OOM killer终止；
  无worker JSON落盘，正式计数仍0/5。profiler已收缩为重放一个已计时queue中的真实child
  selected-CROWN call，关闭shape/stack收集但保留kernel/launch/sync/memory事件；timing矩阵和门槛未改。
- retry-2 host peak降至2 GiB，证明profiler OOM已关闭；但worker在最终exact gate发现
  `profile/control semantics differ`后退出，仍无有效worker、正式计数0/5。runner现会在validation失败时
  原子写入`*.json.invalid`并保留完整raw profile/control语义，下一次只跑repeat-0诊断差异，不改门槛。
- repeat-0诊断保留33 MiB raw：两clause的profile/control和全部chunk共12组均无结构差异；最大absolute
  float diff=`1.52587890625e-05`，最大relative diff=`1.710717646052519e-04`，均低于原冻结`2e-4`；
  差异仅在raw浮点及其alpha/beta/bounds/score派生hash。实现改为结构exact+逐叶finite/tolerance双门禁，
  同时保存并绑定完整raw payload hash与派生hash差异计数；门槛和production均未改变。
- retry-3以user systemd unit完成5/5 fresh workers，exit 0、wall 30分54秒、主机峰值2.1 GiB；正式
  queue share中位=`0.0709863183`、complete share中位=`0.0705232890`，paired perturbation中位=
  `0.999304/1.006747`，60组结构exact且数值最大absolute/relative diff=
  `2.288818359375e-05/1.710717646052519e-04`。
- 最大allocated/reserved仅占物理显存`0.996%/1.353%`，合法domain batch上限1、无OOM；memory path=
  `N/A`。CUPTI代表调用含5954 kernels、5486 launches、398 sync与5364 memory events。
- artifact summary/manifest hash=`7eefe6a7…ab50`/`d0272fe4…c81f`；独立replay exit 0、stdout exact，
  文件digest与manifest hash独立重算通过；5 raw/50 normalized/2 query/0 failure rows。

## Decisions

- 完整九子句 floor直接 GPU probe在deadline后触发cache-coverage校验，不作为 G1入口；
- 使用与 floor相同的shared→objective root数学构造后直接进入冻结31-node queue；
- chunk sweep仅为 harness override，禁止回写默认32；
- semantic-valid domain batch当前上限1，禁止复制输入伪造memory pressure。
- 按冻结门禁，selected-CROWN不是GPU winner，queue/complete目标均超过该单一区域的 Amdahl
  无限加速上限；停止 selected-CROWN-only incremental G2/G3 和该区域专属 TIR 实现。该决定不关闭
  BoundFlow 全栈路线；`gpu-winner-reselection` 保留为冻结历史输出，当前路线由 Full-Stack 计划取代。

## Follow-Ups（历史，已被取代）

1. 提交G1 closure、回归与DocOps证据，交外部模型审计；
2. 新开只读GPU whole-queue winner归因，不复活selected-CROWN G2/G3；
3. 只有新winner通过独立share/Amdahl门禁后，才预注册下一优化变量。

上述列表记录 G1 closure 当时的后续动作，不再是当前执行指令；当前应执行 Full-Stack 计划。

## Links

- plan: [NRIR49A G1 plan](BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- current route: [Full-Stack GPU Baseline and Attribution v1](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
