# RVIR-v4 V4-2B Policy/Trace Cross-Binding 修改记录

> 后续状态：本文记录的重启门禁已经解除；正式artifact/replay/tamper closure见
> `change_2026-08-13_rvir_v4_optimizer_step_formal_closure.md`。

日期：2026-08-13

## 起因

V4-2B formal GPU artifact仍等待系统重启。对已实现runner做攻击面复审时发现，原合同虽然能拒绝raw
tensor、ordinal、copy-in和外层digest篡改，但两个内部重哈希场景还缺独立证据交叉绑定：

1. mutation policy只检查“类型与允许模式”，没有把固定ResNet2B production run的全部数值钉死；
2. step trace内的raw state/lower即使自洽，尚未逐step与同一worker独立生成的call-tree metadata绑定。

这两个缺口不影响上一提交的capture-ready结论，但会削弱正式artifact对“攻击者同步重算内部hash”的
抵抗力，因此在GPU恢复前先关闭。

## 修改

- `ProductionMutationPolicyV4.validate()`现在精确准入：
  - iteration=`10`，alpha/beta LR=`0.01/0.05`；
  - lower-only、fixed-intermediate、deterministic=`false`、batch-any stop；
  - Adam、LR decay=`0.98`、keep-best、sum reduction、patience=`10`、start-save=`0.5`；
  - last-fp64=`false`、pruning=`true`、threshold=`0.2`、max-time=`60.0`；
  - alpha/beta enabled、init-alpha=`false`、shared-alpha/output constraints/direct optimize/input
    tightening/cuts继续按冻结协议fail closed。
- artifact semantic replay对每个depth-1 beta call新增两条独立交叉绑定：
  - `call.pre_state`必须逐字段等于step的24个typed tensor metadata；
  - `call.result_tensors[result[0]]`必须与step lower的shape/dtype/content digest一致，且call result
    device必须是严格`cuda`或`cuda:<integer>`。
- 新增内部重哈希攻击：修改mutable α并重建tensor/step/trace hash，仍因call pre-state不匹配被拒绝；
  修改lower并重建lower/trace hash，仍因call result digest不匹配被拒绝。
- 新增固定policy负向矩阵：iteration、双LR、deterministic、decay、patience、start-save、pruning、
  threshold及max-time任一漂移均拒绝。

## 验证与边界

- focused optimizer/capture=`26 passed`；扩展RVIR-v4回归=`47 passed`；
- mypy四文件clean；Pylint四文件=`10.00/10`；
- full suite=`1118 passed, 39 skipped`；39项skip均为既有CUDA、TVM重复编译或冻结
  VNN-COMP checkout环境边界；
- 当前仍没有formal GPU trace，V4-2B状态保持`IMPLEMENTED-CAPTURE-READY / FORMAL-BLOCKED`；
- 本修改不实现optimizer replacement，不准入V4-2C/B2，也不产生性能claim。

## 下一步

重启进入已安装的新内核，使loaded NVIDIA module `610.43`与userspace `610.57`恢复一致；随后从本轮
clean commit生成正式artifact，并对state/result/step/policy分别执行“内部语义重哈希 + 外层manifest
重签名”探针。全部通过后才关闭V4-2B。
