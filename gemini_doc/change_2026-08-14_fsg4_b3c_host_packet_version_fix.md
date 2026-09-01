# FSG4/B3-C Host Packet Version 边界修复

日期：2026-08-14
状态：`FIXED-PENDING-FRESH-GPU-RERUN`

## 发现

source `a3ac761`的第一次fresh B3-C真实GPU运行在atomic stage前fail closed。真实provider `d_dict`除
`depths/history/thresholds`外，还含`BatchFirstBranchingDecisions`等将被core return丢弃的临时对象。
初版`_host_version()`错误地递归序列化全部value，因此拒绝了合法provider packet。

该失败发生在任何live tensor/host mutation之前；incomplete staging临时目录自动清理，没有生成或追认
正式artifact。

## 修复

- pre-host version改为绑定完整key inventory，但只递归版本化事务实际保留的
  `depths/history/thresholds`；
- candidate/post-host仍要求exact三字段并完整版本化；
- 任意额外key变化仍会改变pre version并在commit前拒绝；
- 额外provider对象只允许作为将被删除的opaque value，不进入可重放artifact或candidate packet；
- 新增不可序列化`object()`额外字段的CUDA回归，确认stage/commit成功且该字段被原子移除。

## 边界

本修复不降低12-path tensor inventory、placement、alias、finite、version或rollback门禁，也不改变
candidate D2H=`0`预注册目标。必须从修复后的新commit重新运行fresh GPU，不能使用失败run追认任何
counter或正确性结论。
