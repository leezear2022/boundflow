# RVIR-v4 V4-3B Native Backward Export 修改记录

日期：2026-08-13

## 修改

- 在通用CROWN backward增加可选lower-adjoint export，不改变既有bounds返回语义；
- 新增typed `NativeBackwardExportV4`，把六个native ReLU preactivation显式映射为provider activation
  lA，并把共享pre-result external intermediate bounds映射为六个provider preactivation keys；
- export路径只消费模型、input/spec、terminal native α/β/split和共享intermediate input，不读取V4-3A
  lA或provider return作为candidate输入；
- provider core/`compute_bounds`/`update_bounds` callback与fallback固定为`0/0/0/0`；
- 新增独立artifact runner，重新执行V4-2 native optimizer形成terminal state，再与V4-3A truth在独立
  comparator中比较。

## Capture-ready诊断

- 六层lA shape与sign全部exact，最大绝对差`9.238719940185547e-07`；
- 12个intermediate lower/upper tensor全部通过，最大绝对差`6.079673767089844e-06`；
- final lower最大绝对差`3.0994415283203125e-06`；
- 全部低于预注册`atol=rtol=2e-4`；
- 当前只标记`IMPLEMENTED-NATIVE-BACKWARD-EXPORT / FORMAL-ARTIFACT-PENDING`，不关闭V4-3B、V4-3
  或B2，不形成性能claim。

## 下一动作

提交clean runner基线后生成正式V4-3B artifact，执行semantic replay和同步重签tamper门禁；通过后才能
准入V4-3C native KFSB candidate evaluation。
