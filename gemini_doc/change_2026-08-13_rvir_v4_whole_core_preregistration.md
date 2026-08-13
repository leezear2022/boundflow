# RVIR-v4 V4-3 Whole-Core Replacement 预注册修改记录

日期：2026-08-13

## 修改

- 在V4-2 optimizer replacement正式关闭后，新建V4-3 whole-core计划；
- 将`UpdateBoundCoreReturn`责任拆为bound result、mutable state、branch inputs、core decision/result四层；
- 明确KFSB内部candidate split也必须由BoundFlow执行，不能通过`LiRPANet.update_bounds`隐性回调provider；
- 冻结V4-3A truth→V4-3B native lA/intermediate→V4-3C KFSB→V4-3D live assembly→V4-3E five-fresh
  correctness顺序；
- 冻结provider core/compute_bounds/update_bounds=`0/0/0`、完整return/post/queue/verdict等价与10项formal
  acceptance；
- 性能claim和B2计时继续关闭。

下一动作只允许V4-3A whole-core truth artifact。

## Capture-ready实现补充

- observer已在KFSB消费前捕获六个activation lA、三组candidate split、三次provider child-lower返回、
  最终branch decision、完整`UpdateBoundCoreReturn`和post packet；
- 新增typed truth validator和formal artifact runner；
- formal replay改为重新执行固定external provider，并对完整truth做`2e-4`数值容差、符号和离散结构门禁；
- 两次独立RTX 4060运行比较：451 tensors、213,060 signs exact、最大绝对差
  `5.066394805908203e-06`、decision exact；
- 当前仍是capture-ready，不将V4-3A、whole-core replacement或B2 timing标记为通过。

## Formal generation前置修正

首次formal generation暴露runner把`.venv/bin/python`调用`Path.resolve()`后解析为uv base interpreter，
从而丢失external αβ-CROWN venv的Torch。现改为保留绝对venv launcher symlink；这是执行环境隔离修正，
不是算法或correctness失败。
