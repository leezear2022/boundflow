# FSG4/B4-B1a Five-Fresh Runner 候选变更记录

日期：2026-08-18
状态：`IMPLEMENTED-B4-B1A-FIVE-FRESH-RUNNER-PENDING-FORMAL`

## 变更

- 新增独立 B4-B1a CUDA worker，生成 B4-B0 base + reference sufficiency amendment；
- 新增 raw-first five-fresh generate/replay runner，绑定 source/model、B4-B0 frozen identity、
  source commit、9个代码文件 blob、protocol、raw PT、summary 与 manifest；
- root replay 先重建新 reference capture，再将 base 投影交给已外审关闭的 B4-B0 semantic replay；
- amendment 对 incoming/operator bias、output adjoints 与 sparse mapping raw 做结构、tolerance、sign
  与计数重算；
- 新增 8 类 outer-resigned 完整性 probe：incoming bias、operator bias value/presence、两个 output
  adjoint、mapping index、reference attrs、base topology。

## Pilot 与验证

- 5 个独立 CUDA pilot worker / 10 captures；
- amendment comparisons=`90 tensors / 63,645 elements`；
- cross-run max diff=`0.0`，sign exact；
- related=`28 passed`；
- Black、scoped Mypy、Pylint=`10.00/10`、`git diff --check`通过；
- full pytest 延至 formal artifact 与正式 artifact test 加入后执行。

Pilot 位于临时目录，不作为正式 artifact 或 claim。正式运行必须从本提交后的 clean source 生成。

## 已披露限制

B4-B1a capture sufficiency 本身只能拒绝单-run动态 bias/adjoint漂移和绝对绑定的mapping/topology
漂移；若攻击者协调一致改写全部run的动态bias/adjoint并全链重签，必须由后续B4-B1 numerical
reference semantic replay拒绝。本阶段不隐瞒该限制，也不以B4-B1a外审关闭整个B4-B1。

下一步：提交冻结runner，生成formal five-fresh artifact，执行8类完整性probe、artifact tests和
full regression。typed IR/reference与B4-B2/TIR仍关闭。
