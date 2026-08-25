# GC-0/FCR-1 Verification Graph ABI + Correctness 预注册变更记录

date: 2026-08-26
scope: documentation-only
performance-claimed: false

## 变更

- 新增独立 GC-0/FCR-1 预注册，冻结 verification program/region/value/op/effect/VJP schema；
- 冻结 analysis-only legality、22 类稳定拒绝原因与 guarded rule registry；
- 冻结 Relax/TIR lowering request/receipt、重编译 replay identity；
- 冻结真实 physical arena、persistent views、lease/epoch 与 prepared runtime ABI；
- 冻结 P empty-β、S active-β、C2→C1→C0 multi-site 10/9 三类 formal signature；
- 冻结双独立 oracle、五组 fresh、数值/离散/trajectory/rollback 门禁；
- 冻结 `10/9` region submission、warm per-op crossing/allocation/PyTorch tensor op=0、dense-A=0；
- 冻结 semantic replay 与 22 类 fully re-signed tamper；
- 冻结 GC0-0—GC2-2 分阶段提交 DAG、AC1—AC7、GO/NO-GO 与后继开放边界；
- 对齐总蓝图的 GC-0/GC-1/GC-2 分层：GC-0 只关 ABI/legality，GC-1 关 rewrite/VJP correctness，
  GC-2 才关真实 arena/runtime；禁止一次提交或一次结论跳级；
- 明确独立外审批准前 `implementation_open=false`，全阶段 `timing_open=false`、
  `performance_claimed=false`。

## 事实修正

- 不把已有 `TaskIRUnit`、R3/CIBC 实验类型描述成已完成的通用 verification graph；
- 显式记录旧 `DifferentiableLowerRegionIRV1` 的 `/49` 与 lower-only/single-consumer 限制；
- 显式记录旧 `R31FullRegionPlanV1` 的固定 6 ReLU/domain/spec/site 限制；
- 不把 MR7-R `required region speedup=1.91213674x` 误写成已经取得的 speedup；
- 不用 P empty-β 的局部结果外推 S active-β 或 multi-site production 性能。

## 未发生

- 无 Python/C++/TIR/runtime 代码改动；
- 无 artifact、benchmark、CUDA Graph、schedule search；
- 无 production default、solver、query/queue 行为变化；
- 无 correctness 或 performance claim 升级。

## 验证计划

- `git diff --check`；
- 新文档路径、证据入口和权威文档引用检查；
- hardcode/claim/timing 负向文本检查；
- DocOps change/validation/lint；
- 独立外部模型按专用 handoff 审计预注册完整性与可证伪性。
