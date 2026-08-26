# GC0-0 Generic Verification Graph Schema 变更记录

date: 2026-08-26
stage: GC0-0
parent: `ad23d86ddd2d8dc95b4ad4dd74d6a02710a34bce`
status: externally-approved-closed
timing-open: false
performance-claimed: false

> **2026-08-26外审关闭**：exchange `gc0-0-schema-20260826` Round 1 已 approve 并由executor
> 关闭；0 blocker、0 major、1 minor、1 info。正式状态为
> `VALIDATED-GC0-0-GENERIC-VERIFICATION-GRAPH-SCHEMA`。唯一后继是GC0-1 capture/analysis
> 预注册，不是实现。F1/F2 的 shallow-policy/full-analysis 分层约束见独立closure文档。

## 1. 门禁与范围

GC-0/FCR-1 预注册已由 DocOps exchange `gc0-fcr1-prereg-20260826` Round 1 外审批准并由 executor
关闭。外审结论为 `approve`，0 blocker、0 major、3 minor、1 info，只开放 GC0-0。

本提交只实现 generic schema 和无需 analysis pass 即可独立触发的负向校验。明确未实现：

- Bound/Task/R3 capture；
- topology/postdominator/effect-order/alias analysis；
- Relax/TIR lowering 或编译；
- physical arena/prepared runtime/custom VJP execution；
- production provider replacement；
- timing、CUDA Graph、多 stream、schedule search 或性能 claim。

## 2. 新增代码

新增 `boundflow/ir/verification_graph.py`：

- `VerificationProgramV1`；
- `VerificationRegionV1`；
- `VerificationValueV1`；
- `VerificationOpV1`；
- `VerificationEffectTokenV1`；
- `VerificationVJPContractV1`；
- `VerificationRuleV1/VerificationRuleRegistryV1`；
- `LegalityResultV1` schema；
- `VerificationGraphModuleV1` canonical container；
- `VerificationGraphValidationError`，携带 stable rejection reason；
- canonical JSON、`allow_nan=false`、SHA-256 identity 与 strict canonical round-trip；
- frozen、non-executable v1 rule registry，包含 V-R1/R2/R3、V-D1、V-C1、V-VJP1、V-M1、V-H1；
- module/program/registry hash 逐层绑定，`timing_recorded=false`、`performance_claimed=false`、
  `execution_enabled=false` fail closed。

## 3. 拒绝原因分层

22 类 `VerificationRejectionReason` 全部成为 first-class enum/schema，并被机械分区：

- `GC0_DIRECT_REJECTION_REASONS`：15 类无需 graph analysis 即可由 typed constructor/identity 检查触发；
- `GC01_ANALYSIS_REJECTION_REASONS`：7 类依赖 external-use、postdominator、effect-order、alias/lifetime、
  residual closure、dense escape 或 queue boundary analysis，留给 GC0-1。

两集合必须 disjoint，union 必须精确等于全部 22 类。GC0-0 不伪造 analysis-only pass 或声称全部
negative graph 已执行。

## 4. 三类 schema fixture

新增 `tests/test_gc0_verification_graph_schema.py`，只证明通用表达和 canonical round-trip：

1. empty-β + Conv affine；
2. active-β + Linear affine，含 location/sign/history 与 β gradient owner；
3. 三个相邻 Conv affine，并由通用 coarse-commit attributes 表达 `10 evaluation/9 mutation`。

这些标识只存在于测试 fixture 的通用 ID/shape 数据中；production schema 源码不包含 ResNet2B、固定
node id、C0/C1/C2 或冻结 shape 常数。测试不执行 production region，也不形成 legality admission。

## 5. 直接负向门禁

当前专项覆盖：

- unsupported op kind；
- dynamic shape；
- dtype/device；
- non-normalizable layout；
- state version；
- α start-node/index/direction；
- β active/empty/location/sign/history；
- bound polarity；
- endpoint policy；
- VJP owner/saved-state/higher-order/dense escape；
- reject-before-launch fallback；
- program/rule/module identity；
- rejected legality result 无 stable reason；
- noncanonical JSON 与 `performance_claimed=true`。

## 6. 外审 findings 处置

- F1 accepted：预注册 §9.4 已拆为 GC-0 ABI、GC-1 semantic/VJP、GC-2 physical runtime 三组门禁；
- F2 accepted：GC0-0 只测试 direct subset，analysis-dependent negative graph 明确留给 GC0-1；
- F3 accepted：§4.1 的“执行”改为 schema construct/admit/lower；三 signature 在 GC-0 的覆盖定义为
  schema construction + canonical round-trip；
- F4 accepted：文档内部验收改名 `Plan-AC1—Plan-AC7`，后续 exchange 使用 `Audit-AC`。

approved exchange 状态不允许写 finding response，因此上述处置在本 changelog 与代码/文档 diff 中闭合，
没有伪造 DocOps response。

## 7. 已执行验证

- targeted：`11 passed`；
- related IR/R3：`54 passed`；
- Mypy `--follow-imports=skip`：2 files clean；
- Pylint：`10.00/10`；
- Black：2 files clean。
- full：`1832 passed, 3 skipped, 6 warnings in 685.82s`；三个 skip 为 TVM 已可用时避免重复
  compilation，以及两个冻结 VNN-COMP checkout 不可用的既有环境边界；warnings 为既有 Torch JIT、
  profiler 与 ONNX/tree spec 兼容性提示。

专项、静态检查、相关回归和全量回归均已通过。最终 DocOps validation 在提交冻结前记录。本状态不是
GC0-0 closure，也不开放 GC0-1。
