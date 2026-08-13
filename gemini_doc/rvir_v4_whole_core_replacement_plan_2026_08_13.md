# RVIR-v4 V4-3 Whole-Core Replacement 预注册计划

日期：2026-08-13

当前状态：V4-3A=`VALIDATED-WHOLE-CORE-TRUTH`，V4-3B=`VALIDATED-NATIVE-BACKWARD-EXPORT`，
V4-3C=`VALIDATED-NATIVE-KFSB`；V4-3D准入；V4-3总体、B2与性能claim仍关闭。

## 1. 目标

把已通过V4-2的pre-state→10/9 native optimizer→12-path atomic copy-out接入真实αβ-CROWN
`update_bounds_core`边界，使official host solver在candidate模式下不调用provider
`update_bounds_core`、`BoundedModule.compute_bounds`或`LiRPANet.update_bounds`，仍产生可被原
`update_bounds_post`和domain queue直接消费的`UpdateBoundCoreReturn`。

V4-3通过后只恢复B2 same-solver AB/BA计时资格，不构成性能claim，也不自动准入B3—B7。

## 2. 冻结基线

- official αβ-CROWN=`e5c7e17`，auto_LiRPA=`5a098e8`，VNN-COMP=`90419aa`；
- workload=`cifar10_resnet:000`，ResNet2B property 0；GPU=RTX 4060 Laptop；
- config：seed=100、timeout=60 s、max BaB iterations=1、batch=64、alpha iteration=5、
  beta iteration=10、KFSB candidates=3；
- V4-2 source artifact=`artifacts/rvir-v4-atomic-copy-out/resnet2b-core-copy-out-v1`，manifest
  SHA256=`b76ee57348f2311996e6b40f013b46acdf39171a3ddc12ae2be9fa0119800136`；
- V4-2已冻结1 core/6 domains/10 evaluations/9 updates/12 receipts/7 changed，不能因V4-3实现改动。

## 3. Whole-Core 输出责任

candidate必须独立生成并绑定以下四层：

1. **Bound result**：`lb/ub`、threshold、pre-last bounds、C/input metadata；
2. **Mutable state**：post α、SparseBeta value、split/history与12-path atomic commit；
3. **Branch inputs**：working intermediate bounds、每个splittable activation的batched `lA`、unstable
   mask、candidate split lower；
4. **Core decision/result**：KFSB decision/points/depth/batch、verified/split count、masked domain state、
   post packet、parent/depth/node accounting、termination与verdict。

不能把provider core的return object、branch decision、lA或post α/β作为candidate输入；它们只允许进入
独立comparator。pre-result、模型、config、branch policy与history是两侧共同输入。

## 4. 执行切片

### V4-3A — Whole-Core Truth Artifact

- 扩展original observer，保存`UpdateBoundCoreReturn`全部语义字段、working intermediate bounds、lA、
  branch decision、post packet与solver accounting；
- raw tensor payload、field schema、lineage和digest均可重放；
- field deletion、lA/intermediate/branch/accounting tamper即使同步重签也fail closed；
- 正式replay必须重新运行固定commit/model/property/config的original provider，再对完整core/post truth做
  shape/dtype/device exact、离散结构exact、sign exact和`atol=rtol=2e-4`比较；不能把内部digest相等
  当成跨进程语义门禁；
- 只冻结truth，不准入replacement。

Capture-ready诊断已确认：`UpdateBoundCoreReturn.batched_lA`在core返回时已被消费为空，因此observer必须在
KFSB入口前捕获六个activation的真实lA；KFSB内部存在3次provider `update_bounds(shortcut=True)`，每次
返回`[24,1]` child lower。两次fresh GPU捕获比较覆盖451个tensor与213,060个浮点符号，最终decision
exact，最大绝对差`5.066394805908203e-06`；49个tensor digest和2个truth hash因合法末位浮点漂移不同。
这也是正式replay采用数值语义重跑而非固定digest比较的直接证据。

### V4-3B — Native Backward Export

- BoundFlow CROWN backward显式导出每个ReLU的lower affine coefficient（provider batched `lA`语义）；
- native interval/intermediate bounds映射为provider node keys；
- fixed trace逐layer比较shape/dtype/finite/sign与数值`atol=rtol=2e-4`；
- 不使用provider `compute_bounds`补齐缺失字段。

### V4-3C — Native KFSB Candidate Evaluation

- 复刻当前KFSB top-3 score/intercept候选和candidate split evaluation；
- 每个candidate child lower由BoundFlow执行，不调用`LiRPANet.update_bounds`；
- top-k候选、两侧child lower、最终decision逐domain与truth比较；离散字段exact、float `2e-4`；
- 任意fallback、候选数/顺序、tie-break漂移fail closed。

### V4-3D — Live Return Assembly

- pre α/β只attach一次；V4-2 terminal state原子提交到真实provider-owned tensors；
- 构造完整`UpdateBoundCoreReturn`并交给未修改的official `update_bounds_post`/queue；
- provider core/compute_bounds/update_bounds callback=`0/0/0`；
- 任一校验/commit/branch/assembly失败时恢复所有live tensors和host packet，禁止partial return或隐式
  fallback。

### V4-3E — Five-Fresh Correctness

- 至少5个fresh process，顺序冻结为`O,C,C,O,C,O,O,C,O,C`五对original/candidate interleave；
- 每对固定cold isolated property、seed/reset/config/GPU；
- call/core lineage、bounds、state、branch、accepted/pruned domains、visited nodes、termination、status/
  success exact或按预注册数值容差；
- 5/5均通过后才设置`b2_same_solver_timing_admitted=true`；任何一轮失败即V4-3 NO-GO。

## 5. Formal Acceptance

V4-3必须同时满足：

1. 三仓/model/property/config/GPU/source artifact identity exact；
2. original truth artifact raw-first replay与完全重签tamper通过；
3. native lower/intermediate/lA逐层通过schema、`2e-4`与sign门禁；
4. KFSB top-3 child lower和最终decision逐domain通过；
5. actual provider-owned 12-path atomic commit且failure rollback；
6. provider core/compute_bounds/update_bounds callback=`0/0/0`，fallback=`0`；
7. 完整`UpdateBoundCoreReturn`、post packet、queue/accounting/termination/verdict等价；
8. 5 fresh correctness pairs全部通过；
9. focused/full/Black/mypy/Pylint/DocOps通过；
10. `performance_claimed=false`，没有用capture/replay时延形成speedup结论。

## 6. Kill Gates

- 若native无法在`2e-4`内恢复lA或KFSB child lower，保留truth/failure artifact并将V4-3标记NO-GO；
- 若必须调用provider bound API才能构造decision或return，视为replacement失败，不降级成“partial whole
  core”；
- 若candidate改变branch、domain accounting、termination或verdict，停止，不进入B2 timing；
- V4-3 correctness阶段禁止同时引入TIR/JIT/fusion/runtime/memory优化，防止语义接入与性能变量混杂。

## 7. 下一动作

V4-3A已由source `bfdeefc`正式关闭：451 tensors/213,060 signs的fresh semantic replay通过，六类
同步重签攻击全部拒绝，full=`1180 passed, 3 skipped`。关闭证据见
`gemini_doc/change_2026-08-13_rvir_v4_whole_core_truth_formal_closure.md`。

V4-3B已由source `762b642`正式关闭：六层lA、12个intermediate tensors和final lower最大差=
`9.2387e-07/6.0797e-06/3.0994e-06`，五类同步重签攻击拒绝，full=
`1183 passed, 3 skipped`。证据见
`gemini_doc/change_2026-08-13_rvir_v4_native_backward_export_formal_closure.md`。

V4-3C已由source `a2097c0`正式关闭：六层mask exact，三组candidate共36项与final decision exact，
72个child lower sign exact、最大差`3.0994e-06`；八类同步重签攻击拒绝，full=
`1187 passed, 3 skipped`。证据见
`gemini_doc/change_2026-08-13_rvir_v4_native_kfsb_formal_closure.md`。

下一动作只实现V4-3D live return assembly。V4-3E five-fresh和B2 timing仍按依赖门禁关闭。
