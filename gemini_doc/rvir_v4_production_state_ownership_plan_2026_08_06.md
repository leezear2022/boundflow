# RVIR-v4 Production-State Ownership 计划与门禁

## 目标

把真实αβ-CROWN solver-core边界上的输入、start-node keyed α、SparseBeta、split/history、
intermediate/reference bounds与optimizer mutation变成BoundFlow独立拥有、可重建、可执行的typed
payload。完成后重新判断B2；在此之前不运行性能比较。

## FSG2事实修正

FSG2 inventory检查了`alpha`、`sparse_beta`、`beta`、`split_beta`，但当前auto_LiRPA实际把
BaB beta对象存放在`node.sparse_betas`（复数）中，每个`SparseBeta`包含：

- `val`：可优化beta值，shape=`[domain_batch,max_history_per_layer]`；
- `loc`：split neuron ordinal；
- `sign`：split方向；
- `bias`：一般branch point时的bias，可为空；
- history owner：`UpdateBoundPreReturn.d_dict["history"]`；
- optimizer copy-out：`UpdateBoundCoreReturn.working_beta`与postprocess后的domain beta values。

因此FSG2“显式beta tensor为0”的结论只能解释为旧探针漏字段，不能作为RVIR-v4的事实基础。
FSG2 initial-CROWN正向结论与“当时B2未准入”仍成立；RVIR-v4必须用修正后的capture重新审计。

## 执行边界

RVIR-v4同时记录两层边界：

1. `BoundedModule.compute_bounds` call tree：保持原24-call parent/depth/phase/result账本；
2. `update_bounds_core(pre_result -> core_result)`：这是production beta/split GPU-heavy ownership边界，
   可在provider attach state之前捕获完整pre-state，在extract之后捕获mutation/result。

只hook第1层会错过`BetaFullData.attach_to_net()`之前的domain/history owner；只hook第2层又会漏掉
initial/alpha初始化。两层必须用共同run/call lineage绑定。

## Typed payload

每个tensor必须包含：

- stable semantic path，不使用Python object address；
- role：input/spec/intermediate/reference/alpha/beta-value/beta-location/beta-sign/beta-bias/
  decision-threshold/result/state-output；
- axes：domain/spec/alpha-polarity/start-node/relu-feature/history-slot；
- shape/dtype/device/content digest；
- ownership：read-only、copy-in、mutable-copy-out；
- alias group与pre/post digest；
- activation layer、start node、split layer等semantic keys。

history必须保存domain ordinal、layer、locations、coefficients、optional bias和depth；optimizer合同必须
保存iteration、learning rates、stop criterion identity、bound polarity、intermediate-bound policy和
determinism设置。

## 阶段与kill gates

### V4-0 — Corrected Capture

- 正式ResNet fixed-one-iteration运行重新捕获24-call tree；
- `sparse_betas.val/loc/sign[/bias]`在beta/split边界均非空且与history一致；
- pre/core/post tensor、alias与mutation closure=100%；
- 旧字段遗漏加入负向回归，禁止再次把plural field漏记为0。

未过：停止，不实现backend。

### V4-1 — Frozen-State Evaluation

- 将provider已优化的post α/β/split state映射到BoundFlow native evaluator；
- 对每个beta/split outer call复算lower，`atol=rtol=2e-4`、soundness方向单独通过；
- state key、domain/spec axes和result shape exact。

未过：记录representation mismatch，不进入optimizer replacement。

### V4-2 — Optimizer Mutation

- 独立BoundFlow backend从pre-state执行相同steps/policy；
- result、post α、post beta逐tensor复核；
- original callback/fallback=`0/0`，mutation copy-out原子提交；
- nested-call count不要求复刻provider内部实现，但outer solver-core work/result/state必须exact。

未过：B2继续NO-GO。

### V4-3 — Whole Core Replacement

- 替换`update_bounds_core`而不调用provider core/compute_bounds；
- branch decision、accepted/pruned domains、parent/depth/node、termination/verdict exact；
- 至少5 fresh correctness runs全部通过，才恢复B2 AB/BA timing。

## 性能纪律

- V4-0—V4-3均`performance_claimed=false`；
- 不用capture/replay开销代表candidate性能；
- B2恢复后仍按原FSG3协议执行5 fresh counterbalanced pairs；
- B3—B7不因RVIR-v4实现而自动升级。

## 提交序列

1. `docs(runtime): preregister RVIR-v4 production state ownership`；
2. `feat(runtime): add RVIR-v4 typed state payload`；
3. `feat(bench): capture corrected αβ-CROWN core state`；
4. `feat(runtime): add frozen-state evaluator mapping`；
5. `feat(runtime): replace optimizer mutation and solver core`；
6. `bench(runtime): close RVIR-v4 and decide B2 admission`。

## V4-0A Typed State 实现记录

- 新增`boundflow/runtime/rvir_v4_production_state.py`：digest-bound tensor、semantic path/axes、
  source device、read/copy-in/mutable-copy-out与alias group；
- alpha按`activation/start-node`保存`alpha_polarity/start_spec/domain/feature`axes；
- beta只从正确的plural `sparse_betas`捕获`val/loc/sign[/bias]`，singular-only对象在beta phase
  fail closed；
- history按domain/layer保存location/coefficient/optional bias，并与SparseBeta loc/sign/bias前缀
  逐项一致性校验；
- pre/post mutation receipt要求path集合、role/axes/shape/dtype均不漂移；tensor tamper和history
  mismatch均拒绝；
- 11项定向单测通过；这只完成typed ownership合同，不代表backend已通过。

## V4-0B Corrected Capture 实现记录

- 新增`scripts/run_rvir_v4_production_state_capture.py`，同一真实GPU run同时观察
  `BoundedModule.compute_bounds` call tree与`update_bounds_core(pre_result -> core_result)`；
- 修正FSG2探针遗漏：beta phase只接受plural `node.sparse_betas`，真实run捕获6个
  `SparseBeta.val/loc/sign`组；singular或只有空容器均fail closed；
- 捕获input/spec/intermediate/decision-threshold、activation/start-node keyed alpha、domain/layer keyed
  history、optimizer policy以及pre/post alpha/beta mutation；history的location/sign/bias/score/depth均进入
  stable hash，ReLU隐式零bias仅在history bias逐项为0时允许省略tensor；
- 诊断GPU run通过冻结门禁：24 calls=`12 initial + 1 alpha + 11 beta`，真实core=`1`，
  history entries=`36`，beta value tensors=`6`，mutation receipts=`12`、changed=`7`；
- capture只允许PyTorch `weights_only=True`安全加载；NumPy scalar在worker边界归一化为Python整数；
- artifact replay会重建typed snapshot、重算tensor/snapshot/mutation/summary digest与JSON projections，
  并验证代码来源；`performance_claimed=false`；
- 本节数字来自正式生成前的诊断run。提交并冻结代码后还必须生成正式artifact和独立replay，
  V4-0才可关闭；V4-1/V4-2/B2当前仍未准入。
