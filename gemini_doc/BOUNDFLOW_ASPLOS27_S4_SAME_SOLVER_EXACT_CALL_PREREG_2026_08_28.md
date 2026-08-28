---
status: draft-preregistered-pending-s3-external-approval
date: 2026-08-28
type: plan
topic: boundflow
slug: asplos27-s4-same-solver-exact-call
stage: s04
execution-authority: false
code-change-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4：production same-solver exact-call 接入预注册

## 0. 直接结论

S4不能把S3 `PreparedS2CrownProgramV1`直接塞进`stage_solve.update_bounds_core`后返回。当前S3只闭合了
P-anchor的一条可变α，而冻结production core在每次evaluation里同时优化6条α和1条非空β；whole-core返回还
需要terminal lA/intermediate bounds、KFSB、12-path atomic commit与queue/post语义。

因此S4的正确路径冻结为：

```text
αβ-CROWN host solver
  → RVIR update_bounds_core exact-call boundary（已有）
  → production pre-state / topology / policy admission（已有）
  → all-mutable-state compiled CROWN evaluation（S4-0—S4-1新增）
  → existing host-owned production 10/9 mutation policy（S4-2接入）
  → terminal lower/lA/intermediate handoff（S4-3）
  → existing KFSB + atomic commit + queue/post（复用）
```

不新造solver execution IR。新增对象只能是typed runtime binding/evaluator receipt；静态图、合法性、计划与
TIR继续复用现有Bound/Plan/Task/Schedule及R3/S2资产。

本稿在S3独立外审批准前`execution-authority=false/code-change-open=false`。当前只允许审阅和修订合同，
不允许实现、计时或升级claim。

## 1. 已独立核对的production事实

### 1.1 exact-call边界

当前真实same-solver路径由`scripts/run_rvir_v4_live_return_capture.py::_LiveExecutor.instrument()`恰一次替换
`activation_split.stage_solve.update_bounds_core`。existing owner链为：

1. `_build_core_pre_snapshot`捕获live state；
2. `initialize_rvir_v4_native_pre_state`按六条ReLU topology建立native state；
3. `execute_rvir_v4_native_optimizer_trace`执行10 evaluation/9 update；
4. `export_rvir_v4_native_backward`生成terminal lower/lA/intermediates；
5. `evaluate_rvir_v4_native_kfsb`生成3组候选并选择branch；
6. device/host atomic transaction提交12条live path；
7. 原solver继续termination、queue与post。

这条边界已经证明provider callback可为0、branch/commit可闭合；S4不重写它，只替换第3步中的CROWN
evaluation primitive，并在第4步消费最后一次evaluation的terminal handoff。

### 1.2 mutable-state覆盖差距

冻结production optimizer trace的每个step包含：

| mutable path | shape | 元素数 | S3是否拥有 |
|---|---:|---:|---|
| `alpha/%2F45/%2F49` | `[2,1,6,178]` | 2136 | 否，S3中为冻结输入 |
| `alpha/%2F48/%2F49` | `[2,1,6,27]` | 324 | 否 |
| `alpha/%2Finput-12/%2F49` | `[2,1,6,132]` | 1584 | 否 |
| `alpha/%2Finput-16/%2F49` | `[2,1,6,121]` | 1452 | 否 |
| `alpha/%2Finput-24/%2F49` | `[2,1,6,86]` | 1032 | 是，P-anchor |
| `alpha/%2Finput-4/%2F49` | `[2,1,6,164]` | 1968 | 否 |
| `beta/%2Finput-28/0/value` | `[6,1]` | 6 | 否，唯一active β |

production α合计8,496元素，S3动态α只覆盖1,032元素，即`12.1468926554%`。这只是state element
coverage，**不是时间share或Amdahl数字**。S3绑定的P β为`beta/%2Finput-20/0/value:[6,0]`，对active β
覆盖为0/6。

### 1.3 whole-core输出差距

S3只返回terminal lower、terminal P α和局部receipt。production whole-core还必须产生：

- 6条最终α与6条β的mutation state；
- terminal lower `[6,1]`；
- 6条lA与6组intermediate lower/upper；
- 3组KFSB candidate与最终6-domain decision；
- 12条device live target的atomic commit/rollback receipt；
- host history/depth/threshold packet与queue/post结果。

任一缺失都禁止exact-call admission。

## 2. 关键设计决定

### 2.1 接入点是evaluation provider，不是whole optimizer wrapper

S3的`execute_asplos27_s3_optimizer_v1`是资格实验：它只对一个α建立简化host Adam。S4 production接入不复用
该简化loop作为owner，而是给已有`execute_rvir_v4_native_optimizer_trace`增加sealed typed evaluator：

```text
PreparedProductionCrownEvaluationV1.evaluate(
    evaluation_ordinal,
    alpha_by_semantic_path,
    beta_by_semantic_path,
) -> ProductionCrownEvaluationResultV1

ProductionCrownEvaluationResultV1:
    lower
    dalpha_by_semantic_path      # 六条，compressed/native shape exact
    dbeta_by_semantic_path       # 六条，含唯一[6,1] active β
    terminal_handoff_or_none     # 只允许ordinal 9拥有
    execution_receipt
```

existing host loop继续唯一拥有两组Adam param group(`lrα=0.01/lrβ=0.05`)、decay=`0.98`、clamp、10/9
ordinal、stop/patience/pruning/keep-best policy identity。candidate evaluator只做纯evaluation/VJP，不得修改
optimizer state、policy或live solver object。

接口必须是精确sealed类型或protocol+receipt双重校验，不接受任意callback；candidate模式不得回退到native
`run_crown_ibp_mlp_from_forward_trace`、`autograd.backward`或provider `compute_bounds/update_bounds`。

### 2.2 全state representation，不复制六个P特判

R3 plan已经含六个`relu_layouts`和全部tensor specs；S4应把“动态参数集合”从单一
`p_alpha_input_ordinal`推广为由production snapshot/topology推导的有序mutable binding：

- key=`semantic_path`；
- shape/dtype/device/feature-index/beta-location/sign全部来自existing typed snapshot/layout；
- α与β分别保持compressed representation；
- empty β是合法零宽tensor，active β必须保持location/sign owner；
- plan/schema中禁止ResNet2B名称、固定node id、固定shape或固定“6”作为通用机制条件；
- formal fixture可以冻结上述六条路径，但只能作为实例证据。

不得创建第二套VerificationGraph或optimizer IR。静态候选继续lower到现有Relax/TIR，runtime binding只负责
把live state映射到已编译参数槽。

### 2.3 forward/VJP必须一次闭合全部mutable state

S4 candidate每个ordinal只允许一个logical CROWN evaluation。它必须同时产生相同lower与全部六α/六β梯度，
禁止：

- 先跑native full evaluation再只替换P梯度；
- candidate和native各跑一次后拼接结果；
- 对五条α或active β使用autograd/native shadow；
- 把未覆盖梯度填零；
- 逐site返回Python对象导致framework crossing重新进入热路径。

实现可复用B4-B2 dense/sparse Linear/Conv TIR与S2 canonical Relax图，但所有site必须通过同一个compiled
evaluation owner和persistent arena。若某site尚无compiled VJP，S4-1直接NO-GO并回到算子/region coverage，
不能静默fallback。

### 2.4 terminal handoff禁止第11次CROWN

ordinal 9必须选择性导出terminal lower、lA及需要的intermediate bounds，复用已有B4-A
`terminal lower/adjoint handoff`与native export assembly合同。S4 whole-core不允许在optimizer结束后为
`export_rvir_v4_native_backward`再执行一次完整CROWN；若terminal handoff不完整，S4-3不准入。

KFSB、atomic commit和queue/post继续由已有RVIR owner执行，不纳入compiled tensor region，也不因candidate
路径改变顺序。

### 2.5 all-state VJP物理实现冻结

详细coverage与算法见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_ALL_STATE_VJP_FEASIBILITY_2026_08_28.md`。审计确认现有整图forward已经消费
六α和active β，缺口是custom backward只导出P gradient，不是其余五site完全没有编译。

S4-1实现固定复用当前两个coefficient arena，按“完整sign pass → 六site effective-value pass → 第二次
coefficient pass逐site即时压缩gradient”执行。不得保存跨层float32 dense A，也不得把六个B4-B2单sitewrapper
串成production路径。B4-B2只作为数学oracle/codegen资产；D1C/D2B residual stage scratch用于暴露site25/site19
的内部incoming coefficient。

S4-1内部顺序固定为1A all-state ABI、1B六site effective values、1C六dα/active dβ emitters、1D single-evaluation
five-fresh closure。四步完成前S4-2继续关闭。

## 3. 分阶段门禁

### S4-0：production signature admission（无GPU执行）

交付：typed mutable binding与coverage receipt，不实现TIR、不计时。

必须机械证明：

- snapshot mutable α key与compiled gradient key完全相等；
- snapshot mutable β key与compiled gradient key完全相等；
- feature index、shape、dtype、device、location/sign与split/history lineage一致；
- P-only计划在当前fixture上明确拒绝，reason=`MUTABLE_STATE_COVERAGE_INCOMPLETE`；
- active β缺失明确拒绝，reason=`ACTIVE_BETA_COVERAGE_INCOMPLETE`；
- 不接受多余、重复、乱序或alias冲突binding；
- performance/timing/same-solver flag全部false。

GO：六α+六β全覆盖，active β=`1/1`，且schema无模型特判。否则S4-1关闭。

### S4-1：all-state single-evaluation compiled correctness

交付：一个prepared evaluator在ordinal 0输入上返回lower、六dα、六dβ和receipt。

门禁：

- 双独立oracle：production captured step与existing provider-independent native CROWN；
- lower `atol=rtol=2e-4`；全部gradient `atol=rtol=2e-5`；sign exact；
- key/shape/dtype/device exact；empty β保持empty，active β 6元素全部比较；
- logical evaluation=`1`；provider/fallback/native-shadow/eager=`0`；
- per-site Python dispatch、warm DLPack、dynamic output allocation=`0`；
- saved/persistent dense A=`0`；terminal handoff在非terminal ordinal必须不存在；
- six-site effective primal arena、sign bitmap与compressed output必须分项披露bytes，禁止通过重命名隐藏内存；
- coefficient arena恰为existing 2个；site25/site19从staged residual scratch即时导出，不得另跑native Conv；
- five fresh correctness，任一site或元素不等价即NO-GO。

S4-1不计时；通过只开放S4-2。

### S4-2：production host-policy 10/9 trajectory

把S4-1 evaluator注入existing production optimizer loop，不使用S3简化loop。逐ordinal比较：

- lower；
- 六α、六β before/after；
- 两param-group Adam step/m/v；
- α/β learning rate与scheduler；
- update/prune/keep-best/restore/stop predicate；
- terminal state与mutation policy hash。

门禁沿用production parity：lower/state max diff均`<=2e-4`，gradient内部门禁`<=2e-5`，sign exact；
evaluation/update=`10/9`，candidate evaluation=`10`，provider/native evaluation=`0`。任何policy shortcut、固定
10/9无条件展开或读取expected trace都拒绝。

### S4-3：whole-core exact-call correctness

在同一`ABCrownSolver.verify`内比较：

```text
R：RVIR whole-call provider-independent reference
C：RVIR + S4 compiled evaluation + existing host policy/export/KFSB/commit
```

B0 original provider只作额外semantic control，不作为S4实现依赖。五fresh顺序预注册后比较：

- solver status/success、visited domains；
- core lower/upper、6 lA、6 intermediate bounds；
- terminal六α/六β；
- KFSB三候选、child lower与最终decision；
- history/depth/threshold、n_splits/n_verified；
- 12-path atomic commit、rollback与queue/post；
- exact-call=`1`，provider compute/update、fallback、native shadow=`0`；
- terminal duplicate CROWN=`0`。

所有离散字段exact；有限浮点沿用已冻结容差。通过后状态只能是
`VALIDATED-S4-SAME-SOLVER-CORRECTNESS`，仍不形成性能claim。

### S4-4：artifact/replay/tamper closure

- raw-first，source commit、TVM submodule、三个外部仓库、model/property与全部code blob绑定；
- five fresh R/C，必要时加B0 control；部分结果不得resume成formal；
- raw逐step保留，不只存summary digest；
- replay从raw重算coverage、trajectory、whole-core、receipt与verdict；
- 至少16类fully outer-resigned tamper，覆盖missing/extra/swapped state、active β、gradient、policy、ordinal、
  terminal handoff、KFSB、commit、provider/fallback counter、source和claim flag；
- targeted/full/static/DocOps全过后，才允许另写S4-P性能预注册。

## 4. Timing与性能门禁仍关闭

S4 correctness完成前不得复用S3 `3.2439x`作为same-solver预测，因为candidate从单P扩到六α+active β后，
kernel数、workspace、VM/launch和region speedup都会变化。

S4-P只能在正确性artifact关闭后预注册，并重新实测：

- exact-call region真实query share `s`；
- all-state compiled/reference region speedup `r`；
- adapter/export/KFSB/commit integration overhead `h`；
- `T = 1 / ((1-s)+s/r) / h`的query feasibility；
- 同solver B0/R/C三方、fresh进程、GPU状态与最差pair。

若新测`required_r >10x`或candidate必须重复native full evaluation，性能路线直接STOP；不得通过排除host policy、
terminal export或commit来制造headline。

## 5. Fail-closed拒绝清单

至少覆盖以下稳定原因：

1. `S3_EXTERNAL_APPROVAL_MISSING`；
2. `MUTABLE_STATE_COVERAGE_INCOMPLETE`；
3. `ACTIVE_BETA_COVERAGE_INCOMPLETE`；
4. `MUTABLE_STATE_KEY_EXTRA`；
5. `MUTABLE_STATE_KEY_DUPLICATE`；
6. `STATE_SHAPE_MISMATCH`；
7. `STATE_DTYPE_MISMATCH`；
8. `STATE_DEVICE_MISMATCH`；
9. `ALPHA_FEATURE_INDEX_MISMATCH`；
10. `BETA_LOCATION_SIGN_MISMATCH`；
11. `STATE_ALIAS_CONFLICT`；
12. `POLICY_HASH_MISMATCH`；
13. `EVALUATION_ORDINAL_MISMATCH`；
14. `NONFINITE_LOWER_OR_GRADIENT`；
15. `TERMINAL_HANDOFF_EARLY_OR_MISSING`；
16. `TERMINAL_DUPLICATE_CROWN`；
17. `PROVIDER_CALLBACK_OBSERVED`；
18. `FALLBACK_OR_NATIVE_SHADOW_OBSERVED`；
19. `KFSB_OR_BRANCH_DRIFT`；
20. `ATOMIC_COMMIT_OR_ROLLBACK_DRIFT`；
21. `CLAIM_FLAG_TRUE_BEFORE_FORMAL`。

拒绝必须发生在对应evaluation launch、live mutation或commit之前；发生异常时existing live state必须保持原样。

## 6. 建议提交序列

S3外审批准后才允许：

1. `docs: preregister ASPLOS27 S4 production exact-call`（批准后将本稿转execution-authority）；
2. `feat(runtime): add production mutable-state binding coverage`（S4-0）；
3. `test(runtime): close S4-0 coverage and negative gates`；
4. `feat(compiler): compile all-state CROWN evaluation and VJP`（S4-1）；
5. `test(runtime): close all-state single-evaluation correctness`；
6. `feat(runtime): inject sealed evaluator into production optimizer policy`（S4-2）；
7. `test(runtime): close six-alpha active-beta 10/9 trajectory`；
8. `feat(adapter): route RVIR exact call through compiled evaluation`（S4-3）；
9. `test(adapter): close whole-core five-fresh correctness`；
10. `artifact: close S4 replay and tamper`（S4-4）；
11. `docs: preregister S4-P same-solver timing`。

每一级独立提交、独立门禁；不得把coverage、compiler、whole-core和timing揉进一个不可归因提交。

## 7. 当前停止点

DocOps task=`asplos27-s3-optimizer-runtime-20260828`当前为`ready_for_audit`。外审批准前，本稿只是一份
基于仓库事实的预注册草案；S4代码、GPU correctness与timing均保持关闭。外审若指出S3 blocker/major，先处理
finding并重开S3 round，不得绕到S4。
