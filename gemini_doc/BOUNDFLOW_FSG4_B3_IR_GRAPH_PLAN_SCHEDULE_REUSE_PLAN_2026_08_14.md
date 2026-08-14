---
status: active
updated: 2026-08-14T10:05:19Z
type: plan
topic: boundflow
slug: fsg4-b3-ir-graph-plan-schedule-reuse
stage: s01
---

# FSG4 B3 IR Graph Plan Schedule Reuse Preregistration

## 1. Goal

FSG3已经建立同一official αβ-CROWN solver内的B0/B1/B2正式分母。当前B2 whole-call reference
replacement正确、无provider callback/fallback，但query/core的B0/candidate geomean仅为
`0.908400x/0.516767x`。FSG4/B3的目标是：

> 在完全冻结solver算法、10 evaluation/9 update、branch、queue、post和数值容差的前提下，让
> Bound/Graph IR、Prepared PlanTemplate/PlanInstance与Schedule真实驱动执行，消除B2为正确性审计
> 保留、但production live path不需要重复执行的编译、全量trace物化、forward重建、CPU digest与状态
> 往返。

B3不是TIR/kernel fusion、CUDA Graph/JIT、multi-stream runtime或arena/buffer reuse阶段；这些分别属于
B4—B7。B3只能优化执行图、静态/动态计划分离、Schedule中间值复用与状态commit计划。

## 2. Frozen Baseline

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- FSG3 closure commit：`5953c50`；PR publication commit：`57cbd40`；
- B2 formal source：`a4ee2910f4039981338fb6d8688ac4af18508b73`；
- artifact：`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/`；
- summary hash：`df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e`；
- workload：VNN-COMP 2021 `cifar10_resnet` ResNet2B property 0；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- solver protocol：max_iterations=1、batch=64、alpha steps=5、beta steps=10、seed=100。

所有B3比较必须从同一B0和冻结B2重新运行，不能把FSG3旧wall与新candidate跨时段直接相除。

## 3. B2 Evidence and Code Diagnosis

FSG3 B2 profile core的geometric-mean wall归因：

| 区域 | wall | core share |
|---|---:|---:|
| optimizer | `121.565 ms` | `43.999%` |
| atomic commit | `68.199 ms` | `24.684%` |
| KFSB | `46.097 ms` | `16.684%` |
| typed pre-state | `29.617 ms` | `10.720%` |
| backward export | `10.160 ms` | `3.677%` |

源码独立审计确认：

1. `scripts/run_rvir_v4_live_return_capture.py::_LiveExecutor.execute`虽然接收precompiled module，仍在
   每次core中移动全部module bindings、重建scope与dynamic state；
2. `execute_rvir_v4_native_optimizer_trace`为formal parity保存10份lower、α和β完整clone，但live路径只
   读取最后一step；
3. optimizer先构造一次forward IBP trace，`export_rvir_v4_native_backward`在split不变时又构造一次；
4. KFSB三个candidate各自执行child forward/CROWN，这是不同split语义，B3不得假装可直接删除；
5. atomic stage把12个candidate tensor逐个搬到CPU、计算content SHA，再经assembly/stage/commit多次
   `validate()`和`stable_hash()`，最后又复制回GPU live target；
6. 当前形式把formal artifact审计结构直接放进timed production core，导致证据生成成本与必要执行成本
   混合。

一次非claim cProfile诊断在进入B2 core后fail closed：provider callback guard读取到
`sys.getprofile()`返回的`cProfile.Profile`对象并把它当普通callback调用，抛出`TypeError`。该次没有
worker result、没有计时或调用次数结论。B3-0改用显式counter，不修改门禁来迁就profiling。

## 4. Configuration Chain

| ID | 启用内容 | 禁止混入 |
|---|---|---|
| B2 | 冻结whole-call reference replacement | 任何B3优化 |
| B3-A | PreparedCoreTemplate + dynamic PlanInstance | terminal-only、commit改写 |
| B3-B | B3-A + terminal-only optimizer Schedule + terminal forward reuse | atomic改写、TIR |
| B3-C | B3-B + device-resident AtomicCommitPlan | TIR/JIT/runtime/arena |
| B3 | B3-C正式累计候选 | B4—B7全部变量 |

每个子阶段单独提交、单独correctness gate；后一步不得掩盖前一步失败。最终artifact同时报告B2→B3-A、
B3-A→B3-B、B3-B→B3-C和B0→B3。

## 5. Typed IR and Ownership Contracts

### 5.1 PreparedCoreTemplate

新增不可变template，至少绑定：

- program/module canonical hash与input value name；
- six-ReLU topology hash、operator order与static parameter binding；
- device/dtype、objective shape、batch shape与optimizer policy structure；
- mutable α/β 12-path inventory、read-only intermediate inventory与host packet keys；
- optimizer→backward→KFSB→commit的typed dataflow edges；
- backend identity=`torch-eager-reference`和feature activation identity。

template必须在query前构建；module bindings只迁移一次。graph、topology、device、dtype、policy结构或
mutable path inventory变化必须cache miss或fail closed，不能复用stale template。

### 5.2 CorePlanInstance

每次真实call只绑定动态对象：input lower/upper、linear spec、threshold、external intermediates、split/
history、α/β live tensors与state version。instance不得复制static parameters，也不得重新lower graph。

PlanInstance必须显式区分：

- provider-owned live view；
- private optimizer working state；
- shared forward trace；
- terminal export；
- staged device commit；
- post-commit version receipt。

### 5.3 TerminalOptimizerSchedule

production schedule只返回terminal state、terminal lower和可安全复用的forward trace。formal逐step parity
仍由旧trace executor保留在测试/审计路径；production timed path不得物化10份完整step snapshots。

必须保持evaluation/update=`10/9`、双学习率、decay、clamp、gradient与early-stop语义exact。split在本
固定production call内不变，故optimizer forward trace可以供terminal backward复用；若未来dynamic
split改变，reuse必须fail closed。

### 5.4 AtomicCommitPlan

commit plan在template阶段冻结12个目标path的shape/dtype/device/alias与rollback顺序。每个call：

- candidate始终留在GPU；
- commit前做finite、shape/dtype/device、version和完整inventory检查；
- 12个live target先生成device backup，再从private candidate直接copy；
- 任一路径或host packet失败时恢复全部tensor和host pre-image；
- timed core内不执行GPU→CPU content SHA或把candidate转成CPU snapshot；
- artifact audit digest在query同步完成后、headline timing之外生成，并与commit version/receipt交叉绑定。

这不是取消fail-closed，而是把静态schema验证、动态transaction验证和artifact digest放到各自正确阶段。

## 6. Physical Activation Ledger

B3正式worker必须保存以下counter，不能只记录对象存在：

| Counter | B2预期 | B3目标 |
|---|---:|---:|
| template compile | cold `1` | cold `1` |
| template hit in core | `0` | `1` |
| module binding move in core | `1` | `0` |
| scope construction | `>=2` | `1` |
| optimizer evaluations / updates | `10/9` | `10/9` |
| full optimizer step snapshots | `10` | `0` |
| forward-trace builds | `5` | `4` |
| KFSB candidate/child batches | `3/3` | `3/3` |
| timed candidate D2H copies | `12` | `0` |
| committed mutable paths | `12` | `12` |
| device rollback backups | implementation-dependent | `12` |
| provider core/compute/update | `0/0/0` | `0/0/0` |
| fallback dispatch | `0` | `0` |

B3-0必须先用显式instrumentation独立重算B2实际counter。若B2预期不成立，先修正文档/schema，不得静默
调整B3目标。

## 7. Correctness Gates

每个B3子阶段都必须：

- 对同一pre-state的lower、upper sentinel、α、β、intermediate、lA、candidate child lower、final
  decision、branch packet、queue add、termination与solver status逐项比较；
- float使用冻结`atol=rtol=2e-4`且sign exact；离散字段exact；
- provider/fallback=`0/0/0/0`；
- 10/9 optimizer control exact，KFSB 3 candidates/72 child lower不减少；
- stale template、wrong topology、wrong device/dtype、wrong policy、wrong mutable inventory、NaN、
  mid-commit failure与host packet failure全部fail closed并rollback；
- 不允许以FSG3 formal truth作为candidate运行时输入。

至少完成5个fresh B2/B3 correctness pairs后，才准入正式计时。

## 8. Measurement Protocol

正式计时使用六个`B0/B2/B3`全排列block；每个配置在每个block各一个control和一个profile，共36个
fresh GPU进程。环境、温度、排他、profile closure、raw-first replay与outer-resigned tamper沿用FSG3
schema v4，不得读取结果后修改。

headline ratio：

- cumulative：`B0/B3` query、core、GPU、memory；
- incremental：`B2/B3`、`B2/B3-A`、`B3-A/B3-B`、`B3-B/B3-C`；
- control only用于latency，profile only用于attribution；
- 报告raw、median、range、MAD、geomean和每pair退化；
- cold compile、process-hit execute与post-audit digest分开。

## 9. Go / Reduced / No-Go

所有correctness、activation、environment、closure与tamper门禁是硬条件。性能分类：

- `VALIDATED-B3`：`B2/B3 core geomean >=1.15x`，且`B0/B3 query geomean >=1.00x`，任一pair相对
  B2不得退化超过5%；
- `VALIDATED-REDUCED-B3`：`B2/B3 core geomean >=1.05x`、B3 query相对B2 geomean不退化，且全部
  correctness成立；B3可作为后续B4候选，但不能声称已回到B0 parity；
- `VALIDATED-NO-GO-B3`：correctness成立但core geomean `<1.05x`或query退化；默认关闭B3 feature，
  B4从B2或最佳已准入子配置继续；
- `BLOCKED-B3-CORRECTNESS`：任一语义、ownership、rollback、provider/fallback或replay失败；停止B4。

最终`1.20x queue / 1.15x complete-query`仍只施加到B7 vs B0，本阶段不得提前宣称。

## 10. Tasks

1. [x] B3-0：显式call/copy/validation/hash counter diagnostic，状态=`VALIDATED-B2-COUNTERS`，不形成speedup；
2. [ ] B3-A：PreparedCoreTemplate、CorePlanInstance与cache/reject tests；第一版实现候选已落地，等待fresh
   GPU counter/correctness artifact后才能勾选关闭；
3. [ ] B3-B：terminal-only optimizer Schedule和forward-trace handoff；
4. [ ] B3-C：device-resident AtomicCommitPlan、rollback与audit digest分层；
5. [ ] 五fresh B2/B3 correctness；
6. [ ] 36-process B0/B2/B3正式artifact、replay与tamper；
7. [ ] external audit与FSG4/B3 closure。

## 11. Validation

- unit：template/instance/hash/cache、stale reject、terminal schedule、trace reuse、device commit/rollback；
- targeted：B2/B3 counter、5 fresh correctness、provider/fallback zero；
- static：Black、mypy、Pylint；
- regression：`pytest tests`；
- DocOps：每次修改`dol ch add`，验证`dol va add`，handoff前`dol lint --soft`。

## 12. Rollback

- B3 feature默认关闭，B0与B2实现不改；
- template、terminal schedule与commit plan各自独立feature flag；
- 任一子阶段失败回到最近的已验证配置；
- B3不得修改FSG3 v5 artifact、summary/hash或历史结论。

## 13. Links

- changelog：
  `gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_CHANGELOG_2026_08_14.md`；
- FSG3 closure：`gemini_doc/change_2026-08-14_fsg3_same_solver_formal_baseline.md`；
- full-stack roadmap：
  `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`。

## 14. B3-0 Formal Closure（2026-08-14）

- source=`419536126504e2666a5db14681668b7d1add166a`；
- artifact=`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1/`；
- manifest hash=`ccf15ee17cb1ee74b95984a203cb4893e52d70becbc3ba2d3db70618490bb376`；
- report hash=`4304ffe87ce09c6e14ff633ae72f469b6b1fb7c60d297179e74176a3a41ad68e`；
- event journal=`4625`条，replay通过，FSG3 v5六个B2 control语义锚定6/6通过；
- 固定结构实测：template compile/hit=`1/0`，module move=`1`，scope=`2`，optimizer=`10/9`，
  full snapshots=`10`，forward traces=`5`，KFSB candidate/child=`3/3`，candidate D2H=`12`，
  committed/backup/copy=`12/12/12`，provider/fallback全零；
- 观察型成本：tensor content hash=`4417`，其中GPU tensor hash=`45`；typed validate=`84`；stable
  hash=`10`；
- 六类outer-resigned counter/journal/semantic/provider/code攻击6/6拒绝；tamper report hash=
  `f6392fa609c02d043b2397e36e54e52124630aa93fe51679892058efff644d1d`；
- targeted=`25 passed`，full=`1248 passed, 3 skipped`，mypy clean，Pylint 10.00/10；
- 状态=`VALIDATED-B2-COUNTERS`，`diagnostic_timing_claimed=false`、`performance_claimed=false`。

B3-0证明预注册的重复工作真实存在，并把B3-A/B/C的物理分母冻结；它不证明任何B3 speedup。下一动作仅
允许B3-A PreparedCoreTemplate/CorePlanInstance，B3-B/C与B4—B7继续关闭。

## 15. B3-A Implementation Candidate（2026-08-14）

- 新增`PreparedCoreTemplateV1`：冻结primal graph/参数内容、六层ReLU topology、device/dtype、输入与
  objective shape bucket、admitted policy contract、12条mutable copy-out path和实际binding placement；
- 新增动态`CorePlanInstanceV1`：每次重新绑定snapshot/mapping/input/objective/intermediate bounds/
  split/α/β/policy并仅调用一次`build_native_alpha_beta_scope`；
- 新增exact `PreparedCoreTemplateCache`，miss/compile与core hit分别可观测；错误topology、device、dtype、
  mutable inventory、module parameter drift及跨state receipt全部fail closed；
- `_LiveExecutor`仅在显式cache/hash pair下启用B3-A；默认B2仍使用原precompiled module、core内move和两次
  scope路径；prepared路径把binding move放到query/core外，并把typed plan receipt交给optimizer；
- counter schema预注册B3-A相对B2只允许三项变化：module move `1→0`、scope `2→1`、template hit
  `0→1`；optimizer `10/9`、forward `5`、KFSB `3/3`、D2H/commit `12`等全部保持；
- 定向验证=`31 passed`，mypy touched clean，Pylint=`10.00/10`。

当前状态仅为`IMPLEMENTED-PENDING-FRESH-GPU`。上述三项物理counter、真实语义等价及性能均未由fresh
artifact证明；B3-B/C和B4—B7继续关闭。
