---
status: validated-reduced-b3-external-audit-pending
updated: 2026-08-14T21:52:00+08:00
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

测量卫生进一步冻结为：control只启用原有计时观察器，不保留详细physical counter；B2/B3 profile使用
不保留event journal的轻量直接counter。B3每个control/profile worker仍必须提供prepared template、
PlanInstance、terminal Schedule、assembly、atomic commit与post-query audit的直接activation receipt。
profile/control query perturbation必须`<=1.05`；超过即measurement fail closed，不得用profile数字替代
control headline。

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
2. [x] B3-A：PreparedCoreTemplate、CorePlanInstance与cache/reject tests；状态=
   `VALIDATED-B3-A-COUNTERS`，不形成timing/speedup；
3. [x] B3-B：terminal-only optimizer Schedule和forward-trace handoff；状态=
   `VALIDATED-B3-B-COUNTERS`，不形成timing/speedup；
4. [x] B3-C：device-resident AtomicCommitPlan、rollback与audit digest分层；状态=
   `VALIDATED-B3-C-COUNTERS`，不形成timing/speedup；
5. [x] 五fresh B2/B3 correctness：10/10独立GPU worker、5/5 direct semantic pair、root replay与
   7/7 tamper通过，状态=`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`；
6. [x] 36-process B0/B2/B3正式artifact、replay与tamper；36/36、root replay与10/10 tamper通过，
   B2/B3 core=`1.071617x`、B0/B3 query=`0.910001x`，状态=`VALIDATED-REDUCED-B3`；
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

本节的`IMPLEMENTED-PENDING-FRESH-GPU`是正式运行前历史状态，已被下方closure取代。

## 16. B3-A Formal Closure（2026-08-14）

- source=`c7851c8bae1bc943aa9e3d458e5105deafc553f1`；
- artifact=`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1/`；
- manifest hash=`205978cb69238598dfcb860922e3202677d5b1775f0bd6062218f0369e982c95`；
- report hash=`89a3584dddb47d2a835bca689bdb0ba6b936d26fa5aff20a968c2323dc6cd05b`；
- 5157条event独立重放：template compile/hit=`1/1`、module move=`0`、scope=`1`；optimizer=`10/9`、
  full snapshots=`10`、forward=`5`、KFSB=`3/3`、candidate D2H=`12`、committed/backup/copy=
  `12/12/12`，provider/fallback全零；
- 六个冻结B2 control语义一致，artifact replay通过；六类outer-resigned counter/journal/semantic/
  provider/code攻击6/6拒绝，tamper report hash=
  `92a1900a8cdba5f42833dbd02efd2aa510d6027d58d43a1152d9a20f280d9997`；
- targeted=`34 passed`；full=`1257 passed, 3 skipped, 6 warnings`；Black clean，mypy touched clean，
  Pylint=`10.00/10`；
- 状态=`VALIDATED-B3-A-COUNTERS`，`diagnostic_timing_claimed=false`、`performance_claimed=false`。

B3-A证明PreparedCoreTemplate/CorePlanInstance在一个fresh真实solver call中激活且保持冻结语义；它没有
证明延迟改善，也不满足完整B3正式计时前的5 fresh B2/B3 pair门禁。下一动作只允许B3-B terminal-only
optimizer Schedule与forward-trace handoff；B3-C和B4—B7继续关闭。

## 17. B3-B Implementation Candidate（2026-08-14）

- 新增`NativeTerminalOptimizerScheduleV1`：10个typed evaluation action、前9个update action、双学习率
  与0.98 decay全部进入稳定Schedule hash；
- 新增production terminal executor：执行10/9语义但不构造`NativeProductionOptimizerStepV4`，只保留
  terminal lower、terminal α/β和一次父forward trace；旧formal trace函数不变；
- backward新增exact typed forward-trace handoff；graph/scope/split、完整value inventory和tensor placement
  不一致时fail closed，否则跳过父forward rebuild；
- `_LiveExecutor`仅在显式prepared core + terminal schedule组合下启用B3-B；B2/B3-A默认路径不变；
- counter schema预注册B3-B相对B3-A只允许full snapshots `10→0`、forward builds `5→4`；template、
  scope、optimizer 10/9、KFSB、D2H和commit全部保持；
- CPU冻结case确认terminal lower/α/β与formal trace最后一步逐元素相同，forward handoff前后backward
  lower/lA/intermediate逐元素相同；targeted=`42 passed`，mypy touched clean，Pylint=`10.00/10`。

本节的`IMPLEMENTED-PENDING-FRESH-GPU`是正式运行前历史状态，已被下方closure取代。

## 18. B3-B Formal Closure（2026-08-14）

- source=`42df2dcae2d5c5a10f27ab707d8d7aff7686d15e`；
- artifact=`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1/`；
- manifest hash=`2960c85c9b6dfe1382bef39804a9a88b618b438b9b2cb55d629aa24a99c18644`；
- report hash=`f7c24e9080a51fba990bf67502ee91519b8d67047a37d29a64654cbe4ea77061`；
- 5157条event独立重放：full optimizer step snapshots=`0`、forward builds=`4`；template compile/hit=
  `1/1`、module move=`0`、scope=`1`、optimizer=`10/9`、KFSB=`3/3`、D2H/commit/backup/copy=
  `12/12/12/12`，provider/fallback全零；
- 六个冻结B2 control语义一致，artifact replay通过；六类outer-resigned攻击6/6拒绝，tamper report
  hash=`6c1dde930b250d62a9eb00026729888363ea02bae42eb3331daa384ece73dbcf`；
- targeted=`45 passed`；full=`1265 passed, 3 skipped, 6 warnings`；Black clean，mypy touched clean，
  Pylint=`10.00/10`；
- 状态=`VALIDATED-B3-B-COUNTERS`，`diagnostic_timing_claimed=false`、`performance_claimed=false`。

B3-B证明terminal Schedule和forward-trace handoff在一个fresh真实solver call中激活并保持冻结语义；它
没有证明延迟改善，也不满足完整B3计时前的5 fresh pair。下一动作只允许B3-C device-resident
AtomicCommitPlan；B4—B7继续关闭。

## 19. B3-C Implementation Candidate（2026-08-14）

- 新增`DeviceAtomicCommitPlanV1`，在prepared template阶段冻结12条path的role、shape、dtype、CUDA
  device、alias equivalence与rollback ordinal；
- 新增动态transaction，绑定core instance/pre snapshot、live tensor version、host version和12个GPU
  candidate；α sparse projection与β location projection不再生成CPU candidate snapshot；
- 12个live target先做device backup，再同设备直接copy；tensor/host任一失败均恢复全部tensor与host
  pre-image；五个`(6, 0)`beta用empty-object identity验证alias；
- provider assembly的headline metadata不做GPU content SHA；query计时结束并同步后才生成artifact audit
  digest，并绑定plan/transaction/commit receipt；
- counter预注册B3-C相对B3-B只允许timed candidate D2H `12→0`，candidate/commit/backup/copy继续
  `12/12/12/12`，其他B3-B固定结构不变；
- CUDA事务与assembly测试=`10 passed`，相关定向回归=`50 passed`；Black、mypy clean，Pylint
  `10.00/10`。

当前状态仅为`IMPLEMENTED-PENDING-FRESH-GPU-ARTIFACT`。尚无fresh真实worker artifact、全量回归、
5 fresh pair或任何timing/speedup claim；B4—B7继续关闭。

上段为实现候选历史状态，已被下方正式closure取代。

## 20. B3-C Formal Closure（2026-08-14）

- source=`72bec5ee1bdabfdefbf51201ac49395489eeef65`；
- artifact=`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1/`；
- manifest/report internal hash分别为`091f6ac8…e1c2`/`72812a35…cb44`；
- 1484条event独立重放：candidate/commit/backup/copy=`12/12/12/12`、timed candidate D2H=`0`；
  template compile/hit=`1/1`、module move=`0`、scope=`1`、optimizer=`10/9`、snapshots=`0`、
  forward=`4`、KFSB=`3/3`，provider/fallback全零；
- headline assembly content digest=`0`；24次GPU content hash全部在query计时结束和CUDA同步后的audit，
  audit/commit hash交叉绑定12条path；
- 六个冻结B2 control语义一致，artifact replay与6/6 outer-resigned tamper通过；
- targeted=`54 passed`；full=`1279 passed, 3 skipped, 6 warnings`；Black/mypy clean，Pylint
  `10.00/10`；
- 状态=`VALIDATED-B3-C-COUNTERS`，`diagnostic_timing_claimed=false`、`performance_claimed=false`。

B3-C只证明一个fresh真实call的结构和正确性。下一动作是至少5组fresh B2/B3 correctness pairs；未
5/5通过前不得启动36-process正式计时，B4—B7继续关闭。

上段是五组correctness门禁关闭前的历史下一动作，已被下方正式closure取代。

## 21. B3 Five-Fresh Correctness Formal Closure（2026-08-14）

- source=`75dfd8103e8e3dfe824a63e15c2222f8742e28c1`；
- artifact=`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1/`；
- 五组固定交替顺序下10/10独立fresh GPU worker全部完成，B2/B3-C各5个；
- 5/5 direct semantic pair无failure，source/protocol/runtime/GPU identity一致，environment、provider/
  fallback、B2/B3-C counter与B3-C post-query audit门禁全部通过；
- B2每次4625 events，B3-C每次1484 events；后者持续满足template hit=`1`、module move=`0`、scope=
  `1`、optimizer=`10/9`、snapshots=`0`、forward=`4`、KFSB=`3/3`、candidate D2H=`0`；
- root internal manifest/report hash=`457ab1adc8…1573`/`0d649200f4…2827`；独立replay通过；
- 七类outer-resigned report/protocol/nested counter/semantic/audit/swap/delete攻击7/7拒绝；
- targeted=`56 passed`；full=`1289 passed, 3 skipped, 6 warnings`；Pylint=`10.00/10`；
- 状态=`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`，只将`b3_timing_admitted`推进为`true`；五组artifact本身
  仍为`timing_admitted=false`、`performance_claimed=false`。

下一唯一动作是实现并静态验证本计划第8节冻结的B0/B2/B3六全排列、36-process正式计时artifact，然后
从clean source运行、replay和tamper。B4—B7继续关闭。

上段“实现runner”动作现已由下方实现候选取代；尚未取代的是clean-source正式运行、replay与tamper。

## 22. B3 36-Process Formal Timing Runner Candidate（2026-08-14）

- 新增typed B0/B2/B3 run、activation receipt、36-run sequence、ratio与decision重算合同；
- 新增raw-first/resumable generator、每worker envelope/log、formal manifest、独立replay与十类
  outer-resigned tamper probe；
- control不启用详细counter，B2/B3 profile只启用不保留event journal的轻量counter；每个B3 worker仍以
  direct receipts证明template/instance/schedule/assembly/commit/audit真实激活；
- formal preflight、source/code digest、five-fresh admission、benchmark/external repo/runtime/GPU identity、
  固定36-run顺序均进入protocol；机器绝对路径从artifact projection与日志中清除；
- 四个独立GPU worker冒烟覆盖B0 control、B2 profile、B3 control/profile，结构门禁通过；冒烟时间不是
  performance sample；
- targeted=`108 passed`，full=`1308 passed, 3 skipped, 6 warnings`，Black/mypy clean，Pylint=
  `10.00/10`；
- 状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`，没有正式artifact、replay/tamper结果或speedup
  claim。

下一唯一动作是冻结clean source，从position 0执行完整36-process run，随后在同一artifact上执行root
replay和十类tamper probe。B4—B7继续关闭。

上段正式运行指令现已由下方关闭结果取代；当前下一唯一动作是external audit，B4—B7在外审前继续关闭。

## 23. B3 36-Process Formal Timing Closure（2026-08-14）

- source=`36e9069ca4f21183c9b36d74024de0ca8b20f59c`；
- artifact=`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`；
- 六个B0/B2/B3全排列共36/36 fresh worker，correctness、environment、measurement、activation与18/18
  profile closure全部通过；
- B2/B3 core/query geomean=`1.071617x/1.006623x`，六个core pair最差仍`1.063588x`；
- B0/B3 query/core=`0.910001x/0.535965x`，故仍未回到B0 parity；显存reserved不变；
- root replay独立重建相同summary；十类outer-resigned attack=`10/10 rejected`；
- frozen artifact=`6 passed`，targeted=`114 passed`，full=`1314 passed, 3 skipped, 6 warnings`；
- 状态=`VALIDATED-REDUCED-B3`，不是full B3或全栈speedup；`performance_claimed=false`保持。

下一唯一动作是按
`gemini_doc/fsg4_b3_formal_timing_external_audit_handoff_2026_08_14.md`进行外部审计。外审通过后只开放
B4 cumulative candidate；B5—B7与最终system gate继续关闭。
