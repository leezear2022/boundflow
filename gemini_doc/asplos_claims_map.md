# BoundFlow ASPLOS Claims Map

> **2026-08-23 FSG4/B4-B1 Round 1纠正**：外审状态=`REQUEST-CHANGES-B4-B1-R1-F1-F2`。
> 原v2数值重算仍成立，但receipt exact inventory/target binding与deterministic warn/debug restore
> 不满足接口门禁；工作树候选修复不升级claim。clean-source v3与Round 2批准前，B4-B1未外审
> 关闭，B4-B2/TIR/performance/memory/ASPLOS-ready继续关闭。

> **2026-08-18 FSG4/B4-B1内部关闭**：状态=
> `VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。支持两个冻结production
> 锚点的typed lower-region IR/instance、sparse α/β重建、pure-PyTorch forward/local VJP与eligible
> production gradient parity，以及协调bias/adjoint全链重签的数值拒绝。v2=5 fresh/10 captures、
> 60 metrics/196,380 elements、max diff=`6.109476089477539e-07`、sign exact；v1执行策略未冻结
> 已被superseded。该claim不支持B4-B2/TIR、性能、显存、whole-core/query或ASPLOS-ready。

> **2026-08-18 FSG4/B4-B1a内部关闭**：状态=
> `VALIDATED-B4-B1A-FIVE-FRESH-CAPTURE-SUFFICIENCY`。支持5-fresh bias/output-adjoint/
> sparse-layout capture sufficiency与root replay；不支持typed numerical reference/gradient parity、
> coordinated动态重签拒绝、B4-B2/TIR或performance claim。

> **2026-08-18 FSG4/B4-B1a runner候选**：只支持worker/runner机制与临时5-process pilot；
> formal artifact/8-case报告/full regression尚未产生。协调动态bias/adjoint重签仍是显式限制，
> 因此无B4-B1 typed reference correctness/gradient、TIR或performance claim。

> **2026-08-18 FSG4/B4-B1a capture contract候选**：状态=
> `IMPLEMENTED-B4-B1A-CAPTURE-CONTRACT-PENDING-FIVE-FRESH`。仅支持bias/output-adjoint/
> sparse-layout raw capture与单次real CUDA replay机制；不支持five-fresh、typed reference
> correctness/gradient、B4-B2/TIR或performance claim。

> **2026-08-18 FSG4/B4-B1预注册**：仅冻结计划与门禁，不新增实现claim。现有B4-B0 capture
> 不足以自包含重建bias/局部gradient；B4-B1必须新增incoming bias、operator bias、region output
> adjoints与sparse layout raw，并通过five-fresh typed reference parity。B4-B2/TIR/performance未开放。

> **2026-08-18 FSG4/B4-B0 Round 2外审关闭**：状态=
> `VALIDATED-B4-B0-EXTERNALLY-APPROVED`。独立外审关闭Round 1 F1，确认code/protocol/manifest
> 绝对身份绑定、all-run topology/lineage全链重签拒绝、5 fresh/10 captures与数值/ownership证据。
> 该claim只支持capture correctness/ownership；不支持B4-B2/TIR/performance/memory/ASPLOS-ready。

> **2026-08-18 FSG4/B4-B0 v2内部关闭**：source=`422a3ee`，5 fresh/10 captures与
> 108 tensors/664,744 elements重新生成；max diff=`1.1920928955078125e-07`、sign exact；
> 绝对source/topology/lineage身份绑定与`11/11`完整性负例通过。状态=
> `VALIDATED-B4-B0-V2-PENDING-ROUND2-EXTERNAL-AUDIT`，不支持TIR/performance/memory claim。

> **2026-08-18 FSG4/B4-B0 Round 1纠正**：外审以1个major否决原内部关闭；协调一致topology/
> lineage重写可绕过相对run校验。v2绝对身份绑定已实现但formal artifact未生成，状态=
> `IMPLEMENTED-B4-B0-R1-F1-IDENTITY-BINDING-PENDING-V2`。因此B4-B0仍未外审批准，B4-B1/
> TIR/performance无新增claim。

> **2026-08-18 FSG4/B4-B0 five-fresh内部关闭**：状态=
> `VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`。支持production evaluation-0双锚点
> capture correctness/ownership：5 fresh、10 captures、raw typed replay、max diff=`1.192e-7`、
> sign exact、9/9 outer-resigned tamper。仍不支持TIR correctness、region/whole-core/query speedup、
> memory或ASPLOS-ready；`performance_claimed=false`、`tir_admitted=false`。

> **2026-08-18 FSG4/B4-B0 five-fresh runner候选**：状态=
> `IMPLEMENTED-B4-B0-FIVE-FRESH-RUNNER-PENDING-FORMAL-RUN`。只支持runner/mechanism与单fresh
> CUDA smoke；formal 5-fresh、root replay和9/9 tamper尚未产生，故不升级capture correctness，
> 不开放B4-B1/TIR/performance。

> **2026-08-18 FSG4/B4-B0 live observer候选**：状态=
> `IMPLEMENTED-B4-B0-LIVE-OBSERVER-PENDING-FIVE-FRESH`，只支持显式opt-in evaluation-0 live
> observation与单次CUDA snapshot mechanism。未支持5 fresh、artifact replay/tamper、TIR correctness或
> speedup。真实事实为S-anchor active-beta gradient存在/P-anchor empty-beta无gradient，不伪造
> 全零pre-add。

> **2026-08-18 FSG4/B4-B0 typed capture contract候选**：状态=
> `IMPLEMENTED-B4-B0-CAPTURE-CONTRACT-PENDING-LIVE-HOOK`。仅支持schema/admission/tamper-unit
> mechanism；不支持live capture、5 fresh、gradient parity、TIR或speedup。合同同时绑定production
> compressed与native dense状态，防止把压缩源伪造为exact-region autograd leaf。

> **2026-08-18 FSG4/B4-B v1预注册**：状态=`PREREGISTERED-B4-B-V1-NOT-IMPLEMENTED`，无
> implementation/correctness/performance claim。必须同时关闭active-beta Gemm语义锚点和高占比Conv
> 性能锚点；首先只允许gradient-active evaluation-0 read-only exact-call capture。B4-A不进入
> 基线，PR-12 plain path不放宽，B4-C/D、whole-core/query/B0/ASPLOS claims全部关闭。

> **2026-08-18 FSG4/B4-A外审关闭**：Round 1独立重算hash链、24/24 environment、6/6
> correctness、activation/profile、core/query ratio、root replay、14/14 tamper和回归，AC1—AC7全部
> PASS；exchange=`closed/approved`。最终claim=
> `EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`；B4-A只支持correctness/mechanism，
> 不支持累计performance、memory、B0 parity或ASPLOS-ready。仅开放单独预注册B4-B。

> **2026-08-18 FSG4/B4-A正式计时内部关闭**：source=`46a8493`、v5 24/24 worker、6/6 control pair
> correctness/environment/activation/profile通过，max tensor diff=`4.4107437e-06`、sign exact；core wall
> geomean=`1.018995x`未过`1.03x`，query worst=`0.996947x`通过`0.98x`，memory无收益；replay与14/14
> tamper通过。状态=`INTERNALLY-VALIDATED-NO-GO-B4-A-PERFORMANCE-PENDING-EXTERNAL-AUDIT`，无
> performance claim；fixed related=`73 passed`、full=`1356 passed, 3 skipped`。该pending状态已由上方
> Round 1外审批准取代。

> **2026-08-18 FSG4/B4-A正式计时v4环境投影失败**：source=`03043a3`的v4有19个admitted worker；
> run 19 raw的thermal/power累计值存在`54579 µs`历史偏移，但worker区间增量严格同为`2062477 µs`。
> 旧门禁错误比较累计绝对值，现改为interval delta exact，并由formal replay从raw重算；tamper扩为14类。
> v4不进入ratio，只允许clean-source v5从0重跑，当前无performance claim。

> **2026-08-18 FSG4/B4-A正式计时v3环境拒绝**：source=`be2fa96`的v3完成20个worker后，worker 20
> 因执行期software thermal counter独立增长而`environment.admitted=false`；correctness、activation与
> profile结构未失败，但v3整体不进入ratio。正式协议现绑定`nvidia-powerd=inactive`与
> `enforced.power.limit=55.0 W`，逐worker和replay重验，tamper扩为13类。其v4指令已被上方失败处置
> 与v5指令取代。

> **2026-08-18 FSG4/B4-A正式计时v2环境拒绝**：source=`ee73bc2`的v2在5个complete worker后，
> worker 5因独立software thermal slowdown而`environment.admitted=false`；v2不进入ratio。B4-A formal
> preflight加固为每worker前`<=45°C`且software thermal完全inactive；只允许clean-source v3从0重跑，
> 无performance claim。

> **2026-08-18 FSG4/B4-A正式计时v1失败诊断**：source=`292a035`的worker 3因B4-A profiler alias覆盖
> 缺口fail closed；不是performance/correctness失败。修复后live diagnostic物理计数为forward=4、bound
> eval=10、optimizer=`1/10/9`、handoff/rerun=`1/0`。v1不完整raw不进入任何ratio；只允许clean-source
> v2从position 0重跑，当前无性能claim。

> **2026-08-18 FSG4/B4-A正式计时Runner候选**：冻结的24-process B3/B4-A control/profile runner、
> raw-first/resume、root replay及14类outer-resigned tamper probe已实现；固定related=`70 passed`，
> Black/Mypy/Pylint及全量`1353 passed, 3 skipped`通过。状态=
> `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`，无性能claim，B4-B/TIR关闭。

> **2026-08-16 FSG4/B4-A five-fresh正确性关闭**：source=`43d4117`的10/10 fresh worker、5/5
> B3/B4-A pair及每pair 19个raw tensor比较全过，最大差=`6.109476e-06`，sign/discrete exact，5/5
> handoff=1/rerun=0/lineage=6/provider-fallback=0，root replay PASS。状态=
> `INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`；无性能claim，只开放独立正式计时。

> **2026-08-16 FSG4/B4-A实现候选**：typed terminal producer、one-shot handoff、no-rerun assembly、
> post-query content audit与five-fresh runner已实现，状态=
> `IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。GPU smoke只验证机制/语义，单pair约1.02894x core
> 不构成性能claim；five-fresh与正式计时尚未执行，B4-B/TIR关闭。

> **2026-08-16 FSG4/B4-A预注册**：状态=`PREREGISTERED-B4-A-NOT-IMPLEMENTED`。冻结第10次optimizer
> evaluation的terminal lower/六层lA typed handoff、export CROWN rerun=0、handoff=1，以及
> state/graph/split/topology/producer-op/shape/dtype/layout/content lineage。必须先过5 fresh correctness，
> 才允许检验B3/B4-A core `>=1.03x`与query worst pair `>=0.98x`。当前无B4-A correctness/performance
> claim，B4-B/TIR与B5—B7关闭。

> **2026-08-16 FSG4/B4-0外审关闭**：Round 1外部模型从formal raw独立复算AC1—AC7全PASS，
> 无blocker/major；exchange=`closed/approved`。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。只开放B4-A terminal lower/lA handoff；shape必须
> 从correlation parent operator恢复并绑定lineage，B4-A数值复用正确性与性能尚未成立。B4-B不得混入，
> B4-C/D与B5—B7继续关闭。

> **2026-08-16 FSG4/B4-0内部关闭**：source=`66154e4`正式artifact含270609 raw events、
> 35367/35367 CUDA kernel closure、14-call/4-forward exact marker；semantic max diff=`4.768e-7`且
> discrete/sign exact，replay与9/9 outer-resigned tamper通过。CROWN14覆盖9196 kernels，按B3冻结share
> 约为67.72% core；terminal export是完整重复CROWN call。状态=
> `INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`；只支持attribution/opportunity，
> 不支持speedup/memory/B0 parity。外审批准后只开放B4-A。

> **2026-08-16 FSG4/B4-0 Runner候选**：read-only profiler schema/runner已实现并通过15项B4、54项
> B3/B4相关及全量`1329 passed, 3 skipped`；raw保存operator/kernel/stream/shape/memory、明确记录
> correlation/temporal/unattributed方法，并以确定性gzip和多层digest绑定。状态=
> `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`，只支持runner机制claim；正式opportunity、B4-A/B
> 准入、B4 speedup、B0 parity与ASPLOS-ready均未成立。

> **2026-08-16 FSG4/B4预注册**：状态=`PREREGISTERED-NOT-IMPLEMENTED`，无B4 performance claim。
> B3 raw重算显示optimizer-only占query约7.933%，无限加速上限约1.0862x，无法追回B0所需1.0989x；
> 10 optimizer + 1 terminal export + 3 KFSB child的14-call lower-only CROWN合计约12.010%，单独追回
> B0 parity需约3.9897x。当前只准入B4-0 attribution，B5—B7与最终system gate关闭；BoundConv 40x
> 继续标为`USER-REPORTED`。

> **2026-08-15 FSG4/B3外审关闭**：Round 2外部模型从raw独立重算44项检查，AC1—AC7全PASS，
> 无blocker/major/minor；exchange=`closed/approved`。`VALIDATED-REDUCED-B3`正式成立，只支持B3相对B2
> core `1.071617x`、query不退化的reduced claim，仍不支持相对B0或ASPLOS全栈speedup。当前仅开放
> B4 cumulative fusion candidate；B5—B7及最终`1.20x/1.15x`门槛关闭。

> **2026-08-14 FSG4/B3正式计时内部关闭**：source `36e9069`的36/36 fresh worker、correctness、
> environment、measurement、direct activation、root replay与10/10 tamper通过。B2/B3 core/query=
> `1.071617x/1.006623x`，但B0/B3 query=`0.910001x`，所以状态=`VALIDATED-REDUCED-B3`；只支持B3相对
> B2的reduced机制/计时结论，不支持BoundFlow快于原始auto_LiRPA或ASPLOS全栈performance claim。
> 该“外部审计待完成”历史状态已由上方Round 2批准取代。

> **2026-08-14 FSG4/B3正式计时Runner候选**：B0/B2/B3六全排列36-process schema、direct activation
> receipts、raw-first/resume、root replay与十类tamper probe已实现；targeted=`108 passed`、full=
> `1308 passed, 3 skipped`。状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`，只支持runner机制与
> 合同实现claim；该“正式artifact未运行”历史状态已被上方内部关闭结果取代。

> **2026-08-14 FSG4/B3 Five-Fresh关闭**：source `75dfd81`按预注册顺序完成10/10独立fresh GPU
> worker与5/5 B2/B3-C direct semantic pairs；environment、provider/fallback、physical counter、
> post-query audit、root replay与7/7 outer-resigned tamper均通过。状态=
> `VALIDATED-B3-FIVE-FRESH-CORRECTNESS`，支持五组fresh correctness/admission claim，并将正式B3计时
> 准入设为true；不支持timing/speedup，五组artifact继续写`performance_claimed=false`。B4—B7仍关闭。

> **2026-08-14 FSG4/B3-C关闭**：source `72bec5e`的fresh GPU artifact用1484条event确认12个device
> candidate/commit/backup/copy与timed candidate D2H=`0`；headline digest为0，24次GPU hash全部属于
> post-query audit；六个B2 control语义、replay和6/6 tamper通过。状态=`VALIDATED-B3-C-COUNTERS`，只
> 支持device transaction/audit mechanism、correctness和physical counter claim，不支持timing/speedup；
> 5 fresh pairs、正式B3计时及B4—B7仍未关闭。五组门禁现已由上方closure取代；正式计时仍未执行。

> **2026-08-14 FSG4/B3-B关闭**：source `42df2dc`的fresh GPU artifact用5157条event确认full step
> snapshots=`0`、forward builds=`4`，其余B3-A结构、六个B2 control语义、replay与6/6 tamper保持。状态=
> `VALIDATED-B3-B-COUNTERS`，仅支持terminal Schedule/forward handoff mechanism/correctness claim，不支持
> timing/speedup；B3-C及B4—B7仍未关闭。

> **2026-08-14 FSG4/B3-A关闭**：source `c7851c8`的fresh GPU artifact用5157条event确认template
> compile/hit=`1/1`、module move=`0`、scope=`1`，其余optimizer/forward/KFSB/D2H/commit结构保持；六个
> B2 control语义、replay和6/6 tamper通过。状态=`VALIDATED-B3-A-COUNTERS`，仅支持prepared-core
> mechanism/correctness claim，不支持timing/speedup；“B3-B未关闭”已被上方closure取代，B3-C及
> B4—B7仍未关闭。

> **2026-08-14 FSG4/B3-0关闭**：source `4195361`的fresh B2 artifact用4625条显式event确认
> module/scope=`1/2`、optimizer=`10/9/10 snapshots`、forward=`5`、KFSB=`3/3`、D2H/commit=`12/12`，
> tensor hash/typed validate/stable hash=`4417/84/10`；六个冻结B2 control语义与6/6 tamper通过。状态=
> `VALIDATED-B2-COUNTERS`，不是performance claim；“B3-A未实现”已被上方closure取代，B3-B/C与
> B4—B7继续关闭。

> **2026-08-14 FSG4/B3预注册（历史）**：FSG3关闭后只准入IR/graph/Plan/Schedule复用层。B3分为
> PreparedCoreTemplate、terminal-only optimizer Schedule与device-resident AtomicCommitPlan，并冻结
> physical counter、correctness、36-process timing和rollback门禁。该段的
> `PREREGISTERED-NOT-IMPLEMENTED`已被上方B3-0 closure取代；仍没有B3 candidate/timing/speedup，
> B4—B7继续关闭。

> 本表是动态证据账本。`planned` 不代表已经实现；只有代码、测试和工件均存在时才能改为
> `validated`。当前执行基线为 PR-12 validated-reduced；PR-13 已以
> `VALIDATED-REDUCED` 关闭；PR-14B 真实 replay 为 `VALIDATED-NO-GO`，C3 已降级为 C1/C2
> 基础设施，不再主张 non-toy verifier acceleration。2026-08-03 独立 RVIR correctness
> 路线已 validated-reduced，但不改变 ASPLOS performance No-Go。2026-08-04 production
> Schedule-memory P0 同样为 `NO_GO`。随后 NRIR-1 已把固定 ResNet main CROWN backward
> lower 为 native multi-region IR；NRIR-2 完成 storage switch/runtime last-use，NRIR-4 完成
> representation→execution binding，NRIR-5 产生真实 spec-sliced child execution；NRIR-6 已把
> 两轴联合到同一 template/selector 并执行四组合；NRIR-7 加入 9 条真实 property query 的
> packed/serial/cache/lineage；NRIR-8 加入 8 个不同 input-box leaf、exact child state 与
> domain-axis execution。NRIR-34 已进一步加入 same-parent sibling-group Plan/Task/Schedule 与 packed
> dynamic queue；NRIR-35 又以一等六阶段 IR 先执行九子句 floor，再用相同 global start 对 clause 0
> 做 additive packed work。三轮均保留 9/9 original-ordinal accounting，但仍 9/9 unresolved，故
> ASPLOS-ready No-Go 不变。2026-08-06 NRIR49A又证明CPU侧selected-CROWN winner在RTX 4060 GPU
> queue/complete中只占约7.10%/7.05%，Amdahl与physical-memory门禁均失败，selected-CROWN专属
> G2/G3已gated off。`1.0764x`只是删除该单区域的deletion-only上限，不是BoundFlow全栈上限；FSG0
> schema/replay合同已验证；FSG1 official B0 full-stack trace亦已关闭。FSG2现以
> `VALIDATED-REDUCED initial-only`关闭：完整production alpha/beta/split replacement在该历史时点未准入，
> 因而B2曾为NO-GO，FSG3—FSG5按依赖门禁未运行。该结论不否决B3—B7各层潜力。2026-08-13
> RVIR-v4已把V4-1 frozen post-state evaluation关闭；重启后
> V4-2B 10-evaluation/9-Adam-step正式GPU typed trace、original replay与5类同步重签名tamper通过，以
> `VALIDATED-PRODUCTION-TRACE`关闭。V4-2C又以正式artifact关闭6组native pre-state初始化、12/12
> round-trip及6类双层重签名tamper，状态为`VALIDATED-PRE-STATE-INITIALIZER`。这些结论尚未执行
> optimizer mutation。V4-2D现又以formal 10/9 native loop逐step parity与6类双层重签攻击关闭；V4-2E
> 随后以12-path atomic copy-out、rollback、formal replay和6类完全重签攻击关闭。V4-2整体状态升级为
> `VALIDATED-OPTIMIZER-REPLACEMENT`。V4-3A现又以451-tensor fresh semantic replay和六类同步重签攻击
> 关闭original whole-core truth。它不是whole `update_bounds_core` live replacement；B2与性能claim仍
> 关闭。V4-3B又以零provider callback导出六层native lA、12个shared-input intermediate tensors与
> final lower，通过五类同步重签攻击，以`VALIDATED-NATIVE-BACKWARD-EXPORT`关闭。V4-3C又以零provider
> callback推导六层mask、复现三组top-3 candidate、执行72个child lower并恢复final decision，通过八类
> 同步重签攻击，以`VALIDATED-NATIVE-KFSB`关闭。V4-3D随后完成真实CUDA whole-core→official
> post/queue接通、fresh replay与八类完全重签攻击，以`VALIDATED-LIVE-RETURN`关闭。V4-3E再以十个
> fresh CUDA进程完成5/5 counterbalanced pairs及六类tamper，V4-3整体升级为
> `VALIDATED-WHOLE-CORE-REPLACEMENT`。历史ownership blocker已被取代；B2 same-solver timing现已准入
> 并于2026-08-14完成正式FSG3：六个全排列block、36个fresh GPU进程的correctness、environment、
> measurement与replay全部通过。B1 query wall=`0.995657x`，当前B2 query/core=
> `0.908400x/0.516767x`（B0/candidate），故FSG3=`VALIDATED-FSG3-B0-B1-B2-BASELINE`、B2=
> `MEASURED-B2-SLOWER`，不是speedup；B3—B7未实现，当前仍无BoundFlow全栈性能claim。

| Claim | 当前状态 | 代码/设计落点 | 必需测试 | 必需工件 |
|---|---|---|---|---|
| C1：显式物化语义的 Structured Bound-Operator IR | native ResNet correctness/representation binding validated-reduced | typed Bound IR + lowering + dense/structured interpreter + source Plan/Schedule→execution Bound/Task/Launch binder | ResNet 17 Primal→21 source ops；structured execution 49 ops，含 14 cast + 14 materialize；dense/structured max diff 9.54e-7 | NRIR-1/2/4 artifacts；structured storage 仍 dense-equivalent，不能升级 compression/performance |
| C2：Method/Autograd/Memory-Aware Materialization Planner | real-graph joint policy + exact repeated-query/domain Plan selection validated-reduced；IR-5 final 仍 NO-GO | NRIR-6 joint selector + NRIR-7 cache key + NRIR-8 full/domain-size-4 candidate | 四组合；cache invalidate；8 domains 的 full/packed PlanInstance/Schedule identity 不同；历史 Global p90 No-Go 保留 | NRIR-6/7/8 artifacts；无 physical CUDA peak/OOM/Pareto，paper performance claim 不成立 |
| C3：Verification Query Runtime Infrastructure | property-query + input-box parent/child domain formation/packing/lineage validated-reduced；performance downgraded | typed query/domain specs、exact child state、packed execution、serial same-policy restore | NRIR-7 9 queries→3 vs 9；NRIR-8 8 leaf domains→2 vs 8；8/8 parent/result restore；packed/full/serial bitwise equal | fused replacement 0/394 历史事实保留；ReLU/β split queue、prune、termination 与公平 timing pending |
| BoundFlow Schedule IR | real-network storage/representation/spec/domain-slice ownership validated-reduced；production-performance claim 仍 NO-GO | typed ScheduleModule + native ResNet lifetime、transitions、spec/domain loops 与 child stacks | NRIR-5 spec ranges；NRIR-8 domain ranges `[0,4)/[4,8)`、2 child vs serial 8、bitwise equal | deterministic NRIR-1/2/4/5/8 replay；sample axis、full BaB、OOM rescue/GPU evidence pending |
| BoundFlow Task IR | IR-3 per-task semantic closure validated-reduced；production backend pending | TaskIRModule/Unit + typed op/shape/parameter/external/state/memory/backend refs + stateful Bound stepping | 12 个 tests（含 4 graph families、structured materialize、skip/reorder rejection） | per-task output hashes 与 final bound hashes 已入 fresh-process artifact v2 |
| backend 执行 typed Planner/Task 结果而非定义核心抽象 | IR-4 validated-reduced；IR-5 final performance No-Go | composite typed registry + query adapter + real fused/unfused/fallback；prepared capsule 将静态 validate/hash/dispatch 移出 query hot path | residual v3 all backend correctness；ordinary batching p90 regret 1.008×，Global 1.262× | v3 可 replay；backend correctness 成立，但 adaptive production performance claim 失败 |
| 相同浮点语义下保持 reference bound computation | local-semantics 历史 No-Go；external-semantics initial-CROWN validated-reduced | dense reference + explicit external intermediate-bound source/adaptive policy | allclose、gradient、auto_LiRPA、replay | ResNet historical local max diff 796.765；新 external-semantics max diff 3.10e-6、sign 9/9；CPU only |

### 2026-07-20 IR-first claim 纠偏

历史 PR-10/11/12 的数值、OOM、held-out 和 backend 证据不撤销，但其 claim 范围必须与代码对象
层级一致：

- runtime `LinearOperator` 证明结构化表示机制，不自动证明一等 Bound IR；
- `MaterializationPlan`、`MaterializationPlacementPlan` 和 `ExecutionCandidate` 证明局部决策机制，
  不自动证明统一 Plan IR；
- TaskGraph 拓扑序和 `FusedCrownExecutionStep` 不自动证明 Schedule IR；
- PR-13 batching 证明保持 ordinary batching 收益，不自动证明 adaptive runtime 贡献；
- cached specialization/JIT 在新 break-even 证据出现前只属于 planned hypothesis。

新的升级门禁见
`gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。

### 2026-07-28 IR-1A 进度

- `boundflow.bound_ir/v1.0` 已新增 typed value/type/spec/domain/op/graph/module；
- graph verifier 已覆盖 SSA/use-def、类型/极性、batch axes、representation change、method state；
- module verifier 会把 input/spec bind 与 concretize ID 交叉解析到 typed VerificationSpec；
- module 已有 canonical JSON 与 SHA-256 stable hash；
- Bound IR 源模块不依赖 runtime、backend、PyTorch 或 TVM；
- 旧 `DomainState` 兼容路径保留；
- builder、reference interpreter、CROWN lowering 和 IR-driven E2E 尚缺，因此不升级完整 C1。

实现与测试边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_schema_foundation.md`。

### 2026-07-28 IR-1B 进度

- `BoundAffineStateRef` 显式表示 `A_u/b_u/A_l/b_l`，不再把真实 CROWN state 压成单值；
- residual/concat backward route 和 fanout compose 已成为 typed BoundOp，并验证 bias-once 语义；
- `boundflow/frontends/plain_crown_bound_ir.py` 已把单任务 plain-CROWN Task/trace lower 为
  validated `BFBoundModule`；
- `boundflow/runtime/bound_ir_interpreter.py` 已独立执行 dense Bound IR，不 import CROWN oracle；
- identity/multi-spec MLP、chain CNN、residual/concat fanout 的 final lower/upper 已与现有
  `run_crown_ibp_mlp` 对齐；
- stale parameter/objective、缺失 ReLU trace fail closed；
- 专属测试 20 passed，全量 392 passed、1 skipped；
- materialize/representation rewrite、structured execution、生产 runtime 迁移和 IR-driven artifact
  尚缺，因此仍不升级完整 C1。

实现与门禁边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_plain_crown_lowering.md`。

### 2026-07-28 IR-1C / IR-1 closure

- affine-state verifier 禁止 Linear/Conv/ReLU/Reshape/route/compose 隐式改变 representation；
- 新 verified rewrite 在 affine region 入口插入 dense→structured cast，在 ReLU/concretize
  dense boundary 前插入 materialize；
- reference interpreter 已执行 structured LinearOperator region 和显式转换；
- multi-spec MLP、chain CNN、residual/concat fanout 的 dense/structured rewrite final bounds 对齐；
- 非法隐式转换和重复 rewrite fail closed；
- 专属测试 25 passed，相邻 47 passed，全量 397 passed、1 skipped；
- IR-1 契约的最小 reference semantic closure 门禁已通过；
- 完整 C1 仍需 IR-2/3/4 的 Plan/Schedule/backend integration 和 IR-driven E2E artifact。

实现与门禁边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_representation_rewrite.md`。

### 2026-07-28 IR-2A 进度

- 新 `boundflow.plan_ir/v1.0` 已区分 `PlanTemplate` 静态候选空间与 `PlanInstance` 动态选择；
- region/representation/materialization/backend/domain-spec-sample batch/storage/state 成为独立
  typed candidate/decision；
- cross verifier 已检查 Bound hash、partition coverage、capability、transition、memory、
  storage lifetime/alignment/alias、state version 和候选全量记账；
- template/instance canonical JSON/hash 已有，instance strict JSON replay 已拒绝 noncanonical 和
  tampered selection；
- PR-11/12 六类旧对象已有 adapter/partial/unsupported 代码级迁移表；
- 专属 12 passed，相邻 88 passed，全量 409 passed、1 skipped；
- reference template builder、query-time selector、多预算选择和 artifact 尚缺，因此 IR-2/C2
  均不升级为 complete。

实现与门禁边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_schema_and_legacy_migration.md`。

### 2026-07-28 IR-2B 进度

- 新增 typed evidence → `PlanTemplate` reference builder，自动推导 Bound IR region boundary、
  storage lifetime/alignment 和 capability rejection；
- 新增有界 deterministic selector，memory/deadline 改变时产生不同且完整记账的
  `PlanInstance`，无可行计划时 fail closed；
- 新增不可变 Bound/Template/Instance artifact API、逐文件 SHA-256、精确 replay 与 tamper
  rejection；
- Plan IR 专属 11 passed，连同 migration 共 16 passed；相邻 92 passed；全量
  413 passed、1 skipped；
- 尚缺旧 PR-11/12 真实 artifact 批量 assembly/report、query-time state-validity 和独立 replay
  CLI，因此 IR-2/C2 仍不能升级为 complete。

实现与门禁边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_reference_builder_selector.md`。

### 2026-07-28 IR-2C / IR-2 closure

- `PlanInstance.state_validities` 和 `StateAction.REUSE` 已把 query-time exact cache validity 纳入
  canonical verifier/hash/replay；stale state 转 recompute，伪造 valid stale state fail closed；
- legacy migrations 可原子组装到同一 template，accepted/unsupported/rejected 形成稳定报告；
- reference artifact 已有 fresh-process generate/replay CLI；
- 对当前 `artifacts/` 扫描 58 个 JSON/JSONL、4,911 个 JSON objects，三种 PR-11/12
  planner raw schema 记录均为 0；因此只关闭对象族级 migration，不声称历史逐记录迁移；
- 专属 21 passed，相邻 97 passed，全量 418 passed、1 skipped；
- IR-2 最小 reference contract 关闭为 `VALIDATED-REDUCED`；C2 仍需 IR-3 Schedule IR、
  runtime/backend migration 和 IR-driven E2E，不能升级为 paper-level complete。

实现与 closure 边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`。

## PR-10 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-10A Materialization instrumentation | validated | `25225e5`；ReLU barrier opt-in trace |
| PR-10A.1 Trace schema v1 | validated | `boundflow.materialization/v1`、schema contract tests、164 passed |
| PR-10B.1 workload characterization | validated | `8f2c998`；180/180 clean GPU profile；mini-ResNet s128/d32 |
| PR-10B.2 真实 BaB fixed-domain replay | superseded | 不再执行；由 PR-14A/B 真实 verifier trace/replay 取代 |
| PR-10C.1 Dense/gradient reference oracle | validated | 显式 `A_u/A_l/b_u/b_l` oracle；独立 α sign-gradient；170 passed |
| PR-10C.2 Dense/structured 双路径 oracle | validated | local/full/gradient、plain/α/αβ、真实 solve_bab 搜索等价 |
| PR-10D.1 Exact SignSplit operator | validated | exact dense/gradient；composition 包裹而不下推 sign；26 passed |
| PR-10D.2 Structured ReLU 主路径 | validated | main coefficient 不永久 dense；ephemeral bias；operator dump；177 passed |
| PR-10E 全路径回归与 benchmark | validated（guarded） | 360 rows；354 ok/6 structured OOM；179 passed；dense 默认 |

## 当前 Gate 0 证据

- PyTorch 2.12.1+cu132、CUDA 13.2、LLVM 20.1.8、TVM 与单一内嵌 tvm-ffi 已完成现场验证；
- MLP/CNN reduced artifact 已生成：small matrix、warmup 3、iters 10，2 行均通过 correctness；
  它是 Gate 0 回归，不替代论文要求的至少 5 次独立重复；
- Gate 0 已冻结在本地提交 `4e0e059`，全量验证为 162 passed、1 个预期 skip；
- Gate 0 已完成；PR-10 已在 `263ea81` 结项，ReLU structured path 为 feature-gated，dense 默认。

## PR-10 第一版 profile claims

- `C1-E1a` validated：persistent ReLU logical bytes 在固定结构下随 spec×domain 线性放大；
- `C1-E1b` validated：mini-ResNet αβ s128/d32 为 939,524,096 logical bytes、3.45 GB
  trace-off peak allocated；
- `C2-M1` partial：query axes 会改变 materialization 规模，但尚未证明不同计划各有最优 regime；
- 详细口径与限制：`gemini_doc/pr10_materialization_profile_summary_2026_07_12.md`。

## PR-10 完成判定

- `C1-E2` validated：local/full/gradient、CROWN/α/αβ/solve_bab 与 dense reference 对齐，
  360 行矩阵中 0 correctness failure；
- `C1-E3` validated：代表性 plain CROWN 大点 structured peak 降低约 29.8%；
- `C1-L1` validated limitation：同一点 structured latency 增加约 9.17×，不适合默认启用；
- `C1-L2` validated limitation：α/αβ structured 显存恶化，并在 6 个大点 OOM；
- `C2-M1` validated motivation：不存在跨 method/grad/memory regime 的统一最优表示；
- `C2-H1` planned hypothesis：最优可行计划必须感知 method、differentiation stage、capability
  与 memory budget；PR-10 数据只能作为动机/校准数据，尚不是 Planner 有效性证据；
- PR-10 状态：**complete, feature-gated**；默认 dense，structured 由环境开关启用；
- 对照证据：`gemini_doc/pr10_dense_structured_comparison_2026_07_12.md`。

## PR-11 内部门禁

- 0 bound/gradient correctness failure，0 unexpected OOM；
- 若任一合法候选可运行，Planner 应找到可运行计划；α/αβ structured 不得被误选；
- workload-family held-out 上 median latency regret 相对 Oracle 研发目标不超过 20%，并报告 p90；
- 至少选择 dense 与 structured 两类计划；
- 至少一个预算下，让 Always Dense OOM 的 plain CROWN case 成功运行；
- 与 Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy 和 Oracle
  公平比较。

## PR-11 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-11A Context/capability/action/plan dump | validated | `materialization.py`；真实 CROWN shape-derived context；JSON plan |
| PR-11A.1 Runtime guard | validated | CROWN 显式 plan；α/αβ structured capability 拒绝；reduce-batch re-plan signal |
| PR-11A.2 Per-case measured Oracle | validated foundation | fastest observed feasible action；capability/OOM 不可绕过 |
| PR-11B Cost model calibration/held-out | validated foundation | calibration + validation/refit + final mini-ResNet held-out；method/action linear model |
| PR-11C Local/Global benchmark matrix | partial | 1728 rows；Global 239/239、0 unexpected、median/p90 1.0；但与 Memory-Threshold 相同 |
| PR-11C.1 Multi-barrier placement mechanism | validated foundation | synthetic Local re-plan vs Global mixed feasible；两 ReLU mixed execution 与 dense 对齐 |
| PR-11C.2 Measured barrier-level held-out | partial | shuffled calibration 56 rows + held-out mini-ResNet 128 rows，184/184 correct；one-shot Global 未过 feasibility gate |
| PR-11C.3 Global Retry held-out replay | validated reduced | 7/7 feasible、0 unexpected、median 1.159×、p90 1.562×；仅一个 held-out query |
| PR-11D Host OOM retry | validated reduced | 380 MiB cap；dense real OOM→structured success，3/3 独立重复；仅 plain CROWN 单配置 |
| PR-11D.1 Bounded stratified retry | validated reduced | s32/d8 与 s128/d8 均 7/7、0 unexpected；median 1.159×/1.171×；最多 3/5 次；真实 OOM 3/3 |
| PR-11D.2 Scheduler reduce-batch execution | planned | 当前 reduce-batch 仍主要返回 host re-plan signal |
| PR-11E Independent-topology held-out | failed gate | branched ResNet 128/128 correct、9/9 feasible、0 unexpected，但 median/p90 regret 1.976×/4.494×；需 static topology/liveness cost |
| PR-11E.1 Static topology/liveness cost | validated reduced | 不读取 candidate trace；显式 shape/FLOPs/bytes/reuse/batch axes；3× replicated 1,416/1,416 correct |
| PR-11E.2 Ridge/factor LOO calibration | validated reduced | topology-density v3；6-family/36-budget LOO 选择 ridge=.001、factor=1.30；manifest 固化 |
| PR-11E.3 Replicated held-out | validated reduced | 聚合后 23/23 feasible、0 unexpected；median 1.000×/1.194×/1.880×；p90 1.747×/1.194×/2.377× |
| PR-11E.4 Production candidate foundation | validated foundation | static summary→model load→candidate generator→plain-CROWN bounded runtime；真实 OOM v3 3/3 |

## PR-11 冻结 Claims

- `C2-E1` validated-reduced：三组 replicated held-out 共 23/23 产生可行计划，0 unexpected OOM；
- `C2-E2` validated-reduced：380 MiB CUDA cap 下 dense OOM 后 structured recovery 3/3；
- `C2-E3` partial：mini s32/s128 median regret 为 1.000×/1.194×；
- `C2-L1` validated limitation：branched topology median regret 仍为 1.880×；
- `C2-L2` validated limitation：9 个 regret>=1.5 case 全部首先归因为 bounded candidate set
  未包含 measured oracle；7 个仅带待验证的 backend-gap flag；
- `C2-S1` pending：full-scale same-solver BaB 与 time-to-verify 尚未验证。

归因细节见 `gemini_doc/pr11_regret_attribution_2026_07_13.md`。PR-12 只验证 fused backend
是否改善 Pareto frontier，不改写 PR-11 历史 Planner/profile 结论。

## PR-12 当前证据

- `C1-E4` validated kernel foundation：fused ReLU+Linear/Conv PrimFunc 在 reduction 中内联
  sign/slope/bias，pre/post schedule 0 intermediate allocation，不写回完整 `A_scaled`；
- `C2-E4` validated foundation：placement/backend 已拆分，Linear/Conv capability 对
  grad/α/β/split/dtype/device/dynamic shape 和不支持的 Conv 属性显式拒绝；
- `C2-E5` partial sanity：4 个 calibration 点中 3 个快于 PyTorch dense eager，stride-2 medium
  为 1.717× slowdown；尚无正式 latency-memory Pareto、end-to-end 或 final held-out；
- `C2-E6` validated correctness closure：显式 single-consumer Affine→ReLU step、graph/contract
  runtime validation、fanout safe fallback、后端无关 executor、DLPack zero-copy storage alias、
  TVM-FFI custom-stream bridge，以及 chain/residual/multi-block mini-ResNet 最终 bound 对齐；
  尚不等价于正式性能验证；
- `C2-L2` validated current limitation：只支持 static FP32 CUDA plain CROWN、Linear 与
  groups=1/dilation=1 的有限 Conv 子集；
- `C3-M1` pending：compile amortization 与 repeated-query stream 尚未测量。

PR-12E/F 正式证据更新：

- `C2-E7` validated mechanism/Pareto：calibration 12/12、frozen held-out 24/24 candidate rows
  correctness 通过；default/custom stream 均用同 stream CUDA Events，无 timed global sync；
- `C2-E8` validated memory frontier：5 个 held-out 的 fused peak 全部低于 eager；64 MiB
  memory-sensitive Linear 中 eager 68.599 MiB、fused 29.282 MiB，只有 fused 满足预算；
- `C2-E9` guarded Planner：5/5 预算可行、0 unsafe、median/p90/max regret
  1.000×/1.262×/1.262×；fanout fallback 1/1，但 profitable 或 budget-required 仅 3/5；
- `C2-L3` validated limitation：unseen Conv 与三 block mini-ResNet warm speedup 仅
  0.792×/0.968×，memory-sensitive Linear 0.238×；当前 schedule 不能作为 latency headline；
- `C3-M1` partial：warm-faster 点 compile break-even 约 2.2k–7.4k queries；尚未接真实
  repeated-query runtime/BaB stream；
- 工件链：`artifacts/phase7a-pr12/pr12e-calibration-v1-20260713/` →
  `pr12f-final-heldout-v1-canonical-20260713/` → `pr12ef-report-v1-canonical-20260713/`。

PR-12G 多后端证据更新：

- `C2-E10` validated reduced：新增 `pytorch_chunked_r512`，每次只物化有限 query rows 的
  scaled-A，并复用 cuBLAS/cuDNN；Linear/Conv、default/custom stream 和真实 CROWN execution
  step backend contract 均有回归；
- `C2-E11` validated reduced Planner：全新 v2 split 上 calibration 48/48、held-out 36/36
  candidate rows 正确；5/5 budget feasible、0 unsafe、exact Oracle 3/5、median/p90 regret
  1.000×/1.054×，eager/chunked/TIR 各选择 1/2/2 次；
- `C2-E12` validated budget Pareto：memory-sensitive Linear 中 chunked 2.217 ms / 54.08 MiB，
  eager 3.284 ms / 65.69 MiB，64 MiB 下只有 selected candidate 可行；
- `C2-L4` validated limitation：selected geomean 仅为 eager 的 1.081×，尚无 structured eager/
  TVM-unfused 完整正式对照；TIR long-reduction schedule 仍不是 latency headline；
- authoritative 工件链：`pr12g-multibackend-v2-freeze-20260713/` →
  `pr12g-multibackend-v2-calibration-canonical3-20260713/` →
  `pr12g-multibackend-v2-final-canonical3-20260713/` →
  `pr12g-multibackend-v2-planner-replay-canonical3-20260713/` →
  `pr12g-multibackend-v2-report-canonical3-20260713/`。

PR-12H benchmark contract freeze：

- `C2-M2` validated evidence boundary：机器可读合同区分 preallocated kernel、region-runtime 与
  complete final-bound 三层 inclusion/allocation/synchronization；
- `C2-L5` validated limitation：PR-12 fused-sanity 的 PyTorch/TVM allocation contract 不同；
  PR-12E/G candidate timing 又把 region matching/Planner 放在 timed call 外，二者均标记
  `compliant=false`，历史数据不得冒充正式三层合同；
- freeze tag：`pr12g-validated-reduced` → `44f87ae`；规范见
  `docs/pr12_benchmark_contract.md`，持续状态见 `gemini_doc/pr12_execution_status.md`；
- `C2-E13` validated baseline：PR-12I 新合同下 72 rows 为 54 ok、18 N/A、0 correctness
  failure；structured eager 只在 complete final-bound 比较，TVM-unfused 在 region/E2E 都显式
  物化 scaled-A；default/custom stream 均通过；
- `C2-E14` validated attribution/limitation：TVM fused E2E geomean speedup 仅 0.546× eager，
  但 median peak ratio 为 0.512 且 3/3 Pareto；TVM-unfused 为 0.481×、0/3 Pareto，说明显存
  收益来自 fused materialization elimination，但当前 latency 不能成为 headline；
- `C2-L6` validated limitation：`torch.compile(fullgraph=True)` 在 3 workloads×2 streams 均因
  final-bound host path 的 `ContextVar.set` 无法 capture，结构化记录为 N/A，没有改写 workload；
- PR-12I 工件：`pr12i-baseline-v2-20260714/` →
  `pr12i-baseline-report-v2-20260714/`；下一门禁为 PR-12J compile/load/cache amortization。

PR-12J compile/cache 证据更新：

- `C3-M1` validated measurement：cache key 覆盖 signature/target/code schema/TVM ABI，`.so` 与
  manifest SHA 校验；3/3 workload 的 fresh compile、memory hit 与独立进程 disk hit 数值正确，
  worker 0 hidden recompile；
- `C3-E1` partial regime：mini-ResNet fused warm 6.847 ms vs eager 7.234 ms，fresh/disk-first/
  process restart break-even 为 4668/1062/4450 queries；均超过 Q=1024，且不优于 chunked
  6.513 ms；
- `C3-L1` validated limitation：Linear/Conv fused warm 分别 8.557/3.301 ms，均慢于 eager 与
  chunked，因此严格为 `not_amortizable`；3 个 workload 在 Q≤1024 内 0 个可对 eager 摊销；
- v1 tuple/list manifest bug 与 v2 warm-path SHA 污染保留；authoritative 工件为
  `pr12j-amortization-v4-20260714/` → `pr12j-amortization-report-v4-20260714/`。

PR-12K profiler 证据更新：

- `C2-E15` validated activity profile：6 workload×5 backend 共 30/30 complete final-bound rows
  correct；raw Chrome trace、kernel/API activity CSV、图与 SHA manifest 闭合；
- `C2-E16` validated mechanism boundary：fusion 对 TVM-unfused 每个 eligible region 只减少
  2 launch，六点最大整体 launch 降幅 1.96%；按 5% CUPTI device-time 阈值为 3/6 退化、
  1/6 改善、2/6 中性；
- `C2-L7` validated tooling limitation：Nsight Compute 2026.1.1 实测 `ERR_NVGPUCTRPERM`，
  禁止 SpeedOfLight、bandwidth/cache、occupancy 和 stall claim；不根据缺失 counter 猜测；
- `C2-D1` validated decision：PR-12L 唯一分支为 `E_STOP_OPTIMIZING_TIR`；保留 fused 为
  Planner candidate，但停止无 counter 支撑的孤立 schedule 调优；
- authoritative 工件：`pr12k-cupti-v3-20260714/` →
  `pr12k-cupti-report-v4-20260714/`。

PR-12L 止损决策：

- `C2-D2` validated scope freeze：唯一选择 `E_STOP_OPTIMIZING_TIR`，PR-12 closure 不再增加
  Linear tile、CUDA Graph、chunk-size family 或 Conv capability；
- `C2-D3` validated backend boundary：不删除 fused backend；PR-12M 仍可在预算或 amortized
  latency 合适时选择它，避免把局部负结果误写成后端全面失败；
- `C2-L8` validated evidence limit：如果未来获得硬件 counter 或新 workload，必须用新假设/
  新 split 重新开启，不能回写 PR-12K 或消费冻结 final held-out。

PR-12M compile-aware Planner 证据：

- `C2-E17` validated-reduced：capability→budget→risk→amortized latency 决策显式使用 expected
  reuse、memory/disk cache probability 与 fresh/disk setup；
- `C2-E18` validated held-out isolation：v3 split 在 final 未消费时冻结，calibration/final 各
  25/25 correct，fit/replay model SHA 完全一致；
- `C2-E19` validated multi-regime：75 decisions 中 72 个存在可行 candidate，Planner 72/72
  选到可行 backend、0 unsafe；feasible median/p90/max regret 1.000×/1.000×/1.016×；
- `C2-E20` validated nontrivial selection：总选择 eager/chunked/structured/fused 为 47/12/3/13；
  fused 从 cold/mixed 各 1 次增至 warm Q1024 的 11 次，32 MiB 下四类 backend 都出现；
- `C2-L9` validated capacity limit：memory-heavy Linear 在 16 MiB 下 3 个 policy 均无实测可行
  candidate；单独报告，不用不可行区 regret 污染/美化 feasible gate；
- authoritative 工件链：`pr12m-compile-aware-v3-freeze-20260714/` → calibration → model-freeze
  → final-heldout → `pr12m-compile-aware-v3-replay-v2-20260714/` → report。

PR-12N closure：

- `C2-CLOSE` validated-reduced：H–M 门禁、hash、失败与限制已审计，closure tag 为
  `pr12-validated-reduced`；
- 不能升级 `VALIDATED`：Q≤1024 compile amortization 0/3、硬件 counter unavailable、收益仅限
  部分 regime、尚无真实 BaB/VNN-COMP；
- 不降级 `MECHANISM-ONLY`：non-toy mini-ResNet/Conv E2E Pareto、预算可行性、自动多 regime
  selection 与独立 held-out 已成立；
- PR-13 gate 为 GO/READY，但尚未启动；closure audit 与 Artifact Appendix 分别见
  `gemini_doc/pr12_closure_audit_2026_07_14.md`、
  `gemini_doc/pr12_artifact_appendix_2026_07_14.md`。

## PR-13A Query/State Contract 证据

- `C3-M2` validated foundation：`BoundQuery` 显式覆盖 parent、model/weight/input/spec/split、
  method/stage、α/β/cuts、dtype/device/numeric policy 与 requested outputs，canonical JSON 确定；
- `C3-M3` validated foundation：完整 `QueryCompatibilityKey` 分组；αβ/split 强制
  `alpha_beta_dense_split` capability，不会误选 PR-12 plain-CROWN fused TIR；logical
  `QueryBatch` 拒绝 mixed key，并验证 pack/unpack order/result restoration；
- `C3-M4` validated foundation：state validity 对 graph/kernel/planner/intermediate/α/β/cuts/final
  显式返回 EXACT/CONDITIONAL/WARM_START/INVALIDATE；父 β/final 不可 exact reuse；
- `C3-E2` validated smoke：真实 `solve_bab_mlp` driver 产生 8-query 父子流，8/8 replay、
  max abs diff 0、0 query loss、0 duplicate；
- `C3-L2` validated limitation：工件为 CPU two-ReLU smoke，尚无 dynamic batch、OOM split、
  same-solver multi-backend、non-toy/TTV/tail-latency，不能作为性能或完整 C3 claim；
- 工件：`artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714/`；持续状态：
  `gemini_doc/pr13_execution_status.md`。

## PR-13B Dynamic BatchManager 证据

- `C3-M5` validated foundation：exact-key buckets、budget first-fit、fill/timeout/deadline wakeup、
  deterministic OOM bisection 与 ID-based order restoration；
- `C3-M6` validated foundation：physical αβ executor pack/unpack center/spec/split/α/β，并继续强制
  dense split capability；perturbation 与 execution-options 进入 compatibility；
- `C3-E3` validated smoke：真实 8-query stream 动态形成 3 batches，8/8、max diff 0、0 loss/
  invalid；deadline flush 与 queue-wait 分位数字段存在；
- `C3-E4` validated fault path：显式 OOM fault 触发 8→4+4→2+2+2+2，3 events/splits，最终
  8/8、0 loss；
- `C3-L3` validated limitation：CPU、逻辑 clock、fault OOM；尚无 same-solver live adapter、真实
  GPU OOM、non-toy throughput/TTV；
- 工件：`artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714/`。

## PR-13C Same-Solver Adapter 证据

- `C3-M7` validated foundation：原 `solve_bab_mlp` 继续拥有 branch/heap/node order/termination；
  optional adapter 只替换 single/batched bound-call execution；
- `C3-M8` validated foundation：runtime result 携带真实 α/β tensors，solver 可继续 warm
  start/cache；comparison 强制 state tensor 数值对齐，exact content hash 仅作诊断；
- `C3-E5` validated smoke：αβ steps=3、batch=4 下 original/runtime query IDs 7/7，per-query
  bounds/branch/αβ state 7/7，status/node counters/best bounds 一致，0 loss；
- `C3-E6` validated capability guard：forged plain-CROWN capability 在 αβ physical executor 0 次
  调用时拒绝；alpha-only serial adapter 也与原 solver 对齐；
- `C3-L4` validated limitation：toy CPU smoke，单次 wall time non-authoritative；尚无 non-toy
  fixed-tree/E2E、TTV、真实 GPU OOM/stream、plan/cache ablation；
- 工件：`artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714/`。

## PR-13D/E Reduced GPU 与 Closure 证据

- `C3-E7` validated-reduced：RTX 4060、5 repeats、16-query fixed stream，runtime 相对 per-node
  96.52×，相对 batched original 1.024×，16/16 correctness；
- `C3-E8` validated-reduced：同一 solver hard E2E 16 nodes，runtime 相对 per-node 9.93×，相对
  batched original 0.980×；三 variant status/node count 一致；
- `C3-M9` validated foundation：custom CUDA stream event-only test、dispatch cache 1 miss/4 hits、
  query loss/invalid 为 0；
- `C3-L5` validated limitation：收益主要来自 ordinary batching；easy root 为负收益；
  `compiled_plan_cache_applicable=false`、`pr12_planner_dispatches=0`；
- `C3-L6` validated limitation：chain-CNN 16 nodes，不是 VNN-COMP/non-toy；真实 GPU OOM、
  branch/prune/GPU-active 分解未完成；
- 工件：`artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714/`；closure：
  `gemini_doc/pr13_closure_audit_2026_07_14.md`。

## PR-14A/B 真实 Verifier Coverage 与 No-Go

- `C3-E9` validated coverage：官方 MLP/CNN 与 VNN-COMP ResNet-2B 共 540 个真实
  `compute_bounds`；initial 143/146 region-level eligible，activation-BaB 0/394；
- `C3-M10` validated foundation：external observer 可撤销，on/off status 与 visited domains
  一致；exact Box perturbation 保留 VNNLIB per-element clipped bounds 与 query identity；
- `C3-E10` validated narrow equivalence：simple MLP 的 external replay 与 BoundFlow
  eager/chunked/TVM lower 全部 max diff 0；但 external lower-only、BoundFlow lower+upper，公平
  performance 为 N/A；
- `C3-L7` validated non-toy limitation：ResNet nominal forward 对 ONNX max diff `1.67e-6`，但
  BoundFlow whole-query lower 对 external max diff `796.765`，符号只一致 3/9；
- `C3-CLOSE` validated-no-go：activation route 与 initial whole-query replacement 均未过门禁，
  PR-14C blocked；C3 降级为 C1/C2 基础设施，禁止 verifier acceleration claim；
- 证据：`gemini_doc/pr14a_real_query_coverage_2026_07_19.md`、
  `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`。

### 2026-08-03 RVIR correctness follow-up

上述 `0/394` 与 `796.765` 保留为 PR-14 当时 local/fused 路径的历史结论。独立 RVIR 路线
新增两条不互相替代的证据：

- external intermediate bounds + adaptive slope 使 ResNet initial-CROWN max diff 降为
  `3.09944e-6`、sign 9/9；
- provider-owned external exact-call typed admission 为 394/394，当前 CPU 在线 dispatch 为
  377/377，observer on/off 均访问 380 domains 且 final lower 一致。

fused kernel coverage 仍为 0/394；历史 adapter v1 identity limitation 与当前 CPU-only 边界
均已冻结，因此 C3 只升级 correctness/integration，不升级 performance。

外部审计 minor M4 的后续 v2 artifact 已加入 377 条在线 query 与 377 条 typed execution
record 原文。replay 会独立复核 parent 顺序、query/result accounting、observer projection 与
五层 IR hash；该证据强化可审计性，不把 CPU correctness 升级为 CUDA 或 performance claim。

该段 PR-11 early evidence 当时为专项 21 passed、全量 200 passed/1 skipped；其“Global 与
Memory-Threshold 决策相同”的历史限制已由后续 PR-11E 和 PR-12G 证据分别补充，不能再读作
当前全量状态。PR-12G 收尾全量为 318 passed、1 skipped。

第三切片与 profiler 完成后全量为 208 passed、1 skipped。Global 已在 multi-barrier 合成案例中做出非阈值式
mixed placement，但在真实 held-out workload 上尚无 barrier-level cost/Oracle 证据，C2 状态
仍为 `partial`。有界分层 retry 已把第二 query scale 的最坏 56 次 replay 限制到 5，并在两个
reduced held-out query 上通过 median/feasibility 门禁；证据仍局限于一个 architecture family，
不足以把 C2 整体标记 validated。

有界分层 retry 切片收尾验证：全量 216 passed、1 skipped；Mypy 11 files success；Planner 与
PR-11 脚本逐文件 Pylint 10.00/10；`git diff --check` 通过。

独立 branched-ResNet topology 明确否决了当前 v1 aggregate cost model：feasibility 成立但 regret
门禁失败；同时 evaluator 仍依赖 candidate-specific trace logical bytes，属于 profile-guided replay。
C2 保持 partial，下一实现切片改为 static topology/liveness-aware cost summary。
加入独立 topology contract 后最新全量为 217 passed、1 skipped，profiler Mypy/Pylint 与 diff
check 通过。

Static-v3 已消除 candidate-trace feature 依赖，并显式覆盖 shape/FLOPs/bytes/reuse/batch axes。
3× replicated profiles 共 1,416/1,416 correct；聚合后三组 held-out 全部通过 feasibility/median
门禁，p90/max 最坏为 2.377×/3.160×。Production candidate foundation 与真实 OOM 3/3 已成立；
C2 标记 validated-reduced，不能解释为论文级 complete。

### 2026-08-04 Production Schedule + Memory P0

- 两个 residual workload × 四 backend 的 current-code structural regeneration 共 8/8 case；
  每个 10-op Bound graph 被完整 region partition 覆盖，arena budget/allocate/free 与 batch/launch
  均进入 Schedule IR；
- 8/8 case 均无 `MaterializeAction`，batch/storage candidate 数均为 1；
- 64/512 MiB 下 PlanInstance hash 不同，但 decision signature 8/8 相同；峰值减 1 byte 时
  selector 以 `memory_budget_exceeded` fail closed，只证明预算约束有效，不证明预算优化有效；
- 51/51 VNN-COMP ResNet activation calls 五层 hash 可精确重编译，但都是单个 external op/
  launch，`semantics_owner=external_verifier`；
- P0 判定 `NO_GO`。下一假设必须先实现 real-network native Bound IR；不得把 typed wrapper、
  hash 变化或 reduced structural coverage 写成 production memory/performance claim。

### 2026-08-04 Native Real-Network IR v1

- 固定 VNN-COMP ResNet2B prop0：17 个 Primal ops lower 为 21 个 native Bound ops、21 个
  Task units 与 21 次 Schedule launch；Bound/Task external-call count 均为 0；
- 6 组 external intermediate bounds 可 safe-load，并以 aggregate digest 进入 ReLU state version
  与 Plan provenance；同形状不同内容会改变全链 IR identity；
- 五层 hash fresh replay 完全一致；native final lower 对 αβ-CROWN oracle max diff
  `7.152557373046875e-07`、sign 9/9；
- 该 evidence 将 C1/Task/Schedule 的 real-network compiler ownership 升为 validated-reduced，
  但不证明完整 native verifier：forward intermediate bounds 仍来自 external provider；
- Plan 当前只有 1 storage、1 batch、0 materialization，`performance_claimed=false`。C2/ASPLOS
  performance No-Go 不变；下一门禁是 real-graph multi-plan + budget switch。

### 2026-08-04 Native Real-Network Memory Plans v1

- 同一固定 ResNet Bound hash `16e27f31...80fb` 与 PlanTemplate hash `359ee68f...43f3`
  包含 retain-all/lifetime-reuse 两个 storage candidate；高/低预算选择不同 PlanInstance 与
  Schedule，而不是只改变 query identity；
- retain-all Schedule/runtime peak 为 `1,860,912` bytes；lifetime-reuse 为 `442,656` bytes，
  有 386 对 lifetime-safe physical alias、85 个 final-task 前 runtime release；`442,655` bytes
  以 `memory_budget_exceeded` fail closed；
- 两计划 lower/upper bitwise 一致，对 external lower max diff
  `7.152557373046875e-07`、sign 9/9；NRIR-1 原 artifact 五层 hash replay 不变；
- 该证据把 C2/Schedule 的 real-graph storage decision mechanism 升为 validated-reduced，但
  logical arena 与 reference release 不能写成 CUDA peak-memory reduction、OOM rescue、latency
  或 speedup；artifact 明确 `performance_claimed=false`；
- representation audit 发现 Plan metadata 尚不能驱动 runtime structured rewrite，
  `MaterializeAction` 也尚无数值转换效果；因此 0 real-graph materialization 与单 full batch
  仍是明确缺口。

### 2026-08-04 Native CUDA Memory Protocol v1

- `C2-M-NRIR3` validated mechanism：retain/reuse 的 fresh-process CUDA runner、prepared lower-only
  timing、5×5×20 重复矩阵、allocated/reserved counter、alternating order、raw/summary/manifest
  与 semantic replay 已实现；
- `C2-G-NRIR3` frozen gates：模型/intermediate-bound digest、worker PID 唯一、稳定环境、同一
  Bound/PlanTemplate、result identity、reuse allocated delta ≥20%、latency ratio ≤1.20×；
- `C2-E-NRIR3` environment evidence：PyTorch `2.12.1+cu132` / CUDA build 13.2，但
  `cuda_available=false`、device count 0、`nvidia-smi` driver failure；probe exit 2，replay exit 0，
  benchmark generate 在创建 artifact 前 exit 2；
- `C2-L-NRIR3` hard limitation：0 measured worker rows，故没有 CUDA peak-memory、latency、
  OOM rescue、Pareto 或 speedup claim；`performance_claimed=false`；
- 工件：`artifacts/native-real-network-cuda-memory-protocol/environment-unavailable-20260804/`；
  验证：聚焦 17 passed，全量 `484 passed, 37 skipped`，静态门禁全过；
- 下一缺口是 representation semantic binding：Plan selection 与 Schedule materialization 必须
  改变实际 Bound/backend execution，不能由 metadata/hash 冒充 C1/C2 系统收益。

### 2026-08-04 Native Representation Semantic Binding v1

- `C1-M-NRIR4` validated mechanism：source Plan 的全局 dense/structured-affine policy 由
  fail-closed binder 转成实际 execution Bound program；structured 路径在固定 ResNet 上插入
  14 cast + 14 materialize，并为 rewritten graph 重建独立 Plan/Task/Schedule stack；
- `C1-G-NRIR4` ownership gate：28/28 selected transitions 与 source Schedule action、execution
  Bound op 一一对应；49/49 execution Bound ops 均进入 Task 与 Launch，tampered action/hash/event/
  mixed policy 被拒绝；
- `C1-E-NRIR4` real-network semantics：dense 与 structured lower 最大差
  `9.5367431640625e-07`；二者均匹配冻结 external lower，sign 9/9；artifact fresh semantic replay
  与 digest gate 通过；
- `C2-M-NRIR4` selector mechanism：storage-compatible prefix pruning 避免 21-region 双 policy 的
  指数混合枚举；高预算选 dense/retain-all，`442,656` bytes 选
  structured-affine/lifetime-reuse，`442,655` bytes fail closed；
- `C1/C2-L-NRIR4` hard limitation：`DenseLinearOperator` 仍存 dense tensor，structured storage
  binding 至少保留 dense logical bytes；storage coupling 不能写成 representation compression。
  `performance_claimed=false`，无 memory/latency/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-real-network-representation-binding/vnncomp21-resnet2b-prop0-cpu-v1/`；
  下一缺口为 real-network sliced batch execution，必须由 batch decision 驱动真实 Task/Schedule
  slicing 与 query accounting。

### 2026-08-04 Native Real-Network Sliced Batch Execution v1

- `C2/C3-M-NRIR5` validated-reduced mechanism：query-time
  `max_spec_batch_size` 进入 Plan selection/provenance；full 9-spec candidate 与 size-3 candidate
  选择不同 PlanInstance/Schedule，默认 context 保持历史 identity；
- `C2/C3-G-NRIR5` ownership gate：source spec Schedule ranges 必须连续、无重叠、完整覆盖；
  `[0,3)/[3,6)/[6,9)` 分别绑定独立 child Bound/Plan/Task/Schedule hash、query ID 与 execution
  trace，3 个 child 共 63 Task/Launch；同步修改 range/query/digest 仍被结构门禁拒绝；
- `C2/C3-E-NRIR5` real-network semantics：full/sliced lower max diff
  `1.9073486328125e-06`；full/external `7.152557373046875e-07`；sliced/external
  `1.9073486328125e-06`，均 allclose、sign 9/9；artifact generate/replay exit 0；
- `C2/C3-L-NRIR5` hard limitation：v1 只实现 spec axis，child 顺序执行，source controller
  storage 为完整 ledger；domain/sample、representation × batch composition、physical allocator、
  latency、CUDA/OOM/Pareto/speedup 均未证明，`performance_claimed=false`；
- 工件：`artifacts/native-real-network-sliced-batch/vnncomp21-resnet2b-prop0-cpu-v1/`；
  下一缺口是 representation × batch 联合 policy execution，不能把两条独立 mechanism 自动
  组合成全局 Planner claim。
- 验证：新旧 native/Plan/Task/Schedule 聚焦 `89 passed`；全量 `508 passed, 37 skipped`；
  Black/Mypy/Pylint 10.00/10/diff check 通过。

### 2026-08-04 Native Representation × Batch Composition v1

- `C2-M-NRIR6` validated-reduced joint mechanism：同一 source template 同时包含
  representation/storage 与 spec-batch candidates；budget × max spec 由一个 selector 选择
  dense/structured × full/sliced 四个 PlanInstance/Schedule；
- `C2-G-NRIR6` policy propagation gate：source storage/representation 显式成为 child required
  policy，并进入 provenance/hash；child shape 变化不能导致重新选 policy，tamper fail closed；
- `C1/C2-E-NRIR6` ownership：四组合 child op/task/launch=`21/63/49/147`；structured 保留
  28 transition/49-op execution binding，sliced 保留 `[0,3)/[3,6)/[6,9)`；四 source
  PlanInstance/Schedule identity distinct；
- `C2/C3-E-NRIR6` semantics：四路径对 external lower max diff 分别为 `7.15e-7/1.91e-6/
  9.54e-7/1.67e-6`，均 allclose、sign 9/9；artifact generate/replay exit 0；
- `C2/C3-L-NRIR6` hard limitation：structured dense-equivalent、child sequential、无跨 query/
  domain physical batching/cache baseline；无 memory/latency/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-real-network-joint-policy/vnncomp21-resnet2b-prop0-cpu-v1/`；聚焦
  `103 passed`、全量 `522 passed, 37 skipped`、静态门禁全过；下一缺口为真实 query stream。

### 2026-08-04 Native Repeated-Query Batching and Cache v1

- `C3-M-NRIR7` validated-reduced query mechanism：frozen ResNet 的 9 个不同 property objectives
  具有独立 query ID/objective digest/range；packed source 按 size-3 实际执行 3 child，serial
  same-policy baseline 执行 9 child；
- `C2/C3-G-NRIR7` exact cache gate：workload/input/intermediate-bound/state、ordered query content、
  budget/policy/batch config 全部进入 key；first miss/second hit，objective/order/state probe 均 miss；
- `C3-E-NRIR7` lineage/semantics：9/9 per-query restore；packed/cache hit max diff 0；packed/serial
  `3.2186508178710938e-06`；packed/external `1.9073486328125e-06`；serial/external
  `3.2186508178710938e-06`；均 allclose、sign 9/9；
- `C3-L-NRIR7` hard limitation：同一 input domain 的 property queries，不是 BaB parent/child
  domain stream；3 vs 9 仅机制计数，无 timing/memory/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-real-network-repeated-query/vnncomp21-resnet2b-prop0-cpu-v1/`；聚焦
  `121 passed`、全量 `540 passed, 37 skipped`、静态门禁全过；下一缺口为 domain state validity。

### 2026-08-04 Native BaB Input-Domain Batching v1

- `C3-M-NRIR8` validated-reduced domain mechanism：固定 ResNet root box 三层确定性二分为 8 个
  不同 leaf queries；每个 leaf/parent box、tree depth/branch 与 result lineage 显式冻结；
- `C3-G-NRIR8` state-validity gate：每个 leaf 独立重算 IBP exact state，8 个 state hash 全不同；
  parent state 只允许 `warm_start_only`，任何 promotion、range/state/lineage 篡改 fail closed；
- `C2/C3-E-NRIR8` execution：full-domain Plan 执行 1 child，domain-size-4 Plan 执行 2 child，
  same-policy serial 执行 8 child；packed/full/serial 8×1 lower/upper bitwise equal，8/8 restore；
- `C3-L-NRIR8` hard limitation：input-box branch 不是 ReLU/β branch-and-bound；无 queue、prune、
  termination、verified verdict 或 timing/memory/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-real-network-domain-batch/vnncomp21-resnet2b-prop0-cpu-v1/`；下一缺口
  为 native ReLU-split state/queue/control flow；聚焦 `19 passed`、全量
  `559 passed, 37 skipped`、静态门禁全过。

### 2026-08-04 Native ReLU-Split BaB Queue v1

- `C3-M-NRIR9` validated-reduced split/queue mechanism：每个 ReLU split 是 native Bound graph
  的 typed int8 input，并进入 Plan workload/capability、Task partition、Schedule launch 与五层 hash；
- `C3-G-NRIR9` ownership/validity gate：split key/shape/dtype/device/range/content、preactivation
  active/inactive feasibility、node parent/branch/order、IR stack link与同步重哈希后的 artifact
  tamper 均 fail closed；local forward 与 external verifier provenance 分离；
- `C3-E-NRIR9` real-network control flow：固定 ResNet 形成 7 个节点/3 expand/4 frontier 的
  best-first bounded queue；packed-4/serial-1 实际执行 3/7 个 native stacks，lower/upper max diff
  `1.8310546875e-04/1.220703125e-04`，queue signature 与 split identity 相同；
- `C3-S-NRIR9` state rule：child 只继承 discrete split；每个 child batch独立重算 IBP exact state，
  `parent_state_consumed_as_exact=false`。packed/serial CPU batch layout 的 exact tensor hash 可不同，
  因此只按冻结数值容差声明语义一致，不伪称 bitwise；
- `C3-L-NRIR9` hard limitation：plain CROWN bounded run，明确 `budget_exhausted` 与
  `property_status=not_claimed`；无 α/β optimization、完整 verifier verdict 或 timing/memory/CUDA/
  OOM/Pareto/speedup claim；
- 工件：`artifacts/native-real-network-relu-split-bab/vnncomp21-resnet2b-prop0-cpu-v1/`；下一缺口
  为 native α/β optimization state、beta constraint 与 warm-start validity；聚焦 `68 passed`、
  全量 `577 passed, 37 skipped`，静态门禁与 fresh replay 全过。

### 2026-08-04 Native Alpha/Beta Optimization State v1

- `C3-M-NRIR10` validated-reduced state mechanism：固定 ResNet 6 个 ReLU 各有 typed split/alpha/
  beta inputs；19 个 graph inputs、6 optimized ReLU ops 进入 Bound/Plan/Task/Schedule 与五层 hash；
- `C3-G-NRIR10` state/warm-start gate：scope 绑定 model/input/objective/intermediate/split/policy；
  exact same scope 可 exact reuse，monotonic split refinement 只能 initialization，reversal/removal 与
  semantic drift fail closed；
- `C1/C3-E-NRIR10` beta execution：native lower-dual 实际消费 `-beta*split`，对 legacy αβ oracle
  lower/upper max diff=`0.0/0.0`；beta sum=`0.04999999701976776`，zero-beta 对照 lower 差
  `0.34039306640625`，证明不是 metadata-only；
- `C3-L-NRIR10` hard limitation：冻结 state 已编译/执行，但 Adam iteration/gradient/update 仍
  runtime-owned；无完整 BaB/property verdict 或 timing/memory/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-alpha-beta-optimization-state/vnncomp21-resnet2b-prop0-cpu-v1/`；聚焦
  `50 passed`、全量 `591 passed, 37 skipped`、静态门禁与 fresh replay 全过；下一缺口为
  optimizer-step Task/Schedule control。

### 2026-08-04 Native Alpha/Beta Optimizer-Step Schedule v1

- `C3-M-NRIR11` validated-reduced control mechanism：Optimizer Plan 绑定 NRIR-10 的 10 个 source
  compiler hash、state/scope/policy/ReLU keys；固定 step 被 lower 为 evaluate/reduce/backward/
  Adam/project/select-best Task 与一一对应 Schedule actions；
- `C3-G-NRIR11` control/trace gate：task dependency、state version、action order、逐 value hash chain、
  evaluation bound/metric/state、gradient、projection、best iteration 与 runtime scope 篡改均
  fail closed；warm start 复用 NRIR-10 exact/refinement/rejected classifier；
- `C3-E-NRIR11` fixed ResNet execution：1-step program 为 8 Task/Action、2 evaluations、1 backward/
  update/project；alpha/beta gradient L1=`169.23175295069814/12.862210273742676`。Schedule/legacy/
  selected-state native compiler lower/upper max diff 均为 `0.0`，state hash 与 legacy 相同；
- `C3-L-NRIR11` hard limitation：fixed-step static unroll，不含 dynamic early stop；尚未接入多节点
  ReLU-split queue，也无完整 termination/property verdict 或 timing/memory/CUDA/OOM/Pareto/speedup；
- 工件：`artifacts/native-alpha-beta-optimizer-schedule/vnncomp21-resnet2b-prop0-cpu-v1/`；replay
  hash=`31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`；聚焦 `35 passed`、
  全量 `612 passed, 37 skipped`、静态门禁全过；下一缺口为 optimizer Schedule × ReLU-split
  BaB queue integration。

### 2026-08-04 Native Optimized ReLU-Split BaB v1

- `C3-M-NRIR12` validated-reduced integrated mechanism：每个 queue node batch 执行 NRIR-11
  optimizer Plan/Task/Schedule，再执行 NRIR-10 selected-state native Bound/Plan/Task/Schedule；fixed
  ResNet 每 stack 为 8 optimizer actions + 21 native tasks；
- `C3-G-NRIR12` parent/state gate：child parent states 按 batch layout 重组并重建 scope，只允许
  monotonic-refinement initialization；parent/selected hash、optimizer/native IR hash、action count、
  gradient、projection、re-execution 与 queue lineage 篡改 fail closed；
- `C3-E-NRIR12` real-network control：7 nodes/3 expands/4 frontier，packed/serial 3/7 stacks；bounds
  max diff=`1.220703125e-04/1.8310546875e-04`，alpha/beta tensor max diff=
  `4.172325134277344e-07/7.450580596923828e-09`；active child beta gradients 非零，selected native
  re-execution diff 为 0；
- `C3-S-NRIR12` numeric disclosure：packed/serial stable scope fields 与 split tensors 相同，但
  batch-layout intermediate hash 与 exact selected state hash 不相等；只按冻结 tensor/bound tolerance
  声明一致，不伪称 bitwise；
- `C3-L-NRIR12` hard limitation：fixed-step、7-node bounded run，明确 budget-exhausted/not-claimed；
  无 complete termination/property verdict 或 timing/memory/CUDA/OOM/Pareto/speedup；
- 工件：`artifacts/native-optimized-relu-split-bab/vnncomp21-resnet2b-prop0-cpu-v1/`；replay
  hash=`e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`；聚焦 `18 passed`、
  全量 `630 passed, 37 skipped`；下一缺口为 sound property termination/verdict。

### 2026-08-04 Native Property Termination and Verdict v1

- `C3-M-NRIR13` three-state mechanism：单标量 `C f(x) >= threshold` 有独立
  verified/unsafe/unknown trace，绑定 immutable optimized queue hash、objective、threshold 与
  resolved/unresolved leaf sets；
- `C3-G-NRIR13` soundness gate：verified 仅接受 `lower >= threshold` 的 sound prune
  closure；budget/depth/unproven prune 一律 unknown；unsafe 必须重执行 input-box、ReLU split
  path、primal Task IR 和 strict objective violation；同步重哈希篡改 fail closed；
- `C1/C3-E-NRIR13` concrete execution：toy 分别产生 verified/unsafe/unknown；非 root
  witness 实际检查 active ReLU margin。固定 ResNet 完整 primal center objective=
  `0.8564349412918091`，7 nodes/4 frontier 输出 `unknown/node_budget_frontier_open`；
- `C3-L-NRIR13` hard limitation：candidate discovery 由 caller 提供；只支持单标量 property；
  无 timeout/dynamic early stop、real complete closure 或 timing/memory/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/native-property-verdict/vnncomp21-resnet2b-prop0-cpu-v1/`；replay
  hash=`9e3dceed23c8759c910938ba7c9f84caaeb949c8f19b72fab104ce4e1b733405`；聚焦 `19 passed`、
  全量 `649 passed, 37 skipped`；该阶段当时的 complete verifier query 缺口已由 NRIR-14
  执行，但 NRIR-13 单独 claim 不升级。

### 2026-08-04 Complete Verifier Query v1

- `C3-M-NRIR14` multi-clause mechanism：conjunction query 按 ascending clause index 串联
  deterministic PGD candidate search、optimized native queue 与 sound verdict；全部 verified
  才 verified，replayed violation 立即 unsafe，其余 unresolved/pending 为 unknown；
- `C3-G-NRIR14` sound control gate：candidate `not_found` 明确不是 proof；found candidate
  必须 concrete primal replay；unsafe suffix、deadline pending、objective/policy/config 与三层
  trace hash 全部 fail closed；
- `C1/C3-E-NRIR14` real/toy execution：toy 分别闭合 verified/unsafe/attack-not-found unknown/
  deadline unknown。固定 ResNet 九个真实 clauses 全部执行，candidate best objectives 均为正，
  但 scalarized lower bounds 均为负，故 9/9 unresolved、整体 sound unknown；
- `C3-S-NRIR14` scale-aware replay：execution 仍以 `allclose(atol=2e-6, rtol=2e-6)` 守护；
  serialized diff 另设 `2e-3` ceiling 并拒绝 non-finite。固定 clause 6 合法 max diff=
  `6.103515625e-05`；
- `C3-L-NRIR14` hard limitation：center-start single-restart search、cooperative deadline、
  single box/conjunction；固定真实 property 未闭合，无 dynamic optimizer/branch heuristic，
  无 timing/memory/CUDA/OOM/Pareto/speedup claim；
- 工件：`artifacts/complete-verifier-query/vnncomp21-resnet2b-prop0-cpu-v1/`；replay
  hash=`d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`；相关
  `39 passed`、全量 `670 passed, 37 skipped`；下一缺口为公平 end-to-end phase/tightness
  baseline，再按证据推进 bound strength 与执行优化。

### 2026-08-04 End-to-End Tightness and Performance Baseline v1

- `C3-M-NRIR15` external-semantics mechanism：external intermediate ReLU intervals 与 typed
  provenance 进入 optimizer Plan/source compiler/state scope/selected native stack/queue child；
  child external interval 叠加 per-node split，parent state 保持 initialization-only；
- `C3-G-NRIR15` fail-closed gate：missing/wrong provenance、key/order/shape/dtype/device/finite/
  lower≤upper、未施加 split 的 child preactivation 与 artifact semantic/timing/claim tamper 均拒绝；
  默认 constant policy canonical hash 不漂移；
- `C1/C3-E-NRIR15` tightness：fixed ResNet local 为 0/9；external-adaptive 为 6/9 verified，
  仅 clauses `0/2/4` unknown。九个 lower 对 external initial 无退化，最大改善
  `0.0072252750`、sign `9/9`；
- `C3-D-NRIR15` phase diagnosis：三组轮换 clause-0 audit queue median 为 local/external-
  constant/external-adaptive `6.7178/6.7969/6.7317 s`，candidate/verdict 约 `3.6/3.9 ms`；
  只定位 fixed compile/hash/re-execution blocker，不声称 production latency 或 speedup；
- `C3-L-NRIR15` hard limitation：单 fixed ResNet、CPU audit path、6/9 而非完整证明；CUDA、
  multi-workload、production path 与竞品 E2E 均 pending；
- 工件：`artifacts/end-to-end-tightness-performance/vnncomp21-resnet2b-prop0-cpu-v1/`；fresh
  replay hash=`14c3b9dc2e5376156be1f33f3e8804ec21f60e11096bd3bdc95225b7e1474376`；下一门禁为
  prepared production fast path，随后才处理 hard clauses 的 branching/tightness。

### 2026-08-04 Prepared Production Fast Path v1

- `C2/C3-M-NRIR16` prepared mechanism：九个 root objectives 各自冻结 exact optimizer
  Plan/Task/Schedule、source compiler hashes 与 semantic scope；steady-state 仍消费全部 optimizer
  actions，但不重做静态 validation、audit hash chain 或 selected-native validation execution；
- `C3-G-NRIR16` identity gate：program/module/input/objective/intermediate source/scope 任一漂移
  fail closed；production trace 明示 `audit_hash_chain_constructed=false` 与
  `selected_native_reexecution=false`，不能冒充 audit evidence；
- `C1/C3-E-NRIR16` semantics：prepared/audit lower max diff=`1.90735e-6`、candidate exact、
  status exact，fixed ResNet 仍为 6 verified / clauses `0/2/4` unknown；
- `C2/C3-D-NRIR16` three-group CPU diagnosis：audit median=`59.078 s`，prepared warm median=
  `110.950 ms`，内部 evidence-overhead ratio=`532.47×`；cold prepare+first=`16.139 s`，
  retained payload=`2,076,372 B`，两项成本均显式披露；
- `C3-L-NRIR16` hard limitation：root-only exact capsules、单 fixed ResNet CPU；audit-removal ratio
  不是 competitor/verifier speedup，无 child queue、CUDA、多 workload 或 complete closure；
- 工件：`artifacts/prepared-production-fast-path/vnncomp21-resnet2b-prop0-cpu-v1/`；fresh replay
  hash=`e14fcd62b322c0bc60d45c726cf94a7aa6cfb8d7aa3212662d08996db169b6b2`；全量
  `698 passed, 37 skipped`；下一门禁为 hard-clause branching/stronger-bound。

### 2026-08-04 Hard-Clause Objective Branching v1

- `C2/C3-M-NRIR17` branch IR：candidate enumeration、inactive/active child materialization、
  child-bound evaluation、worst-child reduction、selection 均进入 Plan/Task/Schedule 与 runtime；
- `C3-G-NRIR17` exact gate：objective/split/selected state/scope/policy/candidate/score/schedule
  identity 与 argmax selection 均 fail closed；旧 widest 与 NRIR-16 replay 不漂移；
- `C1/C3-E-NRIR17` tightness：同 7-node/depth-2/25-step 预算下，clauses `0/2/4` 的 objective
  worst leaf 相对 widest 改善 `0.120752/0.071564/0.057901`；
- `C3-L-NRIR17` hard limitation：三组 objective terminal leaves 全部仍为负，property status
  `unknown`，整体仍只 6/9；单 fixed ResNet CPU、single-run audit timing，无 competitor/GPU claim；
- 工件：`artifacts/hard-clause-objective-branching/vnncomp21-resnet2b-prop0-cpu-v1/`；fresh
  replay hash=`1193bee8817e4acc9ec33f8ddadc00a671d0ac3c9411f14f62978eb5ab1a95bd`；全量
  `707 passed, 37 skipped`。下一门禁为 multi-workload/device/competitor E2E 与 stronger-bound。

### 2026-08-04 Multiworkload Competitor E2E Baseline v1

- `C1/C3-M-NRIR18` ingest/control mechanism：原生 VNNLIB box + single-linear-unsafe-disjunct
  frontend 编译 immutable Query IR；三 workload selection 再编译为 21-task、6-fresh-process 的
  workload Plan/Task/Schedule；source/policy/device/timeout/hash 全显式；
- `C3-G-NRIR18` fail-closed gate：source digest、CSV ordinal、ONNX shape/op inventory、完整 input
  box、连续 X/Y identity、single inequality DNF、Plan/Task/Schedule linkage、worker result/log digest
  和六组合 coverage 任一漂移均拒绝；
- `C1-E-NRIR18` parser parity：MNISTFC 784-input、ResNet2B/OVAL21 3072-input 的 lower/upper、
  九条 C/rhs 与固定 αβ-CROWN parser 逐字段一致；同时修复 flatten/reshape-first BoxPerturbation
  shape-transform/trace 缺口；
- `C3-E-NRIR18` real execution：BoundFlow 对 MNISTFC/ResNet2B/OVAL21 均 sound unknown；
  αβ-CROWN 为 verified/unknown/verified。BoundFlow unresolved/pending 分别为 `3/0`、`2/7`、
  `1/0`；ResNet 两个 native root lower=`-543.717/-789.331`；
- `C3-D-NRIR18` CPU diagnosis：fresh-process E2E native/competitor 分别为
  `38.644/4.312 s`、`66.910/64.198 s`、`31.498/4.527 s`；算法能力不同且均为单次，
  `performance_claimed=false`，不得计算 speedup；
- `C3-L-NRIR18` hard limitation：三 workload 仍不足以形成完整 benchmark suite；BoundFlow
  0/3 complete verified，CUDA driver/device 不可用，未做 5-repeat 公平性能矩阵；VNNLIB v1
  不支持 general multi-inequality disjunct；
- 工件：`artifacts/multiworkload-competitor-e2e/vnncomp21-three-topology-cpu-v1/`；fresh replay
  hash=`473b287bb88e4c52426b405aeb4164aa72a98d7b1bbd74c00471fe1d1451deb0`；focused
  `16 passed`、全量 `723 passed, 37 skipped`。下一门禁为 native intermediate-bound refinement。

### 2026-08-04 Native Intermediate-Bound Refinement v1

- `C1/C2-M-NRIR19` refinement IR：top ambiguous-width target、selected plain-CROWN backward、
  intersection、forward propagation 和 emit 均进入 Plan/Task/Schedule；任意中间 rank 与
  flattened selected objective 受支持；
- `C3-G-NRIR19` native provenance gate：primal/input/split/initial bounds/policy/target/action trace
  全哈希绑定，`native_refined` 与 `external_verifier` 分离，schema/order/tamper fail closed；
- `C1/C3-E-NRIR19` real tightness：MNISTFC unresolved `3→1`、关闭 `3/7`、nodes `31→21`；
  OVAL21 `unknown→verified`、关闭 `8`、nodes `15→11`；ResNet 两 root lower 改善
  `+70.496/+160.551`，但状态仍 unknown；
- `C3-L-NRIR19` hard limitation：仅 OVAL21 1/3 complete verified；ResNet 仍有 2 unresolved +
  7 pending，root-global width policy 不是 objective-directed/per-child refinement；CUDA 和公平重复
  性能矩阵未执行，`performance_claimed=false`；
- 工件：`artifacts/native-intermediate-refinement/vnncomp21-three-topology-cpu-v1/`；fresh replay
  hash=`f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；focused
  `9 passed`、全量 `732 passed, 37 skipped`。当时的下一门禁 objective-directed target
  selection 已由下节 NRIR-20 完成。

### 2026-08-04 Objective-Directed Intermediate Refinement v1

- `C1/C2-M-NRIR20` objective-aware refinement IR：single-clause objective hash、逐 ReLU
  CROWN coefficient influence、influence×width target score 与真实 selection dependency 已进入
  Plan/Task/Schedule；默认 width payload/hash 保持兼容；
- `C3-G-NRIR20` fail-closed gate：objective policy 与 scalar objective 必须同时出现；多子句、
  nonfinite、shape/dtype/device/influence/score/task dependency 漂移均拒绝；selection heuristic
  不改变 selected-CROWN/intersection soundness；
- `C1/C3-E-NRIR20` fixed root tightness：ResNet clauses `0/1` same-budget 96-target 对照的
  target overlap=`16/96`、`27/96`，objective 相对 width 的 root lower 改善
  `+55.928741/+26.228943`；
- `C3-L-NRIR20` hard limitation：最终 root lower 仍为 `-417.292480/-602.551392`，没有 property
  closure；root-global、CPU 单次、无 per-child/CUDA/竞品/重复性能，`performance_claimed=false`；
- 工件：
  `artifacts/objective-directed-intermediate-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`；
  fresh semantic replay hash=
  `8fce1c7c3e5c63adb14a7ab5b9f23407e4a7a1406353750e4f150ee745b4e88e`；focused
  `16 passed`、全量 `739 passed, 37 skipped`。下一门禁为 per-child exact-state refinement。

### 2026-08-04 Per-Child Objective Refinement v1

- `C1/C2-M-NRIR21` per-node refinement ownership：每个 queue node 依据 exact split 独立编译/
  执行 objective-directed refinement Plan/Task/Schedule，再拼接为 optimizer batch；默认关闭的
  旧 payload/hash 结构不增加字段；
- `C3-G-NRIR21` lineage gate：node/evaluation/refinement 一一绑定 split、Plan/Task/Schedule、
  semantic trace、initial/final intermediate hash 与 target count；parent alpha/beta 只作 warm
  initialization，parent refinement 不作为 child exact result；tamper fail closed；
- `C1/C3-E-NRIR21` fixed bounded-tree result：clauses `0/1` root lower exact matching，但
  per-child worst depth-2 leaf 相对 root-global 退化 `-0.847961/-0.936646`；
- `C3-L-NRIR21` closure=`VALIDATED-NO-GO`：该 shortlist recomputation 不形成 tightness claim；
  单模型、两 clause、CPU、7 nodes，无 property closure/CUDA/competitor/repeated-performance，
  `performance_claimed=false`；下一门禁为 ancestral-constraint carry-forward refinement；
- 工件：
  `artifacts/per-child-objective-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`；fresh replay、
  focused/full/static 的最终 hash/count 见 NRIR-21 changelog。

### 2026-08-04 Ancestral-Constraint Refinement v1

- `C1/C2-M-NRIR22` typed source ownership：child refinement Plan 绑定 parent final/Plan/semantic
  trace，materialize Task/Schedule 显式消费 source constraint；compiler 只接受 validated parent
  execution，不接受裸 mapping；
- `C3-G-NRIR22` sound lineage gate：local exact-split→constrained initial→final 双重单调；root 无
  source、child source 一一指向已完成 parent；consumption=`sound_constraint_only`，exact reuse=false；
  source/task/queue/tamper fail closed；
- `C1/C3-E-NRIR22` fixed-tree tightness：clauses `0/1` ancestral worst leaf 相对 independent
  提升 `+73.615173/+75.022095`，相对 root-global 提升 `+72.767212/+74.085449`，root exact；
- `C3-L-NRIR22` hard limitation：所有 worst leaf 仍为负，单 ResNet/两 clause/CPU/7 nodes/depth 2；
  无 complete closure、CUDA、competitor parity、重复性能或 ASPLOS-ready，`performance_claimed=false`；
- 工件：`artifacts/ancestral-constraint-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`；
  generate hash=`72c0c2a66b82cea425bf7486817c0ce39ae186ef2961cc1271acb31cb7a31b6f`；
  最终 replay/full/static 见 NRIR-22 changelog。下一门禁为 hard-clause convergence expansion。

### 2026-08-04 External-Seeded Ancestral Refinement v1

- `C1/C2-M-NRIR23` typed external seed：external provider/ownership、primal/input、ordered bounds、
  effective local-intersection constraints 与 source artifact/model/property/objective-set 均进入 seed
  IR；refinement Plan/Task/Schedule/action trace 显式消费 seed；
- `C3-G-NRIR23` source gate：raw external 与 local forward 必须可行相交；root seed 与 native
  parent source 互斥；child parent final/Plan/semantic trace、queue record 与 execution 逐项一致，
  provenance/content/lineage tamper fail closed；
- `C1/C3-E-NRIR23` fixed hard-clause tightness：clauses `0/2/4` ancestral worst leaf 相对 external
  baseline 改善 `+0.001512/+0.001133/+0.000534`，相对 seeded root-global 为
  `+0.000823/+0.000004/0`；
- `C3-L-NRIR23` hard limitation：三条 terminal worst leaf 仍为
  `-0.318287/-0.425477/-0.504142`；单 ResNet、CPU、7 nodes/depth 2、无 complete closure、
  CUDA、multi-workload、competitor parity、重复性能或 ASPLOS-ready，`performance_claimed=false`；
- 工件：
  `artifacts/external-seeded-ancestral-refinement/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；
  semantic replay hash=
  `9f52b99a74dab448626061f5b8f060f3b8c43b6c03f6deb0899d9fe91883d9f7`；全量
  `766 passed, 37 skipped`。下一门禁为 external-seeded hard-clause depth/node convergence。

### NRIR-24：External-Seeded Depth/Node Convergence v1

- `C1/C3-E-NRIR24` convergence：固定 external seed、ancestral carry、objective branch、25-step
  optimizer 与 16-target/ReLU 单 pass refinement，只增加 `7/15/31 nodes × depth 2/3/4`；clauses
  `0/2/4` worst terminal lower 曲线为
  `-0.318287→-0.299506→-0.282360`、
  `-0.425477→-0.413456→-0.401845`、
  `-0.504142→-0.479104→-0.459939`；
- `C3-G-NRIR24` replay/nesting gate：九个 fresh-process checkpoint shards 全部 semantic replay；
  `7⊂15⊂31` 按 split-state logical domain、parent lineage、branch selection 与 normalized
  refinement semantics 校验，最大公共 lower 漂移 `1.13249e-6 ≤ 1e-5`；
- `C3-L-NRIR24` hard limitation：depth-4 proof deficits 仍为
  `0.282360/0.401845/0.459939`，无 fixed-tree closure；单 ResNet/三 clause/CPU，无 complete
  property、CUDA、multi-workload、competitor parity、repeated performance 或 ASPLOS-ready，
  `performance_claimed=false`；
- 工件：
  `artifacts/external-seeded-depth-node-convergence/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；
  semantic replay hash=
  `db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`。本阶段
  `VALIDATED-REDUCED`；下一门禁为 dynamic ancestral refinement budget/multi-pass。

### NRIR-25：Dynamic Ancestral Refinement Budget v1

- `C1/C2-M-NRIR25` typed budget ownership：新增 parent-lower generated-group budget policy/decision；
  policy、group、node/split/depth、parent risk、assigned cap 与 conservation totals stable-hashed；派生
  cap 精确进入逐 node refinement Plan policy，Task/Schedule/execution/queue trace 交叉绑定；
- `C3-G-NRIR25` conservation/replay gate：root/single/tie 使用 16，two-parent risk group 使用 24/8，
  每组与全树 planned cap 精确守恒；六个 fresh-process shards、decision→Plan linkage、aggregate
  comparison 与 source-to-semantics replay fail closed；
- `C1/C3-E-NRIR25` same-planned-cap tightness：clauses `0/2/4` 的 fixed16→dynamic8_24 worst lower
  delta=`+0.0003859997/+0.0002329946/+0.0002717972`，三条均不弱且严格改善；每 mode planned
  cap=`496`、actual selected=`2976`；
- `C3-L-NRIR25` hard limitation：dynamic proof deficits 仍为
  `0.2819737196/0.4016119838/0.4596676826`，三条 bounded tree 均 unknown；单 ResNet/三 clause/
  CPU/单 pass，无 complete property、CUDA、multi-workload、competitor、performance 或
  ASPLOS-ready，`performance_claimed=false`；
- 工件：
  `artifacts/dynamic-ancestral-refinement-budget/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；
  evidence hash=`85d9f274c6e17614bcbf318bdbfea18219b03876024be16aea3329ee4d3c56bd`。
  本阶段 `VALIDATED-REDUCED`；下一门禁为 typed multi-pass refinement/termination。

### NRIR-26：Typed Multi-Pass Refinement v1

- `C1/C2-M-NRIR26` typed pass control：新增 total-cap partition/reselection/termination policy 与逐
  pass decision；Plan/Task/Schedule 显式包含两组 enumerate/select/decide/backward/intersect/
  propagate，execution/queue trace 绑定 bounds、target ledger、decision 与 action hash；
- `C3-G-NRIR26` compatibility/replay gate：legacy lowering/hash 条件不变；4/4、8/8、12/12
  per-node cap partition、disjoint ledger、no-target passthrough、program/decision/claim tamper 与六分片
  fresh replay fail closed；
- `C1/C3-E-NRIR26` same-total-cap result：clauses `0/2/4` 的 single/split worst lower 完全相同，
  均为 `-0.2819737196/-0.4016119838/-0.4596676826`；三条 delta=`0.0`，logical-domain
  overlap/union=`31/31`；planned cap=`496`、actual targets=`2976`；
- `C3-L-NRIR26` closure=`VALIDATED-NO-GO`：没有任何严格改善，三条 bounded tree 仍 unknown；
  仅单 ResNet/三 clause/CPU，node-initial influence 固定，无 complete property、CUDA、multi-workload、
  competitor、performance 或 ASPLOS-ready，`performance_claimed=false`；
- 工件：`artifacts/typed-multipass-refinement/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；evidence
  hash=`38992cace70214ffcbd670f03dcfca182e0925bee31eb4df885dab4dab03494d`。typed mechanism
  保留；停止静态 influence 的同总 cap 拆 pass 路线。

### NRIR-27：Production Prepared Verifier v1

- `C1/C2-M-NRIR27` production execution ownership：每个 dynamic node batch 编译为一等 verifier
  Plan、四任务 TaskModule 与 sequential Schedule；runtime action trace 与 node/split/parent、optimizer
  IR、input/intermediate/objective/policy identity 逐项绑定；
- `C3-G-NRIR27` parity/compatibility gate：production 明确
  `audit_hash_chain_constructed=false`、`selected_native_reexecution=false`；旧 audit 默认
  query/hash 条件兼容，production/audit clause lower/upper、status、logical queue 与 selected state
  exact/allclose，identity/tamper/action order fail closed；
- `C1/C3-E-NRIR27` repeated internal CPU result：三组交替 fresh-process clause-0
  audit→production median speedup 为 MNISTFC `1.3663×`、ResNet2B `2.4723×`、OVAL21
  `1.4511×`；该 claim 只比较 BoundFlow 同算法 execution mode；
- `C3-L-NRIR27` hard limitation：full production median=`14.834/60.754/11.964 s` 且三类
  query 全部 unknown；竞品仅历史单次、完整性不同，`performance_claimed=false`。无 GPU、complete
  property、公平 competitor 或 ASPLOS-ready claim；
- 工件：`artifacts/production-prepared-verifier/vnncomp21-three-topology-cpu-v1/`；evidence
  hash=`7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`。本阶段
  `VALIDATED-REDUCED`；phase evidence 将下一门禁冻结为 parametric dynamic-batch compiler/cache。

### NRIR-28：Parametric Dynamic Batch Compiler v1

- `C1/C2-M-NRIR28` template/instance ownership：静态 PlanTemplate 绑定 graph、tensor contract、
  ReLU layout、policy/provenance 与 reusable Task/Schedule；动态 PlanInstance 绑定 input/objective/
  intermediate/split/scope/initial-state、batch 与 warm-start，query cache event 显式记录 miss/hit；
- `C3-G-NRIR28` exact cache/replay gate：每 query 只有一个 miss，后续 instances 必须 exact
  contract hit；contract/runtime tensor/event/instance/IR digest tamper fail closed。NRIR-27 frozen
  artifact 在新增 v2 后仍 fresh replay；
- `C1/C3-E-NRIR28` repeated full-query CPU result：MNISTFC、ResNet2B、OVAL21 的三组交替
  production-v1→v2 median speedup 分别为 `4.2849×/9.8630×/3.5024×`；instances/miss/hit=
  `19/1/18`、`27/1/26`、`11/1/10`，语义逐 clause/queue/state 对齐；
- `C3-L-NRIR28` hard limitation：三类 solver status 仍全部 unknown；这是相同 BoundFlow 算法的
  CPU internal speedup，不是 αβ-CROWN 或 GPU speedup。无 complete-property、跨进程 cache、CUDA
  Pareto 或 ASPLOS-ready claim；
- 工件：`artifacts/parametric-dynamic-batch-compiler/vnncomp21-three-topology-cpu-v1/`；evidence
  hash=`117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`。本阶段
  `VALIDATED-REDUCED`；下一门禁为 fixed-wall-clock parametric BaB search scaling。

### NRIR-29：Wall-Clock Parametric BaB Scaling v1

- `C1/C2-M-NRIR29` experiment compiler ownership：三 budget、三 workload、三 repeat、60 秒
  deadline 与轮转顺序均进入 search-scaling Plan/Task/Schedule；27 个 fresh-process task 与 source、
  budget、repeat、order 逐项 hash 绑定；
- `C3-G-NRIR29` replay/nesting gate：同预算三次 semantic signature 一致；三 workload 全部
  `domains(7)⊂domains(31)⊂domains(127)`，公共 domain lower 最大漂移 `0.0`；budget/task、
  compiler template/cache/instance、semantic digest 与同步篡改 fail closed；
- `C1/C3-E-NRIR29` fixed-deadline search coverage：MNISTFC 三次都由 `6/9` verified 提升到
  `8/9`，且 31/127 node closure 相同；27/27 workers 均 `completed=9,pending=[]`；
- `C3-L-NRIR29` saturation boundary：ResNet 在 7/31/127 budget 分别评估 total
  `63/279/1143` nodes 后仍 `0/9` verified，OVAL21 始终 `8/9`；三类 query 仍 unknown。不同预算
  不计算 speedup，无 CUDA、competitor、完整 property 或 ASPLOS-ready claim；
- 工件：`artifacts/wall-clock-parametric-bab-scaling/vnncomp21-three-topology-cpu-v1/`；evidence
  hash=`e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`。本阶段
  search-coverage `VALIDATED-REDUCED`；下一门禁为 unresolved-clause typed hard-clause escalation，
  而不是继续盲目扩同一 search budget。

### NRIR-30：Typed Hard-Clause Escalation v1

- `C1/C2-M-NRIR30` staged compiler ownership：baseline、Decision admission、shared refinement、
  hard-clause projection、parametric escalation、original-ordinal aggregate 与 emit 编译为一等
  Plan/Decision/Task/Schedule；八 action 与 exact query/source/policy/budget/deadline hash 绑定；
- `C3-G-NRIR30` admission/fallback gate：escalated ordinals 恰等于 baseline unresolved；verified/
  unsafe 不重跑，projected↔original 双射；child aggregate、deadline discard、compiler cache/instance、
  refinement semantic trace 和同步 digest 篡改 fail closed；
- `C1/C3-E-NRIR30` repeated property coverage：OVAL21 三次都由 `[0..7]` 提升到 `[0..8]` 且
  query status=`verified`；MNISTFC 三次 `6/9→8/9`；baseline 与 NRIR-29 n7d2 semantics 对齐；
- `C3-L-NRIR30` hard limitation：ResNet 三次仍 `0/9`；统一 shared top-width refinement 不是
  per-clause objective-directed/ancestral/external-seeded bound。仅三个 CPU workload，无 performance、
  GPU、competitor、complete suite 或 ASPLOS-ready claim；
- 工件：`artifacts/typed-hard-clause-escalation/vnncomp21-three-topology-cpu-v1/`；evidence
  hash=`df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`。本阶段
  property-coverage `VALIDATED-REDUCED`；下一门禁为 per-clause objective-directed escalation。

### NRIR-31：Objective-Directed Hard-Clause Escalation v1

- `C1/C2-M-NRIR31` per-clause compiler ownership：保留 NRIR-30 base program，九子句静态展开为
  33-task TaskModule/Schedule；每个 original clause 独占 guarded objective compile/refine/query，
  并绑定 shared source Plan/semantic trace、scalar objective hash 与 ordinal；
- `C3-G-NRIR31` sound-source/fallback gate：objective child 只能消费 validated shared execution；
  final verified 必须包含 baseline/NRIR-30 coverage，deadline 后 child proof 丢弃；source lineage、
  aggregate 与同步 gate tamper fail closed；
- `C1/C3-E-NRIR31` repeated root tightness：ResNet 9 条 root lower delta 三 fresh repeats 逐值一致，
  为 `+123.842712/+179.970459/+81.522583/+89.696289/+96.595642/+98.525497/`
  `+147.607101/+162.138519/+142.715607`；OVAL clause 8 额外改善 `+0.0018788278`；
- `C3-L-NRIR31` hard limitation：MNIST/ResNet/OVAL final coverage 仍为 8/9、0/9、9/9，没有新增
  closure；CPU 三 workload，无 performance、GPU、competitor、完整 suite 或 ASPLOS-ready claim；
- 工件：`artifacts/objective-hard-clause-escalation/vnncomp21-three-topology-cpu-v1/`；evidence
  hash=`fb9e503bdf93cb9ce56f52915f1965f1f542e092945d4d7d77d8b8c4bd91764a`。本阶段 root-bound
  tightness `VALIDATED-REDUCED`；下一门禁为 objective-ancestral dynamic-child propagation。

### NRIR-32：Objective-Ancestral Hard-Clause Escalation v1

- `C1/C2-M-NRIR32` dynamic queue compiler ownership：static Plan 绑定 graph/input/objective/threshold、
  typed root refinement、optimizer、31/depth4、child refinement policy 与 whole deadline；每个已提交
  root/child/transition/emit 都拥有 exact Task IR 与 1:1 Schedule action，emit 依赖完整 committed proof；
- `C3-G-NRIR32` ancestral lineage/deadline gate：每个 child 的 source final-bound/Plan/semantic trace
  必须等于 exact parent execution；split-state 与 parent ordinal fail closed；deadline-crossing child
  work 被整体丢弃并保留 accepted frontier，parent-lineage/aggregate/committed-hash tamper 被拒绝；
- `C1/C3-E-NRIR32` repeated frontier tightness：固定 ResNet property 0 clause 0、31/depth4/60 s 下，
  三轮 root parity exact；ancestral 均提交 7 nodes，worst active lower=`-104.76541137695312`，相对
  31-node root-global `-200.46539306640625` 改善 `+95.69998168945312`；committed queue/Task/
  refinement hashes 重复一致；
- `C3-L-NRIR32` hard limitation：只覆盖单一 property/clause、CPU serial audit evaluator；late stage
  可使 wall clock 超过 cooperative 60 秒边界，timing 不形成 claim。无 property closure、GPU、
  competitor、multi-workload、完整 suite 或 ASPLOS-ready；`performance_claimed=false`；
- 工件：`artifacts/objective-ancestral-hard-clause-escalation/vnncomp21-resnet2b-clause0-cpu-v1/`；
  evidence hash=`8fba8deca18dcbf0b4b258aa390c1dd48d250c71ea1a48ddb991388765411bfc`。
  本阶段 committed-frontier tightness `VALIDATED-REDUCED`；下一门禁为 child refinement cap/resource
  Pareto，使 bound gain 转化为 fixed-deadline node coverage。

### NRIR-33：Objective-Ancestral Child Budget Pareto v1

- `C1/C2-M-NRIR33` budget compiler ownership：candidate caps/order、90% gain-retention rule、
  calibration rows/evidence、winner 与 exact child refinement policy 进入 Policy/Decision/Plan hash；
  Plan 通过结构协议复用 frozen NRIR-32 Task/Schedule/queue engine；
- `C3-G-NRIR33` selection/replay gate：root parity 与 parent lineage valid；winner 必须由 rows 重算，
  winner/decision/source/digest tamper fail closed；cap128 frozen replay 保持有效；
- `C1/C3-E-NRIR33` cap curve：cap `8/16/32/64/128` 的 worst active lower 为
  `-173.078613/-162.253326/-148.134460/-126.962929/-104.765411`，但 accepted nodes 全为 7；
- `C3-L-NRIR33` closure=`VALIDATED-NO-GO`：90% retention winner=cap128，较小 cap 没有 node
  coverage 收益。pilot timing 不形成 performance claim；无 property/GPU/competitor/multi-workload/
  ASPLOS-ready claim；
- 工件：`artifacts/objective-ancestral-child-budget-pareto/vnncomp21-resnet2b-clause0-five-cap-cpu-pilot-v1/`；
  pilot hash=`db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`。
  下一门禁为 cap128 sibling packed/parametric evaluator。

### NRIR-34：Sibling-Packed Objective-Ancestral Evaluator v1

- `C1/C2-M-NRIR34` compiler ownership：显式 source/evaluator objective projection、same-parent
  `(-1,+1)` SiblingGroup、每 child refinement lineage、packed optimizer/native IR 与 atomic commit
  已进入 Plan/Task/Schedule/Group hash；late pair 不得半提交；
- `C3-G-NRIR34` semantics gate：first-pair bounds exact；formal common 7 domains lower/upper max diff
  均 `7.62939453125e-06`，split/branch/final refinement bounds exact，alpha/beta max diff
  `1.0728836e-04/8.9406967e-08`；projection、dependency、group commit、ordinal accounting tamper
  fail closed；
- `C1/C3-E-NRIR34` repeated same-algorithm coverage：31/depth4/60 s 三 fresh repeats 的 serial
  accepted nodes=`[7,7,7]`，packed=`[15,15,15]`，minimum gain=`+8`；packed depth `3`、worst active
  lower=`-76.077194`，优于 serial depth `2`/`-104.765411`；该项只主张 cooperative-deadline
  committed-node coverage improvement；
- `C3-L-NRIR34` hard limitation：formal wall time 为约 `64.5—66.2 s`，不是硬实时 speedup；九子句
  global-60s integration 只完成 clause 0，unresolved `[0]`、pending `[1..8]`、property unknown。
  CPU 单 workload/单 hard clause，无 GPU、competitor、property closure 或 ASPLOS-ready；
- 工件：profile hash=`7bece7f04459df37dad115622fe3bab5bc16145a4b82190ab003950317117ce9`；
  formal hash=`9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`；
  full-query evidence hash=`dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`。
  本阶段 `VALIDATED-REDUCED`；下一门禁为 cross-clause shared root/parametric evaluator + anytime
  global-budget allocation。

### NRIR-35：Cross-Clause Anytime Objective Evaluator v1

- `C1/C2-M-NRIR35` cross-clause compiler ownership：NRIR-31 floor、Decision、guarded NRIR-34 packed
  compile/execute、monotone original-ordinal aggregate 与 emit 被 lower 为一等 Plan/6-task
  TaskModule/Schedule；objective/threshold/policy、exact clause-0 source 与 single global deadline
  逐项 hash-bound；
- `C3-G-NRIR35` admission/fallback gate：只有 floor completed `[0..8]` 且 clause 0 unresolved 才
  admit；packed unknown 保留 exact floor。wrong ordinal/source、deadline reset、baseline omission、
  non-monotone aggregate、partial sibling group 即使同步重算 worker hash 也 fail closed；
- `C1/C3-E-NRIR35` repeated additive coverage：三 fresh repeats 的 NRIR-31 floor elapsed=
  `[22.227251,21.622773,21.834220] s`，每轮完成 9/9 ordinals；剩余预算内 packed accepted nodes=
  `[7,7,9]`，所有 group 均为 atomic pairs；
- `C3-L-NRIR35` hard limitation：三轮 floor/packed/final 均为 sound unknown、9/9 unresolved；whole
  cooperative elapsed=`[61.991720,62.598928,68.042604] s`，不是 60 秒硬实时或 wall-clock speedup。
  CPU 单 workload、只 escalation clause 0，无 property/GPU/competitor/multi-workload/ASPLOS-ready；
- 工件：`artifacts/cross-clause-anytime-objective-evaluator/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/`；formal hash=
  `74533c9c211a3007bf5af43c08865febd95c3f9ccf1a268e56738793ec9d14d5`。本阶段 cross-clause
  control/original-ordinal preservation `VALIDATED-REDUCED`，`performance_claimed=false`；下一门禁
  为 multi-clause anytime priority/time slicing。

### NRIR-36：Multi-Clause Anytime Priority v1

- `C1/C2-M-NRIR36` multi-clause compiler ownership：root-lower priority、top-2 selection、dynamic
  equal-remaining slice、one-shot cutoff、exact source lineage 与 multi-outcome original-ordinal aggregate
  已 lower 为一等 Plan/8-task TaskModule/Schedule/Slice IR 并进入 canonical hash；
- `C3-G-NRIR36` replay/tamper gate：validator 从 floor candidates 独立重算 rank、selected ordinals、
  source lineage 和每次 dispatch allocation；wrong rank/selection/source、slice inflation、deadline reset、
  ordinal omission、non-monotone aggregate、partial group 与 trace binding 同步重哈希后仍 fail closed；
- `C1/C3-E-NRIR36` repeated allocation result：三 fresh repeats 的 priority 都为
  `[2,3,4,5,0,8,6,7,1]`、selected 都为 `[2,3]`；packed nodes=
  `[[3,3],[3,3],[3,1]]`。repeat 2 的第二条只提交 root，未满足每轮两个 atomic pairs；
- `C3-L-NRIR36` hard limitation：三轮 final 仍 9/9 unresolved；whole cooperative elapsed=
  `[67.213556,66.833706,60.228863] s`，不是硬实时或 speedup。预注册 multi-clause coverage gate
  失败，CPU 单 workload/property，无
  property closure、GPU、competitor、multi-workload 或 ASPLOS-ready，`performance_claimed=false`；
- 工件：`artifacts/multi-clause-anytime-priority/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/`；formal hash=
  `2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；全量
  `890 passed, 37 skipped`。本阶段
  `VALIDATED-NO-GO`；IR/control 可保留，下一门禁为 shared parametric compiler/root/evaluator +
  stronger bound/candidate。

### NRIR-37：Shared Parametric Objective Evaluator v1

- `C1/C2-M-NRIR37` compiler ownership：新增 shared-parametric ancestral Plan/Batch/Task/Schedule；
  frozen NRIR-28 Template 只拥有静态 graph/input/objective-shape/ReLU-layout/policy/provenance，exact
  Instance 拥有 objective content、split、intermediate bounds、warm state、refinement lineage 与 batch；
  一个 query cache owner 跨 batch/跨 clauses 2/3 复用同一 template；
- `C3-G-NRIR37` fail-closed gate：cache key/contract/instance/event、source lineage、batch ordinal、partial
  sibling group、bound drift、selected-native reexecution、deadline reset、rank/selection/source/allocation、
  template count 与 compiler coverage tamper 均被 runtime 或 artifact validator 拒绝；
- `C1/C3-E-NRIR37` same-algorithm parity/coverage：真实 clause 2 frozen audit vs shared root+pair 的 lower、
  branch、split、α、β 与 refinement hashes exact，upper max diff=`1.52587890625e-5` 且 allclose 通过；
  三 fresh repeats 均 rank=`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`、packed nodes=`[31,31]`，
  每轮恰好 1 miss + 31 hits；
- `C3-L-NRIR37` hard limitation：whole elapsed=`[51.996191,52.251681,52.695640] s` 只证明固定
  global deadline coverage，不是 competitor speedup；final 仍 9/9 unresolved，clauses 2/3 depth-4
  worst active lower=`-37.574287/-35.900215`，无 GPU、multi-workload、property closure 或 ASPLOS-ready；
- 工件：`artifacts/shared-parametric-objective-evaluator/`
  `vnncomp21-resnet2b-property0-cpu-pilot-v1/` 与
  `vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/`；pilot/formal hash=
  `c96fff3fa2bc2563b4d46886d69b33f51ac985b19ad80d916309db57fe6cfefa` /
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`；全量
  `917 passed, 37 skipped`。本阶段 `VALIDATED-REDUCED`；下一门禁为 full-depth frontier tightness
  attribution + 单变量 stronger-bound/candidate，不再调 control/cache 常数。

### NRIR-38：Full Frontier Tightness Attribution v1

- `C1/C2-M-NRIR38` attribution ownership：source/active nodes、metric contract、baseline/candidate policy、
  exact sibling batches、Decision 与 emit 已 lower 为一等 Plan/7-task TaskModule/Schedule；两个 policy
  只允许 `steps=5→15`，objective/threshold/split/refinement/parent warm/dtype/device 全部冻结；
- `C3-G-NRIR38` fail-closed gate：validator 独立重算 31-node source 与 16-node active frontier、parent
  path、refinement/state metrics、baseline replay、candidate Decision 和 Task/Schedule；active omission、
  policy/delta/decision/Task/cache/sibling/evidence tamper 均被拒绝；
- `C1/C3-E-NRIR38` exact-frontier result：clauses 2/3 baseline lower/upper max diff=0；steps15 改善
  32/32 nodes、0 regressions，median delta=`+0.107208/+0.132715`，但 worst-active improvement 只有
  `+0.055496/+0.028557`，未过两条均 `>=+1.0` 的预注册门禁；
- `C3-L-NRIR38` hard limitation：本阶段 `VALIDATED-NO-GO`，只排除当前 fixed frontier 的 optimizer
  steps 单轴；不形成 full-query/property/performance/GPU/competitor/multi-workload/ASPLOS-ready claim；
- 工件：`artifacts/full-frontier-tightness-attribution/`
  `vnncomp21-resnet2b-property0-cpu-pilot-v1/`；pilot hash=
  `2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`；全量
  `930 passed, 37 skipped`。下一单变量为已有 objective branch IR 的 shared-evaluator 接入。

### NRIR-39：Objective Branch Shared Evaluator v1

- `C1/C2-M-NRIR39` composite ownership：frozen shared Plan 与 historical objective branch policy 已组合为
  一等 Plan/6-task TaskModule/Schedule；31/31 node evaluations 各自绑定 exact branch Plan/Task/Schedule、
  score trace、selected candidate、queue evaluation 与 child split；
- `C3-G-NRIR39` fail-closed gate：policy/source/coverage/selected ordinal/trace score/Task/claim/control tamper
  即使同步重算外层 hash 仍被拒绝；large-scale float32 width/mean equality 使用
  `rel_tol=1e-6,abs_tol=1e-6`，`+0.1` drift 专用测试仍拒绝；
- `C1/C3-E-NRIR39` fixed-budget tightness：clauses 2/3 root exact，control/candidate 均 31 evaluations、
  16 depth-4 active nodes；worst-active lower 从 `-37.574287/-35.900215` 到
  `-35.530926/-30.258448`，改善 `+2.043362/+5.641768`；median 改善
  `+2.537640/+5.885233`，两条均通过 `+1.0` 门禁；
- `C3-L-NRIR39` hard limitation：logical fixed-budget evidence 不证明墙钟速度或 global-deadline coverage，
  也没有 property/GPU/competitor/multi-workload/ASPLOS-ready claim；
- 工件：`artifacts/objective-branch-shared-evaluator/`
  `vnncomp21-resnet2b-property0-cpu-pilot-v1/`；pilot hash=
  `dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`。本阶段 fixed-budget branch
  selection `VALIDATED-REDUCED`；16 focused、40 predecessor-inclusive tests、全量
  `940 passed, 37 skipped` 与静态门禁通过；下一门禁为 three-repeat whole-query/global-deadline formal。

### NRIR-40：Objective Branch Whole Query Formal v1

- `C1/C2-M-NRIR40` production composition：新增 raw objective-branch shared production queue 与
  multi-clause anytime composition；objective scoring 消费真实 slice/global monotonic deadline，仍由
  composite branch Plan/Task/Schedule 和 exact per-node binding 拥有语义；
- `C3-G-NRIR40` fail-closed gate：worker 从 floor 重算 rank/selected/allocation，验证 atomic groups、
  branch-node coverage、policy hash、source lineage、cache ownership 与 nine-ordinal aggregate；formal
  重算三轮 gate 并交叉绑定 shard。重复 branch node 即使同步改 formal/shard 并重算 worker/formal/
  manifest hashes 仍被拒绝；
- `C1/C3-E-NRIR40` repeated correctness：三 fresh processes 都完成 9/9 floor，rank=
  `[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`；accepted nodes=
  `[[29,23],[29,21],[29,21]]`，每个 accepted node 均有 branch execution，每轮 cache miss=1，无
  partial sibling commit；
- `C3-L-NRIR40` hard limitation：三轮均未达到两条 clause 各 `31 nodes/15 groups`，worst-active lower=
  `[[-48.315041,-43.299690],[-48.315041,-44.731468],[-48.315041,-44.731468]]`，相对 frozen widest
  更差；final 仍 9/9 unresolved。状态为 objective-branch global-budget `VALIDATED-NO-GO`，不形成
  property/performance/GPU/competitor/multi-workload/ASPLOS-ready claim；
- 工件：`artifacts/objective-branch-whole-query/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/`；formal hash=
  `d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`；whole cooperative elapsed=
  `[63.357098,63.161128,62.485366] s`；全量 `944 passed, 37 skipped`。下一步只做 scoring/queue
  cost 与 frontier-order 因果归因，不事后扫 policy 常数。

### NRIR-41：Objective Branch Production Cost Attribution v1

- `C1/C2-M-NRIR41` attribution ownership：新增 Plan/6-task TaskModule/Schedule、16 条 prefix IR、12 条
  unprofiled wall IR、8 条 cProfile phase IR 与 causal Decision；source admission→prefix reconstruction→
  paired execution→profile→decision→emit 全部 hash-bound，NRIR-39/40 frozen 文件未改；
- `C3-G-NRIR41` fail-closed gate：replay 从 NRIR-39 ordered evaluations 按 parent lineage 独立重建
  active sets 与 worst/median，重算三轮 median/MAD、queue ratio、profile share、Decision 和 Task/Schedule；
  formal/manifest 同步重哈希 prefix tamper 仍被拒绝；
- `C1/C3-E-NRIR41` frontier result：clauses 2/3 在 `21/23/29/31` same-node prefix 的 objective-widest
  worst improvement 分别为 `[+2.171364,+2.416264,+2.947929,+2.043362]` 与
  `[+4.988102,+6.255299,+6.350922,+5.641768]`，`frontier_order_retained=true`；
- `C1/C3-E-NRIR41` cost result：widest/objective queue median 为 clauses 2
  `10.515292/18.387675 s`、clause 3 `10.619606/18.591097 s`，ratio=
  `1.748660/1.750639`；branch-program share=`21.9371%/21.9139%`，31 次 branch program 对应 341 次
  candidate enumeration，`scoring_cost_dominant=true`；
- `C3-L-NRIR41` claim boundary：本结果只准入下一 scorer ownership/validation reuse 单变量，不是
  production/system speedup，也没有 property/GPU/competitor/multi-workload/ASPLOS-ready claim；NRIR-40
  global-budget `VALIDATED-NO-GO` 不变，`performance_claimed=false`；
- 工件：`artifacts/objective-branch-production-cost-attribution/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-v1/`；formal hash=
  `fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`。本阶段 attribution
  `VALIDATED-REDUCED`，全量 `948 passed, 37 skipped`，Decision=`optimize_scorer_ownership`。

### NRIR-42：Objective Branch Scorer Ownership v1

- `C1/C2-M-NRIR42` compiler ownership：新增 typed validated capsule、Plan-owned candidate scorer
  Task/Schedule、prevalidated executor、additive production queue 与 multi-clause composition；候选表只在
  branch Plan compile 生成一次，execute/validation 消费 immutable table/token，historical NRIR-39/40
  文件不改；
- `C3-G-NRIR42` fail-closed gate：replay 将 JSON candidate Plan、scorer Task、Schedule 与 capsule
  重建成类型对象，并重算 candidate width/table hash、score reduce/hash、selection、branch binding、
  enumeration ownership、Phase-A median/MAD 与 Phase-B global deadline；同步 token/score/call/deadline
  tamper 被拒绝；
- `C1/C3-E-NRIR42` Phase-A exact/cost：clauses 2/3 六组 old/new 31-node execution 的 branch、全部
  score、child lower、queue lower/upper、split、α/β、refinement exact；enumeration `341→31`，new
  compile=`31`、execute=`0`；new/old median ratio=`0.706888/0.698486`，节省严格大于 MAD；
- `C1/C3-E-NRIR42` Phase-B production：three fresh global-60s queries 均 selected `[2,3]`，accepted
  nodes=`[[31,31],[31,31],[31,31]]`、每条 15 groups/31 capsules，whole=
  `[57.175184,57.697757,58.114412] s`；worst active lower=`-35.530926/-30.258448`，相对 widest
  `+2.043362/+5.641768`；
- `C3-L-NRIR42` claim boundary：只在固定 ResNet2B property 0、CPU8、内部 global-60s 协议下恢复
  objective-branch production admission；final 仍 unknown，不是 property closure、GPU、multi-workload、
  fair competitor speedup 或 ASPLOS-ready，`performance_claimed=false`；
- 工件：`artifacts/objective-branch-scorer-ownership/` 与
  `artifacts/objective-branch-scorer-ownership-global/`；Phase-A/Phase-B formal hash=
  `0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58` /
  `7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`；targeted `10 passed`，
  全量 `958 passed, 37 skipped`，静态门禁通过。本阶段 `VALIDATED-REDUCED`。

### NRIR-43：Cross-Axis Verification Batch Schedule v1

- `C1/C2-M-NRIR43`：typed ragged Plan/Instance/Task/Schedule/Trace 显式拥有 clause/node/candidate
  segments；一个 lower launch 后按 owner 还原 legacy-compatible trace，NRIR-42 frozen 文件不改；
- `C3-G-NRIR43`：6/6 组 queue/branch/score/child-bound/state/refinement exact，segment/launch/objective
  owner synchronized outer-rehash tamper fail closed；每条 physical scorer launch `31→16`；
- `C3-E-NRIR43-NOGO`：clauses 2/3 NRIR-42/cross-axis median ratio=`1.051134/1.044573`，两条预注册
  `<=0.85` timing gate 均失败；减少 launch 在 CPU 上没有转化为加速；
- `C3-L-NRIR43`：Phase B gated off，production 仍使用 NRIR-42；`performance_claimed=false`，不得外推
  公平竞品、GPU、多工作负载、property closure 或 ASPLOS-ready；
- 工件：`artifacts/cross-axis-verification-batch/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1/`；formal hash=
  `692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`；全量
  `968 passed, 37 skipped`。本阶段 `VALIDATED-NO-GO`。

### NRIR-44：Root-Projection Floor Schedule v1

- `C1/C2-M-NRIR44`：typed consumer/liveness Plan/Instance/7-task Task/Schedule/Trace 显式绑定 source
  floor、9 clause owners、被消费的 root/status/evidence fields 与 full/projected budget；additive runtime
  只把 ranking floor child queries 从 `9×n31d4` 投影为 `9×n1d0`，一般 complete verifier 不静默启用；
- `C3-G-NRIR44`：Phase A 三轮 baseline/refinement/root lower/upper/branch、9/9 unknown、rank=
  `[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]` exact，objective evaluations=`279→9`；typed replay
  重建 Plan/Instance/Task/Schedule/Trace，同步外层重哈希 budget/consumer tamper fail closed；
- `C3-E-NRIR44`：old/projected floor median=`24.235039/9.876515 s`、ratio=`0.407530`；Phase B
  floor=`8.538814/8.622447/8.648849 s`，whole=`43.571040/44.144990/44.095736 s`，相对 frozen
  NRIR-42 whole median ratio=`0.764254`；每轮 top-2 production `[31,31]` nodes，worst lower exact；
- `C3-L-NRIR44`：root projection 是 sound-but-less-complete ranking-only specialization；结论仅为
  fixed ResNet2B property 0 CPU8 internal admission，final 9/9 unknown，不是公平竞品 speedup、GPU、
  multi-workload、property closure 或 ASPLOS-ready，`performance_claimed=false`；
- 工件：`artifacts/root-projection-floor/` Phase A/B；formal hash/payload hash=
  `ecb553d88be065054abb0a480b79086ae12cec55a84e5c0ba537572e904ff0fe` /
  `2f22d44fe9f57f233c8a853b66f67f404b03a087d097451e10f663ee257272d9`；本阶段
  `VALIDATED-REDUCED`；全量 `979 passed, 37 skipped`。

### NRIR-45：Prepared Intermediate Refinement Capsule v1

- `C1/C2-M-NRIR45`：typed capsule/receipt、5-stage Task/Schedule/Trace 与 additive prepared
  Program/Execution/shared queue/global composition 已实现；每个 exact child 首次完整准入，runtime 用
  owner/container/Tensor-version receipt 与 cached admitted hashes，显式 full replay 仍调用原始 validator；
- `C3-G-NRIR45`：mutation、wrong input/capsule、source/target/result binding fail closed；Phase A 每条
  prepared queue 的 30 capsules 与 Phase B 每轮 60 capsules 均 full replay，typed artifact reconstruction
  与 synchronized outer-rehash tamper probe 通过；
- `C3-E-NRIR45`：target selection=`246→98`、full Program validation=`186→38`、full hash=`217→39`；
  clauses 2/3 prepared/control median ratio=`0.727519/0.736603`。Phase B trace=
  `31.262521/31.319772/31.470078 s`、measured=`36.396631/36.513683/36.611709 s`，相对 NRIR-44
  median ratio=`0.710268/0.615738`，每轮 `[31,31]` nodes 与 worst lower exact；
- `C3-L-NRIR45`：仅 fixed ResNet2B property 0 CPU8 internal production admission
  `VALIDATED-REDUCED`；final 9/9 unknown，`performance_claimed=false`，不得外推 fair competitor、GPU、
  multi-workload、property closure 或 ASPLOS-ready；
- 工件：`artifacts/prepared-intermediate-refinement/` Phase A/B；formal/payload hash=
  `be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439` /
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；全量
  `984 passed, 37 skipped`，Pylint `10.00/10`。

### NRIR-46（Phase 0 NO-GO）：Intermediate Refinement Template/Instance v1

- `C1/C2-M-NRIR46-NOGO`：原拟把静态 primal graph/policy/selection recipe/Task/Schedule topology
  归 `PlanTemplate/ScheduleTemplate`，但 Phase 0 strict static ceiling 未过，故未实现；node-specific
  target ledger 经验证 60/60 互异，不能跨 child 共享；
- `C3-D-NRIR46`：raw trace floor median=`10.818262 s`、packed slice median=`9.932808 s`；一次
  diagnostic repeat0 的 60 child prepared compile/execute=`5.300590/5.659414 s`，只用于选路线；
- `C3-G-NRIR46-NOGO`：three-repeat compile total median=`5.366369 s`，strict static topology
  median=`1.071197 s < 1.5 s`；ownership-convertible ceiling median=`2.102134 s`。预注册门禁失败，
  Phase A/B gated off；
- `C3-E-NRIR46`：每轮 target selection observed/semantic=`124/60`、冗余=`64`，估计冗余成本=
  `1.026058/1.039642/1.038153 s`；三轮 selected `[2,3]`、nodes `[31,31]`、60/60 full replay exact；
- `C3-L-NRIR46`：formal hash=`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`，
  replay/tamper 通过，`performance_claimed=false`。未实现 Template/Instance，也不构成 10x、公平竞品、
  GPU、多 workload、property closure 或 ASPLOS-ready。

### NRIR-47（Phase A NO-GO）：Single-Pass Target Admission Receipt v1

- `C1/C2-M-NRIR47`：typed target admission receipt 将 exact bounds/policy/objective/influence 与
  ordered target table 绑定到 Task/Schedule 和 prepared capsule；additive production candidate 不重选，
  explicit full replay 仍从 exact source 重选；legacy compiler 文件/默认语义保持不变；
- `C3-D-NRIR47`：NRIR46 target selection observed/semantic=`124/60`，64 次冗余 selection 估计
  median=`1.038153 s`；60/60 target ledger 互异，不允许跨 child 共享；
- `C3-G-NRIR47`：candidate 每条 queue compile selector/reselection=`30/0`、runtime selector=`30`、
  receipt/full replay=`31/31`；correctness/ownership exact，186 receipt replay 与同步重哈希 tamper 通过；
- `C3-E-NRIR47-NOGO`：compiler control/candidate median=`2.739226/2.563922 s`，ratio=
  `0.936003 > 0.85`；clauses 2/3 queue ratio=`1.011205/1.019338 > 0.97`，Phase A timing 失败，
  Phase B gated off；
- `C3-L-NRIR47`：formal hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；
  全量 `992 passed, 37 skipped`。candidate 不默认启用，`performance_claimed=false`，不构成 10x、
  公平竞品、GPU、多 workload、property closure 或 ASPLOS-ready。

### NRIR-48：Top-2 Production Execution Cost Attribution v1

- `C3-D-NRIR48`：additive runner 把 frozen NRIR45 clauses 2/3 production queue 分为七个互斥顶层
  类别，并细分 child refinement execute；NRIR47 candidate 禁用，未修改 production runtime；
- `C3-G-NRIR48`：6/6 paired semantic exact，profile/control ratio=`1.023199/1.020221 <=1.05`，
  顶层/内部 closure、6 profile replay 与同步 category tamper 拒绝通过；
- `C3-R-NRIR48`：两条 clause 3/3 dominant 均为 child execute，median/share=
  `3.816002 s/32.1966%`、`3.704755 s/31.1640%`；selected-CROWN 为唯一合格内部子类，median=
  `2.663321/2.694436 s`，parent share=`71.7725%/72.7291%`；
- `C3-L-NRIR48`：formal hash=`571c2e47c0c8906d2486e5e19e8152eb1ef0d3024b08cf561e25ed4f71d177a4`；
  全量 `996 passed, 37 skipped`；attribution `VALIDATED-REDUCED` 不是 speedup，不外推 GPU、
  competitor、multi-workload、property closure 或 ASPLOS-ready；当时准入的 NRIR49 selected-CROWN
  execution 已由下节完成，不是当前路线。

### NRIR-49A（G1 selected-CROWN-only NO-GO）：GPU Selected-CROWN Opportunity Attribution v1

- `C3-D-NRIR49A`：additive read-only runner在RTX 4060 Laptop上执行clauses 2/3、五fresh workers、
  五chunk Latin sweep与paired default32 control；production runtime/TIR/kernel/default policy未修改；
- `C3-G-NRIR49A`：5/5 worker envelope/hash通过，60组离散结构exact，数值最大absolute/relative diff=
  `2.288818359375e-05/1.710717646052519e-04 <=2e-4`；profile/control ratio中位=
  `0.999304/1.006747 <=1.05`；代表调用CUPTI含5954 kernels/5486 launches/398 sync/5364 memory events；
- `C3-E-NRIR49A-NOGO`：queue/complete selected-CROWN share中位=
  `0.0709863183/0.0705232890`，queue机会门槛`>=0.20`失败，queue `1.20x`与complete `1.15x`
  Amdahl目标均不可达；最大allocated/reserved比例=`0.009964/0.013530`、合法batch上限1、无OOM，
  physical-memory path=`N/A`；
- `C3-L-NRIR49A`：summary/manifest hash=`7eefe6a7…ab50`/`d0272fe4…c81f`；5 raw/50 normalized/
  2 query/0 failure rows，独立replay stdout与所有digest重算通过。G1
  `VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`，只将selected-CROWN专属G2/G3
  gated off；`1/(1-0.070986)=1.0764x`只是假设该region变为零耗时的单区域上限，不约束BoundFlow
  operator/graph/JIT/runtime/memory累计收益。artifact中的`next_route=gpu-winner-reselection`是冻结
  历史输出，已由
  `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`取代；FSG0合同已以
  20项定向测试与`1079 passed, 3 skipped`回归关闭，外部审计三项minor已修复；当前FSG1只采集
  official control full-stack trace，
  不构成speedup、competitor、multi-workload、
  solved verdict、memory headline或ASPLOS-ready claim。

### RVIR-v4 V4-2B：Production Optimizer Step Truth Trace

- `RVIR-V4-2B-M`：一个真实ResNet2B property 0 CUDA production core捕获10 evaluations、9个真实
  Adam steps、双LR schedule及每步24项α/SparseBeta state；18项optimizer controls与call lineage进入
  typed payload/hash；
- `RVIR-V4-2B-G`：1 core/24 calls、phase=`12/1/11/0`、每步24项state、9个transition各7项mutable
  改变、state source CUDA均由raw artifact replay重建；state/lower/call-result/step-lineage/policy五类
  同步重哈希和manifest重签攻击均fail closed；与冻结capture-v2的source/protocol/call topology/
  tensor schema/history/policy/branch/mutation structure exact，GPU float max diff=`6.0797e-06`、lower=
  `3.5763e-07`，sign/finite mask exact并通过`2e-4`预注册容差；
- `RVIR-V4-2B-L`：结论仅为provider production truth trace
  `VALIDATED-PRODUCTION-TRACE`；`optimizer_replacement_admitted=false`、
  `b2_same_solver_timing_admitted=false`、`performance_claimed=false`。没有BoundFlow mutation parity、
  atomic copy-out、same-solver timing或ASPLOS-ready claim；
- 工件：`artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1/`；manifest/trace/summary=
  `7d7745e4...fbe6` / `fa070bb0...31f4` / `8ae8be3f...05b7`；source-parity report=
  `c2b48275...8aec`。该时点下一切片为V4-2C；现已由下节正式关闭。

### RVIR-v4 V4-2C（关闭）：Pre-State Native Initializer

- `RVIR-V4-2C-M`：共享typed mapper按topology/alpha-indices/beta-location/history恢复6组dense
  α/β/split和external intermediate bounds，并在真实ResNet2B BoundFlow module/scope构造
  `NativeAlphaBetaOptimizationState`；12个mutable path mapped/full round-trip exact，6组未消费upper α
  plane显式copy-through；V4-1 evaluator复用同一实现；
- `RVIR-V4-2C-G`：formal artifact由clean runner `96c45a6`生成，source snapshot/topology/history/
  intermediate/mapping/native-state/summary hash分别为`2a775b66...a256`、`9be36162...ca35`、
  `8921a052...08a`、`f82523fb...cf06`、`cfcebf92...f8df`、`e3587dd9...bff0`、
  `6702a39d...899c`。original semantic replay通过；topology、α index、history score、intermediate、
  upper α、beta location+history六类攻击在重算内部hash并重签source/outer manifest后，外层provenance
  与内层semantic identity/cross-binding均6/6拒绝；
- `RVIR-V4-2C-V`：focused=`8 passed`、full=`1164 passed, 3 skipped`、mypy七文件clean、Pylint=
  `10.00/10`；artifact manifest SHA256=`daee2fa0...0218`，tamper report SHA256=
  `894c30c4...d858`/report hash=`cfe3f9cd...0033`；
- `RVIR-V4-2C-L`：以`VALIDATED-PRE-STATE-INITIALIZER`关闭。该时点没有执行optimizer mutation、
  post-state parity、atomic copy-out或性能计时；V4-2D现已由下节关闭。

### RVIR-v4 V4-2D（关闭）：Native Step Mutation Parity

- `RVIR-V4-2D-M`：provider-independent native executor已从V4-2C state执行10 evaluations/
  9 Adam updates；production trace不进入executor，只进入独立dense comparator；双LR、decay、sum loss、
  α/β projection与10/9 loop全部typed/fail closed；
- `RVIR-V4-2D-G`：single-thread formal replay的10/10 step lower/α/β allclose且sign exact，最大绝对
  误差=`4.5300e-06`/`1.4663e-05`/`3.9861e-07 <=2e-4`；native trace/parity/summary=
  `d53cc7fc...7c8c`/`2a74e735...2c44`/`0b28f9c9...b8aa`；六类完全重签攻击在provenance/
  semantic两层6/6拒绝；manifest/tamper SHA256=`0b4ae1a8...8493`/`47af58e1...5e36`；
- `RVIR-V4-2D-V`：formal focused=`5 passed`、expanded=`38 passed`、full=`1169 passed, 3 skipped`、
  mypy六文件clean、Pylint=`10.00/10`；
- `RVIR-V4-2D-L`：以`VALIDATED-NATIVE-STEP-PARITY`关闭；terminal copy-out已由下一节关闭。

### RVIR-v4 V4-2E / V4-2（关闭）：Atomic Copy-Out 与 Optimizer Replacement

- `RVIR-V4-2E-M`：terminal dense α/β投影为12个production mutable paths；全部先进入
  private immutable candidate，通过path/schema/finite/post parity/final lower/read-only门禁后才提交；
- `RVIR-V4-2E-G`：formal runner从冻结source重做pre-state、10/9 native mutation、12/12 stage+commit，
  显式结构为`1 core/6 domains/6 topology/12 receipts/7 changed`；α/β/final lower max diff=
  `1.4663e-05/3.6135e-07/2.6226e-06 <=2e-4`且sign exact；正向12-path commit、NaN pre-write拒绝、
  stale-target pre-write拒绝、mid-copy fault rollback均通过；original replay通过，topology/initial α/
  post α/final lower/recorded copy-out/recorded commit六类完全重签攻击两层6/6拒绝；
- `RVIR-V4-2E-V`：manifest/tamper SHA256=`b76ee573...0136`/`621d5485...f70`；focused=
  `11 passed`、full=`1175 passed, 3 skipped`、Black/mypy clean、Pylint=`10.00/10`；
- `RVIR-V4-2E-L`：V4-2E=`VALIDATED-ATOMIC-COPY-OUT`；V4-2 §6八项formal acceptance全部通过，
  整体=`VALIDATED-OPTIMIZER-REPLACEMENT`。whole-core live integration、branch/queue/termination/verdict
  在该时点仍待V4-3；该时点B2和性能claim关闭，现已由后续V4-3 closure取代。

### RVIR-v4 V4-3A（关闭）：Whole-Core Original Truth

- `RVIR-V4-3A-M`：original observer在KFSB消费前冻结六层lA，同时保存六层intermediate、三组
  candidate child lower、final decision、完整core/post和solver accounting；
- `RVIR-V4-3A-G`：fresh provider replay比较451 tensors/213,060 signs，shape/dtype/device、sign、离散
  结构exact，最大差`8.821487426757812e-06 <=2e-4`；
- `RVIR-V4-3A-T`：lA/intermediate/candidate/decision/accounting五类full resign和字段删除攻击6/6拒绝；
- `RVIR-V4-3A-V`：source=`bfdeefc`，manifest/tamper SHA256=`0e6ed721...9818`/
  `dafcb893...a52`，targeted=`12 passed`，full=`1180 passed, 3 skipped`；
- `RVIR-V4-3A-L`：状态=`VALIDATED-WHOLE-CORE-TRUTH`。只准入V4-3B native lA/intermediate；
  该时点`whole_core_replacement_admitted=false`、B2与performance claim关闭，现已由后续closure取代。

### RVIR-v4 V4-3B（关闭）：Native Backward Export

- `RVIR-V4-3B-M`：通用native CROWN backward显式导出六层lower adjoint，并把共享pre-result external
  intermediate bounds映射为六个provider preactivation keys；
- `RVIR-V4-3B-G`：六层lA/12 intermediate/final lower最大差=
  `9.238719940185547e-07/6.079673767089844e-06/3.0994415283203125e-06 <=2e-4`，sign exact；
- `RVIR-V4-3B-C`：provider core/compute_bounds/update_bounds/fallback=`0/0/0/0`；
- `RVIR-V4-3B-T`：lA/intermediate/lower full resign与topology/truth source outer resign共5/5拒绝；
- `RVIR-V4-3B-V`：source=`762b642`，manifest/tamper SHA256=`110dfd63...8269`/
  `4cdd2231...e355`，targeted=`9 passed`，full=`1183 passed, 3 skipped`；
- `RVIR-V4-3B-L`：状态=`VALIDATED-NATIVE-BACKWARD-EXPORT`。formal native为CPU semantic replay；
  KFSB现由V4-3C另行关闭；该时点GPU live integration、whole-core、B2与performance仍未准入，现已由
  后续closure取代。

### RVIR-v4 V4-3C（关闭）：Native KFSB Candidate Evaluation

- `RVIR-V4-3C-M`：六层unstable mask只由native/shared intermediate bounds与terminal split推导；
  BaBSR alpha/intercept score、top-3、min reduction、invalid threshold与tie-break均由BoundFlow执行；
- `RVIR-V4-3C-G`：mask 37464 elements/4200 true逐元素exact；三候选36项和final六项exact；72个
  child lower sign exact，最大差`3.0994415283203125e-06 <=2e-4`；
- `RVIR-V4-3C-C`：provider core/compute_bounds/update_bounds/fallback=`0/0/0/0`；
- `RVIR-V4-3C-T`：candidate/child/final/mask/score/reduction六类full resign与topology/truth source两类
  outer resign共8/8拒绝；
- `RVIR-V4-3C-V`：source=`a2097c0`，manifest/tamper SHA256=`28e4da09...2ed8`/
  `c197b5d5...45f9`，targeted=`16 passed`，full=`1187 passed, 3 skipped`；
- `RVIR-V4-3C-L`：状态=`VALIDATED-NATIVE-KFSB`；该时点只准入V4-3D，现已由下节closure取代。

### RVIR-v4 V4-3D（关闭）：Live Return Assembly

- `RVIR-V4-3D-M`：BoundFlow在真实RTX 4060进程执行pre-state→10/9 optimizer→backward→三候选
  KFSB，原子提交12条provider-owned α/β与host packet并构造完整`UpdateBoundCoreReturn`；
- `RVIR-V4-3D-G`：native lower/child lower/lA source device均为`cuda:0`；未修改official post/queue
  消费成功，visited domains=`[6]`，final decision exact；对V4-3A truth的451 tensors/213,060 signs
  最大差=`1.0669231414794922e-05 <=2e-4`且sign exact；
- `RVIR-V4-3D-C`：provider core/compute_bounds/update_bounds/fallback=`0/0/0/0`；12/12 path committed、
  7 changed，live tensor与host packet联合rollback单测通过；
- `RVIR-V4-3D-T`：lA/intermediate/child lower/α/decision/accounting/provider callback/atomic flag八类
  完全重签攻击8/8拒绝；
- `RVIR-V4-3D-V`：source=`dc7038a`，manifest/tamper SHA256=`272ac92c…2d10`/
  `1e4acb65…ddb1`，targeted=`23 passed`，full=`1196 passed, 3 skipped`；
- `RVIR-V4-3D-L`：状态=`VALIDATED-LIVE-RETURN`，`whole_core_replacement_admitted=true`仅限固定一次
  live core；该时点`five_fresh_correctness_admitted=false`、B2/performance仍关闭、只准入V4-3E，
  现已由下节closure取代。

### RVIR-v4 V4-3E / V4-3（关闭）：Five-Fresh Whole-Core Replacement

- `RVIR-V4-3E-P`：sequence exact=`O,C,C,O,C,O,O,C,O,C`，pair mapping exact=
  `(0,1)/(3,2)/(5,4)/(6,7)/(8,9)`；10个独立CUDA进程、cold isolated property；
- `RVIR-V4-3E-G`：5/5 pairs通过完整core/post/state/branch/queue/termination；合计2255 tensors、
  1,065,300 signs，最大差=`1.0669231414794922e-05 <=2e-4`，sign exact；每run accepted/pruned=
  `6/0`、visited=`[6]`、status/success=`verified/true`；
- `RVIR-V4-3E-C`：original provider call总数120；candidate provider core/compute/update/fallback=
  `0/0/0/0`；
- `RVIR-V4-3E-T`：candidate/original lA、candidate decision三类inner+outer resign与queue/callback/
  sequence三类outer resign共6/6拒绝；
- `RVIR-V4-3E-V`：source=`17d2d61`，manifest/tamper SHA256=`ca37bd56…ada2`/
  `bc41cde5…9fc2`，closing targeted=`8 passed`，full=`1200 passed, 3 skipped`；
- `RVIR-V4-3E-L`：V4-3E=`VALIDATED-FIVE-FRESH-CORRECTNESS`，V4-3整体=
  `VALIDATED-WHOLE-CORE-REPLACEMENT`；`b2_same_solver_timing_admitted=true`，但B2未执行且
  `performance_claimed=false`。

### FSG3/B2 Same-Solver Timing（正式基线已关闭）

- `FSG3-P`：B0/B1/B2物理模式、六个全排列block、每配置6 control+6 profile及36-process exact顺序已
  在`fsg3_b2_same_solver_timing_preregistration_2026_08_13.md`冻结；
- `FSG3-M`：cold total、process-hit query、whole core、GPU event、compile与post-validation分离；
- `FSG3-G`：correctness/no-fallback、GPU排他、profile/control `<=1.05`、raw semantic replay与同步重签
  tamper为硬门禁；
- `FSG3-S`：typed schema/replay已实现，顺序/删run/provider/scope/semantic/profile/environment等13项合同
  与1项lower-only upper-mask amendment测试通过；初始full=`1213 passed, 3 skipped`，post-amendment
  full显式延后到real-worker切片；
- `FSG3-R`：source=`a4ee291`完成36/36 fresh进程；每配置6 control+6 profile，correctness、environment、
  profile closure/扰动与static replay全过，summary hash=`df852590d…1318e`；
- `FSG3-X`：B0/B1 provider core/compute/update分别=`1/14/3`，B2=`0/0/0`且fallback=0，证明物理
  whole-call replacement而非original callback；
- `FSG3-T`：B1 query wall geomean=`0.995657x`；B2 query/core=`0.908400x/0.516767x`
  （B0/candidate），显存ratio=`1.0`，break-even=`not_reachable`；因此B2=
  `MEASURED-B2-SLOWER`，不得写成加速；
- `FSG3-A`：B2 core的optimizer/atomic/KFSB/typed-pre share约=
  `43.999%/24.684%/16.684%/10.720%`，是FSG4/B3优先归因依据；
- `FSG3-F`：8类payload修改+manifest文件digest/hash同步更新的outer-resigned攻击全部拒绝；
- `FSG3-L`：状态=`VALIDATED-FSG3-B0-B1-B2-BASELINE`，raw仍
  `performance_claimed=false`。它不关闭B3—B7；下一门禁为FSG4/B3，ASPLOS-ready仍为NO。
