# BoundFlow 面向 ASPLOS 的总体研发与论文执行计划 v1.0

> 状态：**顶层执行计划 v1.0；后续研究工作受本文门禁约束。**  
> 基线日期：2026-07-12  
> 原始计划代码基线：`263ea81`（PR-10 complete）；当前 integration base：`f194034`
> 投稿策略：ASPLOS 2027 September Cycle 为有条件冲刺；ASPLOS 2028 为稳健主目标。

> **路线修订（2026-07-20）**：本文保留 2026-07-12 的研究问题、历史门禁和 PR-10—13
> 计划，但其对当前实现完成度的判断已由
> `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md` 纠正。
> 当前 Bound IR 仍为占位，统一 Plan IR 与 Schedule IR 尚未实现；不得仅凭 runtime operator、
> PlanBundle 或 TaskGraph 将 C1/C2 视为 paper-level 完成。

> **2026-08-03 最终修订**：上句是 2026-07-20 当时状态。IR-1—IR-4 narrow
> plain-CROWN compiler/runtime 已 validated-reduced；external α/β 仍为显式 No-Go。
> IR-5D prepared execution 已把 host overhead 从 70.263× 降到接近公平 baseline，
> 但 fresh residual-final-v3 的 Global p90 regret 仍为 1.26160×，gray 无 compiler
> Pareto，且无多预算切换。IR-5 最终 VALIDATED-NO-GO，IR-6 不启动，当前 ASPLOS
> system-performance 路线停止。最终证据见
> `gemini_doc/change_2026-07-28_ir5h_residual_final_v3_nogo.md`。

> **2026-08-03 RVIR 修订**：IR-5 后独立 correctness 路线已 CPU
> VALIDATED-REDUCED：ResNet external-semantics initial-CROWN 等价恢复，activation external
> exact call 完成 typed IR admission/dispatch artifact。该结果只修复集成语义，不撤销
> IR-5 performance No-Go，不启动 IR-6，也不形成 verifier acceleration claim。

> **2026-08-04 P0 修订**：production Schedule-memory ownership audit 为 `NO_GO`。
> Reduced residual path 能由 Schedule IR 控制 arena 与 region launch，但没有 materialization、
> storage choice 或 budget decision switch；VNN-COMP ResNet 主计算仍是 external opaque call。
> 下一工作是 `feat/native-real-network-bound-ir-v1`，不是重开 IR-5、IR-6 或孤立 TIR 调优。

> **2026-08-05 NRIR-36 修订**：NRIR-31 九子句 floor 已由 typed root-lower priority 在同一 global
> 60 秒预算内选择 top-2 clauses 2/3；三 fresh repeats 都复现相同 rank/selection，但 packed nodes=
> `[[3,3],[3,3],[3,1]]`，repeat 2 第二条未提交 atomic sibling group，final 仍 9/9 unresolved。
> 预注册 multi-clause coverage gate 失败，本阶段 `VALIDATED-NO-GO`；IR/control 可保留，但不是
> property、硬实时或 performance claim。下一门禁为 shared parametric compiler/root/evaluator 与
> stronger bound/candidate，不继续调 top-k/slice 常数；ASPLOS-ready=NO 不变。

> **2026-08-05 NRIR-42 修订**：NRIR-41 定位的 scorer ownership 重复已由 typed validated capsule
> 消除；每条 31-node queue 的 candidate enumeration 从 341 次降至 compile-only 31 次，old/new
> branch/score/child-bound/queue/state/refinement exact。Phase-A new/old median ratio=
> `0.706888/0.698486`。条件 Phase B 三 fresh global-60s queries 都完成 selected clauses 2/3 的
> `[31,31]` nodes，whole=`57.175184/57.697757/58.114412 s`，production admission
> `VALIDATED-REDUCED`。final property 仍 unknown，且没有 GPU/multi-workload/fair competitor speedup；
> ASPLOS-ready=NO 不变。下一单变量为 cross-clause/node/candidate batch Schedule。

> **2026-08-05 NRIR-43 预注册**：只允许把 NRIR-42 已 ready 的 clause/node/candidate lower work
> 改为 typed ragged batch Schedule，不改变算法、queue、预算、deadline、dtype 或 workload。先以
> sibling-node scorer pack 的 exact parity、launch `62→<=32` 和 paired queue ratio `<=0.85` 过
> Phase A；再以 two-clause ready-set 的 optimizer/scorer launch `<=16`、three-repeat whole
> `<=45 s` 且 median ratio `<=0.80` 过 Phase B。当前没有新 claim，ASPLOS-ready=NO。

> **2026-08-05 NRIR-43 关闭**：typed ragged scorer batch 的 6/6 semantic groups exact，每条 launch
> `31→16`，但 clauses 2/3 median ratio=`1.051134/1.044573`，CPU wall time 退化，故 Phase A
> `VALIDATED-NO-GO`、Phase B gated off。下一单变量为 root-projection floor Schedule；不把 launch
> reduction 当性能结果，ASPLOS-ready=NO 不变。

> **2026-08-05 NRIR-44 关闭**：typed ranking-only consumer contract 将 floor objective evaluations
> `279→9`，三轮 root/rank/selected exact，old/projected floor median ratio=`0.407530`。Phase B
> floor=`8.538814/8.622447/8.648849 s`，whole=`43.571040/44.144990/44.095736 s`，相对 NRIR-42
> median ratio=`0.764254`；两条 production queue 仍各 31 nodes 且 worst lower exact。状态为 fixed
> ResNet2B property 0 CPU8 `VALIDATED-REDUCED`，但 final 仍 unknown、无公平竞品/multi-workload/GPU
> 证据，ASPLOS-ready=NO。下一单变量来自剩余 top-2 production queue 的成本归因。
> 功能/证据提交 `437680e` 已由 PR #55 合入 `main@f194034`。

> **2026-08-05 NRIR-45 预注册**：cProfile 显示 top-2 production queue 的 per-child refinement 存在
> prepare/execute/aggregate 间重复 validation/target-selection；单 queue 246 次 `_select_targets` 中 186 次
> 来自 `Program.validate()`。下一唯一变量为 typed prepared refinement capsule，首次完整准入不删，
> refinement/optimizer/branch/queue 语义冻结。Phase A queue ratio 要求 `<=0.80`；Phase B whole trace/
> measured 要求每轮 `<=40/50 s`。当前无正式 claim，ASPLOS-ready=NO。

> **2026-08-05 NRIR-45 关闭**：typed prepared refinement capsule/receipt 将每条 31-node queue 的
> target selection=`246→98`、full Program validation=`186→38`、full hash=`217→39`；clauses 2/3
> 三轮 median ratio=`0.727519/0.736603`。Phase B whole trace=
> `31.262521/31.319772/31.470078 s`、measured=`36.396631/36.513683/36.611709 s`，相对 NRIR-44
> median ratio=`0.710268/0.615738`；每轮 `[31,31]` nodes 与 60/60 full replay exact。状态为 fixed
> ResNet2B property 0 CPU8 `VALIDATED-REDUCED`，final 仍 unknown、ASPLOS-ready=NO；下一步先做
> residual phase attribution，再冻结新的单变量。

> **2026-08-05 NRIR-46 Phase 0 NO-GO**：NRIR45 residual attribution 将 trace 拆为 floor median
> `10.818262 s` 与两条各约 `9.932808 s` packed slice；diagnostic repeat0 中 60 child prepared
> compile/execute=`5.300590/5.659414 s`。正式 Phase 0 strict static topology median=
> `1.071197 s`，低于预注册 `1.5 s` 门槛；60/60 target ledgers 全部互异。因此不实现
> Plan/Schedule Template/Instance，Phase A/B gated off，状态为 `VALIDATED-NO-GO`。formal hash=
> `712ce359…cf846`，replay/tamper 通过，`performance_claimed=false`。下一候选路线是独立预注册
> single-pass exact target admission receipt；ASPLOS-ready=NO。

> **2026-08-05 NRIR-47 Phase A NO-GO**：typed receipt/Task/Schedule、additive single-pass compiler、
> prepared binding、candidate route 与 explicit full replay 已完成；每条 candidate queue compile
> selector/reselection=`30/0`、receipt/full replay=`31/31`，correctness/ownership exact。compiler ratio=
> `0.936003 > 0.85`，clauses 2/3 queue ratio=`1.011205/1.019338 > 0.97`，故 Phase A timing 失败、
> Phase B gated off。formal hash=`a7561e51…042ce`；全量 `992 passed, 37 skipped`。candidate 不默认
> 启用，下一门禁转 top-2 production execution math/queue attribution；ASPLOS-ready=NO。

> **2026-08-05 NRIR-48 预注册**：PR #58 已将 NRIR47 NO-GO 合入 `main@1e44949`。本轮只对
> NRIR45 default production 的 clauses 2/3 做 three-fresh-process paired execution-cost attribution，
> 不启用 NRIR47 candidate、不实现优化。七类互斥成本须 `<=1%` 闭合，profile/control ratio
> `<=1.05`；dominant category 还须跨 clause/repeat 满足 `>=20%` share 与稳定性门禁。当前无结果，
> ASPLOS-ready=NO。

> **2026-08-05 NRIR-48 判定**：6/6 paired semantic exact，profile/control ratio=
> `1.023199/1.020221`。两条 clause 的 3/3 winner 均为 child refinement execute，queue share=
> `32.1966%/31.1640%`；内部 selected-CROWN 占 parent=`71.7725%/72.7291%`，为唯一过门禁子类。
> formal hash=`571c2e47…d177a4`，replay/tamper 通过。状态为 attribution `VALIDATED-REDUCED`，不是
> speedup；全量 `996 passed, 37 skipped`。当时准入的下一门禁只做 NRIR49 selected-CROWN execution，
> 该历史动作已由下段完成，不是当前指令；ASPLOS-ready=NO。

> **2026-08-06 NRIR49A G1 GPU selected-CROWN-only Opportunity判定**：RTX 4060 Laptop五fresh workers的selected-CROWN
> queue/complete share中位=`7.0986%/7.0523%`，paired perturbation中位=`0.999304/1.006747`。
> 测量有效，但20%机会门槛失败；queue 1.20x和complete 1.15x均超过Amdahl无限加速上限。
> 最大reserved仅1.353%物理显存，合法batch上限1、无OOM，memory path=`N/A`。summary hash=
> `7eefe6a7…ab50`，replay/digest通过。G1以
> `VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`关闭，selected-CROWN G2/G3 gated off；
> `1/(1-0.070986)=1.0764x`只是假设selected-CROWN region变为零耗时的deletion-only单区域上限，
> 不是BoundFlow全栈上限；只停止selected-CROWN专属TIR/JIT/融合。正式artifact的
> `next_route=gpu-winner-reselection`是冻结历史输出，已由
> `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`取代。FSG0
> schema/replay合同已以19项定向测试和`1078 passed, 3 skipped`关闭；当前下一步为FSG1 official
> original executor control full-stack trace。该阶段只建立B0分层基线，尚无BoundFlow全栈GPU性能
> claim，ASPLOS-ready=NO。

> **2026-08-04 NRIR-21 修订**：per-child exact-split objective refinement 的 IR/control、lineage
> 与 replay 已实现，但固定 ResNet clauses 0/1 的最差 depth-2 leaf lower 相对 root-global 分别
> 退化 `0.847961/0.936646`，故该策略 `VALIDATED-NO-GO`。下一方法门禁是把祖先已证明 refined
> constraints 与 child exact split-forward 单调合并后再 refinement；不得靠扩大树或单次 timing
> 掩盖当前 tightness 退化。

> **2026-08-04 NRIR-22 修订**：祖先 refinement execution 已成为 child Plan/Task/Schedule 的
> typed constraint source。固定 clauses 0/1 的 worst leaf 相对 independent 提升
> `73.615173/75.022095`，相对 root-global 提升 `72.767212/74.085449`；fixed-tree tightness
> `VALIDATED-REDUCED`。所有 worst leaf 仍负，下一门禁是 hard-clause/depth/node convergence
> 曲线，不得升级 complete verifier 或 performance/ASPLOS-ready claim。

> **2026-08-04 NRIR-23 修订**：external-owned intermediate constraints 已通过 typed seed
> IR 接入 native root refinement，再由 validated parent execution 逐节点传递。固定 clauses
> `0/2/4` 的 ancestral worst leaf 相对 external baseline 改善
> `0.001512/0.001133/0.000534`，相对 seeded root-global 为
> `0.000823/0.000004/0`；全部 terminal leaves 仍负。本阶段 `VALIDATED-REDUCED`，下一门禁为
> external-seeded depth/node convergence，不形成 property/performance/CUDA/ASPLOS-ready claim。

> **2026-08-04 NRIR-24 修订**：固定 `7/15/31 nodes × depth 2/3/4` 的九单元 fresh-process
> convergence matrix 已完成。clauses `0/2/4` worst terminal lower 均随预算严格改善，depth-4 为
> `-0.282360/-0.401845/-0.459939`，但仍无 bounded-tree closure。结论只为 convergence trend
> `VALIDATED-REDUCED`；下一方法门禁是 dynamic ancestral refinement budget/multi-pass，不把继续
> 堆 fixed depth 当作 complete-verifier 或性能证据。

> **2026-08-04 NRIR-1 修订**：上述 native correctness 门禁已通过。固定 ResNet2B 的
> 17-op Primal graph 已 lower/execute 为 21-op native Bound/Task/Schedule path，五层 hash 绑定
> external-bound payload，final lower max diff `7.15256e-7`、sign 9/9。当前仍是单 dense
> storage/full batch、0 materialization、CPU only。下一工作收窄为 NRIR-2 real-graph 多计划与
> budget decision switch；未通过前不得启动性能主张。

> **2026-08-04 NRIR-2 修订**：real-graph storage-axis 门禁已通过。同一 ResNet
> Bound IR/PlanTemplate 在高/低预算下选择 retain-all（`1,860,912` B）与 lifetime-reuse
>（`442,656` B）；低内存 runtime 提前释放 85 个值，双计划 bitwise equal。该结果仍不包含
> CUDA allocator peak、OOM rescue、latency、real materialization 或 sliced batching，故顶层
> performance No-Go 与 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-3 修订**：双 storage 的 fresh-process CUDA physical-memory protocol 已
> 冻结并实现：5 repeats × 5 warmup × 20 measured、prepared lower-only timing、allocator
> allocated/reserved delta、交替进程顺序、identity/replay 与 20%/1.20× 门禁。本机 CUDA
> driver/device 不可用，只有 `environment_unavailable` probe，0 measured rows、无 performance
> claim。下一工程路线为 representation semantic binding，而不是停等 GPU 或调整冻结阈值。

> **2026-08-04 NRIR-4 修订**：representation semantic binding 已在固定 ResNet 上执行。
> structured-affine source policy 生成 14 cast + 14 materialize 的独立 49-op execution Bound
> graph；每个 transition 均绑定 source Schedule action、Task 与 Launch。dense/structured lower
> 最大差 `9.53674e-7`，external sign 9/9。当前 operator/storage 仍是 dense-equivalent，故只把
> C1/C2 mechanism 升为 validated-reduced，不形成 compression 或 performance claim。下一工程
> 路线为 real-network sliced batch execution。

> **2026-08-04 NRIR-5 修订**：spec-axis BatchDecision 已真实驱动 source Schedule
> `[0,3)/[3,6)/[6,9)`、三个独立 21-op child compiler stack 与结果聚合；full/sliced lower
> 最大差 `1.90735e-6`，external sign 9/9，artifact generate/replay 通过。该结果仍只为
> correctness/ownership：domain/sample、representation × batch composition、physical
> memory/latency/CUDA 均 pending。全量 `508 passed, 37 skipped`，correctness/integration
> VALIDATED-REDUCED；顶层 performance No-Go 不变。

> **2026-08-04 NRIR-6 修订**：representation/storage × spec-batch 已由同一 source
> template/selector 联合选择并执行。四组合 child op/task/launch=`21/63/49/147`，四路径
> external sign 9/9，全量 `522 passed, 37 skipped`。这关闭 cross-axis ownership，不证明
> 跨 query/domain batching、cache 或物理性能；顶层 performance No-Go 不变。

> **2026-08-04 NRIR-7 修订**：9 个 ResNet property objectives 已形成显式 query stream；
> packed size-3 执行 3 child，same-policy serial 执行 9 child，exact cache first miss/second hit，
> 9/9 lineage 恢复。packed/serial max diff `3.21865e-6`，全量 `540 passed, 37 skipped`。
> 这只关闭同域 property-query mechanism；BaB domain validity 与性能 No-Go 不变。

> **2026-08-04 NRIR-8 修订**：固定 ResNet root box 已形成 8 个不同 leaf domains；leaf exact
> state 独立重算，parent 只允许 warm-start。full/packed-4/serial 分别执行 1/2/8 child stacks，
> lower/upper bitwise equal、8/8 lineage 恢复。该结果关闭 input-domain execution mechanism，
> 但没有 ReLU/β split queue、prune、termination 或任何 performance 证据；顶层 No-Go 不变。

> **2026-08-04 NRIR-9 修订**：native plain-CROWN 已支持 first-class ReLU split inputs 与
> deterministic best-first bounded queue。固定 ResNet 7-node run 的 packed/serial stacks 为 3/7，
> bounds allclose、queue/branch/split identity 一致，终止状态明确为 budget-exhausted/not-claimed。
> 该结果关闭 split/queue/control-flow ownership，不包含 α/β optimization、完整 verdict 或性能；
> 顶层 No-Go 不变，下一门禁为 native α/β state 与 warm-start validity。

> **2026-08-04 NRIR-10 修订**：frozen alpha/beta state、beta split constraint 与 warm-start
> validity 已进入 native Bound/Plan/Task/Schedule。固定 ResNet native/legacy αβ bounds bitwise
> 相同，非零 beta 对 zero-beta lower 改善 `0.34039306640625`；parent→child monotonic split 只允许
> initialization。Adam loop 仍 runtime-owned，无完整 verdict/performance；下一门禁为 native
> optimizer-step Task/Schedule control。

> **2026-08-04 NRIR-11—14 修订**：fixed-step optimizer Schedule 已接回 multi-node queue，
> three-state verdict、concrete witness replay、九子句 conjunction、deterministic candidate search、
> unsafe short-circuit 与 cooperative deadline 均形成 typed/replayable control contract。固定
> ResNet 九子句仍因 native scalarized bounds 过松而 9/9 unknown；因此只关闭 correctness/control
> VALIDATED-REDUCED。下一阶段是公平 end-to-end phase/tightness baseline，不能从这些 artifact
> 推导 speedup 或 ASPLOS-ready。

> **2026-08-04 NRIR-15 修订**：external intermediate semantics 与 adaptive α 已贯穿 optimized
> queue/query。固定 ResNet 从 local 0/9 提升为 6/9 verified，仅 clauses 0/2/4 unknown；九个
> lower 对 frozen external initial 无退化。三组 CPU audit queue 均约 6.7 s，candidate/verdict
> 只有约 3.6/3.9 ms，定位 fixed compile/hash/selected-native re-execution 为主耗时。下一工程
> 门禁为 prepared production fast path，之后再推进 hard-clause branching；顶层 performance
> No-Go 与 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-16 修订**：root-only exact prepared capsules 将 fixed ResNet 三组
> complete-query warm median 从 audit `59.078 s` 降至 `110.950 ms`；cold prepare+first=
> `16.139 s`，retained payload=`2.076 MB`。production/audit lower max diff=`1.90735e-6`、
> candidate/status exact，仍为 6/9。该结果只证明单 workload CPU internal evidence-overhead
> removal，不是 competitor speedup；下一门禁为 clauses 0/2/4 branching/stronger-bound，顶层
> ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-17 修订**：objective-aware branching 已 lower 为 first-class
> Plan/Task/Schedule。same-budget clauses `0/2/4` worst leaf 相对 widest 分别改善
> `0.120752/0.071564/0.057901`，但所有 terminal leaves 仍为负，整体只 6/9。该结果为
> fixed-budget tightness `VALIDATED-REDUCED`，不是完整验证或 performance claim。下一门禁为
> 多 workload/设备/竞品 E2E 与 stronger-bound；顶层 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-18 修订**：原生 VNNLIB Query IR 与三 workload Plan/Task/Schedule 已在
> MNISTFC、ResNet2B、OVAL21 上运行。BoundFlow 为 unknown×3，固定 αβ-CROWN 为
> verified/unknown/verified；单次 CPU E2E 不形成 speedup。ResNet native local root lower=
> `-543.717/-789.331`，说明下一优先级是 intermediate-bound refinement，而不是 CUDA timing、
> 更多 audit overhead removal 或继续堆 fixed-tree depth。顶层 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-19 修订**：native selected-CROWN intermediate refinement 已进入
> Plan/Task/Schedule，并在同 policy 下令 MNISTFC unresolved `3→1`、OVAL21
> `unknown→verified`；ResNet root lower 改善 `+70.496/+160.551` 但仍 unknown。该结果为
> multiworkload tightness `VALIDATED-REDUCED`，不是性能或 3/3 closure。下一路线只推进
> objective-directed target selection（已由下段 NRIR-20 完成）；顶层 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-20 修订**：clause-sensitive CROWN influence×width selection 已成为
> refinement Plan/Task/Schedule 的显式语义。固定 ResNet clauses 0/1 在相同 96-target 预算下，
> root lower 相对 width policy 再改善 `+55.928741/+26.228943`，但仍为负，未关闭 property。
> 该结果只为 fixed-root tightness `VALIDATED-REDUCED`；下一门禁是 per-child exact-state
> refinement，顶层 performance No-Go 与 ASPLOS-ready=NO 不变。

> **2026-08-04 NRIR-11 修订**：fixed-step optimizer control 已进入 typed Plan/Task/Schedule。
> 固定 ResNet 1-step program 为 8 actions，Schedule/legacy/final native execution max diff 均为 0，
> alpha/beta gradient 均非零。该结果只关闭 optimizer control ownership；dynamic early stop、
> multi-node BaB integration、complete verdict 与性能仍缺。下一门禁为 optimizer Schedule ×
> ReLU-split queue integration。

> **2026-08-04 NRIR-12 修订**：optimizer Schedule 已进入每个 native ReLU-split queue node batch，
> selected state 再经 native compiler执行；fixed ResNet 为 7 nodes/3 expands/4 frontier、packed/
> serial 3/7 stacks，bounds/state tensors 在冻结容差内。该结果仍 budget-exhausted/not-claimed；
> 下一门禁是 sound property termination/verdict，不得直接升级 verifier 或性能 claim。

---

## 0. 执行摘要

BoundFlow 的 ASPLOS 论文不能被定义为“用 TVM 加速几个神经网络验证算子”，也不能把
“支持 CROWN、一般 DAG、GPU 或 BaB batching”本身作为核心新意。ASPLOS 版本的统一命题是：

> **BoundFlow 是一个面向重复神经网络边界查询的 verification-aware compiler and runtime。它以 operator-preserving 的线性界表示延迟显式系数张量的构造，由全局 Planner 联合决定物化、融合、批处理、缓存、重算与显存布局，并由 host runtime 在 CROWN、αβ-CROWN、BaB 和 certified training 的相关查询之间复用计划与状态。**

论文的系统研究问题不是“某个 bound 算法能否实现”，而是：

> 当验证工作负载反复生成高度相关、形状规则但容易爆炸的线性算子查询时，编译器应保留什么结构、何时物化、如何跨查询调度，才能改善吞吐、尾延迟和峰值显存，同时保持数值 soundness 与 bound tightness？

这一定位符合 ASPLOS 对体系结构、操作系统或程序语言研究“必须有实质推进”的要求；官方
CFP 明确指出，仅推进其他领域并使用系统技术并不足够，而且 rapid review 只阅读前两页，
多数投稿可能无法进入完整评审。因此，论文前两页必须把“新系统抽象 + 全局决策 + 端到端
收益”讲完整，而不能从验证算法背景或 TVM 工程细节开始。

---

## 1. 论文北极星

### 1.1 一句话论文主张

> **Preserving linear-bound operators across repeated verification queries enables a compiler/runtime to jointly optimize materialization, memory, and batching decisions that eager tensor execution cannot coordinate.**

### 1.2 三项核心贡献

#### C1. Structured Bound-Operator IR with Explicit Materialization Semantics

将 CROWN backward 中的系数对象 `A` 表示为带显式物化语义的结构化线性算子 DAG。目标是
尽可能保留结构，而不是承诺永不生成 dense tensor。ASPLOS 版本至少覆盖：

- `linear` / right-matmul；
- `conv2d` / transpose-convolution composition；
- `reshape` / `flatten` / layout-preserving view；
- `add` merge；
- `concat` / slice；
- ReLU sign-split、relaxation slope/intercept 与 α/β 修正；
- 必要的 row-norm / concretization 接口。

贡献不在于重新定义 CROWN 数学，而在于提供一个既保持参考计算、又暴露系统优化机会的
编译器表示。IR 必须显式表达：结构、形状、批维、spec/domain 维、materialization barrier、
reason、估算字节数、生命周期、复用关系和 dense reference semantics。ReLU sign selection、
row norm、concretization 等允许局部物化，但所有 fallback 必须可观察、可计量并可由 Planner
选择物化位置、分块与生命周期。

#### C2. Method-, Autograd- and Memory-Aware Materialization Planner

Planner 不只是普通 graph fusion。它需要在同一代价模型和显存预算下联合决定：

- 保持 lazy 还是物化；
- 在哪个 barrier、哪个 batch 粒度物化；
- task partition 与 fusion 边界；
- spec batch、domain/BaB-node batch 的组织方式；
- 哪些中间状态缓存、复用或重算；
- logical buffer 到 physical buffer 的映射；
- 在峰值显存约束下的调度顺序与 fallback。

PR-10 已否定“structured 应成为统一默认表示”的假设：在代表性 plain CROWN 大点上，
structured 将峰值显存降低约 29.8%，但慢约 9.17×；在 α/αβ 路径中，structured 的显存
反而恶化并产生 6 个 OOM。因此 Planner 必须显式区分 `bound_method`、`requires_grad` 和
`optimization_stage`，不能把它们折叠成单一 method 标签。至少还应观察 alpha/beta enable、
split state、query reuse、spec/domain batch 与目标设备 capability。

规划问题的输入定义为 `(G, Q, H, B, R)`：operator DAG、查询集合或分布、硬件 profile、
显存预算和参考 bound 配置。计划定义为 `P=(m, π, f, b, c, r, s)`：materialization、
partition、fusion、batch layout、cache、recompute、storage/scheduling。

PR-11 v1 采用“先可运行、再最快”的字典序目标，而不是用任意权重把时间和显存相加：

```text
1. 过滤不满足 capability/correctness 约束的候选；
2. 要求 `M_pred(P) <= η * min(B_user, M_available)`，初始 `η` 取 0.85–0.9；
3. 在可行计划中最小化 amortized compile + execute + queue + transfer latency；
4. 若无可行计划，减小 spec/domain batch 后重新规划；仍不可行时输出结构化 OOM diagnosis。
```

所有 planned path 必须在相同浮点语义下保持 dense reference computation。实现不承诺一次
精确联合求解全部变量，而采用：

```text
candidate generation
  → staged cost-aware heuristic
  → local greedy baseline
  → global heuristic
  → small-graph exhaustive oracle
```

必须至少实现一个会根据 shape、batch、reuse count、query distribution 或 memory budget 选择
不同计划的非平凡 Planner；仅提供手动开关或固定启发式不足以支撑该贡献。

#### C3. BaB-Oriented Repeated-Query Runtime for Multi-Spec and Domain Batches

首篇只将以下场景统一为相关 bound query 流：

- multi-spec verification；
- BaB node batches；
- dynamic BaB domain batches。

host runtime 保留搜索、优先队列、超时、分支状态和动态 batching；TVM Relax/TIR 只执行
粗粒度、批量的 bound tasks。BaB 控制流不写入 Relax。Runtime 必须支持 query compatibility、
计划/kernel cache、状态版本、parent-to-child warm start、批量合并/拆分、OOM fallback、失败
隔离和可观测性。certified training 只作为第二客户端；epsilon sweep、checkpoint reuse、
incremental verification、persistent GPU BaB 与 multi-GPU 属于扩展或未来工作。

### 1.3 北极星指标

主指标按优先级排序：

1. **给定显存预算下的 repeated-query throughput**；
2. **BaB time-to-verify 与 p90/p99 node latency**；
3. **峰值 GPU memory / 最大可运行 batch 或网络规模**；
4. compile、first-run、cold、warm 分离后的端到端时间；
5. 相同 tightness/soundness 下的速度，或相同时间预算下的 verified instances；
6. certified training 的 step time、峰值显存和可训练规模。

不能只报告单个 kernel 的平均延迟，也不能以 compile time 被 warm cache 隐藏后的数字作为
headline result。

---

## 2. 与现有工作的边界

### 2.1 不能作为 BoundFlow 核心新意的内容

- 自动从一般计算图派生 bound；
- 支持 IBP、CROWN、α-CROWN、β-CROWN 或 GCP-CROWN；
- 支持一般 DAG、CNN、ResNet 或一般非线性；
- 在 GPU 上运行 bound propagation；
- BaB node batching；
- 将 Python 实现改写为 C++；
- 将 neuron-level certifier 描述自动变成 tensor implementation。

auto_LiRPA 已覆盖一般计算图上的自动 LiRPA，并作为 α,β-CROWN 的核心库；α,β-CROWN
已经提供 GPU bound propagation、BaB 和广泛模型支持。2025 年的 tensor-based certifier
compiler 已针对 neuron-level specification → tensor implementation 提出专用 IR、shape analysis
和稀疏运行时。2026 年 Luna 又提供了 C++ 的一般图 IBP/CROWN/α-CROWN propagator。

因此 BoundFlow 的差异必须收敛到：

1. **operator representation**：保留而非提前打平线性界结构；
2. **global materialization planning**：跨算子、跨任务、跨查询联合决策；
3. **compiler/runtime co-design**：针对 repeated bound queries 的动态批处理与状态复用；
4. **evidence**：在完整 solver/training 场景而非 toy kernel 上证明收益。

### 2.2 TVM 的角色

TVM 是可替换后端，不是 BoundFlow 的核心抽象：

- Primal IR / Bound IR / Task IR 不得依赖 Relax 的表达边界；
- Planner 的物化、复用和 batch 决策必须在进入 TVM 前可解释、可测试；
- TVM 负责粗粒度 task lowering、fusion、TIR/CUDA code generation 与执行；
- 论文需通过 Python reference backend 与至少一个 TVM backend 证明抽象独立性；
- “TVM 默认 pass 已有的收益”必须作为 baseline 分离，不能算作 BoundFlow 贡献。

### 2.3 certified training 的位置

certified training 是重要 repeated-query 应用，但不是唯一故事。CROWN-IBP 已说明 tight linear
relaxation 与 IBP 在稳定性、紧度、时间和显存之间存在关键权衡。BoundFlow 的目标是改善这条
成本—规模 Pareto frontier，不提出新的训练算法，也不以最终鲁棒准确率单独证明系统贡献。

### 2.4 Query state 与 cache validity

Runtime 不能笼统声称相关查询可以共享中间状态。每个缓存对象必须标记为：

- `EXACT_REUSE`：可直接作为当前查询的有效结果；
- `CONDITIONAL_REUSE`：只有 key/version/shape 等条件满足时可直接复用；
- `WARM_START_ONLY`：只能初始化后续求解，不能当作当前查询的精确结果；
- `INVALIDATE`：必须失效并重算。

| 对象 | Multi-spec | BaB 父→子 | 参数更新后 |
|---|---|---|---|
| Primal/Bound 图结构 | EXACT_REUSE | EXACT_REUSE | EXACT_REUSE |
| Planner 计划模板 | EXACT_REUSE | EXACT_REUSE | CONDITIONAL_REUSE |
| 编译 kernel | EXACT_REUSE | EXACT_REUSE | shape/dtype 不变时 CONDITIONAL_REUSE |
| 参数相关常量折叠 | EXACT_REUSE | EXACT_REUSE | INVALIDATE |
| intermediate bounds | CONDITIONAL_REUSE | WARM_START_ONLY 或 INVALIDATE | INVALIDATE |
| α 参数 | CONDITIONAL_REUSE | WARM_START_ONLY | 通常 INVALIDATE |
| β/split state | INVALIDATE | 子节点专属 | INVALIDATE |
| 最终输出 bounds | INVALIDATE | INVALIDATE | INVALIDATE |

尤其不能把父节点 intermediate bounds 直接视为子节点的有效精确结果；split constraint 会改变
后续传播语义，父状态最多作为 warm start 或参考。

### 2.5 三层 correctness/soundness 术语

1. **数学 soundness**：由 CROWN/IBP/αβ abstract transformer 与 solver 规则保证；
2. **编译变换语义保持**：dense、operator、planned、fused path 在相同浮点执行语义下保持
   reference bound computation；
3. **实现验证**：通过 dense reference、allclose、gradient comparison、auto_LiRPA
   comparison、sampled concrete sanity 与 deterministic replay 建立证据。

除非未来加入 outward rounding、误差 envelope 或 proof checker，否则不得宣称 GPU FP32
编译路径对实数语义具有严格 numerical soundness。论文统一优先使用：

> preserving the reference bound computation under the same floating-point semantics

---

## 3. 当前代码基线与缺口

### 3.1 已完成并可复用的基础

| 层次 | 当前状态 | ASPLOS 价值 |
|---|---|---|
| Frontend / Primal IR | Torch、ONNX、normalize、general DAG 子集 | 提供真实图输入与双前端一致性 |
| Bound/runtime semantics | IBP、CROWN-IBP、α-CROWN、αβ-CROWN、BaB | 作为系统优化的正确性载体，不作为算法贡献 |
| LinearOperator | dense、right-matmul、conv、reshape、SignSplit、deterministic dump | C1 已完成 PR-10 基础 |
| General DAG | residual add、concat；PR-9 去掉 merge/slice dense fallback | 已完成第一批 operator-preserving path |
| Planner | task graph、partition、liveness、storage reuse、memory stats | C2 的工程基础，但尚未形成 materialization 联合决策 |
| TVM backend | Relax/TIR、compile cache、fusion、memory-plan 对照 | C2/C3 的执行后端 |
| Runtime | multi-spec、α/β、BaB node batch/cache/prune | C3 的起点，但尚未统一为 query abstraction |
| Artifact | JSONL schema、CSV、figure、manifest、quick/full runner | ASPLOS 证据链基础 |
| 环境 | PyTorch 2.12.1+cu132、LLVM 20.1.8、TVM、单一 tvm-ffi | 可复现实验基础 |
| Native real-network compiler | ResNet2B native IR；joint policy；query/domain batching；optimized ReLU-split queue；frozen α/β state；optimizer Schedule；sound verdict；complete query control | C1/C2/C3 的真实图 correctness/decision/query/control-flow 载体；real proof tightness 与 device-level 性能仍缺 |

Gate 0 与 PR-10 是历史已完成节点；当前 integration base 已推进到 `f191034`，NRIR-1—19 已合并，
NRIR-20 在当前 feature branch 完成验证，NRIR-3 CUDA protocol 已完成但本机 device unavailable。PR-10 的 structured 路径保留为 opt-in research capability，dense
继续作为默认；不得把历史基线 `263ea81` 当作当前工程入口。

### 3.2 论文成立前必须补齐的缺口

| 缺口 | 当前问题 | 必须达到的证据 |
|---|---|---|
| ReLU barrier | structured mode 已消除 persistent dense；dense 保持默认 | 需 Planner/fused lowering 解决 eager 重算与 α/β OOM |
| 物化决策 | NRIR-6 已联合 NRIR-4 的 28 transitions 与 NRIR-5 的 spec child execution；structured storage 仍 dense-equivalent | 在真实 repeated-query/domain stream 后，于可用 CUDA 设备按冻结协议测物理 memory/latency；无物理证据不升级 Pareto |
| fused CROWN task | TVM 后端以 IBP/task 基础设施为主 | CROWN 粗粒度 task lowering 与正确性/性能门禁 |
| repeated-query abstraction | NRIR-7/8 有 property/domain batching；NRIR-9—14 已形成 optimized split queue、sound verdict 与 complete query control | 先量化 fixed real query 的 proof gap 与 phase timing，再提升 tightness/closure 并建立公平 same-solver/竞品 baseline |
| 真实 workload | VNN-COMP ResNet2B correctness/storage 已进入 native IR；性能仍无真实 device protocol | ResNet/basic-block、更多 VNN-COMP 代表实例、至少一个训练 workload 的公平性能证据 |
| headline result | 当前结果证明链路正确，不证明系统主张 | 端到端吞吐/显存/TTVerify 的显著、可解释收益 |
| baseline 完整性 | 已有 auto_LiRPA/TVM 对照，但缺 Luna/系统竞品定位 | 公平版本、硬件、算法/tightness 和计时口径 |

---

## 4. 研发路线与每个 PR 的论文义务

### Gate 0：冻结环境迁移与基线（开始 PR-10 前）

目标：把当前未提交的 CUDA 13.2/LLVM/TVM/钩子/reshape 工作整理为独立工程提交。

验收：

- 去除 `crown_ibp.py` 全文件格式化噪声，只保留必要兼容改动；
- 激活/反激活、nvcc、TVM CUDA、TVM↔Triton ABI smoke 全通过；
- 全量测试重新记录；
- MLP/CNN reduced baseline 形成稳定多次计时，而非单次 quick；
- 建立统一 build/run workflow 文档。

论文义务：只作为 artifact foundation，不宣称研究贡献。

### PR-10：Structured ReLU Barrier 与 Materialization Instrumentation

目标：先建立 materialization instrumentation，再保持现有 ReLU/α/β 数学语义，将全局 dense
fallback 改成结构化 operator transform，必要时仅局部物化。PR-10 是 representation-enabling
PR，不要求 Python lazy path 当场变快。

实现要点：

- 表达 sign-dependent row scaling 与 bias accumulation；
- stable/unstable ReLU 分路；
- α/β 参数与 split constraint 不复制顶层 solver；
- operator composition 可继续穿过 chain CNN 与 general DAG；
- 首先增加 materialization reason/count/estimated bytes/lifetime trace；
- dense reference path 保持可独立运行；
- 增加 α gradient comparison。

验收指标：

- 数值与当前 dense reference 对齐，α gradient 对齐；
- chain CNN、residual DAG、α、αβ、BaB 回归全通过；
- 主 coefficient 不永久退化为 dense；
- fallback 全部可追踪，并明确减少 dense bytes/materialization count；
- 不接受无法解释的严重 runtime 或显存退化；
- 端到端速度硬门槛放在 PR-12 fused lowering。

### PR-11：Method- and Autograd-Aware Materialization Planner

目标：把“是否/何处物化”从 operator 内部硬编码提升为全局 Planner 决策。

需要新增或明确：

- `MaterializationContext`：bound method、`requires_grad`、optimization stage、alpha/beta/split
  state、batch/spec/domain axes、operator summary、memory budget/available memory、reuse 与 target；
- `OptimizationStage` 至少区分 inference、alpha init/optimize/reuse、final bound、BaB node eval、
  training；不得只从 `requires_grad` 反推阶段；
- operator cost summary：shape、estimated FLOPs、dense/structured/temporary bytes、reuse count、
  batch axes、operator depth/nodes 与 autograd state estimate；
- `MaterializationDecision` / `MaterializationPlan` 及确定性的 reason/plan dump；
- v1 候选仅包含 `DENSE`、`STRUCTURED`、`REDUCE_BATCH`；
- capability constraint：当前 structured autograd 与 optimized-bound structured 未通过门禁，
  因而 α/αβ optimize 或其他 requires-grad 路径不得选择 structured；
- memory budget、safety margin 与 deterministic reduce-batch/re-plan/OOM diagnosis；
- deterministic heuristic baseline；
- 至少一个 cost-aware 自动策略；
- 解释性 cost model 优先采用 profile lookup/shape bucket 或 piecewise linear model，不以黑盒
  预测器作为 v1 前置条件。

必须做的消融：

1. Always Dense；
2. Always Structured；
3. Method-Only；
4. Memory-Threshold；
5. Local Greedy；
6. Global Planner；
7. 每个 case 实测所有合法候选的 Oracle；
8. Global Planner 在多个显存预算下。

数据不得随机拆分相邻 shape；采用 workload-family held-out 或 leave-one-architecture-out，并将
mini-ResNet 与 unseen memory budgets 保留为最终测试。验收要求：0 bound/gradient correctness
failure、0 unexpected OOM；任一合法候选可运行时 Planner 应找到可行计划；α/αβ structured
不得被误选；held-out median latency regret 相对 Oracle 的研发目标不超过 20%，同时报告 p90；
至少选择过 dense 与 structured，并在至少一个预算下让 Always Dense OOM 的 plain CROWN case
通过。上述 regret 数值是内部 Go/No-Go 目标，不预写成论文结果。

### PR-12：Fused TVM CROWN-Task Lowering

执行版进一步收敛为 **Fused CROWN-Task Lowering for Memory-Efficient Plain CROWN**：v1 只支持
static-shape、FP32、CUDA、`requires_grad=False` 的 plain CROWN，先覆盖 ReLU+Linear，再覆盖
ReLU+Conv2d；α/αβ autograd、training 与新 BaB scheduling 明确排除。PR-11 高-regret 归因与
详细门禁见 `gemini_doc/pr11_regret_attribution_2026_07_13.md` 和
`gemini_doc/pr12_fused_crown_task_plan_2026_07_13.md`。

目标：将 Planner 选择后的 operator region 降到粗粒度 TVM task，而不是逐小算子调用。

范围：

- linear/conv/ReLU/view/add/concat 的代表性 fused region；
- static shape first，dynamic batch 只在必要处引入；
- compile cache key 包含 operator DAG、shape、dtype、batch axes、materialization plan；
- Python reference、TVM unfused、TVM fused 三方对齐；
- compile/cold/warm 分开统计。

验收：证明收益来自 BoundFlow region/plan，而非仅来自 TVM 默认 fusion；报告 kernel launch、
intermediate bytes 和 compile amortization break-even point。

PR-12 分为两项 capability：plain CROWN fused structured path 用于减少 Python dispatch 与重复
临时物化；differentiable optimized-bound path 必须定义独立的 forward、saved-state、backward
和 autograd registration contract。在完成 forward/gradient equivalence、gradcheck/opcheck、
saved-tensor profile、优化收敛和 peak-memory 门禁前，不向 PR-11 暴露
`FUSED_STRUCTURED_AUTOGRAD` 候选。

截至 2026-07-13，PR-12D single-consumer fusion、general-DAG fallback 与跨 runtime CUDA stream
correctness 已关闭；PR-12E/F v1 的正式证据保留了 memory-sensitive Linear、unseen Conv 与
mini-ResNet 的 latency 负结果。PR-12G 又增加 budgeted chunked 候选并冻结全新 held-out-v2：
5/5 预算可行、0 unsafe，Planner median/p90 regret 1.000×/1.054×，在 eager/chunked/TIR 间做出
1/2/2 的非平凡选择。该结果关闭 reduced Planner quality，但 selected geomean 相对 eager 仅
1.081×，structured eager/TVM-unfused、profiler 与 2× headline 当时仍缺失。PR-12 后续 H–N 已
完成公平 baseline、compile/cache、CUPTI activity、止损和 compile-aware multi-budget Planner，
最终以 `VALIDATED-REDUCED` 关闭在 `3492d79` / `pr12-validated-reduced`。PR-13 已获 GO 并进入
13A；详细 closure 见 `gemini_doc/pr12_closure_audit_2026_07_14.md`。

### PR-13：Multi-Domain Runtime 与真实 BaB Adapter

目标：统一 repeated-query execution，并接入真实 BaB query stream。

截至 2026-07-14，PR-13A/B 已完成 state-versioned query/compatibility/validity contract、dynamic
budget/deadline batcher、OOM 二分和 physical αβ dense pack/unpack。真实 BaB driver 的 8-query
smoke fixed/dynamic replay 均为 8/8、0 mismatch/loss。该证据只标记 validated foundation。
PR-13C 又把 runtime 作为 optional bound-call adapter 接回同一 solver，αβ smoke 的 query/
state/branch/status/node counters 7/7 对齐。PR-13D/E 已完成 RTX 4060 reduced fixed/E2E 并以
`VALIDATED-REDUCED` 关闭：fixed 与 hard E2E 相对 per-node 为 96.52×/9.93×，但 hard 相对公平
batched original 仅 0.980×，说明收益来自 batching。non-toy multi-backend/TTV 仍未完成。持续状态见
`gemini_doc/pr13_execution_status.md`。

最小抽象：

- `QueryState`：输入域、spec、α/β、split、版本；
- `QueryBatch`：可合批性与 batch-axis metadata；
- `CompiledPlanKey`：结构与动态状态分离；
- scheduler：形成/拆分 batch、timeout、OOM fallback；
- cache：forward bound、operator plan、compiled module、warm-start state；
- observable counters：cache hit、batch fill、queue wait、compute、prune。

验收：

- 不只使用合成 node list，而是由现有 BaB driver 产生查询；
- 输出与逐节点 reference 一致；
- 报告 time-to-verify、verified/timeout 数、p50/p90/p99 node latency、batch utilization；
- 分离 batching、cache、planner、fusion 各自收益。

### PR-14：Certified Training Adapter

目标：证明同一 compiler/runtime abstraction 能覆盖参数周期性变化的重复查询。

范围控制：

- 不提出新训练目标；
- 先支持一个 CROWN-IBP/IBP-CROWN 训练配方；
- 明确参数更新后哪些 plan 可复用、哪些 compiled code 可复用、哪些 bound state 必须失效；
- 比较 eager auto_LiRPA 或仓库内 reference。

验收：训练 loss/gradient 数值门禁、step time、peak memory、最大 batch/模型规模；最终准确率只作
sanity check，不作为唯一 headline。

### PR-15：Workload、消融与 Artifact 封箱

目标：形成论文表图、匿名 artifact 和复现实验。

要求：

- 固定 workload/version/hash；
- 固定 warmup/repetition/timeout/seed；
- smoke/reduced/full 三档；
- 所有表图只从原始 JSONL 生成；
- 失败、OOM、timeout 也写结构化记录；
- 自动生成 manifest、claims map 和 expected outputs；
- 独立机器或干净环境复跑 reduced workflow。

---

## 5. 实验设计

### 5.1 Workload 梯度

#### A. 语义与 microbenchmark

- MLP、MNIST CNN；
- residual basic block；
- concat/branch DAG；
- isolated linear/conv/ReLU operator chain；
- 可控 spec count、domain count、稳定 ReLU 比例与显存预算。

用途：机制解释、代价模型校准、回归，不支撑 headline。

#### B. 中等规模系统 workload

- CIFAR-10 CNN / ResNet-like model；
- 至少一个 VNN-COMP 风格 ONNX + VNNLIB workload 子集；
- multi-spec 与 BaB domain scaling；
- 真实 BaB node stream。

#### C. 两个主应用场景

1. **主场景——完整/限时验证**：当前算子覆盖可支撑的 ReLU CNN/ResNet VNN-COMP
   representative instances，使用真实 ONNX、VNN-LIB、BaB query stream 和 timeout；
2. **第二客户端——CROWN-IBP certified training**：一个 CIFAR-10 配方，报告 step time、peak
   memory、compile amortization、loss/gradient 对齐和 verified-accuracy sanity。

首篇不加入 ViT、Transformer、控制器或大量新非线性；若资源允许，更大 ResNet/TinyImageNet
只能在主链闭环后作为扩展。

### 5.2 Baseline

必需 baseline：

- PyTorch eager；
- `torch.compile` / TorchInductor（支持则报告结果，不支持则记录失败原因）；
- auto_LiRPA eager bound propagation；
- α,β-CROWN 对应 solver 路径；
- BoundFlow Python dense reference；
- BoundFlow always-lazy / always-materialize / fixed barrier；
- BoundFlow local planner / global planner / small-graph oracle；
- TVM unfused / default pipeline；
- BoundFlow planner + fused backend；
- **相同 α,β-CROWN host solver + original executor 与 BoundFlow executor**；
- Luna：若其公开实现、模型和方法可公平复现，则作为 general-graph/C++ propagator baseline；
  否则在 related work 中定性比较并明确不可复现原因。

公平性约束：同一模型、输入域、spec、dtype、device、bound method、优化步数、timeout 与正确性
容差。算法参数不同的结果不能被解释为系统速度差。

### 5.3 表与图的最小集合

1. 端到端主表：吞吐、peak memory、time-to-verify、verified/timeout；
2. Planner 消融表：eager/lazy/local/global × memory budget；
3. 机制图：materialized bytes、operator depth、launch count、cache hit；
4. repeated-query scaling：spec/domain/node batch size；
5. compile amortization：query count 对总时间的影响；
6. Pareto 图：latency/throughput vs peak memory；
7. training 图：step time/peak memory vs batch/model size；
8. tightness/soundness 表：确保系统优化不改变算法结果。

### 5.4 统计口径

- 至少 5 次独立重复；长时间 E2E 可按实例集合统计并给置信区间；
- 报 median、p90，尾延迟场景报告 p99；
- compile、first-run、cold、warm、queue wait 分离；
- 明确 CPU/GPU 同步点；
- GPU peak memory 使用统一采样/allocator 口径；
- OOM、timeout 不删除，进入结果表；
- headline 数字必须可从 JSONL 自动追溯到命令、commit、环境和图表。

---

## 6. 论文结构与 rapid-review 约束

### 6.1 前两页必须独立成立

ASPLOS 2027 rapid review 明确只看前两页。前两页建议固定为：

1. **问题与规模证据**：重复边界查询中的 eager `A` materialization、显存和 launch/cache
   瓶颈，配一张真实 profile 图；
2. **关键洞察**：`A` 不是普通 dense tensor，而是跨查询可组合、可延迟、可复用的 operator DAG；
3. **系统方案图**：Bound IR → global planner → repeated-query runtime → TVM task backend；
4. **三项贡献**：表示、Planner、runtime；
5. **一个 headline result**：端到端而非 microbenchmark；
6. **边界声明**：不提出新 verifier 算法，保持 soundness/tightness。

如果前两页需要读者阅读后文才能理解贡献，视为不具备投稿条件。

### 6.2 建议全文结构

1. Introduction；
2. Background and Motivation；
3. Repeated Bound Query Model；
4. Structured Bound-Operator IR；
5. Query- and Memory-Aware Materialization Planner；
6. BaB-Oriented Repeated-Query Runtime and TVM Backend；
7. Implementation；
8. Evaluation；
9. Related Work；
10. Limitations and Conclusion。

### 6.3 必须避免的叙事

- “我们首次支持 general DAG/CROWN/αβ-CROWN”；
- “TVM 比 Python 快”；
- 用 toy MLP 的高倍 speedup 作为摘要数字；
- 将算法参数或 bound tightness 差异包装为系统收益；
- 只展示平均 kernel time，不展示 compile、memory、E2E；
- 先写大量神经网络验证背景，第二页才出现系统抽象。

---

## 7. 投稿时间表与 Go/No-Go

ASPLOS 2027 September Cycle 的 full-paper deadline 为 **2026-09-09 AoE**。从本计划基线
到截止约八周，因此它只能是条件冲刺，不是默认承诺。ASPLOS 2028 截止日期尚未在本计划中
假设，待官方 CFP 发布后更新。

### 7.1 2027 September 冲刺节奏

| 日期 | 必须完成 |
|---|---|
| 7/12–7/16 | Gate 0：环境提交、稳定 baseline、统一 workflow |
| 7/17–7/26 | PR-10 + materialization instrumentation/profile |
| 7/27–8/05 | PR-11 + 非 toy workload + 首个 latency–memory Pareto |
| **8/05** | **第一次硬 Go/No-Go** |
| 8/06–8/14 | PR-12 fused CROWN task + headline v0 |
| 8/15 | PR-13 BaB adapter prototype + 两页初稿 |
| 8/20 | 主实验与核心消融基本冻结 |
| 8/24 | 最终投稿决定 |
| 8/25 后 | 禁止新增技术功能，只完成实验、论文和复现 blocker |
| 9/01–9/05 | 内部评审、匿名化、artifact reduced workflow |
| 9/06–9/08 | 只修论文与复现 blocker |
| 9/09 AoE | 仅在全部门禁满足时投稿 |

### 7.2 8 月 5 日硬门禁

以下条件必须全部满足，才继续冲刺 ASPLOS 2027：

- ReLU operator-preserving 主路径可用；
- 自动 materialization Planner 已经存在且会做非平凡决策；
- 至少一个非 toy workload；
- 至少一个 latency–memory Pareto 结果；
- Planner 在不同显存预算下选择不同计划；
- 测试矩阵中 0 unexpected OOM；任一合法候选可运行时 Planner 能找到可行计划；
- α/αβ optimized path 不会错误选择尚无 capability 的 structured action；
- held-out workload 上报告相对 per-case Oracle 的 median/p90 latency regret；
- 至少在一个预算下，让 Always Dense OOM 的 plain CROWN case 通过；
- correctness 与 materialization profile 证据完整；
- C1/C2 的前两页故事不依赖未来 PR 才成立。

任一项缺失，即转为 ASPLOS 2028，不以“先投再说”处理。

### 7.3 8 月 24 日最终投稿门禁

必须全部满足：

- 真实 BaB query adapter；
- 主表和核心消融已出，不再依赖待跑实验；
- headline result 在重复运行中稳定；
- 前两页经至少两位外部读者认为是系统/PL贡献；
- 所有 claim 有 JSONL→table/figure 证据路径；
- 没有靠隐藏 OOM/timeout 或不公平 baseline 获得的结果。

若失败，停止 2027 投稿，保留完整研发成果并扩展到训练/更多 workload 后投 2028。ASPLOS
官方对被拒论文的后续周期重投有限制，过早提交会损害稳健版本。

---

## 8. Artifact 与复现计划

### 8.1 三档工作流

- **smoke（≤15 min）**：环境、import、CUDA/TIR、一个 correctness case；
- **reduced（≤2 h）**：代表性 MLP/CNN/ResNet block、Planner 消融、少量 BaB instances；
- **full（论文全结果）**：完整 workload、重复、timeout、训练与所有图表。

### 8.2 证据链

```text
command/config
  → raw JSONL（含失败记录）
  → schema validation
  → normalized CSV
  → table/figure
  → MANIFEST + CLAIMS map
  → Artifact Appendix expected outputs
```

每条记录至少包含：git commit、submodule commit、dirty flag、Python/PyTorch/CUDA/LLVM/TVM、
GPU、driver、seed、workload hash、planner config、compile cache key、计时分解、memory、正确性、
status/error。

### 8.3 提交前工件

- 匿名仓库；
- build/install/doctor 脚本；
- 固定输入、模型和预期输出；
- `docs/genai_usage.md`；
- Artifact Appendix；
- 软件、硬件、数据和运行时间说明；
- 公开归档计划。ASPLOS 2027 AE 要求 Artifact Appendix 描述依赖、关键结果和验证流程；
  Artifact Available badge 最终需要公共归档仓库及 DOI。

---

## 9. 项目治理规则

从本计划定稿后，每个研究 PR 必须在描述和变更文档中回答：

1. 消除了哪个系统瓶颈？
2. 改善哪个北极星指标？
3. 哪个 ASPLOS contribution 获得了新证据？
4. correctness/soundness 如何验证？
5. eager、local、TVM-default 等 baseline 是否公平？
6. 原始 JSONL、表图和 manifest 在哪里？
7. 如果没有收益，学到了什么，是否应停止该路线？

另外遵循：

- 一项 PR 只推进一个主要研究假设；
- 算法语义与系统优化分开提交；
- third-party 修改隔离；
- 不为漂亮结果删除失败实例；
- 不在主路径未闭环时扩大量新算子或新模型；
- 文档中的“已完成”必须有代码、测试和工件三方证据。

---

## 10. 风险登记与止损

| 风险 | 早期信号 | 止损措施 |
|---|---|---|
| operator-preserving 无实际收益 | composition 最终总在 ReLU/concretize 打平 | 用 profile 决定局部物化；不宣称全程 lazy |
| Planner 退化成手工开关 | 所有 workload 选择同一计划 | 增加预算/shape/query 变化；否则降级贡献等级 |
| TVM compile 开销吞噬收益 | break-even query count 过高 | 加 plan/code cache；明确适用 repeated-query 区间 |
| BaB batch 不稳定 | 分支异质导致 fill rate 低、尾延迟升高 | bucketing、timeout、拆批和 eager fallback |
| tightness 漂移 | fused/low precision 与 reference 不一致 | 默认 FP32；每次运行强制 correctness gate |
| workload 太 toy | 只有 MLP/MNIST CNN 有结果 | 8/05 前必须加入至少一个非 toy 代表负载 |
| 竞品已覆盖表面贡献 | Luna/新 certifier compiler 提供类似功能 | 始终围绕 representation/planning/repeated-query co-design |
| 2027 时间不足 | 8/05 前无 Planner/Pareto，8/14 前无 headline v0 | 立即转 ASPLOS 2028，不消耗重投机会 |

---

## 11. 执行版维护检查清单

每次里程碑复审时，需要多模型/人工评审明确回答：

- 三项贡献是否都是系统/PL贡献，而不是验证算法功能列表？
- C1 与 auto_LiRPA/Luna 的边界是否足够清楚？
- C2 是否真的需要全局 Planner，还是局部 heuristic 已足够？
- C3 是否有真实重复查询，而非人工复制 batch？
- 2027 的时间表是否现实，哪些任务可以删除而不破坏论文？
- headline result 应优先选择 BaB、multi-spec 还是 memory-constrained CROWN？
- certified training 是主评估还是扩展评估？
- 哪个非 toy workload 能在 8/05 前稳定，第二个能否在 8/15 前进入主表？
- 前两页能否在不解释大量验证术语的情况下成立？

执行中必须同步更新：

- `gemini_doc/README.md` 顶层索引；
- 当前阶段/下一步计划；
- PR 模板或协作 workflow；
- 论文 claims map 与实验 schema（若新增字段）。

---

## 12. 参考来源

- [ASPLOS 2027 Call for Papers](https://www.asplos-conference.org/asplos2027/cfp/)：研究推进标准、rapid review、2026-09-09 AoE deadline 与重投限制。
- [ASPLOS 2027 Artifact Evaluation](https://www.asplos-conference.org/asplos2027/artifact-evaluation/)：Artifact Appendix、workflow、expected outputs、公共归档与 DOI。
- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)：一般计算图、自动 LiRPA、α/β/GCP-CROWN 与 GPU 支持边界。
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)：GPU bound propagation、BaB 与 verifier 生态。
- [A Tensor-Based Compiler and a Runtime for Neuron-Level DNN Certifier Specifications](https://arxiv.org/abs/2507.20055)：neuron-level specification 编译、shape analysis 与 g-BCSR runtime。
- [The Luna Bound Propagator for Formal Analysis of Neural Networks](https://arxiv.org/abs/2603.23878)：C++ general-graph IBP/CROWN/α-CROWN propagator。
- [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://arxiv.org/abs/1906.06316)：CROWN-IBP 的训练稳定性、紧度与成本权衡。
