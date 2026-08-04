# BoundFlow 当前状态：PR-13 Closure 之后

> 状态日期：2026-08-05
> 当前 integration base：`8969064`（NRIR-42 / PR #53 merge）；PR-13 历史基线：`57a854b` / tag `pr13-validated-reduced`
> 当前研发分支：`main`；下一预注册分支：`feat/cross-axis-verification-batch-schedule-v1`
> 总判定：IR-5 final **VALIDATED-NO-GO**；PR-14B 同为 No-Go、PR-14C/IR-6 不启动；
> ASPLOS-ready 为 **NO**。
> 2026-08-05 NRIR-37 后续：frozen NRIR-28 parametric Template/Instance/Cache 已接入
> objective-ancestral sibling evaluator，并新增独立 Plan/Batch/Task/Schedule IR 与跨 clause 单一 cache
> owner。真实 ResNet clause 2 root+pair 与 frozen audit lower/branch/split/α/β/refinement exact，upper
> max diff=`1.52587890625e-5` 且既有 allclose guard 通过。三 fresh repeats 的 rank/selected 均固定为
> `[2,3,4,5,0,8,6,7,1]`/`[2,3]`，两条每轮均提交 `31 nodes/15 groups`，whole=
> `[51.996191,52.251681,52.695640] s`，每轮恰好一次模板编译。NRIR-37 以 shared compiler ownership +
> fixed-deadline coverage `VALIDATED-REDUCED` 关闭；final 仍 9/9 unresolved，ASPLOS-ready 与 performance
> No-Go 不变。下一门禁转向 full-depth frontier tightness attribution 与单变量 stronger-bound/candidate，
> 不继续调 top-k/slice/cache。
> 2026-08-05 NRIR-38 已关闭：两条 clause 均覆盖 31 evaluations / 16 active depth-4 nodes，baseline
> replay lower/upper max diff=0。optimizer `steps=5→15` 虽改善 32/32 nodes，但 worst-active lower 只
> 改善 `+0.055496/+0.028557`，未过预注册 `+1.0` 门禁，以 `VALIDATED-NO-GO` 冻结 optimizer-step
> 轴。下一单变量为已有 objective branch IR 接入 shared ancestral evaluator。
> 2026-08-05 NRIR-39 fixed-budget pilot 已通过：新增 composite Plan/6-task TaskModule/Schedule，将既有
> objective branch 五阶段程序接入 shared ancestral queue，并为每条 clause 的 31/31 evaluations 绑定
> branch execution。clauses 2/3 worst-active lower 由 `-37.574287/-35.900215` 提升为
> `-35.530926/-30.258448`，改善 `+2.043362/+5.641768`，两条均过 `+1.0` 门禁；median 亦提升
> `+2.537640/+5.885233`。状态为 fixed-budget branch selection `VALIDATED-REDUCED`，下一门禁是
> three-repeat whole-query/global-deadline formal；performance/property/ASPLOS-ready 尚未升级。
> 2026-08-05 NRIR-40 已完成：objective branch 进入原始 production queue 和 single-global-60s
> multi-clause runtime，三 fresh repeats 的 correctness、rank/selected、typed branch coverage、cache 与
> original-ordinal aggregate 全过；但 accepted nodes 只有 `[[29,23],[29,21],[29,21]]`，clauses 2/3
> worst-active lower 为 `-48.315041` 与 `-43.299690/-44.731468`，相对 NRIR-37 widest formal 更差。
> whole cooperative elapsed=`[63.357098,63.161128,62.485366] s`；production coverage/tightness gate
> 三轮均失败；全量 `944 passed, 37 skipped`。按预注册以 objective-branch global-budget
> `VALIDATED-NO-GO` 关闭。NRIR-39 fixed-budget
> 机制结论不撤销，但不得升级为 production/performance/property/ASPLOS-ready claim。下一步只允许先做
> scoring 成本与 frontier-order 因果归因，再冻结新的单变量。
> 2026-08-05 NRIR-41 已完成上述归因：objective 在 clauses 2/3 的 `21/23/29/31` same-node prefix
> worst lower 全部优于 widest，frontier-order gate 成立；三 fresh counterbalanced paired runs 的
> objective/widest queue median ratio=`1.748660/1.750639`，cProfile branch-program share=
> `21.9371%/21.9139%`，且 31 次 branch program 实际触发 341 次 candidate enumeration。
> attribution 以 `VALIDATED-REDUCED` 关闭、`performance_claimed=false`；NRIR-40 production NO-GO 和
> ASPLOS-ready NO 不变；全量 `948 passed, 37 skipped`。下一单变量已限定为 scorer
> ownership/validation reuse，不允许同时改 policy、
> node/depth、slice、optimizer、refinement、cache 或 deadline。
> 2026-08-05 NRIR-42 已完成：typed validated capsule 使每个 node 的 candidate table 只在 Plan compile
> 枚举一次，scorer Task/Schedule 的第一阶段显式读取 `branch.plan.candidates`，execute 与下游 validation
> 不再重建候选。Phase A 三 fresh paired runs 中 clauses 2/3 enumeration 都从 `341→31`，new/old
> queue median ratio=`0.706888/0.698486`，六组 31-node branch/score/child-bound/queue/state/refinement
> exact。Phase B 三轮 whole=`57.175184/57.697757/58.114412 s`，selected 均 `[2,3]`，两条每轮均
> `31 nodes/15 groups/31 capsules`，worst-active lower=`-35.530926/-30.258448`。Phase A/B formal hash=
> `0d310c2f…25b58` / `7274e834…7d759`；全量 `958 passed, 37 skipped`。本阶段以固定
> ResNet2B property 0 CPU production admission `VALIDATED-REDUCED` 关闭；final 仍 unknown，
> performance/GPU/multi-workload/competitor/ASPLOS-ready 均未升级。下一单变量是
> cross-clause/node/candidate batch Schedule，而不是继续调 scorer validation。
> NRIR-42 已由 PR #53 合入 `main@8969064`；功能提交为 `264365f`。
> 2026-08-05 NRIR-36 后续：九子句 NRIR-31 floor 已由 typed root-lower priority 选择 clauses 2/3，
> dynamic equal-remaining slices 在同一 global start 下执行。三 fresh repeats 都复现
> rank=`[2,3,4,5,0,8,6,7,1]`，packed nodes=`[[3,3],[3,3],[3,1]]`；repeat 2 第二条未提交
> atomic pair，预注册 coverage gate 失败，final 仍 9/9 unresolved。状态为 multi-clause allocation
> VALIDATED-NO-GO；下一门禁转 shared parametric compiler/root/evaluator 与 stronger bound/candidate，
> ASPLOS-ready 与 performance No-Go 不变。
> 2026-08-04 NRIR-19 后续：native selected-CROWN intermediate refinement 已成为一等
> Plan/Task/Schedule。MNISTFC 关闭 clauses 3/7，OVAL21 从 unknown 变 verified；ResNet 两个 root
> lower 改善 `+70.496/+160.551` 但状态仍 unknown。下一门禁为 objective-directed intermediate
> target selection；该门禁现已由 NRIR-20 关闭：同预算 ResNet clauses 0/1 root lower 再改善
> `+55.928741/+26.228943`，但仍为负。NRIR-21 per-child exact-state refinement 已完成并在
> clauses 0/1 上使最差 depth-2 leaf lower 退化 `-0.847961/-0.936646`，故为 NO-GO；下一门禁为
> ancestral-constraint carry-forward refinement；该门禁已由 NRIR-22 以 fixed-tree
> `VALIDATED-REDUCED` 关闭，clauses 0/1 worst leaf 相对 independent 提升
> `+73.615173/+75.022095`。NRIR-23 随后连接 external typed seed；NRIR-24 已完成
> `7/15/31 nodes × depth 2/3/4` convergence，三条 hard clause 持续改善但仍无 closure。
> NRIR-25 已进一步完成 same-planned-cap dynamic ancestral budget：三条 hard clause 均有小幅
> 正向 tightness，但仍无 closure。NRIR-26 typed split-two-pass 在同总 cap 下三条 worst lower
> delta 全为 `0.0`，按预注册门禁 NO-GO。NRIR-27 已把 audit verifier 转为显式 production
> prepared queue，并在三真实拓扑相同算法 clause-0 上获得 `1.3663×/2.4723×/1.4511×`
> repeated CPU internal speedup；full query 仍全部 unknown。NRIR-28 随后把 optimizer 编译拆为
> parametric PlanTemplate/PlanInstance，并在相同 full query 上把 v1→v2 median 降至
> `14.807→3.456/61.239→6.209/13.021→3.718 s`。NRIR-29 已把搜索预算冻结为
> `7/depth2→31/depth4→127/depth6`，27/27 fresh workers 完成且 domain nesting 成立；MNISTFC
> verified `6/9→8/9`，ResNet 保持 `0/9`，OVAL21 保持 `8/9`。下一门禁转为只对 remaining
> clauses 使用更强 bound/branch 的 typed hard-clause escalation，不继续单轴堆节点。
> NRIR-30 已完成该门禁：baseline unresolved 被 exact ordinal 投影到 shared native-refined 31-node
> stage；OVAL21 三次由 `8/9 unknown→9/9 verified`，MNIST `6/9→8/9`，ResNet 仍 `0/9`，全部
> 在 60 秒 whole deadline 内且无 fallback。下一门禁是 per-clause objective-directed refinement，
> 不改 admission/search budget 来混淆因果。
> 顶层
> ASPLOS-ready 与 performance No-Go 不变。
> 2026-07-20 修订：本文保留 PR-13/14 历史证据，但第 4 节下一路线已由 IR-first 复审取代。
> 2026-07-28 进度：IR-1 Bound IR、IR-2 Plan IR、IR-3 Task/Schedule IR 的最小
> synchronous reference contract 已分别关闭；IR-4 production backend/runtime migration
> 已以 validated-reduced 关闭。IR-4A typed dispatch key + PyTorch reference
> adapter 已完成 foundation；IR-4B dense/structured/chunked typed registry 已通过，
> IR-4C TVM fused/unfused、dispatch-namespaced cache 与 semantic fallback 已通过；
> IR-4D typed plain-CROWN query→Plan/Task/Schedule、精确 state payload 与计算跳过已通过；
> IR-4E 已把 PR-13 manager 接入 typed compiler，并把 legacy α/β 改为默认关闭的
> historical opt-in。IR-5C3 independent workload-family + fair batching 已完成并给出
> VALIDATED-NO-GO；如继续，唯一补救是 IR-5D prepared execution capsule。
> IR-5A 已完成 query-time memory/deadline/cache/distribution context 与 amortized selector；
> 这只是 mechanism。
> IR-5B 已完成统一 observation 上的 fixed/local/global/oracle evaluator 与 synthetic
> contract artifact。IR-5C2 已产出 fresh CUDA typed MLP measured artifact：Global 8/8
> feasible，p50/p90 regret 1.000×/1.00766×，但同-family split、fair batching baseline
> 与 non-toy workload 仍缺。IR-5C3 随后用 MLP→CNN architecture-held-out 和 fair
> batched-original 补齐口径，Global p50/p90 regret 恶化为 68.065×/70.263×，且无多预算
> 切换/Pareto，因此当前 IR-5 v1 为 VALIDATED-NO-GO。
> IR-5D 已把静态 validate/hash/dispatch 移入 prepared execution capsule，并在已消费
> CNN 上以 from-forward-trace 公平边界得到 `0.880×`/`0.896×` 最快 median 诊断；
> 该结果仅为 calibration，不撤销 No-Go；其后 residual-CNN v3 final 已完成并失败。
> IR-5E 已冻结 CUDA-only chain-CNN calibration→residual-CNN final v2 协议；正式
> v2 首次执行因 fixed-single 重新采样的 input 与 batch 第一 query 不同而
> PROTOCOL-INVALID；未生成 manifest，`7401/7402` 已退役。不得将此写成系统性能结果。
> IR-5G 已用 exact batched-input slice 修复方法学，并冻结 v3 `7501/7502`；
> backend/budget/shape/阈值均未按 v2 timing 调整，随后只运行一次。
> IR-5H v3 final 已完整生成并 replay：correctness 全过，但 Global p90 `1.26160×`
> 超过 `1.20×`，gray 无 compiler Pareto，且无多预算切换。IR-5 最终
> VALIDATED-NO-GO；停止当前 ASPLOS system-performance 路线，IR-6 不启动。
> 2026-08-03 RVIR 后续：真实 verifier correctness/integration 已 CPU
> VALIDATED-REDUCED；这不撤销 IR-5/ASPLOS performance No-Go。
> 2026-08-04 P0 后续：production Schedule-memory 准入审计为 `NO_GO`。Residual reduced
> 路径有完整 arena/launch ownership，但没有 materialize、storage 选择或预算决策切换；真实
> ResNet 的 51 个 activation call 仍各自是一个 external opaque launch。下一分支改为
> `feat/native-real-network-bound-ir-v1`。
> 2026-08-04 NRIR-1 后续：固定 ResNet2B initial-CROWN 已生成 21-op native Bound graph、
> 21 Tasks 与 21 launches，Bound/Task external-call count 为 0；五层 hash fresh replay 一致，
> lower max diff `7.15256e-7`、sign 9/9。该结果只关闭 CPU correctness/compiler ownership；
> external intermediate bounds、单 storage/batch、0 materialization 与无性能 claim 的边界保留。
> 2026-08-04 NRIR-2 后续：同一真实 ResNet Bound IR/PlanTemplate 已加入 retain-all 与
> lifetime-reuse 两个 storage plan。1,860,912/442,656 bytes 预算阈值会切换 PlanInstance 与
> Schedule arena；低内存路径在 Task 边界提前释放 85 个 runtime values，并有 386 对合法
> physical aliases。两计划 bitwise 相同、external max diff `7.15256e-7`、sign 9/9。该结果只
> 关闭 CPU storage-plan correctness/ownership；不是 CUDA allocator peak、OOM rescue 或性能证据。
> 2026-08-04 NRIR-3 后续：fresh-process CUDA protocol 已冻结并实现，包含 5 repeats ×
> 5 warmup × 20 measured、allocator allocated/reserved delta、交替进程顺序、prepared lower-only
> timing、20% memory 与 1.20× latency 门禁及 raw semantic replay。本机
> `cuda_available=false`，所以只生成 `environment_unavailable` probe artifact；正式 benchmark
> 在创建输出目录前 exit 2，`performance_claimed=false`。下一步转 representation semantic bridge。
> 2026-08-04 NRIR-4 后续：fixed ResNet representation decision 已驱动 21-op dense 与
> 49-op structured-affine execution stack，14 cast + 14 materialize 与 Task/Launch 一一绑定；
> dense-equivalent hard limitation 保留，PR #15 已合并。
> 2026-08-04 NRIR-5 进度：query-time spec batch limit 已切换 full/sliced PlanInstance 与
> Schedule；sliced path 执行 `[0,3)/[3,6)/[6,9)` 三个各 21-op 的 child stack，full/sliced
> max diff `1.90735e-6`、external sign 9/9，artifact generate/replay 通过，全量
> `508 passed, 37 skipped`。状态为 VALIDATED-REDUCED；domain/sample、representation ×
> batch 联合执行和性能/内存证据仍 pending。
> 2026-08-04 NRIR-6 后续：representation/storage × spec-batch 已进入同一 source template/
> selector，四组合 child op/task/launch=`21/63/49/147`，source policy 显式传播；四路径
> external sign 9/9，全量 `522 passed, 37 skipped`。状态为 joint compiler ownership
> VALIDATED-REDUCED；下一缺口为跨 query/domain batching、cache 与公平性能证据。
> 2026-08-04 NRIR-7 后续：9 个真实 property objectives 已成为 9 条 explicit queries；packed
> 3 child vs same-policy serial 9 child，9/9 lineage 恢复；first miss/second exact hit，
> objective/order/state 均进入 cache key。packed/serial max diff `3.21865e-6`、external sign 9/9，
> 全量 `540 passed, 37 skipped`。状态为 repeated-query correctness/ownership VALIDATED-REDUCED；
> BaB domain state 与性能仍 pending。
> 2026-08-04 NRIR-8 进度：固定 ResNet root box 已三层二分为 8 个不同 leaves；每个 leaf
> 独立重算 exact IBP state，parent 仅 `warm_start_only`。domain-size-4 Plan/Schedule 执行
> 2 child，full-size-8 执行 1 child，same-policy serial 执行 8 child；三路径 lower/upper
> bitwise equal、8/8 lineage 恢复。状态为 input-domain batching/state ownership
> VALIDATED-REDUCED；ReLU/β split、BaB queue/prune/termination 与 performance 仍 pending。
> 2026-08-04 NRIR-9/10 后续：first-class ReLU split queue 与 frozen alpha/beta state 已分别合并；
> split/alpha/beta 均进入 native Bound/Plan/Task/Schedule，warm-start 只允许 exact 或 monotonic
> refinement initialization。完整搜索/verdict 与性能仍未关闭。
> 2026-08-04 NRIR-11 进度：fixed-step optimizer 已 lower 为 typed Plan/Task/Schedule。固定 ResNet
> 1-step program 执行 8 actions，alpha/beta gradient L1=`169.23175295069814/12.862210273742676`；
> Schedule/legacy/final native execution max diff 均为 `0.0`。状态为 optimizer control ownership
> VALIDATED-REDUCED；下一缺口为接回 multi-node ReLU-split BaB queue。
> 2026-08-04 NRIR-12 进度：optimizer Schedule 已接入每个 ReLU-split queue node batch；固定 ResNet
> 为 7 nodes/3 expands/4 frontier、packed/serial 3/7 stacks。bounds/state tensors 在冻结容差内，
> active child beta gradients 非零，selected native re-execution diff=0。状态为 integration
> VALIDATED-REDUCED；fixed run 仍 budget-exhausted/not-claimed，下一缺口为 sound verdict。
> 2026-08-04 NRIR-13 后续：three-state sound verdict 与 concrete witness replay 已关闭；固定
> ResNet 7-node frontier 正确返回 unknown，未把开放 frontier 伪装为 verified。
> 2026-08-04 NRIR-14 后续：multi-clause complete query、deterministic candidate search、unsafe
> short-circuit 与 cooperative deadline 已关闭为 control/correctness VALIDATED-REDUCED。固定
> ResNet 九个真实 clauses 全部执行，但 9/9 native scalarized lower bounds 仍过松并返回 unknown；
> 下一阶段必须建立端到端 phase/tightness baseline，再攻 dynamic optimization、branching 与执行性能。
> 2026-08-04 NRIR-15 后续：external intermediate semantics 已贯穿 optimizer/queue child/query，
> adaptive 1-step 把固定 ResNet 从 0/9 提升到 6/9 verified，仅 0/2/4 unknown；三组 CPU audit
> queue 均约 6.7 s，而 candidate/verdict 仅约 3.6/3.9 ms。下一门禁确定为 prepared production
> fast path；6/9 仍不是完整 verifier 或 ASPLOS performance claim。
> 2026-08-04 NRIR-16 后续：root-only exact prepared capsules 已把 fixed ResNet 三组 warm
> complete-query median 从 audit `59.078 s` 降为 `110.950 ms`；cold prepare+first=`16.139 s`，
> payload=`2.076 MB`，semantic/status 不变。该比值只归因内部 audit evidence overhead；下一门禁
> 为 clauses 0/2/4 branching/stronger-bound，ASPLOS-ready 仍为 NO。
> 2026-08-04 NRIR-17 后续：objective branch score 已成为 first-class Plan/Task/Schedule；同预算
> hard-clause worst leaf 相对 widest 改善 `0.120752/0.071564/0.057901`，但全部 terminal
> leaves 仍为负，6/9 与 ASPLOS-ready=NO 均不变。下一门禁是多 workload/设备/竞品协议与
> stronger-bound，不再把继续增加 widest depth 当作主路线。
> 2026-08-04 NRIR-18 后续：MNISTFC、CIFAR ResNet2B、OVAL21 三种拓扑已经由原生
> VNNLIB Query IR 和 21-task/6-fresh-process workload Plan/Task/Schedule 驱动。BoundFlow
> 状态为 `unknown/unknown/unknown`，固定 αβ-CROWN 为 `verified/unknown/verified`；ResNet
> native local root lower 达 `-543.717/-789.331`，明确暴露 intermediate-bound strength 缺口。
> CPU 单次 E2E 仅为诊断，不计算 speedup；下一门禁转为 native intermediate-bound refinement。

## 1. 当前真实阶段

BoundFlow 已经完成从边界表示到 query runtime prototype 的主干：

| 层次 | 状态 | 已验证边界 |
|---|---|---|
| Structured Bound IR | IR-1 reference + IR-4 backend closure validated-reduced | typed schema/lowering/verifier、dense/structured rewrite/interpreter、PyTorch/TVM typed execution |
| Plan/Task/Schedule IR | IR-2/3 reference + IR-4 runtime closure validated-reduced | typed builder/selector/task lowering/schedule verifier/per-task semantics/query/state/backend artifacts |
| Fused/multi-backend CROWN execution | validated-reduced | eager/chunked/structured/TVM fused 多预算选择；收益只在部分 regime |
| Query runtime | validated-reduced | `BoundQuery`、state validity、dynamic batching、same-solver adapter、reduced GPU E2E |
| 真实 complete verifier integration | RVIR CPU correctness/integration validated-reduced | ResNet external-semantics max diff 3.10e-6、sign 9/9；typed external-call admission 394/394；真实在线 dispatch 377/377 |
| Production Schedule + Memory P0 | NO-GO | residual 8/8 完整 arena ownership，但 0 materialize、单 storage、0 budget decision switch；真实 ResNet 51/51 为单 external launch |
| Native real-network IR NRIR-1 | correctness/compiler ownership validated-reduced | ResNet2B 17 Primal ops → 21 native Bound/Task regions/launches；五层 hash 绑定 external-bound payload；max diff 7.15e-7、sign 9/9；仍无 memory choice/GPU/performance |
| Native real-network memory NRIR-2 | storage-plan correctness/ownership validated-reduced | 同一 real graph/template 的 retain-all 1,860,912 B 与 lifetime-reuse 442,656 B；预算决策切换、386 alias pairs、85 early releases、双计划 bitwise equal；无 CUDA allocator/performance claim |
| Native CUDA memory protocol NRIR-3 | protocol implemented / environment unavailable | fresh worker、5×5×20、allocator/timing/identity/replay 门禁已实现；本机 0 CUDA device，只保留 fail-closed probe，不产生 performance claim |
| Native representation binding NRIR-4 | correctness/compiler ownership validated-reduced | ResNet source policy 驱动 21-op dense 或 49-op structured execution；28 transitions 绑定 Schedule/Task/Launch；dense-equivalent、无性能 claim |
| Native spec-sliced execution NRIR-5 | correctness/integration validated-reduced | full 9 specs→1 child；limit=3→3×21-op child 与精确 range/aggregation；CPU semantics/replay 通过；domain/sample/joint representation/performance pending |
| Native joint policy NRIR-6 | cross-axis correctness/ownership validated-reduced | 同一 template/selector 的 dense/structured × full/sliced 四组合；policy propagation、21/63/49/147 child ownership、external sign 9/9；跨 query/domain/performance pending |
| Native repeated-query NRIR-7 | query formation/cache/lineage validated-reduced | 9 property queries→packed 3 child vs serial 9；exact cache miss/hit/key invalidation；9/9 restore；BaB domain/performance pending |
| Native input-domain batching NRIR-8 | parent/child state + domain execution validated-reduced | 8 different leaf boxes；8 exact child states；full 1 / packed 2 / serial 8 stacks bitwise equal；parent warm-start-only；full BaB/performance pending |
| Native ReLU-split queue NRIR-9 | split state + bounded control flow validated-reduced | first-class int8 split；toy complete queue；fixed ResNet 7 nodes/3 expands/4 frontier；plain CROWN、budget-exhausted、无完整 verdict/performance |
| Native alpha/beta state NRIR-10 | frozen optimized-state ownership validated-reduced | 6 ReLU split/alpha/beta inputs；beta lower dual；exact/refinement warm-start；runtime optimizer control 当时仍缺 |
| Native optimizer Schedule NRIR-11 | fixed-step control ownership validated-reduced | typed optimizer Plan/Task/Schedule；fixed ResNet 8 actions、正 alpha/beta gradient、legacy/native 0 diff；尚未接回 multi-node queue，无 verdict/performance |
| Native optimized split queue NRIR-12 | optimizer × queue integration validated-reduced | 每 node batch 8 optimizer actions + 21 native tasks；7 nodes/3 expands/4 frontier；parent warm-only；仍 budget-exhausted/not-claimed |
| Native property verdict NRIR-13 | three-state soundness/control validated-reduced | verified 只接受 sound-pruned closure；unsafe 必须 concrete replay；固定 ResNet frontier 保持 unknown |
| Complete verifier query NRIR-14 | multi-clause query control validated-reduced | conjunction、PGD candidate、witness replay、unsafe short-circuit、cooperative deadline；固定 ResNet 9/9 unresolved，无性能 claim |
| E2E tightness/performance baseline NRIR-15 | external semantics + CPU diagnosis validated-reduced | fixed ResNet 6/9 verified、3 hard clauses；三组 audit queue 约 6.7 s，candidate/verdict 毫秒级；下一步 prepared production path，无 speedup claim |
| Prepared production path NRIR-16 | root-only repeated-query mechanism validated-reduced | audit/prepared warm median 59.078 s/110.950 ms；cold total 16.139 s、payload 2.076 MB；semantic 6/9 不变，仅内部 overhead diagnosis |
| Objective branching NRIR-17 | branch IR/control + fixed-budget tightness validated-reduced | three hard-clause worst-leaf improvements 0.120752/0.071564/0.057901；all remain unknown；单 workload CPU、无 speedup claim |
| Multiworkload competitor E2E NRIR-18 | ingest/control/coverage validated-reduced | MNISTFC/ResNet2B/OVAL21 原生 VNNLIB→Query/Plan/Task/Schedule；BoundFlow unknown×3，αβ-CROWN verified/unknown/verified；CPU diagnostic only、GPU pending |
| ASPLOS 最终系统主张 | IR-5 final VALIDATED-NO-GO | IR-1—4 narrow closure 保留；Global p90/Pareto 失败，当前 system-performance 路线已关闭 |

历史 `main@263ea81` 只到 PR-10 closure，不能再作为项目当前状态入口。跨会话恢复必须同时检查
research branch、annotated tag 与 closure 文档，不能只看 `main`。

## 2. 已经成立的证据

### C1：Structured Bound Representation

- ReLU 后主 coefficient 可以保持结构化 operator；
- materialization barrier 有稳定 trace schema；
- dense/operator/planned 路径在相同浮点语义下有 reference comparison；
- structured 不是统一默认策略：plain CROWN 的部分显存收益伴随明显 latency 代价，α/αβ
  structured 还会增加 autograd peak 并出现 OOM。

### C2：Query- and Memory-Aware Multi-Backend Planner

- PR-11 已完成静态 topology/liveness feature、global placement、bounded retry 和真实 OOM fallback；
- PR-12 已建立 eager、chunked、structured、TVM fused 候选及 compile-aware、多预算选择；
- final held-out 中可行机会 72/72 选到可行 backend，feasible p90 regret 为 1.000×；
- fused 的稳定价值主要是减少中间物化/peak memory，而不是普遍降低 latency。

### C3：Query Runtime Prototype

- PR-13A 已有 state-versioned `BoundQuery`、compatibility key、split lineage、fixed replay；
- PR-13B 已有 dynamic batching、deadline/budget、OOM bisection、顺序恢复和可观测 counters；
- PR-13C 已把 adapter 接回同一 host solver，只替换 bound-call execution；
- PR-13D reduced GPU 中，fixed/E2E 相对逐节点为 96.52×/9.93×，但 hard E2E 相对公平
  batched original 仅 0.980×。

因此 96×/9.93× 必须归因于物理 batching，不能描述成 runtime abstraction 的独立加速。

## 3. PR-14 已关闭的问题

1. **真实 coverage**：540 calls 中 initial 143/146 region-level eligible；activation-BaB 0/394；
2. **真实 fixed replay**：MLP lower 等价，但 requested outputs 不同，性能 N/A；
3. **non-toy bound equivalence**：ResNet nominal forward 正确，whole-query lower max diff
   `796.765`、符号 3/9，不能接入 same-solver；
4. **C3 定位**：无公平 batched-original 净收益证据，已降级为支撑 C1/C2 的基础设施。

## 4. IR-first 路线执行结果与关闭状态

PR-14 implementation 已停止。原定 `docs/asplos-c1-c2-story-freeze` 被代码级复审否定后，
历史工程主线切换到 `feat/compiler-ir-stack-v1`，并已按 Bound IR → Plan IR →
Task/Schedule IR → runtime/backend → adaptive evaluation 完整执行。契约见
`gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。仍不得用 PR-14C
E2E 绕过 bound-equivalence gate。

截至 2026-08-03，Bound IR、Plan IR、Task/Schedule IR 的 synchronous reference closure
与 IR-4 backend/runtime validated-reduced closure 均已完成；IR-5 adaptive PlanInstance
也已执行到 fresh residual final，并以 VALIDATED-NO-GO 关闭。不得回滚重复实现
IR-1/2/3/4，也不得继续旋转 IR-5 final。IR-2
raw historical artifact 缺失边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`；IR-3 closure 证据见
`gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`。

IR-4D 已证明可验证 plain-CROWN 请求能够通过 typed query 入口完成
PlanInstance→TaskIR→ScheduleIR→backend，并已实现 exact-version dense state
load/store/task skip。PR-13 α/β 请求仍因 PR-14 whole-query mismatch 在 compiler 入口显式
No-Go。IR-4E 随后把 `plain_crown_typed_ir` 请求接入 PR-13 DynamicBatchManager，并把旧
`SameSolverQueryRuntime` 设为默认拒绝、仅 historical replay 显式 opt-in。IR-4 现以
validated-reduced 关闭；其后 IR-5 已完成并失败。不得把 IR-4 closure 写成 α/β external
integration 或 ASPLOS 性能结论。

IR-5A 已让 cold/repeated/warm-cache 与 per-query memory/deadline 进入 PlanInstance
identity、provenance 和 runtime cache namespace。同一 template 可合法切换不同 plan。
IR-5B/C2 随后完成四策略 evaluator 与 fresh CUDA typed MLP artifact：Global 在 8/8
contexts 可行，p50/p90 Oracle regret 为 1.000×/1.00766×，高内存选择 dense、冻结低内存
选择 TVM fused。IR-5C3 随后冻结 MLP calibration→chain-CNN held-out，并加入 fixed-single、
ordinary typed batching 与 legacy fair batched-original。全部 correctness/feasibility gate
通过，但 batched-original 约 0.506–0.508 ms/query，Global 约 34.449–35.678 ms/query，
p50/p90 regret 68.065×/70.263×；64/512 MiB 都选择 chunked，无 memory Pareto。

profile 曾将主要问题定位到 query hot path 重复 Plan/Bound/Task validate、stable hash、
canonical JSON 与 dispatch-key 构造；IR-5D 已完成该补救。随后 fresh residual final 仍以
Global p90 `1.26160×` 和 gray Pareto 缺失失败。ASPLOS-ready 判定为 NO，IR-6 不启动，
IR-5 内部不存在仍被证据允许的后续旋转；独立 NRIR 路线按第 8 节推进。

IR-5D remediation 现已实现：prepared Bound/Task program 冻结静态参数与 identity，
Plan cache 复用预计算 dispatch key，production trace 不在 timed path 生成中间 tensor
SHA；同时新增 from-forward-trace legacy baseline，使双方都只计 CROWN backward。
在旧 gray/color CNN 上的 20-sample CUDA calibration 中，最快 typed/legacy median 比值为
`0.880×`/`0.896×`。这些 workload 已被消费，故只能证明优化方向，不能升级 claim。
该 calibration 当时只用于决定是否值得运行 final；其后 residual-CNN v3 final 已完成并失败。

IR-5E 完成了 protocol freeze：新 workload 含真实 residual fanout/`add_backward`，
baseline 固定为 from-forward-trace，并显式输出 p90≤1.20、双 workload latency-memory
Pareto 与 multi-budget switch 字段。v2 因输入身份协议错误失效，`7401/7402` 已退役。

实际首次运行发现同 seed、不同 batch shape 的 `torch.randn` 不保证前缀一致，导致
fixed-single 与 batched-first 输入不同。v2 在 semantic gate fail closed，未形成 summary/
manifest，未进入正式性能判定。当时唯一允许的处置是修复显式 input slicing、升级
protocol 并旋转 fresh identities；该处置已由 v3 完成，IR-6 始终未启动。

v3 runner 先对 fixed-single 与 batched query zero 做 `torch.equal`，再检查 final bounds；
split 记录 exact-clone contract。`7501/7502` 已按预注册协议运行一次并永久冻结。

v3 正式 artifact 已执行并绑定 `971a317`。Global 8/8 feasible，p50 regret
`1.00385×`，但 p90 `1.26160×`；失败来自 color warm-cache context 选择 TVM
（0.53146 ms/query）而 dense 为 0.42577 ms/query。color 有 latency-memory tradeoff，
gray 的 TVM 同时更快更省内存，只有单点 frontier，故双 workload Pareto 门禁失败。
IR-5/IR-6 路线按预注册止损规则关闭。

IR-5 当时冻结的最优先候选是与 `7501/7502` 独立的真实 Verifier IR correctness；该候选现
已由 RVIR 路线执行并按第 6 节关闭。其完成只解除 correctness/integration blocker，不授权
重新提出性能 claim。

明确禁止：

- 回到 `bench/pr10b2-real-bab-fixed-domain-replay`；
- 继续无证据地调孤立 TIR/kernel；
- 新建 persistent GPU BaB queue；
- 把 reduced chain-CNN 结果写成 VNN-COMP/non-toy 结论；
- 把逐节点 speedup 当成相对成熟 batched verifier 的 headline。

## 5. 权威阅读顺序

1. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_PLAN_2026_08_04.md`；
2. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_PLAN_2026_08_03.md`；
3. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`；
4. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`；
5. `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`；
6. `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`；
7. `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`；
8. `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`；
9. 本文（含 PR-13/14 历史状态与第 6—11 节当前修订）；
10. `gemini_doc/asplos_claims_map.md`；
11. `gemini_doc/asplos_execution_memo_v1_0.md`。

## 6. RVIR 关闭后的当前边界

PR-14B 的 `796.765` 与 `0/394` 仍是当时 local whole-query/fused replacement 路径的正确历史
结论；它们已被新的 correctness 路线分解，而不是被删除：

- external intermediate bounds + adaptive slope 的 ResNet initial-CROWN 已通过，max diff
  `3.09944e-6`、sign 9/9；
- fused replacement coverage 仍是 `0/394`；
- provider-owned typed external-call admission 是 `394/394`；
- adapter v2 当前 CPU exact-call execution 是 `377/377`，observer on/off 的 status、380
  domains 与 final lower 一致。

历史 394 行仍缺 split tensor values、requested polarity 与 parent lineage，artifact 已逐行标注；
当前 377 行补齐 lower-only 与 347 parent links。v2 artifact 进一步冻结这 377 条在线 query 与
377 条 typed execution record 原文；fresh replay 会逐条复核 query/record 顺序、parent
precedes child、完成状态和五层 IR hash，不再只信任生成端摘要。全量回归为
`452 passed, 37 skipped`（RVIR closure 基线）；在线 raw replay v2 合并前的最新回归为
`460 passed, 37 skipped`。
当前没有被证据授权的 CUDA/performance claim，下一性能研究必须另立公平 lower-only 合同与
fresh GPU protocol，不能直接复用本 correctness artifact。

## 7. Production Schedule IR + Memory P0 判定

`artifacts/schedule-p0/production-schedule-memory-p0-20260804/` 对 IR-5 residual-final-v3
和 RVIR v2 做了 digest-first、semantic-replay 审计：

- 2 workload × 4 backend 的 8 个 residual structural case 均由 Schedule IR 覆盖 10/10
  Bound ops，并显式执行 check-budget、arena allocate/free、batch loop 与 9/10 个 region launch；
- 但 8 个 template 均只有一个 batch 和一个 storage candidate，且没有任何
  `MaterializeAction`；64/512 MiB 虽生成不同 PlanInstance hash，实际 decision signature
  8/8 完全相同；
- 冻结 residual-final-v3 原有结论仍是 no multi-budget switch、双 workload Pareto 失败；
- VNN-COMP ResNet 51/51 activation call 的五层 IR hash 全部可重编译，但每条 Bound graph
  只有一个 `EXTERNAL_VERIFIER_CALL`，Schedule 也只有一个 external launch，主计算与数值
  语义仍由 αβ-CROWN provider 拥有；
- baseline OOM rescue 没有冻结证据，只能记为 not demonstrated。

因此不能直接启动 `feat/production-schedule-memory-v1`。当时批准的下一代码路线是
`feat/native-real-network-bound-ir-v1`：先把一个冻结真实 residual network 的主计算 lower
为 native multi-region Bound IR，并通过 external-semantics correctness oracle；之后才允许增加
多个 storage/batch 候选、重开 memory feasibility 与 GPU 性能门禁。

## 8. Native Real-Network IR v1 判定

NRIR-1 已在固定 VNN-COMP 2021 ResNet2B prop0 上完成 P0 要求的第一步：

- model/VNNLIB/αβ-CROWN commit 与 6 组 external intermediate bounds 均有 digest；portable
  payload 可由 `torch.load(weights_only=True)` 加载，ordinal/name/shape/dtype/tensor/aggregate
  identity 任一变化均拒绝；
- ONNX/Primal topology 为 17 ops（Conv 6、ReLU 6、Add 2、Flatten 1、Linear 2）；native
  plain-CROWN lowering 生成 21 个 Bound ops、21 个 Task units 与 21 次 Schedule launch；
- Bound IR 与 Task IR 的 `EXTERNAL_VERIFIER_CALL` 均为 0。external-bound aggregate hash 进入
  每个 ReLU relaxation state version，并继续进入 Plan provenance，所以五层 hash 对 oracle
  payload 内容敏感；
- fresh replay 的 native lower 对 αβ-CROWN final lower max diff
  `7.152557373046875e-07`，allclose 门限 `2e-4/2e-4`，sign 9/9；
- artifact 显式 `performance_claimed=false`，当前只有一个 dense storage、一个 full-query batch、
  0 materialization candidate，external verifier 仍负责 forward intermediate bounds。

结论为 CPU correctness/compiler ownership `VALIDATED-REDUCED`，不是完整 native αβ-CROWN 或
性能关闭。其 storage-axis 下一门禁已由 NRIR-2 按第 9 节完成；representation/materialization
与 sliced batch execution 仍未完成，不能因 storage switch 自动升级。

## 9. Native Real-Network Memory Plans v1 判定

NRIR-2 保持 NRIR-1 的 Bound graph、external semantic payload 与 reference backend 不变，只在同一
PlanTemplate 中加入两个可验证 storage plan：

- `native-retain-all-v1` 使用不相交对齐区间，并把所有 value lifetime 延长到 final op，
  Schedule arena 和 runtime observed residency 均为 `1,860,912` bytes；
- `native-lifetime-reuse-v1` 使用 compiler-derived exact last-use，确定性复用不重叠 lifetime
  的 byte ranges，Schedule arena 和 observed residency 均为 `442,656` bytes；
- 高预算选择 retain-all；预算为 `442,656` 时选择 lifetime-reuse；再减 1 byte 时 selector 以
  `memory_budget_exceeded` 拒绝；两者共享 Bound hash `16e27f31...80fb` 与 PlanTemplate hash
  `359ee68f...43f3`，但 PlanInstance/Task/Schedule identity 均不同；
- runtime 在 Task 前检查输入 resident，Task 后按 selected `live_to_op_id` 释放引用。真实图
  lifetime-reuse 有 386 对合法 physical aliases、85 个 final-task 前释放；
- 两计划 lower/upper bitwise 相同，对 external lower max diff
  `7.152557373046875e-07`、sign 9/9。parent NRIR-1 artifact 原五层 hash replay 不变。

结论为 storage-plan correctness/runtime ownership `VALIDATED-REDUCED`。`performance_claimed=false`
必须保留：当前 byte ledger 是 Plan/Schedule logical arena 与 runtime residency contract，不是
`torch.cuda.max_memory_allocated`、真实 allocator reuse、latency、OOM rescue 或 speedup。

representation 审计同时发现：当前 Plan 的 representation decision 不能自动改写 Bound IR；
structured 执行依赖另一份 rewritten module，而 Schedule reference executor 只记录
`MaterializeAction`。因此本轮没有加入假的 structured candidate。下一步应先尝试 fresh CUDA
physical-memory protocol；若 GPU 不可用，则冻结 runner/protocol 并推进 representation semantic
binding bridge，不得用 metadata/hash 代替执行证据。

## 10. Native CUDA Physical-Memory Protocol v1 判定

NRIR-3 已把 NRIR-2 双 storage 的设备测量方法冻结成可运行实现：每个 plan/repeat 使用独立
worker process，5 个 repeats 中交替启动 retain/reuse，每 worker 5 warmup、20 measured；计时
只覆盖 prepared native CROWN backward，并以同步后的 `max_memory_allocated/reserved` baseline
delta、result hash、Bound/PlanTemplate identity 与原始 latency samples 形成 replay-grade artifact。

本机 PyTorch 为 `2.12.1+cu132`，但 `torch.cuda.is_available=false`、device count 0，
`nvidia-smi` 无法连接 driver。因此：

- 冻结 probe artifact 为 `environment_unavailable` 且 `performance_claimed=false`；
- 正式 `generate` 在创建输出目录或 measured row 前 exit 2；
- 没有 CUDA allocator reduction、latency Pareto、OOM rescue 或 speedup 结论；
- 协议测试/全量回归为 `17 passed` / `484 passed, 37 skipped`，静态门禁通过。

协议实现已完成，设备实验待可用 CUDA 主机按原参数执行。当前无需停等硬件；下一代码路线为
representation semantic binding bridge，使 Plan representation 与 `MaterializeAction` 真正改变
Bound/backend execution，并先通过真实 ResNet 双路径语义一致性。

## 11. Native Representation Semantic Binding v1 判定

NRIR-4 已关闭 NRIR-2/3 明确指出的“表示选择只停留在 metadata”缺口：

- source PlanTemplate 对固定 21-op ResNet Bound graph 提供两个全局一致 policy；高预算选择
  `native-dense-v1` + retain-all，`442,656` bytes 选择
  `native-structured-affine-v1` + lifetime-reuse，`442,655` bytes fail closed；
- structured policy 的每个 selected transition 与 source Schedule `MaterializeAction`、rewritten
  execution Bound op 一一绑定。真实图插入 14 个 `REPRESENTATION_CAST` 与 14 个
  `MATERIALIZE`，execution graph 从 21 ops 变为 49 ops；49 个 op 均各自进入 Task 与 Launch；
- rewritten Bound graph 使用独立 execution PlanTemplate/PlanInstance/Task/Schedule hash；没有把
  source PlanTemplate 冒充成对另一 Bound hash 仍有效；
- dense/structured lower 最大差 `9.5367431640625e-07`；二者对 external lower 均 allclose，
  sign 9/9；artifact digest 与 fresh semantic replay 通过；
- selector 新增 storage-compatible prefix pruning，在不改变可行解集合的前提下避免真实图
  dense/structured 全排列的指数枚举。

结论为 representation binding/compiler ownership `VALIDATED-REDUCED`。当前 structured value
由 `DenseLinearOperator` 包装 dense tensor，execution storage 对每个 structured binding 仍保留
至少 dense logical bytes。因此不得声明 compression、memory reduction、latency、CUDA、OOM、
Pareto 或 speedup；source policy 与 NRIR-2 storage 的耦合仅用于确定性预算选择，物理内存收益仍
没有被 NRIR-4 证明。

下一代码门禁是 real-network sliced batch execution：Plan 的 domain/spec/sample batch decision
必须改变实际 Task/Schedule slicing 与 query accounting，并保持 dense/structured、single/batched
语义一致。CUDA NRIR-3 设备实验作为环境可用时的独立待办，不阻塞该代码路线。

## 12. Native Real-Network Sliced Batch Execution v1 进度

NRIR-5 已让 batch decision 进入真实执行，而不再只是 metadata：source template 同时提供 full
与 spec-size-3 candidate；`PlanSelectionContext.max_spec_batch_size` 选择不同 PlanInstance。
source Schedule 的 spec loop 冻结连续半开区间，每个区间生成独立 native child
Bound/Plan/Task/Schedule stack，runtime 校验完整 objective digest 后按 spec 轴聚合结果。

固定 ResNet 的 full path 为 1×21-op child；sliced path 为 3×21-op child，ranges 为
`[0,3)/[3,6)/[6,9)`，合计 63 Task/Launch。两者共享 source Bound/PlanTemplate，source
PlanInstance/Schedule 不同；full/sliced lower max diff `1.9073486328125e-06`，二者均匹配
external oracle、sign 9/9。artifact generate 与 fresh semantic replay 已通过；新旧
native/Plan/Task/Schedule 聚焦 `89 passed`，全量 `508 passed, 37 skipped`，Black/Mypy/
Pylint/diff 门禁通过。

该状态不能写成 batching speedup 或 memory reduction：三个 child 顺序执行，source controller
storage 仍是完整 ledger，未测物理 allocator/latency。v1 只实现 spec axis；domain/sample 与
NRIR-4 representation × batch 四组合是下一联合门禁。完成联合门禁后，再推进真实 repeated-query/
domain batching 与 cache accounting；CUDA NRIR-3 仅在设备可用时按冻结协议执行。

## 13. Native Representation × Batch Composition v1 判定

NRIR-6 已关闭上一节的联合门禁：同一 source Bound/PlanTemplate 同时含 dense/structured-affine
representation/storage policy 和 full/spec-size-3 batch candidate。memory budget 与 query-time
spec limit 由 generic selector 联合决定四组合，四个 source PlanInstance/Schedule identity 均不同。

source selected storage ID 进入 child selection contract；PlanInstance provenance 与 verifier 防止
child 因 shape 变小而改选 policy。固定 ResNet 四组合 child op/task/launch 为
`21/63/49/147`；structured 两组合保留 28 transition 与 49-op execution binding，sliced 两组合
保留三个 exact ranges。四路径对 external lower max diff 均不超过
`1.9073486328125e-06`，sign 9/9；artifact generate/replay、聚焦 `103 passed`、全量
`522 passed, 37 skipped` 与静态门禁全过。

结论为 cross-axis compiler/runtime ownership `VALIDATED-REDUCED`，不是性能关闭：structured
仍存 dense tensor，spec slices 顺序执行，controller storage 仍是逻辑 ledger。下一分支应实现
真实 repeated-query/domain stream 的 batch formation、plan/code cache、per-query lineage/结果恢复
和公平 batched baseline；物理 CUDA protocol 保持环境可用时执行。

## 14. Native Repeated-Query Batching and Cache v1 判定

NRIR-7 已把上一节的“真实 query stream”从计划变成 native execution：9 个不同 property
objectives 各有 query ID、objective digest 与 range，packed runtime 用三个 size-3 child 执行，
serial reference 在相同 source representation/storage policy 下分别执行 9 个 child。结果按 range
恢复到 9 条 query，packed/serial/external 均 allclose、sign 9/9。

cache 是 exact in-process compilation cache：workload/input/state/intermediate-bound、ordered query
contents、budget/policy/batch config 全部进入 key。first miss/second hit，objective/order/state
三个 probe 都产生不同 key 与 miss。artifact replay、聚焦 `121 passed`、全量
`540 passed, 37 skipped` 与静态门禁全过。

结论为 real repeated-query formation/packing/cache/lineage `VALIDATED-REDUCED`。它仍只覆盖同一
input domain 的 property queries；3 vs 9 child 是机制计数而非 timing。下一路线必须加入不同
input boxes 的 BaB parent/child domains、state validity/invalidation、domain packing/restore 与
same-solver baseline，不能把 NRIR-7 自动升级为完整 C3 或 performance claim。

## 15. Native BaB Input-Domain Batching v1 判定

NRIR-8 已关闭 NRIR-7 的“同一 input domain”缺口：fixed ResNet root box 按前三个正宽输入坐标
确定性三层二分为 8 个 leaf queries；每个 leaf/parent box、tree lineage、exact state 与 result
都有独立 digest。child exact state 由 leaf box 重新运行 forward IBP 得到；parent state 单独记录为
`warm_start_only`，编译、验证和执行 trace 均禁止将其作为 child exact input。

同一 source Bound/PlanTemplate 提供 full-domain 与 size-4 candidates。max domain=4 产生
`[0,4)/[4,8)` 两个 Schedule query slices 和两个 child compiler stacks；max domain=8 产生一个
full child；serial reference 以同一 representation/storage policy 执行 8 个单域 child。固定
artifact 的 packed/full/serial lower/upper 均 bitwise equal，8/8 query/parent/result 恢复。
聚焦 `19 passed`，全量 `559 passed, 37 skipped`，fresh replay 与静态门禁全过。

结论为 input-box domain formation、state validity、Plan/Schedule domain-axis execution 与 restore
`VALIDATED-REDUCED`。该机制不是完整 BaB：没有 ReLU split、β state、priority queue、bound prune、
termination 或 property verdict；2 vs 8 也不是性能数据。下一代码路线是 native ReLU-split BaB
queue/state v1，而不是直接书写 speedup 或提交 ASPLOS。

## 16. Native ReLU-Split BaB Queue v1 判定

NRIR-9 已关闭 NRIR-8 的“只有 input-box branch”缺口。plain-CROWN Bound IR 支持 6 个 ResNet
ReLU 的 first-class int8 split inputs；split payload 进入 ReLU op、Bound hash、Plan workload/
capability、Task 和 Schedule。runtime 对 key/shape/dtype/device/range/hash 和 constrained
preactivation fail closed；local forward provenance 与 external verifier ownership 分开。

best-first bounded queue 冻结 node/parent/depth、widest-ambiguous branch、priority、prune/expand/
terminal reason与预算。每个 child 只继承 discrete split state，forward IBP 与 native compiler
stack 重新执行；parent exact state 不可复用。toy 15-node complete tree 的 packed/serial stacks 为
5/15。固定 ResNet 7-node run 形成三代、3 expand、4 frontier，packed/serial stacks 为 3/7；
lower/upper max diff 为 `1.8310546875e-04/1.220703125e-04`，queue/branch/split identity 一致。
artifact generate/replay、聚焦 `68 passed`、全量 `577 passed, 37 skipped` 与静态门禁全过。

结论为 first-class ReLU split、bounded queue/control flow 与 actual node-batch execution
`VALIDATED-REDUCED`。固定 run 正确报告 `budget_exhausted`、`property_status=not_claimed`。没有
α/β optimization、beta constraint、完整搜索/verdict 或性能证据；3 vs 7 不是 speedup。下一代码
路线为 native α/β optimization state + warm-start validity v1。

## 17. Native Alpha/Beta Optimization State v1 判定

NRIR-10 已关闭 NRIR-9 的“只有 plain-CROWN split queue”缺口。optimized ReLU BoundOp 显式绑定
split/alpha/beta；Plan workload/capability/provenance、Task 与 Schedule 均消费同一 frozen state。
fixed ResNet 共有 19 graph inputs、6 optimized ReLU ops、21 Task/Launch。native 与 legacy αβ
lower/upper max diff 均为 0；非零 beta 相对 zero-beta 将 lower 提升 `0.34039306640625`。

state scope 绑定 model/input/objective/intermediate bounds/split/policy/payload。parent→child 单调新增
split 只允许 warm initialization，不允许 exact reuse；split reversal/removal 或 semantic drift 均拒绝。
artifact generate/replay、聚焦 `50 passed`、全量 `591 passed, 37 skipped` 与静态门禁全过。

结论为 frozen alpha/beta state ownership、beta constraint execution、warm-start validity
`VALIDATED-REDUCED`。Adam iteration/gradient/update 尚未 lower 到 Task/Schedule；没有完整 BaB/
property verdict 或性能证据。下一代码路线为 native alpha/beta optimizer-step Task/Schedule control v1。

## 18. Native Alpha/Beta Optimizer-Step Schedule v1 判定

NRIR-11 已关闭 NRIR-10 的“Adam iteration/gradient/update 仍 opaque”缺口。Optimizer Plan 绑定
NRIR-10 source compiler 的 10 个 hash、initial state/scope、policy、ReLU keys 与 warm-start；固定
steps 被静态 lower 为 evaluate/reduce/backward/Adam/project/select-best Task 与同步 Schedule。
runtime 只按 action 顺序执行，并记录完整 value hash chain、gradient、projection、evaluation 和
per-domain best selection。

2-step toy 为 13 actions，与 legacy bounds/alpha/beta 逐张量一致。固定 ResNet 1-step child 为
8 actions；alpha/beta gradient L1=`169.23175295069814/12.862210273742676`，Schedule 对 legacy 与
selected-state native compiler 的 lower/upper max diff 全为 `0.0`。artifact replay hash 为
`31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`；聚焦 `35 passed`、
全量 `612 passed, 37 skipped`，静态门禁全过。

结论为 fixed-step optimizer control ownership `VALIDATED-REDUCED`。这不是 dynamic optimizer，也
尚未进入 multi-node BaB queue；没有完整 termination/property verdict 或性能证据。下一代码路线
是 native optimized ReLU-split BaB integration v1：每个 node 由 optimizer Schedule 产生 selected
state，再经 native Bound stack 执行，parent 只能作为 monotonic-refinement initialization。

## 19. Native Optimized ReLU-Split BaB v1 判定

NRIR-12 已关闭 NRIR-11 的 single-node 边界。每个 best-first queue node batch 都执行固定 1-step
optimizer Schedule（8 actions），selected alpha/beta state 随后进入 native compiler（21 tasks）。
child parent state 按目标 batch layout 重组并重建 scope，NRIR-10 classifier 必须判为 monotonic
refinement；parent exact state 不被 child exact execution 消费。

toy complete queue 为 15 nodes，packed/serial 5/15 stacks，selected state hash 与 bounds 均一致。
固定 ResNet bounded queue 为 7 nodes/3 expands/4 frontier，packed/serial 3/7 stacks；lower/upper
max diff=`1.220703125e-04/1.8310546875e-04`，alpha/beta tensor max diff=
`4.172325134277344e-07/7.450580596923828e-09`，selected native re-execution max diff 为 0。
artifact replay hash=`e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`；
聚焦 `18 passed`、全量 `630 passed, 37 skipped`，静态门禁全过。

结论为 optimized queue integration/control ownership `VALIDATED-REDUCED`。exact batch-layout state
hash 不相等且已披露；fixed run 仍是 `budget_exhausted/property_status=not_claimed`，所以不是完整
verifier。下一代码路线是 native property termination/verdict v1：verified/unsafe 必须有闭合 proof
或 concrete witness，任何未闭合 budget/depth/timeout 都保持 unknown。

## 20. Native Property Termination and Verdict v1 判定

NRIR-13 已将 NRIR-12 的 `property_status=not_claimed` 边界升级为独立、可重放且
fail-closed 的 `verified / unsafe / unknown` 证明层。verified 要求 complete queue 且所有
leaf 都有 `lower >= threshold` 的 sound prune；任何 frontier、depth terminal 或无法证明的
prune 都会成为 unresolved leaf 并返回 unknown。

unsafe 不信任序列化数字：新 concrete Task IR executor 重执行 primal graph，检查 input box、
node ReLU split path 和严格的 objective violation，再绑定 input/output/value-trace hash。toy
verified/unsafe/unknown matrix 与非 root split witness 均已通过。固定 ResNet 中心点 objective
为 `0.8564349412918091`，不是反例；7-node 运行仍有 4 frontier，因此正确返回
`unknown/node_budget_frontier_open`，没有伪造 verified。

结论为 three-state verdict soundness `VALIDATED-REDUCED`。artifact replay hash 为
`9e3dceed23c8759c910938ba7c9f84caaeb949c8f19b72fab104ce4e1b733405`，聚焦 `19 passed`，
全量 `649 passed, 37 skipped`，静态门禁全过。
NRIR-13 closure 当时仍缺 candidate discovery、multi-clause property、timeout/dynamic early stop
与 real complete closure；该下一路线现已由第 21 节 NRIR-14 执行。不能把 NRIR-13 单独升级为
端到端验证器或性能 claim。

## 21. Complete Verifier Query v1 判定

NRIR-14 把 NRIR-13 的单 clause/caller-candidate 边界扩展为可直接执行的 conjunction query。
每个 clause 按 ascending index 顺序执行 deterministic center-start box-projected gradient search、
optimized ReLU-split queue 和 sound verdict；candidate search 的 `not_found` 永远不构成 proof，
found candidate 仍必须经过 concrete primal Task IR replay。任一 replayed violation 立即返回 unsafe
并显式标记后缀 clauses skipped；deadline 只在 stage 边界 cooperative 检查，到期 clauses 显式 pending。

toy matrix 独立产生 verified、unsafe、attack-not-found unknown 与 deadline unknown。固定 ResNet
使用九个真实 property objectives；九个 candidate best objective 均为正，未发现反例，但九个
native scalarized lower bounds 均为负，因此总体正确返回
`unknown/one_or_more_clauses_unresolved`，unresolved 为 9/9。该结果说明 query control 已闭环，
同时也把真正 blocker 定位为 bound tightness，而不是继续增加包装层。

artifact 位于 `artifacts/complete-verifier-query/vnncomp21-resnet2b-prop0-cpu-v1/`，generate/replay
hash=`d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`。相关回归
`39 passed`，全量 `670 passed, 37 skipped`，静态门禁全过。

结论为 complete-query correctness/control `VALIDATED-REDUCED`，不是 real-property closure 或
性能结果。下一工程阶段必须先冻结端到端 phase/tightness baseline，至少分解 candidate、bound
optimization、queue、verdict 的 wall time、proof gap、nodes 与 batching/cache 行为；在公平
same-solver/竞品口径下再决定 dynamic optimizer、branching/tightness 和执行优化的优先级。

## 22. End-to-End Tightness and Performance Baseline v1 判定

NRIR-15 修复了 NRIR-14 optimized queue 丢失 external intermediate semantics 的断层。typed
external provenance、六组 ReLU intervals、adaptive α 初始化与 split-constrained child batches
已经贯穿 optimizer source/Plan/state/native stack/query；任何来源或 tensor schema错配均 fail closed。

固定 ResNet external-adaptive 1-step lower 对 frozen external initial 无退化，并把
`1/3/5/6/7/8` 六个 clauses 证明为 verified；`0/2/4` 保持 unknown。fresh semantic replay 与
artifact hash `14c3b9dc2e5376156be1f33f3e8804ec21f60e11096bd3bdc95225b7e1474376` 一致。

三组轮换 CPU 诊断中，三种 queue variant median 均约 6.7 秒；candidate/verdict 仅约
3.6/3.9 毫秒。结论是 fixed compiler/hash/selected-native validation re-execution 支配当前耗时。
因此下一代码路线是 prepared production fast path，并要求与 audit path 数值/状态一致；之后才对
三个 hard clauses 做 branching/stronger-bound。该阶段只为单 workload CPU
`VALIDATED-REDUCED` diagnosis，不是 production/CUDA/competitor speedup，也不是 ASPLOS-ready。

## 23. Prepared Production Fast Path v1 判定

NRIR-16 新增 exact prepared optimizer/query capsule。cold phase 完整验证 optimizer/native source
compiler、scope 与 hashes；warm phase 仍由 optimizer Task/Schedule 驱动数值更新与 best selection，
但不构造逐 action audit hash chain，也不执行 selected-native validation stack。生产 trace 明示这两项
省略，任何 semantic identity 漂移均 fail closed。

fixed ResNet 三组 audit/prepared warm median 为 `59.078 s/110.950 ms`，内部 evidence-overhead
diagnostic ratio=`532.47×`；cold prepare+first=`16.139 s`，retained payload=`2,076,372 B`。
production lower 对 audit max diff=`1.90735e-6`、candidate/status exact，仍为 clauses
`1/3/5/6/7/8` verified、`0/2/4` unknown。fresh replay hash=
`e14fcd62b322c0bc60d45c726cf94a7aa6cfb8d7aa3212662d08996db169b6b2`。

结论为 root-only repeated-query preparation 与单 workload CPU overhead removal
`VALIDATED-REDUCED`。这不是 competitor speedup、child BaB、CUDA 或完整性质闭合；下一代码路线
是 hard-clause branching/stronger-bound v1。

## 24. Hard-Clause Objective Branching v1 判定

NRIR-17 新增 objective branch Plan/Task/Schedule 与 exact score runtime。top-width-per-ReLU
shortlist 中每个 candidate 的 inactive/active child lower 都由同一 selected alpha/beta state 批量
计算，再按 worst-child、mean-child、stable identity 选择；所有输入与选择结果进入 stable hash。

fixed ResNet clauses `0/2/4` 的 same-budget widest→objective worst leaf 分别为
`-0.440550→-0.319799`、`-0.498173→-0.426609`、`-0.562577→-0.504676`。fresh replay hash=
`1193bee8817e4acc9ec33f8ddadc00a671d0ac3c9411f14f62978eb5ab1a95bd`，全量
`707 passed, 37 skipped`。

结论为 branch IR/control 与 bounded-tree tightness `VALIDATED-REDUCED`。三个 hard clauses 仍
unknown，不能升级 complete verifier；20–22 秒 audit timing 不是 production 或 competitor
performance。下一阶段必须扩展多 workload/设备/竞品 E2E，并研究能实质缩小剩余 frontier
deficit 的 stronger-bound mechanism。

## 25. Multiworkload Competitor E2E Baseline v1 判定

NRIR-18 新增原生 VNNLIB box/property frontend 与 typed Query IR。首批固定 VNN-COMP 2021
CSV ordinal 0 的 MNISTFC 256x2、CIFAR10 ResNet2B、OVAL21 base CNN；三份 property 的 input
lower/upper、九条 C 与 rhs 均与固定 αβ-CROWN parser 一致。workload Plan/Task/Schedule 明确包含
3 source locks、21 tasks 和 6 个 fresh-process native/competitor execution action，source、policy、
timeout、device、thread 与所有 IR hash 均可 replay。

正式 CPU 矩阵中，BoundFlow 对三项均返回 sound `unknown`：MNISTFC 9 clauses 中 3 unresolved，
ResNet 在 deadline 后完成 2 clauses、7 pending，OVAL21 仅 clause 8 unresolved。固定
αβ-CROWN 对 MNISTFC/OVAL21 返回 verified，对 ResNet timeout 后 unknown。对应 fresh-process
E2E 分别为 `38.644/4.312 s`、`66.910/64.198 s`、`31.498/4.527 s`；由于算法、complete
能力和单次样本不同，这些数字只作诊断，禁止计算 speedup。

artifact 位于 `artifacts/multiworkload-competitor-e2e/vnncomp21-three-topology-cpu-v1/`，fresh
replay hash=`473b287bb88e4c52426b405aeb4164aa72a98d7b1bbd74c00471fe1d1451deb0`；全量
`723 passed, 37 skipped`。该阶段关闭 ingest/IR/control/workload coverage
`VALIDATED-REDUCED`，不关闭 verifier parity、GPU/performance 或 ASPLOS-ready。ResNet native
local root lower=`-543.717/-789.331`，下一门禁明确为 native intermediate-bound refinement
Plan/Task/Schedule，再对三 workload 重测 closed clauses 与成本。

## 26. Native Intermediate-Bound Refinement v1 判定

NRIR-19 将 top ambiguous-width target selection、selected plain-CROWN backward、sound
intersection、forward propagation 与最终 emit lower 为独立 Plan/Task/Schedule。Plan 绑定 primal
graph、input box、split state、初始 intermediate bounds、policy 和每个 neuron target；runtime 必须
逐 action 消费 Schedule，任何 source/schema/hash/target/order 漂移均 fail closed。新增
`native_refined` provenance，不能冒充 external verifier bounds。

同一 7-node/depth-2/5-step CPU policy 的正式 fresh-process 对照中，MNISTFC unresolved 从
`{3,7,8}` 降为 `{8}`，关闭 clauses `3/7`，nodes `31→21`；OVAL21 从 unknown 变 verified，
关闭 clause `8`，nodes `15→11`。ResNet 仍为 unknown 且只完成前两个 clauses，但 root lower 从
`-543.717/-789.331` 改为 `-473.221/-628.780`，改善 `+70.496/+160.551`，没有隐藏失败。

artifact 位于 `artifacts/native-intermediate-refinement/vnncomp21-three-topology-cpu-v1/`，fresh
source-to-IR replay hash=
`f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；focused
`9 passed`、全量 `732 passed, 37 skipped`，Black/Mypy/Pylint 全过。

结论为 native refinement IR/control 与 multiworkload tightness `VALIDATED-REDUCED`。BoundFlow
只在 1/3 workload complete verified，CUDA/重复性能矩阵仍缺，单次 CPU timing 不形成 speedup，
ASPLOS-ready 仍为 NO。ResNet 表明纯 width shortlist 不足；下一路线是 objective-directed
intermediate target selection，以 clause-sensitive influence 选择有限 targets，再评估 per-child
recomputation，而不是先扩大树深或做 CUDA timing。

## 27. Objective-Directed Intermediate Refinement v1 判定

NRIR-20 将当前 scalar clause 的 CROWN backward coefficient influence 加入 target selection。
新 policy 以 `ambiguous_width * max(abs(A_u), abs(A_l))` 排序；Plan 绑定 objective hash 和每个
target 的 influence/score，Task/Schedule 显式声明 objective influence dependency。只允许一个
finite scalar clause，旧 width policy payload/hash 保持兼容；selection heuristic 不参与
soundness，最终 bounds 仍来自 selected plain-CROWN 与单调 intersection。

固定 ResNet2B property 0 clauses `0/1` 的 same-budget fresh-process 对照中，两种 policy 均为
96 targets。target overlap=`16/96`、`27/96`；width/objective root lower 分别为
`-473.221222/-417.292480` 与 `-628.780334/-602.551392`，objective 改善
`+55.928741/+26.228943`。结果仍远低于 threshold，没有声称 property closure。

artifact 位于
`artifacts/objective-directed-intermediate-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`，
fresh semantic replay hash=
`8fce1c7c3e5c63adb14a7ab5b9f23407e4a7a1406353750e4f150ee745b4e88e`；focused
`16 passed`、全量 `739 passed, 37 skipped`，Black、targeted Mypy、Pylint 10.00/10 通过。

结论为 objective-directed refinement IR/control + fixed-root tightness
`VALIDATED-REDUCED`。CPU timing 仅诊断，CUDA/竞品/重复性能/完整验证/ASPLOS-ready 均未关闭。
下一路线是 per-child objective-directed refinement：child 必须按 exact split state 重算
intermediate bounds、influence、Plan/Task/Schedule，parent refinement 只能作为 warm-start 提示。

## 28. Per-Child Objective Refinement v1 判定

NRIR-21 已把 child exact split→forward→objective influence→target→selected CROWN→propagation
完整接入 optimized BaB。每个 evaluation 与 refinement Plan/Task/Schedule、semantic trace、
initial/final bounds hash 一一对应；packed/serial 的 per-node IR、bounds 与 logical queue 一致，
parent alpha/beta 仅初始化，旧默认 queue payload 保持无扩展字段。

固定 clauses `0/1`、7-node/depth-2、同 96-target/5-step 预算下，root lower 均与 root-global
相同；但最差 leaf lower 从 `-413.739044/-591.944275` 退化到
`-414.587006/-592.880920`。因此 closure=`VALIDATED-NO-GO`，没有 complete property、CUDA、
competitor parity、重复性能或 ASPLOS-ready claim。下一路线固定为祖先约束单调 carry-forward，
解决“child 重算时丢失 root refinement tightening”的结构性问题。

## 29. Ancestral-Constraint Refinement v1 判定

NRIR-22 将 parent refinement execution 作为 child Plan/Task/Schedule 的 typed source：source
final/Plan/semantic trace 三哈希绑定，materialize Task 显式输入，local→constrained initial→final
双重单调，queue parent lineage 与 `sound_constraint_only` consumption fail closed。默认、root-global
与 NRIR-21 independent payload 均条件兼容。

fixed clauses `0/1` 的 ancestral worst leaf=`-340.971832/-517.858826`，相对 independent
提升 `+73.615173/+75.022095`，相对 root-global 提升 `+72.767212/+74.085449`；root lower
仍为 `-417.292480/-602.551392`。结论为 fixed bounded-tree tightness
`VALIDATED-REDUCED`，不是 property closure、CUDA/competitor speedup 或 ASPLOS-ready。

下一工程门禁为 hard-clause convergence expansion：扩展 hard clause coverage 与 depth/node budget
曲线，量化剩余 closure deficit，再决定动态 BaB budget/termination 或公平 GPU E2E 的先后顺序。

## 30. External-Seeded Ancestral Refinement v1 判定

NRIR-23 新增 external-owned typed constraint seed。seed 对 raw external bounds 与 local forward
求可行交集，并绑定 external ordered digest、effective constraint hash、primal/input 与 source
artifact/model/property/objective-set；Plan/Task/Schedule/action trace 均显式引用 seed。queue root
消费 seed，child 只消费 validated parent refinement，二者互斥且逐节点 hash lineage fail closed。

固定 ResNet clauses `0/2/4` 上，external baseline→seeded root-global→seeded ancestral 的 worst
leaf 分别为 `-0.319799→-0.319110→-0.318287`、
`-0.426609→-0.425481→-0.425477`、`-0.504676→-0.504142→-0.504142`。三条 ancestral
均不弱于 root-global，两条严格改善，但全部仍负。

结论为 typed seed/control + fixed-tree tightness `VALIDATED-REDUCED`。artifact generate/replay hash=
`9f52b99a74dab448626061f5b8f060f3b8c43b6c03f6deb0899d9fe91883d9f7`；全量
`766 passed, 37 skipped`，静态门禁全过。下一工程动作是冻结 7/15/31 nodes、depth 2/3/4 的
hard-clause convergence；不得升级 complete property、GPU/performance 或 ASPLOS-ready claim。

## 31. External-Seeded Depth/Node Convergence v1 判定

NRIR-24 固定 NRIR-23 的 external seed、ancestral carry、objective branch、25-step optimizer、
16-target/ReLU 单 pass refinement 与 batching，只改变 `7/15/31 nodes × depth 2/3/4`。九个
clause/budget 均由 fresh process 生成 checkpoint shard，并在 replay 中逐对象重算。

clauses `0/2/4` worst terminal lower 分别从
`-0.318287/-0.425477/-0.504142` 改善到 depth-3 的
`-0.299506/-0.413456/-0.479104`，再改善到 depth-4 的
`-0.282360/-0.401845/-0.459939`。三条曲线单调且未饱和；logical domains 按
`split_state_hash` 嵌套，lineage/branch/refinement semantics 通过。所有 deepest terminals 仍负，
三条 bounded-tree status 均为 unknown。

结论为 fixed-hard-clause convergence trend `VALIDATED-REDUCED`。artifact/replay hash=
`db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`。不升级 complete
property、GPU/performance、multi-workload、competitor 或 ASPLOS-ready；下一工程动作是冻结
dynamic ancestral refinement budget/multi-pass 对照，不再以纯 fixed-depth 扩展为主路线。

## 32. Dynamic Ancestral Refinement Budget v1 判定

NRIR-25 把 parent-lower risk allocation 冻结成一等 policy/decision IR，并把 24/8/base16 assigned
cap 精确 lower 到逐 node refinement Plan，Task/Schedule/execution/queue trace 与 group conservation
全部 fail closed。旧 fixed16 路径条件兼容。

固定 clauses `0/2/4`、31 nodes/depth 4、单 pass 下，dynamic8_24 相对 fixed16 的 worst lower
分别改善 `+0.0003859997/+0.0002329946/+0.0002717972`；两 mode 的 planned cap 均为 `496`，
actual selected targets 均为 `2976`。按预注册门禁为 `VALIDATED-REDUCED`。

artifact evidence hash=`85d9f274c6e17614bcbf318bdbfea18219b03876024be16aea3329ee4d3c56bd`。
三条 bounded tree 仍 unknown；不声明 complete property、performance、CUDA、multi-workload、
competitor 或 ASPLOS-ready。下一工程动作是 typed multi-pass refinement/termination 与 pass lineage。

## 33. Typed Multi-Pass Refinement v1 判定

NRIR-26 将 multi-pass 总 cap partition、updated-width target reselection、prior-target ledger 与
no-unseen termination 编译为一等 Plan/Task/Schedule/decision trace；旧路径保持条件兼容。dynamic
8/16/24 assigned cap 分别拆为 4+4/8+8/12+12，逐 node/树总 cap 守恒。

固定 clauses `0/2/4` 上，single 与 split-two-pass worst lower 完全相同：
`-0.2819737196/-0.4016119838/-0.4596676826`；planned cap 均 `496`、actual targets 均
`2976`，三棵 logical tree 均 `31/31` 重合，没有 stopped pass。因此 mechanism/control 可保留，
但方法按预注册门禁为 `VALIDATED-NO-GO`。

artifact evidence hash=`38992cace70214ffcbd670f03dcfca182e0925bee31eb4df885dab4dab03494d`。
不声明 tightness、complete property、performance、CUDA、multi-workload、competitor 或
ASPLOS-ready；停止 node-initial static influence 的同总 cap 拆 pass。

## 34. Production Prepared Verifier v1 判定

NRIR-27 新增 production verifier Plan/Task/Schedule 与 complete-query 路径。每个 dynamic batch
显式执行 validate、optimizer、materialize、commit 四类 action；production 不构造 audit tensor
hash chain，也不再次运行 selected-native oracle。旧 audit query/hash 与默认行为保持兼容。

三种真实拓扑的 clause-0 相同算法 fresh-process median audit→production 为：MNISTFC
`4.510→3.301 s`（`1.3663×`）、ResNet2B `22.509→9.104 s`（`2.4723×`）、OVAL21
`5.192→3.578 s`（`1.4511×`）；每个 workload 三组交替次序，semantic parity 全过。full
production median 为 `14.834/60.754/11.964 s`，状态仍全部 unknown；ResNet 三次完成 `9/9`
clauses，只说明 deadline/accounting 改善，不形成 property closure。

artifact evidence hash=
`7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`。结论为 production
runtime + internal CPU overhead `VALIDATED-REDUCED`；竞品参考仅是不同完整性协议下的历史单次
诊断，不得计算 speedup。GPU、公平 complete competitor、verified/unsafe closure 与 ASPLOS-ready
仍未成立。phase evidence 显示 full-query execution 的约 `59%–65%` 尚在四类 action 之外；下一
工程门禁为 parametric dynamic-batch `PlanTemplate/PlanInstance` 与 compile-cache ownership。

## 35. Parametric Dynamic Batch Compiler v1 判定

NRIR-28 新增静态 optimizer PlanTemplate、动态 PlanInstance、可复用 Task/Schedule、query-scoped
exact cache 和 additive parametric queue/query。template 绑定 graph、tensor contract、ReLU layout、
policy 与 provenance；instance 绑定 input/objective/intermediate/split/scope/initial-state content。
contract 或 exact runtime tensor 漂移在执行前 fail closed；NRIR-27 frozen 文件零修改且 artifact
继续 replay。

三组交替 fresh-process full-query production-v1→v2 median 为：MNISTFC
`14.807→3.456 s`（`4.2849×`）、ResNet2B `61.239→6.209 s`（`9.8630×`）、OVAL21
`13.021→3.718 s`（`3.5024×`）。每次 query 只编译 1 个 template；instances/miss/hit 分别为
`19/1/18`、`27/1/26`、`11/1/10`。v1/v2 的 clause accounting、logical queue、selected state 与
root bounds 逐项一致。

artifact evidence hash=
`117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`；全量
`818 passed, 37 skipped`。本阶段以 same-algorithm full-query internal CPU performance
`VALIDATED-REDUCED` 关闭；三类 property 仍 unknown，无 CUDA、竞品 speedup、complete-property 或
ASPLOS-ready claim。下一工程门禁为 fixed-wall-clock parametric BaB depth/node scaling。

## 36. Wall-Clock Parametric BaB Scaling v1 判定

NRIR-29 将三档 search budget、三真实 workload、三 fresh repeats 与轮转次序编译成一等
Plan/Task/Schedule；每个 worker 保存逐 clause split-state logical domains、leaf verdict、compiler
template/cache/instance 与 raw timing。artifact replay 重建 experiment IR，并重新校验 27 个
Task/record、同预算 repeat semantics、跨预算 nesting 与 closure gate。

27/27 worker 都是 `completed=9,pending=[]`；三次重复的 semantic signature 一致，所有 workload
均满足 `domains(7)⊂domains(31)⊂domains(127)`，公共 lower 最大漂移 `0.0`。MNISTFC verified
从 `6/9` 严格提升为 `8/9`，31 nodes 已与 127 nodes 相同；ResNet 三档始终 `0/9`，OVAL21
三档始终 `8/9`。n127d6 median execution 分别为 `2.515/58.566/2.287 s`，只作为固定协议资源
曲线，不计算不同预算之间的 speedup。

artifact evidence hash=
`e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`。按预注册门禁以
search-coverage `VALIDATED-REDUCED` 关闭；三类完整 query 仍全部 unknown，ASPLOS-ready 仍为 NO。
ResNet 在 1143 total nodes 后仍 0/9、OVAL/MNIST 的最后 clause 也未随纯扩树关闭，所以下一工程
门禁为 typed hard-clause escalation：只对 unresolved clauses 编译更强 native intermediate
refinement/branch policy，并继续保持 fixed total deadline、sound fallback 与 artifact replay。

## 37. Typed Hard-Clause Escalation v1 判定

NRIR-30 将 baseline local-forward `7/depth2`、exact unresolved admission、shared native selected-CROWN
refinement、projected `31/depth4` parametric query、original-ordinal aggregate 与 fail-closed fallback
编译为一等 Plan/Decision/8-task TaskModule/Schedule。whole deadline 只有 60 秒；baseline
verified/unsafe 不重跑，over-deadline escalation proof 丢弃而不是升级结果。

三 workload 各三次 fresh process，baseline 与 NRIR-29 n7d2 accounting/root/evaluated nodes 对齐。
MNISTFC admit `[3,7,8]` 后 final verified 稳定为 `[0..7]`；ResNet admit `[0..8]` 后仍 0/9；
OVAL21 只 admit clause 8，并三次都从 unknown 变为完整 query `verified`。median whole-stage
execution 为 MNIST `2.974 s`、ResNet `20.146 s`、OVAL `2.208 s`；9/9 run 都
`fallback=none`，但不形成 speedup claim。

artifact evidence hash=
`df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`。本阶段以 typed staged
control + fixed-deadline property coverage `VALIDATED-REDUCED` 关闭；只覆盖三个 CPU workload，
无 GPU、competitor、完整 benchmark suite 或 ASPLOS-ready claim。下一工程门禁只改变 hard-clause
refinement selection：per scalar objective 编译 influence/target Plan，在相同 admission、31-node 与
deadline 下检验 MNIST clause 8/ResNet root 或 closure 的严格改善。

## 38. Objective-Directed Hard-Clause Escalation v1 判定

NRIR-31 在 NRIR-30 shared refinement 的 validated final bounds 上，为每个 admitted scalar clause
单独编译 objective-influence refinement，再执行 31/depth4 parametric query。全九子句静态展开为
33 个 guarded Task/Schedule action；source Plan/semantic trace、objective hash、original ordinal、
deadline discard 与 aggregate 都 fail closed。NRIR-30 frozen files 未修改。

pilot 先行并因 ResNet root tightness 通过。三 fresh repeats 中，MNIST 保持 8/9、OVAL 保持 9/9；
ResNet 仍 0/9，但九条 root lower 相对 shared top-width 全部严格改善，三轮 delta 逐值一致，最小
`+81.522583`、最大 `+179.970459`。9/9 run 都 `fallback=none`，所有 final verified 都是
NRIR-30 的 superset。

artifact/replay evidence hash=
`fb9e503bdf93cb9ce56f52915f1965f1f542e092945d4d7d77d8b8c4bd91764a`。结论为 objective-root
tightness `VALIDATED-REDUCED`；没有新增 property closure，不声明 performance、GPU、competitor、
完整 suite 或 ASPLOS-ready。下一工程门禁为 NRIR-32 objective-ancestral hard-clause escalation，
把 root objective execution 作为动态 child 的 typed ancestral source，验证 frontier/closure 增益。

## 39. Objective-Ancestral Hard-Clause Escalation v1 判定

NRIR-32 新增 additive static Plan、committed dynamic Task IR、1:1 sequential Schedule 与 native
objective-ancestral queue。root admission 绑定 NRIR-31 typed execution；每个 child compile/refine/eval
逐项绑定 parent final-bound、Plan、semantic trace 与 split-state hash；emit 显式依赖所有已提交
evaluation/transition，deadline 后未提交工作不得进入 proof identity。

ResNet clause 0 two-child pilot 先得到 worst-child `+59.253479` 改善。正式三 fresh repeats 固定
31/depth4/60 s：root lower exact parity=`-204.17315673828125`；ancestral 每次提交 7 nodes、24 tasks、
max depth 2，worst active lower=`-104.76541137695312`；31-node root-global 对照为
`-200.46539306640625`，三轮 delta 均为 `+95.69998168945312`。committed queue/Task/refinement
hash 重复一致，fresh replay 通过。

artifact evidence hash=
`8fba8deca18dcbf0b4b258aa390c1dd48d250c71ea1a48ddb991388765411bfc`。结论为 typed lineage +
frontier tightness `VALIDATED-REDUCED`；当前仍是单 ResNet property/clause、CPU serial audit path，
cooperative deadline 后丢弃 late evaluation。没有新增 closure、performance、GPU、competitor、完整
suite 或 ASPLOS-ready claim。下一门禁为 fixed-deadline child refinement budget/cap Pareto。

## 40. Objective-Ancestral Child Budget Pareto v1 判定

NRIR-33 以 additive Plan 协议复用 NRIR-32 queue engine，selected cap、five-cap calibration、90%
retention selection 与 evidence hash 全部一等化；旧 cap128 source/artifact 未修改且可 replay。

固定 ResNet clause 0 的 cap `8/16/32/64/128` fresh-process pilot 全部只提交 7 nodes、到 depth 2；
worst active lower 从 `-173.078613` 单调改善到 `-104.765411`，但没有任何 coverage 变化。相对
root-global `-200.465393` 的 cap128 gain 为 `+95.699982`，预注册规则只能选择 cap128。

结论为 cap-only coverage `VALIDATED-NO-GO`；pilot hash=
`db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`。没有新 property 或
performance claim。下一工程门禁为 sibling packed refinement/evaluation + parametric evaluator，
保持 cap128、typed ancestral lineage 与 60 秒 deadline。

## 41. Sibling-Packed Objective-Ancestral Evaluator v1 判定

NRIR-34 把已有 packed node helper 提升为一等 source/evaluator projection、SiblingGroup
Plan/Task/Schedule 与 atomic queue runtime。每个 child 仍独立执行 cap128 objective-ancestral
refinement；同一 parent 的 `(-1,+1)` pair 只共享 optimizer 与 selected-native compiler execution。

first-pair profiler 的 serial→packed child elapsed 为 `13.291550→7.018038 s`，optimizer/native group
均 `2→1`，bounds exact。正式三 fresh repeats 固定 31/depth4/60 s，serial accepted nodes
`[7,7,7]`，packed `[15,15,15]`；common lower/upper max diff 都是
`7.62939453125e-06`，minimum node gain=`+8`。packed max depth=`3`、worst active lower=
`-76.07719421386719`，serial 为 depth `2`/`-104.76541137695312`。formal hash=
`9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`。

ResNet property 0 的 9-clause global-60s integration 保持 sound `unknown` 与 original ordinal：完成
clause 0 的 13 nodes/6 atomic groups，unresolved `[0]`，pending `[1..8]`。evidence hash=
`dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`。结论为 single-hard-clause
same-algorithm deadline coverage `VALIDATED-REDUCED`；atomic cooperative wall time 会到
`64.5—66.2 s`，不声明硬实时/wall-clock speedup、property、GPU、competitor 或 ASPLOS-ready。
下一门禁是 NRIR-35 cross-clause objective/root/compiler sharing + anytime global budget，目标是在同一
60 秒内增加 completed original clauses。

## 42. Cross-Clause Anytime Objective Evaluator v1 判定

NRIR-35 用 static Plan/Decision/6-task TaskModule/Schedule 把 frozen NRIR-31 all-clause floor 与
NRIR-34 clause-0 packed queue 串接。Decision 只有在 floor completed `[0..8]`、final unknown、
clause 0 unresolved 且 exact accepted child refinement 存在时才 admit；root Plan/semantic/final-bound
hash、original ordinal 与 global deadline 都 fail closed。Aggregate 始终保留九个 original ordinals，
packed unknown 只能留下 exact floor。

feasibility 先以 floor `22.180303 s`、packed 7 nodes 通过。正式 runtime 三 fresh repeats 的 floor
elapsed=`[22.227251,21.622773,21.834220] s`，每轮 completed/unresolved=`[0..8]`；packed accepted
nodes=`[7,7,9]`。whole elapsed=`[61.991720,62.598928,68.042604] s`，是 cooperative atomic
sibling-group completion，不是硬实时或 speedup。formal hash=
`74533c9c211a3007bf5af43c08865febd95c3f9ccf1a268e56738793ec9d14d5`；replay、六类同步重哈希
tamper、关联 29 tests、全量 `874 passed, 37 skipped` 与静态门禁均通过。

结论为 cross-clause control/original-ordinal preservation `VALIDATED-REDUCED`；三轮 final 仍为
sound unknown、9/9 unresolved，`performance_claimed=false`。没有 property closure、GPU、competitor、
multi-workload 或 ASPLOS-ready claim。下一分支为 `feat/multi-clause-anytime-priority-v1`：在同一
global 60 秒预算内用 typed priority/time slice 覆盖多个 unresolved clauses，不为每条 clause 重置
deadline。

## 43. Multi-Clause Anytime Priority v1 判定

NRIR-36 新增 static Policy/Plan/8-task TaskModule/Schedule、ranked Candidate/Decision、每 dispatch
Slice IR 与 multi-outcome Aggregate。rank 从 NRIR-31 exact floor root lower 独立重算，top-2 固定为
clauses 2/3；每条 slice 按 dispatch 时真实 remaining global budget 动态等分，私有 one-shot clock
将 cutoff 传给 frozen NRIR-34，完整 sibling group 才能提交。

单次 first-class pilot 后的三 fresh repeats 均复现 priority=`[2,3,4,5,0,8,6,7,1]` 与
selected=`[2,3]`。floor elapsed=`[21.637124,21.604930,21.871310] s`，packed nodes=
`[[3,3],[3,3],[3,1]]`；repeat 2 clause 3 只提交 root，worst active lower 保留 floor
`-152.287033`。whole cooperative elapsed=`[67.213556,66.833706,60.228863] s`，final 仍为 sound
unknown、9/9 unresolved。

formal hash=`2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；
replay、九类同步重哈希 tamper、16 focused tests、NRIR-31/34/35 predecessor replay、全量
`890 passed, 37 skipped` 与静态门禁均通过。
“两条 selected clauses 三轮均至少提交一个 atomic pair”的 acceptance criterion 未成立，结论为
multi-clause allocation `VALIDATED-NO-GO`，`performance_claimed=false`；IR/control 可保留，没有
property closure、硬实时、GPU、competitor、multi-workload 或 ASPLOS-ready claim。下一门禁是 shared
parametric compiler/root/evaluator + stronger
bound/candidate：先分解两个 selected clause 的 compile/root/child phase，再冻结可复用合同与 tightness
gate，不继续只调 top-k 或 slice 常数。

## 44. Shared Parametric Objective Evaluator v1 判定

NRIR-37 保持 frozen NRIR-31 floor、NRIR-36 root-lower priority/top-2/dynamic equal-remaining slices、
NRIR-34 cap128 ancestral refinement/sibling atomic commit 与 31/depth4/60 秒不变。新增
`SharedParametricAncestral` Plan/Batch/Task/Schedule：template 只拥有 graph、input non-batch shape、
objective shape/dtype/device、ReLU layout、optimizer policy 与 provenance；objective content、split、
intermediate bounds、warm state、refinement lineage 和 batch size 都属于 exact instance。生产 batch
显式 `selected_native_reexecution=false`，root 或完整 sibling pair 才形成 commit。

真实 clause 2 first-class parity 中，frozen audit root+pair=`14.073795 s`、shared evaluator=
`1.198798 s`；lower、branch、split、α、β、refinement final-bound hashes exact，upper max diff=
`1.52587890625e-5`，满足 frozen `allclose(atol=1e-5,rtol=1e-5)`。单轮 top-2 pilot 的 floor/whole=
`20.291832/50.548707 s`，clauses 2/3 均完整提交 31 nodes，32 个 cache events 只有一次 miss。

正式三 fresh processes 的 floor elapsed=`[21.733539,21.941763,21.925033] s`，whole elapsed=
`[51.996191,52.251681,52.695640] s`；三轮 priority 都为
`[2,3,4,5,0,8,6,7,1]`、selected 都为 `[2,3]`、packed nodes 都为 `[31,31]`，每轮
cache miss count=1。clauses 2/3 depth-4 worst active lower 稳定为
`-37.574287/-35.900215`，verdict 仍 unknown。

pilot hash=`c96fff3fa2bc2563b4d46886d69b33f51ac985b19ad80d916309db57fe6cfefa`；formal hash=
`9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`。artifact replay、11 类
control/compiler 同步重哈希 tamper、Task/Batch commit binding tamper、27 focused tests、全量
`917 passed, 37 skipped`、mypy clean、Pylint `10.00/10` 均通过。

预注册 multi-clause coverage gate 成立，因此 NRIR-37 以 same-algorithm shared compiler ownership +
fixed-deadline coverage `VALIDATED-REDUCED` 关闭。这不把内部 audit→production timing 写成 speedup，
也没有 property closure、硬实时、GPU、competitor、multi-workload 或 ASPLOS-ready claim。下一门禁
先解释完整 depth-4 frontier 的剩余 gap，再只改变一个 stronger-bound/candidate 变量。

## 45. Full Frontier Tightness Attribution v1 判定

NRIR-38 新增一等 `FrontierTightnessAttribution` Plan/Task/Schedule。Plan 绑定 source execution/Plan/
queue、objective/threshold、exact active node/split、baseline/candidate policy 与预注册门禁；七阶段 Task/
Schedule 固定 source admission、frontier enumeration、source summary、baseline replay、candidate evaluate、
decision 与 emit。runtime 从 source 独立枚举 active frontier，并按 source commit 恢复八个完整 sibling
pair；steps5/steps15 分别使用独立单-template cache。

真实 clauses 2/3 均为 31 evaluations、16 active depth-4 nodes。baseline replay lower/upper max diff
全部为 0，split、parent、sibling grouping 与 refinement final-bound hashes exact。steps15 对两条 clause
均改善 16/16 nodes、无退化；median delta=`+0.107208/+0.132715`，但 worst-active lower 只由
`-37.574287→-37.518791`、`-35.900215→-35.871658`，改善 `+0.055496/+0.028557`，远低于冻结
`+1.0` gate。depth-4 alpha interior fraction 也只有 `2.164%/2.518%`，支持 optimizer 已近饱和的归因。

pilot hash=`2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`；replay、8 类同步
tamper、13 focused tests、全量 `930 passed, 37 skipped`、mypy/Pylint 均通过。本阶段以 fixed-frontier
optimizer-step tightness `VALIDATED-NO-GO` 关闭；不启动 steps15 full-queue formal，不形成 property、
performance、GPU、competitor、multi-workload 或 ASPLOS-ready claim。下一单变量固定为把已有
objective-bound-impact branch Plan/Task/Schedule 接入 shared ancestral evaluator，与 widest branch 做
exact fixed-tree 对照。

## 46. Objective Branch Shared Evaluator v1 判定

NRIR-39 保持 NRIR-37 shared template、steps5、cap128 ancestral refinement、parent warm state、best-first
queue、31/depth4 与 sibling atomic commit 不变，只把 branch candidate 从 widest 改为历史 NRIR-17
objective-bound-impact policy（8 candidates/ReLU、batch64、cap256）。新增 composite Plan、6-task
TaskModule/Schedule；每个有候选的 evaluation 都绑定 exact branch Plan/Task/Schedule/score trace 与 selected
candidate，queue decision/child split 逐项 fail closed。

真实 clauses 2/3 两侧 root exact，control/candidate 均为 31 evaluations、16 个 depth-4 active nodes。
worst-active lower 从 `-37.574287/-35.900215` 提升到 `-35.530926/-30.258448`，改善
`+2.043362/+5.641768`；median 改善 `+2.537640/+5.885233`，两条都通过预注册 `+1.0` gate。
pilot hash=`dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`。

本阶段以 fixed-budget branch selection `VALIDATED-REDUCED` 关闭，只证明 branch tightness 与 IR/runtime
ownership；logical fixed-budget clock 不承载墙钟结论，尚无 full-query/global deadline、property closure、
GPU、competitor、multi-workload 或 ASPLOS-ready claim。下一阶段为 objective-branch whole-query three-repeat
formal，必须同时报告 branch scoring 成本、committed coverage 与最终九子句 verdict。
