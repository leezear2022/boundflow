# BoundFlow 当前状态：PR-13 Closure 之后

> **2026-08-28 当前状态：S1 canonical CIBC pipeline资格关闭**。source=`56c494f`，artifact=
> `artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2`；六fresh pipeline/PyTorch geomean=
> `2.5028099854x`、worst=`2.4600205501x`，pipeline/direct=`1.0001854311x`。17-op、6/6 CIBC、
> 2 cuBLAS、fallback/eager/warm-DLPack=0，final correctness/replay/8类tamper通过。下一只允许S2
> coarse CROWN/custom VJP；same-solver、complete-query和总体10×仍未运行、未声明。

> **2026-08-26 当前执行：GC0-1 capture/analysis已预注册，待独立外审**：只冻结source snapshot、
> generic adapter、A0—A8 analysis与full causal witness协议。当前没有GC0-1代码或artifact；下一步只审计
> 预注册，批准后才允许实现。lowering/runtime/timing继续关闭。

> **2026-08-26 当前状态：GC0-0外审批准并关闭，只开放GC0-1预注册**：正式状态=
> `VALIDATED-GC0-0-GENERIC-VERIFICATION-GRAPH-SCHEMA`。外审确认schema、22 reason、三类fixture、
> canonical hash/tamper、非执行registry与1832+3回归全部成立。minor要求GC0-1区分schema-level
> shallow policy rejection与full analysis witness。当前不得实现GC0-1；下一只写预注册并外审。

> **历史（已由上方GC0-0外审关闭取代）：2026-08-26 GC0-0内部验证状态**：通用
> `Program/Region/Value/Op/Effect/VJP/Rule/LegalityResult/Module`、canonical identity与22类拒绝原因
> 分层已落地；三类fixture仅证明schema表达与round-trip。没有capture、analysis、lowering、arena、
> runtime、production执行或计时。下一唯一动作是GC0-0外审；批准后只开放GC0-1预注册，不能越级
> 实现。

> **历史（预注册外审已批准并关闭）：2026-08-26 GC-0/FCR-1 ABI+correctness预注册**：通用graph/effect/
> legality、Relax/TIR lowering identity、physical arena/prepared runtime、minimal-saved-state VJP、
> P empty-β/S active-β/multi-site 10/9、双oracle、five-fresh replay与22类fully re-signed tamper均已
> 冻结。当时尚无代码或raw，`implementation_open=false/timing_open=false/performance_claimed=false`；
> 外审随后只开放GC0-0 schema与direct negative legality tests。当前动作以上方GC0-0状态为准。

> **2026-08-26 当前状态：MR7-R通过，GC-0/FCR-1 correctness预注册开放**：10 fresh/5 pair
> 证明unprofiled ledger低扰动；boundary median=`20.333%/24.684 ms`、5/5过门禁，required region
> speedup=`1.91214x`。这只是opportunity admission，不是speedup。下一步先冻结verification graph ABI、
> guarded rules、Relax/TIR lowering、arena identity和correctness，timing继续关闭。

> **历史（已由MR7-R formal关闭）：2026-08-26 当前状态：MR7已完整执行但归因INVALID，MR7-R预注册**：一个profile/control
> perturbation=`1.239399 >1.10`否决全局资格；其余semantic/launch/host/device/tamper门禁通过。
> `25.891 ms/19.818%`host boundary与`8.692%`device kernel仅为诊断。下一步只跑5 pair unprofiled
> MR6 diagnostic vs MR7 ledger，未通过不得实现FCR-1。

> **2026-08-26 当前状态：MR6 guard dominance NO-GO，MR7 attribution已预注册**：9 fresh显示
> `360→60`同步guard仅回收`1.033126x`，diagnostic/provider仍需`1.107412x`才到parity。安全guard
> fusion不开放；下一步只读拆57 launch、540 DLPack view、layout/materialization与per-site kernel。

> **2026-08-26 当前状态：MR5 multi-site timing正式NO-GO，MR6 attribution开放**：6 pair完整
> outer host geomean=`0.834407x`，所有latency gate失败，candidate约慢19.84%；语义、launch、cache、
> memory和tamper闭合。complete-query继续关闭。下一只量化当前outer至少360次device→host同步guard的
> ceiling；未过预注册路由门禁不得实现安全replacement或扩到query。

> **历史（correctness保留，timing已由上方NO-GO关闭）：2026-08-26 当前状态：MR5 multi-Conv correctness通过，timing预注册激活**：C2→C1→C0
> 三site在真实outer exact call中累计`150/135` launch，5 pair semantics/optimizer/atomic/tamper/full均闭合；
> 当前仍无性能claim。下一步只跑6 pair warm-cache multi-site outer timing，不直接进complete query。

> **2026-08-26 当前状态：MR4 census通过，MR5 multi-site correctness预注册已激活**：三条真实Conv
> edge各50 rows，10/9、absent β、handoff exact；static total=`4.5P`，16/16 tamper，全量=
> `1764 passed,3 skipped`。尚未实现multi-site bridge，不能计时；下一步实现shape/stride-keyed
> generalized TIR与三site cumulative ownership/correctness协议。

> **历史（已由上方formal取代）：2026-08-26 当前执行：MR4 production Conv site census已预注册**：只读真实provider exact call，
> 不计时、不改TIR；确认三条Conv edge是否都有稳定10/9、absent β和足够静态MAC机会。通过也只开放
> multi-site correctness，不覆盖MR3 single-site NO-GO。

> **2026-08-26 当前状态：MR3 single-site production bridge timing NO-GO**：完整outer exact call
> 6 pair的host geomean/bootstrap lower/worst=`0.979727/0.939360/0.916094x`，candidate平均约慢2%；
> 语义、10/9 launch、module稳定、memory与host/event方向均通过。保留correctness，same-solver
> complete-query/multi-site关闭；全量=`1743 passed,3 skipped`。下一步不是继续传播，而是外审本
> closure或另行预注册新结构路线。

> **历史（已由上方NO-GO取代）：2026-08-26 当前执行：MR3 single-site production bridge timing已预注册**：6 pair/12 fresh
> `PB/BP/PB/BP/PB/BP`，headline为完整outer exact-call host wall；CUDA event仅诊断，formal观测、
> compile与dummy warm排除。尚无timing claim，complete-query/multi-site仍关闭。

> **2026-08-26 当前状态：MR3真实production bridge correctness通过**：P-anchor
> `/49: /input-24 → /input-20`已由TIR forward/custom backward接管；5 pair/10 fresh、50/45 launch、
> 10/9 trajectory、atomic rollback与18/18 tamper全过，最坏diff=`3.15905e-6`。没有timing claim；
> 下一只允许预注册single-site timing，S-anchor/multi-site/same-solver仍关闭。

> **2026-08-26 当前状态：MR3-0真实provider hook feasibility通过**：真实beta-split optimized call
> 的outer/inner=`1/10`，`/49`下P ReLU/Conv probe=`20/20`；完整state max diff=`2.02656e-6`、
> 12/12 tamper。下一只实现fail-closed candidate bridge；未开放timing。

> **2026-08-26 当前执行：MR3 P-anchor production bridge correctness已预注册**：只接
> `25/Conv_8`，5 pair/10 fresh逐步核对10 evaluation/9 backward-mutation与outer atomic commit；
> 不计时、不扩S-anchor/multi-site。真实hook已由MR3-0确认，下一实现fail-closed bridge。

> **2026-08-26 当前状态：MR2选出P-anchor，只开放bridge correctness预注册**：P `25/Conv_8`
> site/ABI/ownership/VJP/10×evaluation+9×mutation均proven，multi-site显式single-site bounded，唯一
> missing为production exact-call connection；S-anchor仍有四层缺口。bridge/timing均未实现。

> **2026-08-26 当前状态：MR1-S full-graph static eligibility NO-GO**：394条activation raw全部
> 审计，ResNet2B=`0/51 eligible`；51/51不是IBP整图调用，而是带split state的provider-owned
> activation-BaB/CROWN call。关闭CIBC整图直接替换，same-solver timing/R2不开；下一只做MR2
> production CROWN subgraph/owner contract inventory。

> **2026-08-26 当前状态：MR0 explicit-event budget NO-GO**：17对event的five-fresh
> geomean/bootstrap-upper/worst=`2.137191/2.153191/2.163574x`；12/12 tamper。MR1关闭，
> 下一步只做既有B3/RVIR raw的无计时same-solver static eligibility audit，不实现性能候选。

> **2026-08-26 当前执行：MR0 explicit-event budget预注册**：R3-3 profiler route STOP 后，先在
> CIBC 17-op graph上验证1/4/8/17对预分配CUDA event的扰动；正式只用17对的
> `geomean/bootstrap-upper/worst<=1.05/1.05/1.08x`决策。通过也只开放MR1 correctness。

> **2026-08-26 当前状态：R3-3只读microphysics attribution route=STOP**：5 fresh的profile
> 扰动=`2.4061–2.8053x`、calibration residual=`110–119 us`，故`0/5`准入；12/12 tamper，
> 全量=`1667 passed,3 skipped`。
> bridge/autograd share只是不具准入资格的诊断投影。停止当前fixed S-anchor physical分支，R3-4与
> same-solver继续关闭；后续不得放宽门槛或据此直接实现ABI/autograd优化。

> **2026-08-26 当前执行：R3-3只读microphysics attribution已预注册**：先做CUPTI/
> NVTX/correlation归因和Amdahl route decision，不改TIR/schedule。主口径要达1.05x需总回收
> `1.571209x`，单bucket至少占约`36.35%`才有物理可达性。

> **2026-08-26 当前状态：R3-3 isolated timing NO-GO，只读 attribution 开放**：
> TIR/PyTorch geomean/bootstrap/worst=`0.668275x/0.629157x/0.599089x`，候选约慢1.50x；
> active-β correctness保留。R3-4/same-solver关闭，下一只允许拆分FFI/kernel/autograd/allocation。

> **2026-08-26 当前执行：R3-3 isolated timing 协议已冻结**：6 fresh AB/BA、10 warmup/
> 30 pairs，baseline含dense α/β reconstruction+autograd，candidate含TIR custom wrapper。未生成
> formal raw 前没有active-β性能claim；R3-4/same-solver仍关闭。

> **2026-08-26 当前状态：R3-3 active-β correctness通过，只开放 isolated timing**：
> S-anchor 5 fresh的forward/compressed α/β VJP、ownership、workspace/cache全过，max diff=
> `8.64267e-7`，β=30/30 nonzero，12/12 tamper，全量=`1653 passed,3 skipped`。R3-4与
> same-solver关闭，尚无active-β性能数字。

> **2026-08-26 当前状态：D2-B local wrapper research gate通过，只开放R3-3 correctness**：
> candidate/native geomean/worst=`1.752001x/1.724843x`，region worst=`53.9195x`，12/12 tamper。
> multi-site与same-solver关闭。

> **2026-08-26 当前状态：D2-B correctness通过，只开放timing**：5 pair/10 fresh逐步lower、dα、α、
> Adam moment最大差均`0.0`，ownership/12 tamper/全量回归通过。尚无speedup；R3-3与same-solver关闭。

> **2026-08-25 当前状态：D2-A正式关闭，只开放D2-B correctness**：coefficient-sign 5 fresh
> minimum share=`0.870614`、worst research required=`11.8762x≤15.50x`；residual6/residual11
> dominant signature稳定，14/14 tamper通过。没有性能claim；D2-B timing、R3-3与same-solver关闭。

> **2026-08-25 当前状态：D1-C正式NO-GO，只开放D2-A**：wrapper geomean/worst=
> `0.249369x/0.243233x`，B3 recovery=`1.879305x/1.855758x`，语义/memory/12 tamper通过。
> forward热点已移除但backward成为新主导；R3-3与same-solver关闭。

> **2026-08-25 当前状态：D1-B isolated通过，只开放D1-C**：256-thread winner 5 fresh
> geomean/worst=`58.0619x/56.8625x`，max diff=`9.53674e-7`、10/10 tamper；完整10/9 wrapper
> 尚未运行，不能claim query/queue/ASPLOS speedup。

> **2026-08-25 当前状态：D1-A两个热点正确性均关闭，只开放D1-B**：residual6 source=`52fc62c`，
> 5 fresh/`122,940`元素最大diff=`1.91618e-6`、sign exact、10/10 tamper；未计时且无性能claim。
> D1-C、R3-3与same-solver继续关闭。

> **2026-08-25 当前状态：D1-A residual11通过，residual6 correctness开放**：5 fresh/10 tamper通过，
> max diff=`8.04557e-7`，未计时。D1-B/C、R3-3与same-solver关闭。

> **2026-08-25 当前状态：R3-D0正式关闭，R3-D1-A开放**：5 fresh formal全部通过calibration/sanity，
> Graph route关闭，compiled-region worst required=`9.3180x ≤ 10x`，12/12 tamper拒绝。下一只实现
> residual11 staged factorization correctness；没有performance claim，D1-B/C与R3-3关闭。

> **2026-08-25 当前状态：R3-2B正式NO-GO，当前variant停止**：source=`f43eb76`，5对×30
> wrapper样本geomean/worst=`0.133989x/0.130371x`，约慢`7.46x`；语义保持且memory降至
> `0.0584567x/0.153846x`。R3-3/multi-site/same-solver关闭；下一只允许R3-D0只读microphysics
> attribution预注册。见 `BOUNDFLOW_R3_2B_WRAPPER_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。

> **历史：2026-08-25 当前状态：R3-2A通过，只开放R3-2B**：source=`e7ae590`的P-anchor 5-pair
> 10/9 optimizer trajectory逐步语义、ownership和memory全过；最大lower/dα/α差=
> `8.58307e-6/8.28877e-8/2.38419e-7`，worst allocated/reserved=`0.0586911x/0.166667x`，
> 12/12 tamper，全量=`1602 passed,3 skipped`。当前只开放同轨迹wrapper-inclusive local timing；
> 尚无speedup/query claim。见
> `BOUNDFLOW_R3_2A_OPTIMIZER_TRAJECTORY_FORMAL_CLOSURE_2026_08_25.md`。

> **历史：2026-08-25 当前状态：R3-1b3通过，R3-1已admit**：source=`eeeb1bf`的5对/10 fresh
> correctness/memory全部通过；最坏allocated/reserved=`0.06417x/0.16667x`，lower/dα max diff=
> `4.05312e-6/6.14673e-8`，9/9 tamper。当前只开放R3-2A optimizer trajectory correctness；
> timing仍关闭。见`BOUNDFLOW_R3_1B3_FIVE_FRESH_FORMAL_CLOSURE_2026_08_25.md`。
>
> **2026-08-25 当前执行：R3-1b3协议已冻结待formal**：10 fresh subprocess，顺序=
> `NC/CN/NC/CN/NC`；candidate/native absolute peak allocated与reserved必须逐对`<=1.0x`。
> 协议/worker/replay/tamper/synthetic tests已实现，下一只提交clean source并运行；不计时。见
> `BOUNDFLOW_R3_1B3_FIVE_FRESH_CORRECTNESS_MEMORY_PLAN_2026_08_25.md`。
>
> **2026-08-25 当前状态：R3-1b2关闭，b3开放**：source=`12402da`的compiled custom VJP
> artifact/replay通过；lower/dα max diff=`3.81470e-6/6.14673e-8`、sign exact；2 scratch、
> saved dense A=0、warm allocation=0，12/12 tamper。下一只做five-fresh correctness与physical
> allocated/reserved memory；R3-1未admit且不计时。见
> `BOUNDFLOW_R3_1B2_COMPILED_P_ALPHA_VJP_FORMAL_CLOSURE_2026_08_25.md`。
>
> **2026-08-25 当前状态：R3-1b2实现待clean-source formal**：compiled custom VJP单worker
> lower/dα max diff=`3.93391e-6/6.14673e-8`、sign exact；2 scratch、saved dense A=0、warm
> allocation=0。下一只生成raw-first artifact/replay/tamper；b2尚未关闭，b3/timing关闭。见
> `BOUNDFLOW_R3_1B2_COMPILED_P_ALPHA_VJP_IMPLEMENTATION_2026_08_25.md`。
>
> **2026-08-25 当前状态：R3-1b2数学门禁通过，TIR实现开放**：P-alpha closed-form VJP对
> native autograd max diff=`4.47035e-8`、sign exact、nonzero=`281/281`；无需跨forward/backward
> 保存dense A。当前只实现checkpoint/sign TIR与mandatory custom backward；R3-1仍未admit，
> five-fresh/timing关闭。见`BOUNDFLOW_R3_1B2_P_ALPHA_VJP_MATH_REDUCTION_2026_08_25.md`。
>
> **2026-08-25 当前状态：R3-1b1关闭，b2开放**：fresh-process full-lower artifact/replay通过；
> lower max diff=`3.8147e-6`、15 launches、2 scratch×73,728 B、70/70 DLPack、warm allocation=0，
> 10/10全重签tamper拒绝。当前只开放compiled P-alpha VJP；R3-1仍未admit、不计时。见
> `BOUNDFLOW_R3_1B1_COMPILED_FULL_LOWER_FORMAL_CLOSURE_2026_08_25.md`。
>
> **2026-08-25 R3-1b0关闭/b1开放（已由上方b1 closure取代）**：exact reverse trace正式=
> `VALIDATED-R3-1B0-TRACE-LIVENESS`；12 steps、2 residual、2 scratch，每slot 73,728 B，6/6 tamper。
> 下一只实现compiled no-grad full-lower forward；b2/b3/timing关闭，当前仍无physical memory或性能
> claim。见`BOUNDFLOW_R3_1B0_TRACE_LIVENESS_FORMAL_CLOSURE_2026_08_25.md`。

> **2026-08-25 当前状态：R3-1 M0 Python rematerialization NO-GO**：5对fresh的lower/dα语义
> 全过，最大差=`4.7684e-7/2.3283e-10`；但peak allocated=`1.1181179x`、compiled region=0/5，
> 因此R3-1未admit、R3-2A关闭。下一只允许预注册R3-1b bounded-arena compiled recurrence；当前
> 无timing/performance claim。见
> `BOUNDFLOW_R3_1_M0_PYTHON_REMATERIALIZATION_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。
> R3-1b“已预注册但未实现”是历史状态，已由上方b0/b1 closure取代。见
> `BOUNDFLOW_R3_1B_BOUNDED_ARENA_COMPILED_RECURRENCE_PLAN_2026_08_25.md`。

> **2026-08-25 当前状态：R3-0 compressed-alpha v2关闭，R3-1重新开放**：source=`8941e66`，
> production alpha=`[2,1,6,86]`，saved logical/unique=`207888/109584 B`，replay逐字节一致、
> 12/12全重签tamper拒绝。当前只实现`25/Conv_8`单evaluation mandatory custom backward
> correctness；mutation=0，不计时，R3-2A/2B关闭。见
> `BOUNDFLOW_R3_0_COMPRESSED_ALPHA_V2_FORMAL_CLOSURE_2026_08_25.md`。

> **2026-08-25 R3-0 fixture纠正状态**：v1 validators仍成立，但alpha fixture不是production
> compressed shape；已改为`[2,1,6,86]`并重算saved bytes。当前等待v2 clean-source formal，
> R3-1暂时重关，无performance claim。该待v2状态已由上方正式closure取代。见
> `BOUNDFLOW_R3_0_COMPRESSED_ALPHA_FIX_CHANGELOG_2026_08_25.md`。

> **2026-08-25 当前状态：R3-0关闭，R3-1开放**：contract replay通过，12/12全重签tamper拒绝；
> 8 nodes/8 edges、2 scratch，saved coefficient/dense escape/context=`0/0/0`。当前=
> `VALIDATED-R3-0-CONTRACT`，下一只实现`25/Conv_8` mandatory custom backward correctness；不计时，
> R3-2A/2B关闭；全量=`1568 passed, 3 skipped`。见
> `BOUNDFLOW_R3_0_STRUCTURED_OWNER_FORMAL_CLOSURE_2026_08_25.md`。

> **2026-08-25 R3-0实现待formal（历史，已由上方formal closure取代）**：typed region DAG、closure/liveness、BiasSplit ownership、
> dense-escape/context/saved-state validators和artifact/replay/tamper runner已实现，40 tests通过。
> 当前=`IMPLEMENTED-R3-0-PENDING-CLEAN-SOURCE-FORMAL`；下一唯一动作是从clean commit生成并重放
> contract artifact。R3-1、production和timing保持关闭。见
> `BOUNDFLOW_R3_0_STRUCTURED_OWNER_CONTRACT_IMPLEMENTATION_CHANGELOG_2026_08_25.md`。

> **2026-08-25 当前状态：R1-A正式NO-GO，转R3-0**：6组clean-source Nsight formal已完成。
> 每组均重建42 graph nodes/4400 owner events且unowned/temporal=`0/0`，但六组profile扰动=
> `1.1838—1.1859x`、`0/6`满足冻结`[0.95,1.05]`；clock仅`3/6`通过。因此不得形成op-type或
> same-solver share，R1-B/R1-C/R1-D/R2关闭。当前唯一工程动作是R3-0合同和静态验证器；不接
> production、不计时，R3-1 mandatory custom backward仍关闭。见
> `BOUNDFLOW_CIBC_R1_A_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。

> **2026-08-25 R0完成/R1冻结状态（历史，已由上方R1-A NO-GO取代）**：R0代码与文档卫生已完成；R1 scope/clock/query-local协议已
> 预注册但未实现/运行。当前唯一工程动作是实现R1-0 calibration/topology/schema与negative tests，
> 然后在clean source上先做runner smoke。独立CIBC graph `2.45631x`不得用作真实query的
> `G_query,k`；same-solver必须按op type记录`q_B3,k`并用exact production signature现场重测
> `G_query,k`。R1-D关闭前R2/R3-0继续关闭。见
> `BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`。

> **2026-08-25 当前执行状态修订（该冻结动作已完成，由上方R0完成状态取代）**：下一工程动作不是直接改TIR，也不是启动R3实现；先完成R0的
> 3条新增mypy `arg-type`、1条新增pylint `C0415`与计时披露，再冻结R1三层目标、CUPTI↔host/NVTX
> 校准和raw schema，随后执行CIBC-G1只读归因。归因后还必须测same-solver eligible-IBP query
> share并冻结可solve workload/held-out family；数学可达才开放R2，之后跑B0/B3/cumulative
> candidate三方formal。R3设计评审可并行；R3-1是冻结optimizer mutation但mandatory backward，
> R3-2拆为2A correctness/2B timing，R3-0代码仍关闭。见
> `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`与
> `BOUNDFLOW_RECOVERY_PLAN_TARGET_SCOPE_R3_STAGE_CORRECTION_CHANGELOG_2026_08_24.md`。

> **2026-08-24 R3 structured-owner/custom-VJP设计状态**：已完成独立重设计预注册，状态=
> `PREREGISTERED-DESIGN-REVIEW-ONLY-R3-SO-CVJP`。方案使用closed lower region的DAG owner、一个
> custom VJP、M0 rematerialization和最多两个scratch；dense A不得进入Function output、saved tensor、
> ctx/executor或跨层buffer。当前只进入外部设计评审，未开放实现/性能，CIBC-G1仍是当前工程next。
> 见`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`与
> `BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`。

> **2026-08-24 CIBC外审与当前下一步**：Round 1独立外审`APPROVE`，exchange已由executor
> 关闭为`closed/approved`，最终=
> `EXTERNALLY-APPROVED-VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`。外审独立重算全部
> headline、float64 oracle、replay和13类tamper；0 blocker/major，2 minor+4 info。当前唯一研究
> 动作是先完成R0静态检查/口径卫生，再预注册CIBC-G1 optimized-graph attribution；不得把
> 该IBP结果外推到auto_LiRPA/alpha-CROWN/BaB/query，也不得把未运行的B5—B7写成失败。详见
> `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`。

> **2026-08-24 CIBC-IBP水平融合正式状态**：source=`a52b177`，3个operator schedule worker与
> 6个whole-model fresh worker正式通过。选中128 threads；6 Conv geomean/worst=
> `12.795/9.142x`，完整ResNet2B IBP graph geomean/worst=`2.4563/2.4509x`，输入copy计入，
> max diff=`2.4414e-4`、sign exact，10/10 tamper rejected，全量=`1492 passed, 3 skipped`。
> 当前=`VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`；production default仍不变，auto_LiRPA/
> solver/query/memory/跨模型claim仍关闭。

> **2026-08-23 FSG4/B4-B2 B2-4内部关闭**：compressed alpha=`[6,86]`、empty beta absent；
> P0 five raw与12 candidate共68 metrics/217,770元素通过，max diff=`2.384185791015625e-06`。
> ledger已冻结但未计时/选winner。下一步只开放B2-4外审；B2-5/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-3外审关闭**：`APPROVE`，0 blocker/major/minor；独立
> float64与现场GPU均通过。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS`。当前只开放
> B2-4 P-anchor sparse-source schedule，timing/B2-5/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-3内部关闭**：P-anchor Conv dense correctness 5/5 raw、
> 20/20 metrics、92,190元素全过，max diff=`2.384185791015625e-06`、sign exact；beta gradient
> absent，workspace结构门禁exact。当前只开放B2-3外审；timing/B2-4/B2-5/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-2外审关闭**：`APPROVE`，0 blocker/major/minor；独立
> float64重算、GPU runner、scheduled TIR workspace与全量测试均通过。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS`。下一步只开放
> B2-3 P-anchor Conv dense correctness，timing/B2-4/B2-5/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-2内部关闭**：compressed alpha/beta已直接进入
> S-anchor TIR，compressed gradient projection对native oracle通过；5/20/31,590，max diff=
> `8.642673492431641e-07`，workspace forbidden count=`0`；targeted/related/full=
> `34/88/1448 passed`，3 skipped。当前=
> `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；下一步仅外审。
> 该待审状态已由上方外审关闭状态取代。

> **2026-08-23 FSG4/B4-B2 B2-1外审关闭**：`APPROVE`，0 blocker/0 major；独立
> float64重算36,750元素max diff=`6.988e-07`，GPU现场复跑与三hash逐位一致；
> targeted/related/full=`23/77/1437 passed`，3 skipped。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS`；下一步只开放
> B2-2 S-anchor sparse-source fused forward/backward，其他后续阶段仍关闭。
> 该“下一步B2-2”状态已由上方B2-2内部关闭状态取代。

> **2026-08-23 FSG4/B4-B2 B2-1内部关闭**：5份S raw、20 metrics/36,750元素全部通过，
> max diff=`8.642673492431641e-07`、sign exact；targeted 23、related 77（外审更正）、full=
> `1437 passed, 3 skipped`。当前=
> `VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；下一步仅外审。
> 该待审状态已由上方外审关闭状态取代。

> **2026-08-23 FSG4/B4-B2 B2-0外审关闭**：verdict=`APPROVE`，auditor现场GPU复跑三项
> receipt hash逐位一致；最终=`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`。
> 下一唯一动作=B2-1 S-anchor dense correctness；timing、B2-2/P-anchor/B4-B3继续关闭。

> **2026-08-23 FSG4/B4-B2 B2-0内部关闭**：first-class lowering/receipt与identity CUDA/TIR
> forward/backward已在RTX 4060/sm_89通过；cold miss→warm hit、DLPack与current stream exact、
> launch 1/1、fallback/eager backward 0/0、full=`1426 passed, 3 skipped`。当前状态=
> `VALIDATED-B4-B2-B2-0-ABI-PROBE`。下一唯一动作=B2-1 S-anchor dense correctness；
> region融合、timing、B2-2/P-anchor/B4-B3仍关闭。

> **2026-08-23 FSG4/B4-B2预注册**：dense semantic ABI与sparse-source fused ABI、first-class
> compiler/schedule/module/launch IR、custom-autograd与6-worker物理门禁已冻结；状态=
> `PREREGISTERED-B4-B2-TYPED-CUDA-TIR-NOT-IMPLEMENTED`。下一唯一动作=B2-0 identity-TIR ABI
> probe。尚无TIR实现或性能claim。
> 该预注册状态已由上方B2-0内部关闭取代。

> **2026-08-23 FSG4/B4-B1外审关闭**：exchange=`closed/approved`，Round 2 F1/F2 CLOSED，
> AC1—AC6全PASS且findings=0。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`。Executor RTX 4060 full=
> `1414 passed, 3 skipped`；auditor CPU-only full=`1366 passed, 51 skipped`，集合边界一致。
> 下一步只开放另行预注册B4-B2 typed CUDA/TIR candidate；该动作已由上方预注册完成状态取代。

> **2026-08-23 FSG4/B4-B1 Round 1外审纠正**：正式verdict=`request_changes`，F1/F2均为
> major。F1为receipt metric/gradient target inventory未精确绑定；F2为deterministic warn/debug
> mode未原样恢复。两项修复已在工作树通过targeted=`31 passed`、related=`127 passed, 12 skipped`、
> full=`1365 passed, 51 skipped, 7 warnings`。clean source=`e711e99`；v3已完成10 captures/
> 60 metrics/196,380 elements、max diff=`6.109476089477539e-07`、sign exact、2/2完整性负例，
> targeted=`32 passed`、related=`140 passed`、RTX 4060 full=`1414 passed, 3 skipped, 6 warnings`。
> 下一步是重交Round 2；B4-B2/TIR/performance/memory/
> ASPLOS-ready仍关闭。该待审状态已由上方Round 2批准取代。

> **2026-08-18 FSG4/B4-B1内部关闭**：source artifact=`d9164b8`，deterministic v2完成
> 5 fresh/10 captures的typed IR/instance重建与pure-PyTorch forward/VJP；60 metrics/196,380
> elements、max diff=`6.109476089477539e-07`、allclose/sign exact。2/2 incoming-bias/
> output-adjoint协调all-run全链重签由数值reference拒绝；related=`131 passed`、full=
> `1405 passed, 3 skipped, 6 warnings`，Black/Mypy/Pylint通过。首次full的v1单失败已归因于未冻结
> PyTorch线程策略；v2冻结并恢复threads/determinism/precision/MKLDNN，跨1/4/8线程入口records
> 一致。当前=`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`；下一步仅外审，
> B4-B2/TIR/performance/memory/ASPLOS-ready继续关闭。

> **2026-08-18 FSG4/B4-B1a five-fresh内部关闭**：source=`4a17423`，formal artifact完成
> 5 fresh/10 captures、90 amendment tensors/63,645 elements，max diff=0、sign exact；root replay、
> 8/8完整性负例、related 30与full 1382/3 skip/6 warnings通过。状态=
> `VALIDATED-B4-B1A-FIVE-FRESH-CAPTURE-SUFFICIENCY`；下一步开放typed IR/pure-PyTorch
> reference，协调动态改写须由其代数重算关闭；B4-B2/TIR/performance继续关闭。

> **2026-08-18 FSG4/B4-B1a five-fresh runner候选**：独立worker/runner/root replay与8类
> 完整性probe已实现；5-process pilot比较90 tensors/63,645 elements，max diff=0、sign exact，
> related=`28 passed`。状态=`IMPLEMENTED-B4-B1A-FIVE-FRESH-RUNNER-PENDING-FORMAL`；
> coordinated动态bias/adjoint改写留待numerical reference语义拒绝，typed IR/reference/TIR未开放。

> **2026-08-18 FSG4/B4-B1a capture contract候选**：显式opt-in observer已捕获incoming/
> operator bias与region output adjoints；新payload在B4-B0 base上绑定全部sparse layout raw并可
> root重建。real CUDA双锚点通过，related=`26 passed`、full=`1378 passed, 3 skipped, 6 warnings`。
> 状态=`IMPLEMENTED-B4-B1A-CAPTURE-CONTRACT-PENDING-FIVE-FRESH`；typed IR/reference/TIR未实现。

> **2026-08-18 FSG4/B4-B1预注册**：下一阶段仅为typed pure-PyTorch reference。B4-B0 raw
> 可将两锚点output A重建到约`3e-8`，但缺incoming bias/operator bias时output bias差约
> `0.55/1.11`，且whole-objective `loss_seed`不能替代region output adjoints。因此B4-B1a先扩展
> read-only capture，再建typed IR/reference；禁止target倒推。B4-B2/TIR/performance继续关闭。

> **2026-08-18 FSG4/B4-B0 Round 2外审关闭**：独立外审=`approve`，0 blocker/major/minor/info，
> F1关闭；审计方自行构造all-run topology/lineage全链重签，两案均被root replay拒绝；raw、
> 绝对身份、v1/v2 replay、11/11完整性负例、定向24与全量1376/3 skip/6 warnings均独立复核。
> exchange已关闭，状态=`VALIDATED-B4-B0-EXTERNALLY-APPROVED`。下一步只开放另行预注册的
> B4-B1 typed pure-PyTorch reference；B4-B2/TIR/performance/memory/ASPLOS-ready继续关闭。

> **2026-08-18 FSG4/B4-B0 v2内部关闭**：source=`422a3ee`，绝对身份绑定的v2 artifact
> 已完成5 fresh/10 captures，108 tensors/664,744 elements，max diff=`1.1920928955078125e-07`、
> sign exact；Round 1两类coordinated rewrite进入正式门禁，完整性负例=`11/11 rejected`。
> 定向=`24 passed`，全量=`1376 passed, 3 skipped, 6 warnings`。
> 当前=`VALIDATED-B4-B0-V2-PENDING-ROUND2-EXTERNAL-AUDIT`；B4-B1/B4-B2/TIR/performance
> 继续关闭。

> **2026-08-18 FSG4/B4-B0外审Round 1与F1修复**：外审=`changes_requested`，1 major：
> 全5 run/10 capture同步改写topology或lineage source hashes并全重签可绕过原相对一致性。
> v2修复已冻结source/model/state/primal/split/topology/schedule及逐锚点anchor/lineage绝对身份，
> manifest↔protocol同源；合法v1 replay保持，两类coordinated回归与总计`11/11`完整性负例拒绝。
> 当前=`IMPLEMENTED-B4-B0-R1-F1-IDENTITY-BINDING-PENDING-V2`；须生成v2并Round 2批准，
> B4-B1/B4-B2/TIR/performance继续关闭。

> **2026-08-18 FSG4/B4-B0 five-fresh内部关闭**：source=`1dbb2de`，5个独立CUDA
> subprocess生成S/P各5份capture；root replay从raw重建10份typed capture，比较108组tensor/
> 664,744元素，max diff=`1.1920928955078125e-07`、sign exact；九类outer-resigned tamper
> `9/9 rejected`。定向=`20 passed`，full=`1372 passed, 3 skipped, 6 warnings`。状态=
> `VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`；只关闭capture correctness/ownership，
> 外审前B4-B1/TIR仍关闭，无performance claim。

> **2026-08-18 FSG4/B4-B0 five-fresh runner候选**：状态=
> `IMPLEMENTED-B4-B0-FIVE-FRESH-RUNNER-PENDING-FORMAL-RUN`。typed capture已补齐α-index/lookup、
> β-location/sign、round-trip、CUDA default-stream与alias ownership；新增5-fresh raw-first
> worker/runner、root typed replay和9类outer-resigned tamper。单fresh real CUDA与synthetic 5-run
> summary通过，但formal artifact尚未生成，因此B4-B1/TIR仍关闭且无performance claim。

> **2026-08-18 FSG4/B4-B0 live observer候选**：状态=
> `IMPLEMENTED-B4-B0-LIVE-OBSERVER-PENDING-FIVE-FRESH`。显式opt-in observer只在optimizer
> evaluation 0对`31/Gemm_14`和`25/Conv_8`实施诊断性materialization，默认B3/B4-A
> 路径不变。CPU production-state与real CUDA smoke均通过，确认Gemm incoming-A不可微但
> active-beta gradient存在，Conv incoming-A gradient存在但empty-beta无pre-add/β gradient；weight=
> `(100,1024)`/`(16,16,3,3)`。related=`53 passed`，full=
> `1369 passed, 3 skipped, 6 warnings`，Mypy clean，Pylint 10.00/10。
> 尚无five-fresh/replay/tamper或TIR/performance claim。

> **2026-08-18 FSG4/B4-B0 typed capture contract**：状态=
> `IMPLEMENTED-B4-B0-CAPTURE-CONTRACT-PENDING-LIVE-HOOK`。新schema分离production compressed
> α/β映射源、native dense α/β/`relu_pre_add_coeff_l`输入及native gradients；冻结双锚点
> 和evaluation-0/CUDA/hash/Conv attrs/provider-fallback门禁。新测试10 passed，fixed related 46
> passed，full=`1366 passed, 3 skipped`，Mypy clean，Pylint 10.00/10。尚未接入live solver，
> 无correctness/performance claim，TIR仍关闭。

> **2026-08-18 FSG4/B4-B v1预注册**：状态=`PREREGISTERED-B4-B-V1-NOT-IMPLEMENTED`。冻结
> `node31/Gemm_14` active-beta语义锚点与`node25/Conv_8`候选性能锚点；先在optimizer
> evaluation 0做read-only exact-call双锚点capture，再允许typed reference和独立
> CUDA/TIR forward+backward。旧PR-12 plain-CROWN capability不放宽；单shape speedup不得外推。
> 下一唯一工程动作是B4-B0 capture，TIR实现仍关闭。

> **2026-08-18 FSG4/B4-A外审关闭**：Round 1从formal raw独立复核AC1—AC7全部PASS，
> exchange=`closed/approved`，0 blocker / 0 major / 1 minor / 1 info。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`：core=`1.018995x < 1.03x`，
> query worst=`0.996947x >= 0.98x`。B4-A只保留correctness/mechanism evidence，约1.9%不得
> 计入B4 cumulative performance baseline。下一唯一动作是单独预注册B4-B differentiable
> CUDA/TIR；B4-C/D与B5—B7仍关闭。

> **2026-08-18 FSG4/B4-A正式计时内部关闭**：source=`46a8493`的v5完成24/24 fresh worker，6/6
> semantic pair、19 tensor/pair、activation/environment/profile全部PASS；root replay与14/14 outer-
> resigned tamper通过。core wall geomean=`1.018995x < 1.03x`，query worst=`0.996947x >= 0.98x`，
> memory ratio=`1.0`，故内部状态=`VALIDATED-NO-GO-B4-A-PERFORMANCE-PENDING-EXTERNAL-AUDIT`。
> fixed related=`73 passed`、full=`1356 passed, 3 skipped`。该待外审状态已由上方Round 1批准
> 取代。

> **2026-08-18 FSG4/B4-A正式计时v4环境投影失败与修复**：source=`03043a3`的v4有19/19 worker
> admitted；run 19 raw返回后被旧门禁拒绝。其thermal/power累计值已有`54579 µs`历史偏移，但worker
> 区间增量严格同为`2062477 µs`，故根因是旧代码比较累计绝对值而非区间增量。门禁已改为delta exact，
> formal replay从raw重算投影，tamper扩为14类。v4不形成ratio；下一步验证并以clean source从0生成
> v5，无性能claim，B4-B/TIR关闭。

> **2026-08-18 FSG4/B4-A正式计时v3环境拒绝与功耗策略绑定**：source=`be2fa96`的v3完成20个
> worker，worker 20 correctness/activation/profile计数完整，但执行期software thermal counter独立增长，
> environment=`admitted=false`，v3不形成ratio。根因边界是active `nvidia-powerd`/Dynamic Boost未被
> 原协议约束；runner现冻结service=`inactive`、`enforced.power.limit=55.0 W`并逐worker/replay验证，
> tamper扩为13类。该“下一步v4”指令已被上方v4失败与v5指令取代。

> **2026-08-18 FSG4/B4-A正式计时v2环境拒绝与preflight加固**：source=`ee73bc2`的v2越过原计数
> 失败点并完成5个worker，但worker 5结束时检测到独立software thermal slowdown，environment
> `admitted=false`并fail closed。v2不形成ratio。runner现要求每个worker前GPU `<=45°C`且software
> thermal=`Not Active`，不再接受与power counter暂时耦合的active信号；下一步clean source v3从0重跑。

> **2026-08-18 FSG4/B4-A正式计时v1失败与计数覆盖修复**：source=`292a035`的v1在worker 3
> `B4-A-profile` fail closed；根因是显式计数器未patch B4-A模块持有的terminal optimizer函数引用，导致
> forward少记1、optimizer四项记0。已扩展B4-A alias观测，独立live diagnostic恢复forward=4、bound
> evaluation=10、optimizer trace/evaluation/update=`1/10/9`，handoff/rerun=`1/0`。v1保持不完整且不
> 参与结论；下一步clean source后从position 0生成v2。`performance_claimed=false`，B4-B/TIR关闭。

> **2026-08-18 FSG4/B4-A正式计时Runner状态**：24-process B3/B4-A control/profile runner、raw-first/
> resume、root replay及14类outer-resigned tamper probe已实现，固定related=`70 passed`，Black/Mypy/
> Pylint及全量`1353 passed, 3 skipped`通过。状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`。
> 下一唯一动作是提交clean source并
> 运行正式GPU artifact；无性能claim，B4-B/TIR关闭。

> **2026-08-16 FSG4/B4-A five-fresh状态**：10/10 fresh、5/5 direct pair与19 tensor/pair全部通过，
> 最大差=`6.109476e-06`，root replay PASS。状态=
> `INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`。下一唯一动作是独立正式B3/B4-A计时；无性能
> claim，B4-B/TIR关闭。

> **2026-08-16 FSG4/B4-A实现候选状态**：typed producer/handoff/no-rerun assembly、same-solver opt-in、
> post-query raw content audit与5-pair runner已实现，状态=
> `IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。下一唯一动作是clean-source five-fresh；当前无
> B4-A performance claim，B4-B/TIR与B5—B7关闭。

> **2026-08-16 FSG4/B4-A预注册状态**：第10次optimizer evaluation→terminal lower/六层lA typed
> handoff与no-rerun export合同已冻结，状态=`PREREGISTERED-B4-A-NOT-IMPLEMENTED`。下一唯一动作是实现
> typed lineage、producer与assembly并先过单次/five-fresh correctness；性能门禁尚未开放，B4-B/TIR、
> B4-C/D与B5—B7关闭。

> **2026-08-16 FSG4/B4-0外审关闭状态**：Round 1外审从raw独立复算AC1—AC7全PASS，无blocker/
> major；exchange=`closed/approved`。最终状态=`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。
> 下一唯一工程动作是B4-A预注册与terminal lower/lA handoff；production shape从correlation parent
> operator恢复并绑定lineage。B4-A correctness/performance尚未成立，B4-B不得混入，B4-C/D与B5—B7关闭。

> **2026-08-16 FSG4/B4-0内部关闭状态**：source=`66154e4`正式fresh control/profile artifact含
> 270609 raw events、35367/35367 CUDA kernel closure、14-call/4-forward exact marker；semantic
> max diff=`4.76837158203125e-07`、discrete/sign exact，root replay与9/9 outer-resigned tamper通过。
> CROWN14按冻结B3 share覆盖约67.72% core，B4-A满足消除完整重复terminal export CROWN call。状态=
> `INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`，无性能claim；下一步外审，批准后
> 只启动B4-A，B4-B不得混跑，B4-C/D与B5—B7关闭。

> **2026-08-16 FSG4/B4-0 Runner候选状态**：typed raw profiler schema、control/profile worker、
> 14-call/4-forward marker、CUDA annotation/kernel区分、correlation/temporal归因、确定性gzip与
> operator/kernel/materialization replay、B3冻结semantic comparator与9类outer-resigned tamper已实现；
> targeted=`15 passed`、B3/B4相关=`54 passed`、full=`1329 passed, 3 skipped`，静态门禁通过。状态=
> `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`；无B4 performance claim。下一唯一动作是从clean
> source生成fresh B4-0 artifact并关闭opportunity门禁，B4-A/B/C/D与B5—B7仍未开放。

> **2026-08-16 FSG4/B4预注册状态**：B4 cumulative CUDA/TIR与跨阶段融合已冻结为
> `PREREGISTERED-NOT-IMPLEMENTED`。路线覆盖14次production lower-only CROWN，而不是只优化占query
> 7.933%的optimizer。下一唯一动作是B4-0 read-only kernel/materialization attribution；B4-A/B/C/D、
> B5—B7、B0 parity与最终system gate均未关闭。

> **2026-08-15 FSG4/B3外审关闭状态**：Round 2审计从raw独立重算44项检查，AC1—AC7全部PASS，
> 无blocker/major/minor；exchange=`closed/approved`。B3正式状态=
> `EXTERNALLY-APPROVED-VALIDATED-REDUCED-B3`。只开放以B3为累计基线的B4 fusion candidate；B5—B7、
> B0 parity、complete-query/TTV与最终system gate仍未关闭。

> **2026-08-14 FSG4/B3正式计时内部关闭状态**：source `36e9069`的六全排列36-process artifact已
> 36/36完成，correctness/environment/measurement/activation、root replay与10/10 tamper全过。
> B2/B3 core/query=`1.071617x/1.006623x`，B0/B3 query=`0.910001x`，故状态恰为
> `VALIDATED-REDUCED-B3`，仍未回到原始B0 parity。frozen=`6 passed`、targeted=`114 passed`、full=
> `1314 passed, 3 skipped`。该“external audit待完成”历史状态已由上方Round 2批准取代。

> **2026-08-14 FSG4/B3正式计时Runner候选状态**：B0/B2/B3六全排列、control/profile共36个独立
> worker、direct B3 activation receipts、raw-first/resume、root replay与十类tamper probe已经实现；
> targeted=`108 passed`、full=`1308 passed, 3 skipped`、静态检查全过。状态=
> `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`。正式artifact尚未运行，无timing/speedup claim；下一
> 唯一动作是冻结clean source后从position 0执行36进程；该历史动作现已由上方正式结果取代。

> **2026-08-14 FSG4/B3 Five-Fresh关闭状态**：source `75dfd81`生成5组、10个独立fresh GPU
> worker；固定交替顺序、5/5 direct semantics、全部environment/provider/fallback/counter/audit、root
> replay和7/7 tamper均通过。定向=`56 passed`，全量=`1289 passed, 3 skipped`。状态=
> `VALIDATED-B3-FIVE-FRESH-CORRECTNESS`，只开放B0/B2/B3六全排列36-process正式计时；当前仍无B3
> timing/speedup，B4—B7关闭。

> **2026-08-14 FSG4/B3-C关闭状态**：source `72bec5e`的fresh GPU artifact含1484条event，实测
> candidate/commit/backup/copy=`12/12/12/12`、timed candidate D2H=`0`，其余B3-B结构与六个B2
> control语义保持；headline digest=`0`、post-query audit/replay/6/6 tamper、定向`54 passed`和全量
> `1279 passed, 3 skipped`通过。状态=`VALIDATED-B3-C-COUNTERS`，不是timing/speedup；下一动作是5组
> fresh B2/B3 correctness pairs，B4—B7关闭。该下一动作现已由上方Five-Fresh关闭取代。

> **2026-08-14 FSG4/B3-B关闭状态**：source `42df2dc`的fresh GPU artifact含5157条event，实测full
> step snapshots=`0`、forward builds=`4`，其余B3-A冻结结构与六个B2 control语义保持；replay、6/6
> tamper、定向`45 passed`和全量`1265 passed, 3 skipped`通过。状态=
> `VALIDATED-B3-B-COUNTERS`，不是timing/speedup；该时点下一动作只允许B3-C AtomicCommitPlan，现已由
> 上方B3-C与Five-Fresh关闭取代。

> **2026-08-14 FSG4/B3-A关闭状态**：source `c7851c8`的fresh GPU artifact含5157条event，实测template
> compile/hit=`1/1`、module move=`0`、scope=`1`，其余冻结B2结构与语义不变；replay、六个B2 control
> 语义和6/6 tamper通过，定向`34 passed`、全量`1257 passed, 3 skipped`。状态=
> `VALIDATED-B3-A-COUNTERS`，不是timing/speedup；该“下一动作”已被上方B3-B关闭取代。

> **2026-08-14 FSG4/B3-0关闭状态**：source `4195361`正式B2 artifact的4625条event确认全部预注册
> counter，六个冻结B2 control语义、replay与6/6 tamper通过，状态=`VALIDATED-B2-COUNTERS`。它没有
> speedup claim；该“下一动作”已由上方B3-A关闭取代，B3-C—B7保持关闭。

> **2026-08-14 FSG4/B3启动状态（历史）**：IR/graph/Plan/Schedule复用完成预注册时尚未实现；当时下一
> 动作为B3-0。该指令及其后续“下一动作B3-A”已被上方B3-A/B/C与Five-Fresh关闭取代；当前唯一下一
> 动作是36-process正式B3计时，仍没有B3 speedup，B4—B7保持关闭。

> **2026-08-14 当前状态**：FSG3正式same-solver基线已关闭。source `a4ee291`的
> `resnet2b-prop0-v5`包含六个全排列block、36个fresh GPU进程；correctness、environment、
> profile closure/扰动、static replay与8类outer-resigned tamper门禁全过。B1 query wall=
> `0.995657x`；当前B2 query/core=`0.908400x/0.516767x`（B0/candidate），显存ratio=`1.0`，因此
> FSG3=`VALIDATED-FSG3-B0-B1-B2-BASELINE`，B2=`MEASURED-B2-SLOWER`，不是speedup。下一门禁为
> FSG4/B3 IR/graph/Plan/Schedule复用；B4—B7与最终queue/complete-query门槛均未测试，ASPLOS-ready=NO。

> 状态日期：2026-08-14
> 当前 integration base：`f194034`（NRIR-44 PR #55 merge）；PR-13 历史基线：`57a854b` / tag `pr13-validated-reduced`
> 当前研发分支：`feat/rvir-v4-production-state-ownership-v1`；FSG2历史 implementation/inventory
> revisions=`aa31eae`/`8bf6981`；当前已推进至RVIR-v4 V4-3 whole-core replacement关闭；
> FSG0、FSG1均已验证；FSG2曾以`VALIDATED-REDUCED initial-only`关闭，完整B2 replacement在该时点
> `NO-GO/not admitted`、FSG3—FSG5按依赖门禁未运行；该ownership blocker现已由RVIR-v4 V4-3
> whole-core replacement关闭；FSG3 same-solver timing现亦已正式关闭
> 总判定：IR-5 final **VALIDATED-NO-GO**；PR-14B 同为 No-Go、PR-14C/IR-6 不启动；
> ASPLOS-ready 为 **NO**。
> 2026-08-13 RVIR-v4状态：V4-1 frozen-state evaluator已`VALIDATED-REDUCED`关闭；V4-2A只关闭
> 双LR与10 evaluation/9 update子合同。重启后GPU/NVML恢复，V4-2B正式artifact从`af8db08`生成：
> 1 core/24 calls、10 evaluations/9 observed Adam updates、每步24项raw state、相邻7项mutable变化，
> original replay与state/lower/result/lineage/policy五类同步重签名tamper均通过。因此V4-2B以
> `VALIDATED-PRODUCTION-TRACE`关闭。它只冻结provider真值轨迹；BoundFlow尚未执行mutation，V4-2、
> B2和性能claim仍关闭。V4-2C正式artifact从clean runner `96c45a6`生成：共享mapper在真实ResNet2B
> native scope上恢复6组dense α/β/split与external intermediate bounds，12/12 round-trip bit-exact，
> upper-α显式copy-through；original replay及topology/index/history/intermediate/upper-α/beta-location
> 六类内部重哈希、source/outer重签攻击在provenance与semantic两层全部拒绝。因此V4-2C以
> `VALIDATED-PRE-STATE-INITIALIZER`关闭。V4-2D formal native executor又在不读取reference trace、零
> provider callback下独立执行10 evaluations/9 Adam updates；10/10 step lower/α/β allclose且sign exact，
> 最大误差=`4.5300e-06/1.4663e-05/3.9861e-07 <=2e-4`，original replay与6类双层完全重签攻击通过，
> 以`VALIDATED-NATIVE-STEP-PARITY`关闭。V4-2E随后私有stage并原子提交12个production mutable paths，
> 其中7个改变；α/β/final lower最大误差=`1.4663e-05/3.6135e-07/2.6226e-06 <=2e-4`，
> NaN/stale/mid-copy fault均保持live pre-image。formal original replay和topology/initial α/post α/final
> lower/recorded copy-out/recorded commit六类完全重签攻击在两层6/6拒绝；full=
> `1175 passed, 3 skipped`。因此V4-2E=`VALIDATED-ATOMIC-COPY-OUT`、V4-2=
> `VALIDATED-OPTIMIZER-REPLACEMENT`。它不是whole-core live integration；B2与性能claim仍关闭，
> V4-3A随后从source `bfdeefc`冻结1 core/6 domains/24 calls、6 intermediate、6 pre-KFSB lA、3组
> candidate child lower、final decision和完整post/accounting；两次fresh semantic replay覆盖451
> tensors/213,060 signs，最大差`8.8215e-06 <=2e-4`，六类同步重签攻击6/6拒绝，full=
> `1180 passed, 3 skipped`。因此V4-3A=`VALIDATED-WHOLE-CORE-TRUTH`。它仍不是candidate whole-core
> replacement。V4-3B随后从terminal native state零provider callback导出六层lA、12个shared-input
> intermediate tensors和final lower，最大差=`9.2387e-07/6.0797e-06/3.0994e-06 <=2e-4`，sign
> exact；五类同步重签攻击5/5拒绝，full=`1183 passed, 3 skipped`，以
> `VALIDATED-NATIVE-BACKWARD-EXPORT`关闭。formal native replay为CPU semantic evidence，不是GPU live
> integration。V4-3C随后从native bounds/split/lA独立推导六层mask，复现三组top-3 candidate并执行
> 72个child lower；candidate/final decision exact、child lower最大差`3.0994e-06`，八类同步重签攻击
> 8/8拒绝，full=`1187 passed, 3 skipped`，以`VALIDATED-NATIVE-KFSB`关闭。V4-3D随后在RTX 4060上
> 以零provider bound callback完成whole-core→未修改
> official post/queue端到端运行，451 tensor语义比较最大差`1.0669e-05`、decision exact；fresh replay
> 与8类完全重签攻击通过，以`VALIDATED-LIVE-RETURN`关闭。下一门禁为V4-3E five-fresh；V4-3整体、
> B2与性能claim继续关闭。V4-3E随后按冻结顺序运行10个fresh GPU进程，5/5 original/candidate pairs
> 的完整state/branch/queue/termination通过，六类重签攻击拒绝，以
> `VALIDATED-FIVE-FRESH-CORRECTNESS`关闭；V4-3整体=`VALIDATED-WHOLE-CORE-REPLACEMENT`。B2
> same-solver timing随后按冻结六个全排列block、36个fresh control/profile worker完成；当前正式状态、
> 数字与下一动作以上方“2026-08-14 当前状态”为准。本段保留V4-3E关闭时的历史顺序，不再作为当前
> `PREREGISTERED-NOT-RUN`指令。
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
> 2026-08-05 NRIR-43 已预注册：唯一变量是把已经 ready 的 clause/node/candidate child-lower
> 计算降为 typed ragged batch Schedule；policy、optimizer/refinement、queue、31/depth4、dtype、
> workload 与 global-60s deadline 全部冻结。Phase A 要求 exact ownership/semantics、scorer launch
> `62→<=32` 且每条 queue ratio `<=0.85`；只有 Phase A 全过才进入 two-clause ready-set Phase B，
> 其门禁为 optimizer launch `32→<=16`、scorer launch `62→<=16`、每轮 whole `<=45 s` 且 median
> ratio `<=0.80`。以上为预注册时门禁，不是已实现或性能结论。
> 2026-08-05 NRIR-43 Phase A 已正式关闭：6/6 组 exact，per-clause scorer launches `31→16`，但
> clauses 2/3 median ratio=`1.051134/1.044573`，墙钟分别退化 `0.655621/0.579665 s`；formal hash=
> `692b9e27…30390`，全量 `968 passed, 37 skipped`。状态为 `VALIDATED-NO-GO`，Phase B 不启动，
> NRIR-42 production admission 保持。下一变量是 NRIR-44 root-projection floor Schedule。
> NRIR-43 功能/负结果提交 `00b82c2` 已由 PR #54 合入 `main@2d245d6`。
> 2026-08-05 NRIR-44 已关闭：新增 typed consumer/liveness Plan/Instance/Task/Schedule/Trace，将 ranking
> floor 的 9 条 objective queues 从 n31d4 投影为 n1d0。Phase A 三轮 root/rank/selected exact，evaluation
> `279→9`，old/projected floor median=`24.235039/9.876515 s`、ratio=`0.407530`；Phase B floor=
> `[8.538814,8.622447,8.648849] s`，whole=`[43.571040,44.144990,44.095736] s`，相对 NRIR-42
> median ratio=`0.764254`。两条 production queue 每轮仍为 `[31,31]` nodes，worst-active lower exact。
> Phase A formal hash=`ecb553d8…ff0fe`，Phase B payload hash=`2f22d44f…7272d9`。以 fixed ResNet2B
> property 0 CPU8 `VALIDATED-REDUCED` 关闭；final 9/9 unknown、`performance_claimed=false`、
> ASPLOS-ready=NO；全量 `979 passed, 37 skipped`。下一步转剩余 top-2 production queue 的新一轮
> 单变量归因与优化。
> NRIR-44 功能/证据提交 `437680e` 已由 PR #55 合入 `main@f194034`。
> 2026-08-05 NRIR-45 已预注册：cProfile 定位 top-2 production queue 的最大累计成本为 per-child
> intermediate refinement；单 queue 的 `_select_targets` 246 次中 186 次来自重复 Program validation。
> 唯一变量冻结为 prepared-once refinement capsule/validation ownership，不改 target、CROWN、optimizer、
> branch、queue、31/depth4 或 deadline。路线前 ceiling probe 的 clause 3 queue trace 约
> `12.85→9.761678 s` 且 worst lower exact；正式 Phase A/B 尚未开始。
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

## 47. 2026-08-05 当前生产路线补录：NRIR-42—45

NRIR-42 先以 compile-owned scorer capsule 将每条 31-node queue 的 candidate enumeration 从 341 次
收敛到 31 次，并恢复 fixed ResNet2B property 0 CPU8 objective-branch production admission；NRIR-43
跨节点 scorer batching 虽将 launch `31→16`，CPU wall time 却退化，因此 `VALIDATED-NO-GO`。

NRIR-44 随后用 typed ranking-only consumer/liveness contract 将九子句 floor objective evaluations
`279→9`，whole trace 中位数降到约 44.10 秒。NRIR-45 再把 per-child intermediate refinement 的完整
validation/hash ownership 收敛为 prepare-once capsule：target selection=`246→98`、full Program
validation=`186→38`、full hash=`217→39`；clauses 2/3 queue median ratio=
`0.727519/0.736603`。

NRIR-45 Phase B 三轮 whole trace=`31.262521/31.319772/31.470078 s`，measured wall=
`36.396631/36.513683/36.611709 s`，相对 NRIR-44 median ratio=`0.710268/0.615738`；floor、rank、
selected `[2,3]`、两条 `[31,31]` nodes、worst lower 与 final 9/9 unknown 均保持。Phase A/B hash=
`be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439` /
`4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；全量
`984 passed, 37 skipped`。

当前状态为 fixed ResNet2B property 0 CPU8 internal production `VALIDATED-REDUCED`，但 final 仍
unknown，`performance_claimed=false`，公平竞品、GPU、多 workload、property closure 与 ASPLOS-ready
仍未成立。下一动作不是直接声明“够投”，而是对最终约 31.3 秒 trace 做 residual phase attribution，
再预注册一个 IR/Plan/Schedule 单变量；不得重开 NRIR-43 CPU batching 或事后降低 cap/nodes/depth。

## 48. 2026-08-05 NRIR-46 Template/Instance Phase 0 NO-GO

residual attribution 已把 NRIR45 whole trace 拆为 floor action median=`10.818262 s`、packed slice
median=`9.932808 s`、packed-plan compile median=`0.146457 s` 与 rank median=`0.024966 s`。一次
diagnostic repeat0 的 60 child prepared compile/execute=`5.300590/5.659414 s`、per-child total=
`10.975123 s`、optimizer execute=`1.156098 s`。这些不是 formal claim。

stacked branch 为 `feat/intermediate-refinement-template-instance-v1`。唯一变量原计划把静态 graph/
policy/selection recipe/Task/Schedule topology 冻结到 PlanTemplate/ScheduleTemplate，把逐 child split/
source/objective/bounds/exact target ledger 绑定到 PlanInstance/InstanceSchedule。NRIR46 不做跨节点数值
batching，不改 cap/nodes/depth/policy/deadline；Phase 0 ceiling、Phase A exact+timing、Phase B whole-query
按顺序门禁。

Phase 0 三轮 compile total=`5.356892/5.366369/5.452290 s`，strict static topology=
`1.071197/1.062492/1.071704 s`，ownership-convertible ceiling=`2.097255/2.102134/2.109857 s`。
预注册 strict static gate 要求 median 至少 `1.5 s`，实测 median=`1.071197 s`，因此 NRIR46
`VALIDATED-NO-GO`，Template/Instance 未实现，Phase A/B gated off。

三轮仍保持 selected `[2,3]`、nodes `[31,31]`、60/60 capsules/full replay；每轮 target selection
observed/semantic=`124/60`、冗余=`64`，60 个 target ledger 全部互异。formal hash=
`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`，replay/tamper 通过，
`performance_claimed=false`。下一独立变量只能是 single-pass target admission receipt；公平竞品、
10x、property closure 与 ASPLOS-ready 仍为 NO。

## 49. 2026-08-05 NRIR-47 Single-Pass Target Admission Receipt Phase A NO-GO

NRIR46 已由 PR #57 合入 `main@ca0bcf3`。下一分支为
`feat/single-pass-target-admission-receipt-v1`，唯一变量是把每个 child 的 exact target selection 从
compile + validation 两次收敛为一次 selection 加 typed admission receipt。60 个 target ledger 仍
node-specific 且互异；production 不重选，显式 full replay 必须从 exact bounds/objective/policy 重选。

NRIR46 已测 target reselection ceiling median=`1.038153 s`，只约占 frozen NRIR45 whole trace 3.3%。
Phase A 预注册 compiler ratio `<=0.85`、clauses 2/3 queue ratio 均 `<=0.97`；只有 correctness、
ownership、timing 全过才启动 Phase B，其 trace/measured ratio 均须 `<=0.98` 且改善大于 pooled MAD。

typed receipt/Task/Schedule、additive single-pass compiler、prepared binding、candidate production route
与 explicit full replay 已实现。每条 candidate queue compile selector/reselection=`30/0`、runtime
selector=`30`、receipt/full replay=`31/31`；correctness/ownership exact，186 份 receipt replay 与同步
外层重哈希 tamper 拒绝通过。

正式 compiler ratio=`0.936003 > 0.85`；clauses 2/3 queue ratio=
`1.011205/1.019338 > 0.97`，所以 timing gate 失败，NRIR47 `VALIDATED-NO-GO`，Phase B gated off。
formal hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；
全量 `992 passed, 37 skipped`。candidate 不默认启用；final 9/9 unknown、
`performance_claimed=false`、公平竞品/10x/property closure/ASPLOS-ready=NO 不变。下一步转 top-2
production execution math/queue phase attribution。

## 50. 2026-08-05 NRIR-48 Top-2 Production Execution Cost Attribution 判定

NRIR47 已由 PR #58 合入 `main@1e44949`。新分支
`feat/top2-production-execution-cost-attribution-v1` 只测 NRIR45 default production route，不启用
NRIR47 candidate，也不修改算法/IR/runtime。clauses 2/3 各运行 three fresh paired control/profile，
把 queue wall time 闭合到七个互斥顶层类别，并对子 refinement execute 做五类内部诊断。

预注册要求 correctness exact、category closure error `<=1%`、instrumentation median ratio `<=1.05`；
同一 dominant category 必须在两条 clause 各 3/3 排第一、median share 均 `>=20%`、range `<=10`
percentage points 且稳定超过 pooled MAD。

正式 6/6 semantic exact，clauses 2/3 profile/control ratio=`1.023199/1.020221`。两条 3/3 winner
均为 child refinement execute：median=`3.816002/3.704755 s`，queue share=
`32.1966%/31.1640%`。内部 selected-CROWN median=`2.663321/2.694436 s`，占 parent=
`71.7725%/72.7291%`，为唯一过 `>=30%` 的子类。formal hash=
`571c2e47c0c8906d2486e5e19e8152eb1ef0d3024b08cf561e25ed4f71d177a4`；replay/tamper 与全量
`996 passed, 37 skipped` 通过。

NRIR48 attribution `VALIDATED-REDUCED`；未实现优化、没有 speedup claim。其 closure 当时只准入另立
NRIR49 selected-CROWN execution 单变量；该历史动作已由下述 NRIR49A 完成，不是当前指令。

## 51. 2026-08-06 NRIR49A G1 GPU Selected-CROWN-only Opportunity 判定

G0 post-reboot已确认RTX 4060 Laptop、Torch CUDA 13.2/SM89、TVM CUDA TIR、TVM-FFI stream与
双方workload digest门禁通过。NRIR49A随后只读执行frozen clauses 2/3、31-node production queue；
production runtime、TIR、kernel与默认chunk 32均未修改。

五个fresh worker全部成功，queue/complete selected-CROWN share中位=
`0.07098631834282758/0.070523288963519`；paired profile/control ratio中位=
`0.999304435327957/1.0067470427656482`，测量门禁通过。60组离散结构exact，数值最大
absolute/relative diff=`2.288818359375e-05/0.0001710717646052519 <=2e-4`。代表调用CUPTI记录
5954 kernels、5486 launches、398 sync和5364 memory events。

selected-CROWN share低于20%，queue `1.20x`和complete `1.15x`目标均超过该单区域的Amdahl无限
加速上限；`1/(1-0.070986)=1.0764x` 只是假设 selected-CROWN region 变为零耗时的
deletion-only 上限，不是 BoundFlow 的全栈上限。最大allocated/reserved只占8 GiB物理显存
`0.996%/1.353%`，合法domain batch上限1且无OOM，memory path=`N/A`。summary/manifest hash=
`7eefe6a7…ab50`/`d0272fe4…c81f`，独立replay与digest重算通过。NRIR49A G1为
`VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`，只将selected-CROWN专属G2/G3
gated off；不否定算子、Bound/Graph IR、Plan/Schedule、跨阶段融合、JIT、runtime调度或内存复用的
累计收益。

正式artifact中的`next_route=gpu-winner-reselection`是冻结历史机器输出，现由
`gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`取代。当前按
FSG0现已关闭：typed layer/phase/resource/cache、feature activation ledger、critical-path/residual、
累计消融与tamper-resistant replay合同共20项定向测试通过，全量`1079 passed, 3 skipped`，外部
审计三项minor已修复。当前下一步FSG1采集official original executor的control full-stack trace；该阶段
只建立B0分层基线，尚无BoundFlow
全栈GPU性能claim。ASPLOS-ready仍为NO。
