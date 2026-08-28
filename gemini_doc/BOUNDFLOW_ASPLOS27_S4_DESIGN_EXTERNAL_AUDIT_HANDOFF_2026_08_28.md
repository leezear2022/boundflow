---
status: ready-for-external-design-audit-v18-s4-4-construction-readiness-frozen
date: 2026-08-29
type: external-audit-handoff
topic: boundflow
slug: asplos27-s4-design-audit
audit-kind: preregistration-and-implementation-blueprint
base-commit: ebf45cc72438141d8f0b35dadfd5cf774d7e753f
design-result-commit: c750fafdde8435f56146294faf509221a780057d
execution-authority: false
code-change-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
---

# BoundFlow ASPLOS'27 S4-0—S4-4设计外审交接

## 0. 给外审模型的直接任务

请把本轮当作**设计/预注册审计**，不是实现验收。不要因为文档详尽就推断S4代码、GPU correctness、same-solver
speedup或complete-query性能已经存在。

请独立核对仓库源码、历史artifact和pinned external source，判断：

1. S4把S3 P-only局部owner扩展为六α+active β whole-core exact-call的路线是否完整；
2. S4-0—S4-4的分层是否真正对应production事务，而不是为写IR/receipt而写IR/receipt；
3. all-state compiled VJP、sealed production policy、terminal handoff、KFSB、commit/post是否有遗漏owner；
4. S4-3 failure state是否诚实，尤其PyTorch `_version`与post-after-commit；
5. S4-0 preflight关于offline snapshot不能证明live storage alias/`_version`的纠正是否成立；
6. tensor-free receipt与不可序列化strong-ref ephemeral lease的双层owner是否关闭S4-0→S4-1A→S4-3跨阶段
   object/storage/version ownership且没有新增IR；
7. S4-1A two-phase prepare、12-source private lease retention、16 base view和三阶段失败清理是否闭合；
8. live B0/R phase probe把scratch合同从terminal disposal升级为variant-specific finalization v2是否成立；
9. terminal logical/unique storage、view alias、B0 batch-24 residue与当前R batch-12 stale是否被正确区分；
10. S4-4的indexed-binary stdlib raw、33-worker replay和96类layered tamper是否足以支持第三方独立审计；
11. S4-1B0把site19根因收窄到`Ainput==0→center`、恢复selected-primal lowering是否数学成立；
12. S4-1B0现场发现浮点NaN classifier会被TVM/CUDA错误化简后，改用IEEE-754 exponent位检查、独立cache
    key与module/launch receipt是否足以fail closed；
13. 五张binary selector是否也必须使用`-128` sentinel；七gradient TIR的safe-index/finite poison、46 emitter
    views与base+emitter local 48口径是否完整；
14. S4-1D read-only admission、post-begin `POISONED_NO_RETRY`、opaque result capability、12-worker六全排列
    full-IEEE raw和修正后的`389,574 B` ledger是否共同关闭single-evaluation事务；
15. S4-2 live policy/functional Adam审计及施工修订是否正确关闭checkpoint、patience reachability、
    step/best/shadow memory、post-begin poison和24-worker formal口径；
16. S4-3 live诊断据此引入prepared working-β、prefix-only rollback、post/queue独立计数与细粒度latch是否正确；
17. 是否同意在S3外审批准后仍按S4-0→1A→1B0→1B→1C→1D→2→3→4顺序实施；
18. 是否发现必须在第一行S4代码开工前修正的blocker/major。
19. artifact self-consistency与source/model真实性分层是否正确，external trust anchor是否关闭全量替换后重签；
20. 101 core/4 script/559 TVM Python/3 native的低扰动loaded snapshot加关键receipt，是否是可接受的source closure；
21. per-worker tensor index + content-addressed raw-binary sidecar是否完整保留IEEE、logical path、view和alias语义；
22. 18 positive + 15 isolated fault=33 subprocess是否为最小且充分的poison/freshness拓扑；
23. semantic-root→summary→tamper→stdout→manifest→external-anchor的seal DAG是否无环，外审前pending状态是否正确。
24. S4-0 V4新增keyword-only `exact_call_id`并将raw identity只留在private lease、hash写入receipt是否足以绑定phase；
25. pinned provider的exact built-in `dict/list/Tensor`事实是否支持专用strict extractor，且不应复用历史宽松helper；
26. receipt前后两次live token capture是否足以关闭admission read race，又没有冒充通用并发锁；
27. 两轮12 Tensor content validation的24条logical D2H/`68,016 B`披露是否准确，零candidate kernel/allocation口径是否诚实；
28. generic validator异常的envelope+residual稳定reason方案和56类negative是否可机械实现；
29. S4-0 local single-transfer与S4-3 process-global exact-call exclusivity的claim分层是否正确。
30. S4-1A把evaluation/result lease、Adam、10/9 trajectory和terminal handoff移回S4-1D/S4-2/S4-3是否是正确scope；
31. S4-1A public API删除caller-provided device/stream、改为runtime观察是否关闭伪造边界；
32. existing `(data_ptr,shape)`DLPack key的same-pointer/shape different-stride碰撞是否可复现，V5 full key是否充分；
33. S4-1A自身`32 D2H / 85,056 B`、与S4-0累计`56 / 153,072 B`是否完整且没有把logical冒充physical；
34. single resource owner、清loop local和退出`except`后raise是否在retained traceback下真正关闭CUDA/view lifetime；
35. ticket→resource-owner→prepared的single logical owner adoption是否优于逐字段move，仍有无double/no-owner窗口；
36. gradient/lower未初始化且first full-write前read-forbidden是否足够；upstream是否必须纳入initial content validation；
37. 68 negative与5 positive+7 isolated fault formal是否足以关闭S4-1A，而不冒充provider mapping/global exclusivity。
38. S4-1B0作为fixed backend lowering而不新增endpoint IR是否正确，是否仍有必须由IR表达的新solver/effect语义；
39. isolated pack/select的5 view是否确与S4-1A 16 base view零重叠，`18,432+73,728=92,160 B`账是否完整；
40. `389,574 B` production ledger是否确实隐含selected output复用coefficient arena，S4-1B phase/liveness proof是否充分；
41. precompile lookup key、immutable module receipt、mutable cache observation、formal tensor sidecar四层是否正确；
42. device source作为compile output不进入首次lookup key、但进入module receipt/cache value是否避免循环身份；
43. warm path删除content hash/class count D2H是否正确，generation/descriptor/O(1) counter能否维持fail closed；
44. 20 stable reason、16项test layout与5+1+5 future formal topology是否足以关闭S4-1B0。
45. S4-1B 90个argument descriptor与完整S4-1A/B/C 110个的集合重算是否准确，旧48是否应严格降为局部scope；
46. pass A 19-action及A29/A26/A24/A20/A18/Ainput六个capture点是否与现有coefficient schedule精确一致；
47. selected graph 42-read+7-write、six persistent slot、active-α ABI和site31不消费α/map/sign是否完整；
48. residual11/6 scratch是否确为两个coefficient arena的offset slice，新增physical bytes=0，旧49,152 B是否重复计账；
49. S4-1D/S4-2/S4-3修正subtotal `389,574/491,774/559,838 B`是否逐项成立；alias失败时S4-1D是否为`463,302 B`；
50. Ainput coefficient→selected-input→coefficient的generation/live-reader/stream转换能否在不重建warm DLPack view下实现；
51. S2 prepare的Torch allocated与CUDA driver free delta差异，是否证明future artifact必须双口径披露TVM/VM/cuDNN footprint；
52. 55类S4-1B negative、five-fresh raw/replay/full-resign tamper是否仍有遗漏。
53. S4-1C完整Pass C是否确为10 coefficient+7 emitter=`17` actions，terminal插入6 copy后=`23`；
54. site31是否必须等待dα31和dβ31两个V reader后再copy，逐reader state是否比单`gradient emitted`位更强；
55. terminal copy作为6个额外typed symbol、完整module 13 symbols是否合理，且新增descriptor/storage确为0；
56. 六V/lA slot是否在一个37,464-element storage上无重叠/无空洞，terminal in-place覆盖是否不需要额外arena；
57. argument DLPack 110与result普通Torch view 6的口径是否准确，site31 view是否确可复用；
58. runtime O(1) receipt与formal V-pre-overwrite/lA-post-copy sidecar分层是否避免热路径D2H/hash；
59. native six-clone handoff迁移到one-arena one-shot lease是否保持topology/order/spec-axis与KFSB lifetime；
60. 62类S4-1C negative、5+5 fresh和17/23 action replay是否还有遗漏或无法机械实现之处。
61. S4-1D旧`919,680 B`是否确只是5+5 candidate output，不能支持A/B/C三方独立复核；
62. formal改为两fixture各六全排列、12 fresh是否是关闭执行顺序/owner污染的合理最小拓扑；
63. `3,310,848 B` three-way output+`899,136 B` terminal V sidecar=`4,209,984 B`最低raw是否准确；
64. raw Tensor getter不可撤销反例是否意味着result/terminal必须改成opaque exact sealed-consumer capability；
65. exact consumer type/hash与post-consume retention audit能否充分阻止Tensor/storage逃逸，是否还有更强可实现边界；
66. parent/child 9-state、14 legal transition、67 invalid组合是否完整，transfer是否正确独立于parent close；
67. component→execution→artifact 15-node seal DAG是否无环，result runtime ref是否正确排除于semantic root；
68. 86类S4-1D negative、12-worker raw-first/replay/full-resign tamper是否足够且可机械实现；
69. S4-1D通过后只开放S4-2而不提前暴露raw Tensor/支持re-arm，边界是否正确。
70. S4-2用run-level evaluator family发行10个one-shot generation、完成9次controlled re-arm，而不复活已关闭的
    S4-1D对象，是否是正确所有权边界；
71. opaque result的exact policy consume是否真正禁止lower/gradient/lA raw Tensor逃逸，terminal child原子转移是否
    还有consumer异常或异步引用缺口；
72. evaluation-input version、optimizer mutation count与storage commit generation三轴拆分是否必要且充分，
    terminal restore是否还需第四类version；
73. 7 parameter+7m+7v+7step的28项commit cursor、scheduler后提交和partial failure poison是否诚实；
74. terminal lA只允许与ordinal9 best state handoff、earlier-best稳定拒绝，是否正确关闭adjoint/state错配；
75. 16-state/16-event/32-legal/224-invalid policy模型是否遗漏close、handoff、re-arm或terminal restore状态；
76. A/B与B/C各6对、正反各3的24-worker topology是否为合理最小顺序平衡设计；
77. A/B/C per-run raw floor=`2,837,288/2,871,296/1,511,936 B`及总计`60,550,896 B`是否正确，
    `491,774 B`降为known base lower bound后还有哪些必须在实现前冻结的storage。
78. S4-3 whole-core transaction是否必须从terminal claim起算，而不是从第一条device copy起算；post-claim pre-copy
    failure进入`STAGING_POISONED`是否正确？
79. provider scratch 36项finalization作为独立可失败mutation是否完整；partial sentinel normalization后是否存在任何
    可安全retry/provider reentry路径？
80. 23-state/22-event/40-legal/466-invalid模型是否完整；hash=`6ed3d2fd...a3388a`能否独立重建？
81. 12 device+host final packet+intermediate container clear=`14` mutation是否是正确logical commit边界？
82. prefix-only inverse restore且untouched suffix write=0是否是in-place PyTorch `_version`约束下最强可实现语义？
83. provider `BatchedDomainList.add`的多owner mutation surface是否证明queue不能声称原子；fault后只盘点changed units并
    `QUEUE_POISONED`是否足够？
84. add成功但`check_worst_domain`失败时，candidate add count=1且query poisoned是否正确？
85. R/C由5对改为6对、`RC/CR=3/3`、12 fresh是否为合理最小顺序平衡设计？
86. downstream semantic occurrence=`1,025,952 B`、R/C per-run=`3,897,248/2,537,888 B`、总计
    `38,610,816 B`是否算术和scope均正确？
87. tensor-occurrence、unique content和physical file bytes三口径是否足以避免raw预算claim漂移？
88. `559,838 B`降为known tensor/base lower bound后，prepared transaction/provider scratch/KFSB/post/queue/allocator
    还应强制测量哪些storage与phase peak？
89. S4-4的15-fault registry是否逐项覆盖preclaim clean、staging/commit/post/queue poison，且没有把claim后故障
    错写为可retry clean abort？
90. B0/R/C per-run semantic occurrence=`3,829,232/3,897,248/2,537,888 B`是否逐项成立？
91. 18 positive=`61,586,208 B`、15 fault=`24,209,400 B`、总计=`85,795,608 B`是否正确，是否还有
    mandatory semantic tensor遗漏？
92. 语义出现量、worker-local unique content与最终physical artifact bytes三口径是否已彻底分离？
93. 16-node/36-edge final evidence seal DAG是否可按冻结序列化规则重建且确实无环？
94. semantic root、derived summary、tamper report、replay stdout、manifest、external anchor和anchored record的
    先后边界是否避免任何self-hash循环？
95. 96-case registry的13/5/77/1 enforcement分层是否正确，是否有应移动到另一层的攻击？
96. T19 fresh-process attestation只能返回`OFFLINE_UNATTESTABLE`是否是诚实且必要的限制，现场外审应补何种OS/
    parent/process证据？
97. 其余95类在同步重写payload/summary/manifest的fully re-signed条件下是否都有明确拒绝owner？
98. stdlib replayer是否能从raw独立重建S4-2三个version轴、S4-3 23-state事务、fault terminal与全部headline，
    而不调用production validator？
99. final protocol hash在实现时是否还必须绑定source/input/numeric/schema完整值，而不能误用当前仅用于设计复核的
    structure projection hash？
100. 是否同意本轮仍只关闭S4设计施工合同，不开放S4代码/formal/timing/performance，唯一外部动作仍是S3审计？

## 1. 审计范围和Git边界

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- 本轮设计base：`ebf45cc72438141d8f0b35dadfd5cf774d7e753f`；
- S4-0 live admission/lease、S4-3A scratch finalization、S4-1A prepare transaction、S4-1B0 ternary TIR、
  S4-1B/1C selector/gradient/arena ABI、S4-1D evaluator、S4-2 policy、S4-3 whole-core transaction及S4-4 formal
  evidence readiness、S4-0 V4、S4-1A V5、S4-1B0、S4-1B、S4-1C、S4-1D、S4-2、S4-3与S4-4 construction readiness
  全部设计结果：`c750fafdde8435f56146294faf509221a780057d`；
- 审计范围以`ebf45cc72438141d8f0b35dadfd5cf774d7e753f..c750fafdde8435f56146294faf509221a780057d`
  和下列S4文档的完整版本为准；
- S3 formal实现/结果不在本轮重新验收，但它是S4设计输入；S3独立exchange仍等待审计；
- `.docops/exchange/gc0-1-prereg-20260826`异步audit文件和`docs/CIBC_for_DAC.pdf`是用户保留的范围外dirty文件，
  不得误判为S4设计diff。

本轮不应要求出现：

- S4 production代码；
- S4 compiled all-state TIR；
- S4 GPU formal raw；
- S4 timing/speedup；
- S4 complete-query或10x结果。

这些都明确保持closed。

## 2. 必读文档顺序

1. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_SAME_SOLVER_EXACT_CALL_PREREG_2026_08_28.md`；
2. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_ALL_STATE_VJP_FEASIBILITY_2026_08_28.md`；
3. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_EVALUATOR_ABI_AND_TERMINAL_HANDOFF_2026_08_28.md`；
4. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_MUTABLE_STATE_ADMISSION_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
5. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_ADMISSION_PREFLIGHT_CORRECTION_2026_08_28.md`；
6. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_LIVE_LEASE_IMPLEMENTATION_READINESS_2026_08_28.md`；
7. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V4权威施工合同）；
8. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_ORDERED_BUFFER_ABI_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
9. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_PREPARE_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`；
10. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V5权威施工合同）；
11. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_BOX_ENDPOINT_SUBGRADIENT_CLOSURE_2026_08_28.md`；
12. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_ENDPOINT_IMPLEMENTATION_READINESS_2026_08_28.md`；
13. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_TIR_ABI_IMPLEMENTATION_READINESS_2026_08_28.md`；
14. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（权威施工合同）；
15. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1BC_DAG_ADJOINT_PREFLIGHT_CORRECTION_2026_08_28.md`（历史v1）；
16. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B_SIX_SITE_EFFECTIVE_VALUE_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
17. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
18. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1C_COMPRESSED_GRADIENT_EMITTER_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
19. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1C_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
20. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1BC_SELECTOR_GRADIENT_TIR_IMPLEMENTATION_READINESS_2026_08_28.md`；
21. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_ALL_STATE_EVALUATOR_CLOSURE_BLUEPRINT_2026_08_28.md`；
22. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_EVALUATOR_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`；
23. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
24. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_SEALED_PRODUCTION_POLICY_DRIVER_BLUEPRINT_2026_08_28.md`；
25. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_POLICY_DRIVER_IMPLEMENTATION_READINESS_2026_08_29.md`；
26. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
27. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_EXACT_CALL_TRANSACTION_BLUEPRINT_2026_08_28.md`；
28. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_29.md`；
29. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
30. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3A_PROVIDER_NET_SCRATCH_CONSUMER_AUDIT_2026_08_28.md`；
31. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_4_FORMAL_ARTIFACT_REPLAY_TAMPER_CLOSURE_BLUEPRINT_2026_08_28.md`；
32. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_4_FORMAL_EVIDENCE_IMPLEMENTATION_READINESS_2026_08_29.md`；
33. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_4_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`（V1权威施工合同）；
34. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_CHANGE_LOG_2026_08_28.md`。

## 3. 已冻结的production事实，请独立复核

### 3.1 mutable state

- 六条α source stored=`8,496 float32`；
- lower-only active/preserved=`4,248/4,248`；
- S3 P-only stored/active=`1,032/516`，coverage=`12.1468926554%`；
- 六条β中唯一active为`beta/%2Finput-28/0/value:[6,1]`，其余五条empty；
- S3 P β为`[6,0]`，对production active β覆盖为0。

请从production snapshot/raw和initializer重新统计，不采信表格。

### 3.2 production policy

- evaluation=`10`；
- parameter mutation=`9`；
- scheduler call=`10`；
- terminal scheduler产生的post LR没有后续evaluation消费；
- 上述第10次scheduler只属于当前无early-exit fixed path；early exit在scheduler之前发生；
- checkpoint ordinal=`0,6,7,8,9`，ordinal 5不满足`i > int(10*0.5)`；
- fixed 10-step程序中patience从0开始最多到10，源码`patience>10`不可达；
- fixed physical pruning为inactive，preserve/next-preserve mask均为`None`；
- formal raw中六domain best ordinal均为9，但设计仍保留keep-best/prune/stop/patience/timeout/restore分支。

请亲读pinned `auto_LiRPA@5a098e8/optimized_bounds.py`确认。

### 3.3 terminal/KFSB/return

- terminal six-lA=`37,464 float32 / 149,856 bytes`；
- handoff必须来自ordinal 9，duplicate CROWN=0；
- KFSB=`3 candidates × batch 24 = 72 child lower`；
- live commit path=`12`；
- fixed candidate provider return constructor=`12`；
- C路径provider bound callback=`0`，official postprocess=`1`；
- query total domain add=`2`，candidate post domain add=`1`；
- host `d`必须prune到history/depths/thresholds；
- `pre_result.interm_bounds`必须clear。

### 3.4 known logical memory账

- S4-1D correctness ledger=`389,574 bytes`；
- 该ledger隐含S4-1B把`73,728-byte selected_endpoint`安全复用existing coefficient arena；phase proof未关闭时只
  是conditional design ledger，失败时S4-1D=`463,302 bytes`；
- residual11/6 scratch是两个coefficient arena的offset slice，只增加2个descriptor、additional physical bytes=`0`；
- S4-2 current m/v/step、compressed best、best lower、`ret_0`和transition shadow新增=`102,200 bytes`；
- S4-2 known subtotal=`491,774 bytes`（CUDA `491,718` + CPU `56`）；
- full candidate=`34,008 bytes`；
- candidate+rollback=`68,016 bytes`；
- persistent upper/depths=`48 bytes`；
- S4-3 known subtotal=`559,838 bytes`（CUDA `559,758` + CPU `80`）；
- working-β location/sign=`72 bytes`为external retained liveness，不重复计入new allocation。

旧S4-1D复核曾把`49,152 B` residual scratch当成独立storage，形成`438,726 B`；源码与GPU storage identity证明
它们已包含在147,456 B coefficient arenas内，所以相对最早`386,712 B`只应增加`2,862 B` metadata。这些是
design-time logical bytes，不是实测peak allocated/reserved。请独立检查本次纠正是否仍有重复计数、漏项或错误归属。

## 4. Acceptance criteria

### AC1：路线与IR边界

PASS要求：

- 没有创建第二套solver execution IR；
- S4新增对象仅是typed admission/buffer/evaluator/policy/transaction/artifact合同；
- static graph/legality/planning/lowering继续复用现有Bound/Plan/Task/Schedule/Relax/TIR；
- KFSB、queue、branch、termination仍由host solver拥有；
- S4文档能解释每个新增对象解决哪个真实ownership/legality/replay问题。

若某对象只是重复已有dataclass而没有新的fail-closed责任，请列为finding。

### AC2：S4-0 admission完备性

PASS要求：

- snapshot/topology/plan/live-source/exact-call五方key、shape、dtype、device、feature index、β location/sign、object/storage/
  stride/offset/version全覆盖；
- auditor复现distinct views共享storage反例，确认snapshot alias group不能替代live storage group；
- auditor复现全量same-content clone替换，确认canonical group/hash可完全相同，因此receipt不能替代strong-ref lease；
- prepared lease必须local single-transfer、不可序列化，并从S4-0保持到S4-3 current-provider precommit；
- receipt只保存exact-call hash，private lease保存raw ID并在phase/PID/thread/stream重验；S4-0不得冒充S4-3全局latch；
- live input只接受S4 strict extractor返回的exact built-in dict；不得复用existing helper的宽松`Mapping.get()`；
  lease/wrapper必须为非dataclass `__slots__` class并同时拒绝
  copy/deepcopy/pickle；
- `plan.source_state_hash`只作dense mapping provenance；plan/snapshot projection可不调用dense initializer独立重算；
- topology hash按plan order canonical，输入tuple置换不改变receipt；
- β width与history长度exact，不接受只匹配前缀；
- stored/active/preserved α口径分开；
- five empty β是metadata token，不伪造physical tensor；
- P-only和active-β missing明确拒绝；
- schema没有ResNet2B/node/shape特判；
- same-storage view、empty clone、provider rebind与lease重复使用均在mutation/commit前拒绝；
- `.data`/DLPack alias绕过`_version`时仍由content hash拒绝；hash同步成本保留到S4-P实测，不在correctness阶段删除；
- receipt前后双capture拒绝source read race；宽泛validator异常不通过英文文本解析，而归一为冻结detail code；
- 5 fresh real-provider receipt可由stdlib replay，minimum 56 negative逐项断言detail+reason；
- S4-0无candidate kernel/CUDA allocation、dense initializer、TIR或timing；两轮content validation的24条logical D2H、
  `68,016 B`必须单列，不得写成零GPU活动或物理CUPTI transaction。

### AC3：S4-1 all-state evaluator物理可行性

请从现有R31B1/R31B2/D1C/D2B/B4-B2代码验证：

- S4-1A formal buffer是否为6 α leaf+1 active β leaf+5 empty token、parameter/gradient各4,254元素/17,016 B、
  base DLPack 16/16 exact，且16 candidate storage与12 source storage完全不相交；
- candidate logical storage是否为`34,080 B`；本机allocator `39,936 B`是否只作诊断而不进入canonical gate；
- private lease保留12 source/8,502 elements/34,008 logical B是否确为incremental allocation 0，同时确实延长lifetime；
- `PROVIDER_SOURCE_RETAINED_AFTER_PREPARE`删除是否正确，provider container/callback与lease外source引用是否才应拒绝；
- two-phase prepare是否在validation前锁定single-attempt、在local staging后single-transfer；parameter/buffer/view故障时是否
  逆序释放、source不变、allocated回entry、stream/device恢复，且retry/fallback/empty-cache均为0；
- public API是否应拒绝caller device/stream token并从runtime观察；single private ticket/resource owner是否关闭逐字段move；
- same-pointer/shape不同stride是否会在existing key下碰撞并绕过TVM noncontiguous拒绝；V5 full key/lookup order是否足够；
- source entry/exit加7 parameter/upstream content是否确为S4-1A `32 D2H/85,056 B`、累计`56/153,072 B`；
- gradient/lower未初始化且first full-write前read-forbidden是否比hash uninitialized memory更正确；
- retained exception/traceback时loop local和TVM view是否仍可能保留CUDA storage；stable error context清理是否可机械测试；
- evaluation/result lease、Adam/10-step和terminal handoff从S4-1A移出后是否分别由S4-1D/S4-2/S4-3完整拥有；
- 68 negative与5 positive+7 fault formal是否充分，且receipt保持provider mapping/global exclusivity false；
- forward实际已消费六α+active β；
- 缺口确实是P-only gradient ABI，而非forward根本不支持其他site；
- 旧二元selected-primal在site19失败是否确由606个Ainput zero错误映射到lower导致；
- 三元`positive→lower / negative→upper / zero→center`是否与provider `abs`零点次梯度严格一致；
- 浮点`x==x` NaN检查在当前TVM/CUDA lowering中被错误化简的FAIL探针是否可信；float32 exponent
  bits/mask=`0x7f800000`是否是当前dtype下更强的fail-closed classifier；
- select是否显式区分`+1/-1/0/invalid`，invalid生成bits=`0x7fc00000`的canonical quiet NaN，而非被default
  else吞成center；
- max-finite和min-subnormal反例是否证明midpoint operation order必须绑定`(lower+upper)*float32(0.5)`；
- 独立pack/select的2 launch、5 unique tensor/view、6 argument occurrence、zero center tensor/view、zero workspace
  是否准确，且这5个view是否与S4-1A base零重叠；
- isolated selector/selected output是否确为`18,432/73,728 B`、合计`92,160 B`；旧稿写existing/零新增是否应纠正；
- production `389,574 B` ledger是否以selected output复用coefficient arena为条件；pass A capture→pass B E0→pass C
  recompute的storage/generation/live-reader/stream proof若失败，是否必须加回`73,728 B`至`463,302 B`；
- S4 cache key是否必须超出compute capability，绑定schema/symbol/TIR/target/shape/dtype/threads和三项policy；
- cache hit后重验module receipt、旧binary cache key必须miss/拒绝是否足够防止历史模块冒充；device source作为compile
  output不进入首次lookup key、但进入immutable receipt是否正确；
- mutable hit/miss count是否必须离开module identity；formal content/count sidecar是否必须离开warm receipt以避免D2H/sync；
- A18/A20/A24/A26/A29五张binary selector若仍只有0/1，NaN是否会因`A>=0=false`静默选upper；把六张
  selector统一加入`-128` invalid sentinel是否必要且不改变normal zero branch语义；
- S4 selector总bytes是否应为55,296、相对R31B2增加12,288，而不是“center为0所以总bytes不变”；
- gradient TIR是否必须safe-clamp runtime index/location再poison invalid，避免validator失误时先发生OOB；
- A/V/bound/α/upstream nonfinite、lower>upper或α越界是否必须qNaN poison，避免gate false返回有限0；
- β sign保持int8并在TIR内cast是否比prepare float32 copy更忠实；metadata是否为2,862 B；
- 7 launch、53 argument occurrence、emitter unique view 46、与base重叠14、additional 32、base+emitter local 48
  的口径是否逐项成立；
- reverse `31→28→25→23→19→17`中同stream emitter-read→terminal-copy→transform是否足以保护V/lA slot；
- coefficient-action VJP作为规范oracle、selected-primal作为优化lowering的双层owner是否合理；
- pass C按31→28→25→23→19→17即时导出六dα和site31 active dβ；
- site25/site19可从coefficient arena内部residual scratch slice取incoming coefficient；
- cross-layer saved/persistent dense A可保持0；
- two coefficient arenas足够；
- residual11/6 scratch是否分别与arena1/0共享storage、offset=`6144/12288`，所以additional physical bytes=0；
- Pass A 19-action是否在两个staged residual中间精确插入A26/A20 pack；
- selected graph是否为42 read + selected scratch/six V共7 write=49 argument descriptor；
- S4-1B descriptor是否为`16+49-5+30=90`，full A/B/C是否为`90+20=110`；
- terminal lA与V slot alias的lifetime门禁充分，且复制发生于ReLU transform前并恢复
  `[D,S,*feature]` spec-axis view；若不足是否应默认独立arena。
- Pass C是否只需10个coefficient动作并在A18结束，所以nonterminal完整action count=`17`而非继续执行
  ReLU17/Conv0/concretize；terminal插入6 copy后是否恰为`23`；
- site31的V31是否有dα/dβ两个reader，严格顺序必须为`dα31→dβ31→copy→ReLU31`；单一
  `gradient emitted`布尔位是否会错误放行copy-before-dβ；
- terminal copy的6个额外typed symbol是否使完整module最多13 symbols，但因复用A/V使新增argument
  descriptor/storage均为0；
- 六V/lA interval是否无重叠、无空洞、共一个37,464-element storage；site31 emitter/result shape相同是否使
  result-facing额外普通Torch view总数为6而非7；
- runtime receipt是否只保留O(1) phase/counter/identity；formal sidecar是否必须在覆盖前抓V、覆盖后抓lA且完全
  位于production timing之外；
- native handoff的六份clone storage能否安全迁移为plan-order one-arena one-shot lease，并在KFSB后只释放lA sublease；
- request admission是否在任何counter/buffer/generation mutation前完成，pre-begin reject是否保持`READY`；
- counter reset之后任一pass/finite/receipt失败是否必须`POISONED_NO_RETRY`，禁止reset/reuse generation；
- lower/六dα/六dβ/terminal lA是否应由一个composite lease共同持有，terminal child是否只可transfer一次；
- S4-1D每fresh owner只执行一次、success不隐式回`READY`是否比提前允许10-step复用更安全；
- 5 nonterminal + 5 terminal worker是否足够关闭两种fixture，且每个worker恰一次evaluation；
- 旧candidate-only `919,680 B`为何不足；12-worker A/B/C output+terminal V sidecar最低numeric raw
  `4,209,984 B`是否准确，禁止仅存hash+bounded projection是否合理；
- raw Tensor getter无法撤销已逃逸引用的反例是否成立；result/terminal child是否必须只提供exact sealed consumer/
  formal sink方法而不公开Tensor/dict/DLPack/generic callback；
- parent/child 9-state、9-event、14 legal transition与67 invalid组合是否覆盖embedded、child-first、parent-first和poison；
- 15-node seal DAG是否正确排除receipt/result、raw/summary和manifest self-hash循环；
- 36-buffer logical ledger、allocator delta与existing source lease lifetime三种口径是否被正确区分。

请特别找出文档中可能把“数学可行”误写为“已有production implementation”的地方。

### AC4：S4-2 sealed policy等价性

PASS要求：

- native dense oracle与compiled compressed evaluator是两个closed实现，不接受任意callback；
- sealed driver完整拥有live policy，而非无条件`for range(10)`；
- functional Adam/ExponentialLR、clamp、keep-best/prune/stop/patience/timeout/restore均可逐step比较；
- live two-group `batch_dim=2/0`、Adam defaults和7个CPU float32 step scalar进入ABI；
- pinned functional Adam对9×7 parameter的parameter/m/v/step共63组比较bit exact；
- transition使用out-of-place shadow，stable copy-commit或任何post-begin失败进入`POISONED_NO_RETRY`，不伪称
  content rollback可恢复`_version`/hidden state；
- A/B与B/C各5对/10 fresh worker，总计20 worker；
- terminal compressed→dense→compressed只在ordinal 9执行一次；
- preserved α不进入optimizer；
- `10/9/10`不被错误简化成`10/9/9`；
- fixed checkpoint=`[0,6,7,8,9]`且10-step patience `>10`不可达；不可达分支使用不同program hash的sealed
  synthetic fixture；
- S4-2不执行KFSB/commit/post/timing。

### AC5：S4-3 whole-core transaction与失败语义

请重点攻击：

1. existing device commit v1中commit copy和rollback copy是否确实都会增加PyTorch `_version`；
2. `PRECOMMIT_ABORTED_CLEAN`、`COMMIT_POISONED`、`POST_POISONED`、`QUEUE_POISONED`是否覆盖全部边界；
3. provider constructor failure是否在commit前；
4. official post failure是否必然在commit后；
5. host packet prune和intermediate container clear是否属于同一logical transaction；
6. net scratch是否可能被post/queue/next call继续读取；PlanV2对B0 observe与R/C normalize的variant policy是否安全；
7. `BatchedlA.from_net`导出的六条split-layer lA是否确实只是`gc_lA_from_net`清理的18条all-node lA子集；
8. 当前fixture `6 α + 12 intermediate + 18 lA = 36`是否应作为protocol fixture expectation而非generic schema常数；
9. pointer-swap是否被正确保持为未批准实验；
10. failure后禁止fallback/retry/queue continue是否足够fail closed；
11. query-scoped exclusive core-owner latch是否足以排除provider reentry、multi-core和stale preserve-mask；
12. B0 terminal transfer后被provider KFSB重新写成batch-24 residue、而当前R保持batch-12 stale scratch的phase事实
    是否成立；
13. R/C在native KFSB后normalize 36 path、B0仅observe residue是否是比“伪造三方scratch parity”更安全的设计；
14. logical bytes、unique storage、view alias与attribute sentinel是否被正确区分，尤其clear attribute不等于立即free。
15. current working-β `deepcopy`与temporary upper的allocator delta是否足以要求prepared bridge/persistent buffer；
16. fault rollback是否必须只恢复committed prefix并禁止写untouched suffix；
17. host packet的decision-stage `betas/history/depths`与final `thresholds/history/depths`是否应分列；
18. official post=`1`、query-total add=`2`、candidate-post add=`1`是否必须来自三个独立observer counter。

如果能设计出既恢复内容又保持`_version`/alias/consumer identity的更强方案，请作为替代设计说明，但不要把未证明方案
标成当前实现。

### AC6：core/post semantic coverage

PASS要求：

- 比较不止lower/decision，还覆盖6 lA、6 intermediates、六α/六β、history/depth/threshold、n_splits/n_verified、
  clip/xL/xU/batched_lA；
- official post的CPU materialization、α/β conversion和`max(lb,lb_last)`进入scope；
- B0、R、C的provider counter按variant区分；
- terminal lA one-shot并在KFSB后release；
- KFSB三candidate raw和72 child lower保留，不只存winner。

### AC7：S4-4 artifact/replay独立性

PASS要求：

- 6 B0/R/C全排列×3=`18 positive worker`，15个fault point各用独立fresh worker，总计33；
- 不从partial output resume；
- `.pt`不是唯一formal raw；
- tensor index + content-addressed raw-binary sidecar和stdlib decode足以恢复IEEE、logical path、view/alias；
- stdlib replayer不import BoundFlow/PyTorch/TVM/Numpy/αβ-CROWN，不复用production validator；
- source inventory以低扰动loaded module/native snapshot、declared critical source和compiled receipt共同覆盖，不用高扰动call trace；
- artifact内hash只形成self-consistency，真实性由外部anchor绑定manifest/semantic root/source/input/replayer；
- summary所有字段均能从protocol/source/raw重建；
- seal DAG无summary↔tamper↔manifest hash cycle，replay有derive/self-check/anchored-check三种模式；
- artifact无绝对本机路径/credential泄漏；
- failure artifact与positive worker分离但被同一manifest绑定。
- semantic occurrence floor必须独立重算为B0/R/C per-run=
  `3,829,232/3,897,248/2,537,888 B`、18 positive=`61,586,208 B`、15 fault=
  `24,209,400 B`、总计=`85,795,608 B`，且不得写成physical file size；
- final evidence seal DAG固定为16 nodes/36 edges，按施工包canonical payload重算hash=
  `01e179ea504f94c3e9720d5f63b318e34e912738d30c21d690f283b857ac491c`并完成topological sort；
- scratch按`core-entry/terminal-pre/terminal-post-transfer/post-KFSB/post-finalization/solver-return`投影，live
  finalization keys/sentinel、logical/unique bytes、alias与object/storage/data-pointer lineage均可从raw重建；
- B0六β container/96 B nonempty residue与R/C provider-net β inventory=`0`按variant核验；scratch count与production
  12-path count严格分离，且不把sentinel替换升级为即时CUDA memory free。

请评估33-worker设计是否过度或不足，以及B0/R/C在独立进程下如何证明同一个deterministic pre-state。

### AC8：tamper、claim与执行顺序

PASS要求：

- 96类攻击编号/分区完备，13 external-anchor、5 frozen-protocol、77 raw-semantic、1 execution-evidence；
- 除T19外95类攻击全部fully re-signed并被对应owner拒绝；T19必须诚实写
  `OFFLINE_UNATTESTABLE`，不得虚称offline artifact能密码学证明fresh process；
- raw semantic攻击同步更新payload/file/summary/manifest后仍由self-check拒绝；source/model authenticity替换由external
  anchor拒绝；process freshness只形成execution evidence，不虚称cryptographic attestation；
- 外审另造至少3个未预注册攻击仍能被设计覆盖；
- S4各级code/timing flag仍closed；
- S3外审批准前不得实施S4；
- S4-4通过后只允许另写S4-P timing预注册；
- 没有same-solver speedup、complete-query、10x或ASPLOS-ready claim漂移。

## 5. 重点源码入口

### BoundFlow

- `scripts/run_rvir_v4_live_return_capture.py`；
- `boundflow/runtime/rvir_v4_live_return.py`；
- `boundflow/runtime/fsg4_b3_device_atomic_commit.py`；
- `boundflow/runtime/fsg4_b3_device_live_return.py`；
- `boundflow/runtime/rvir_v4_native_backward_export.py`；
- `boundflow/runtime/rvir_v4_native_kfsb.py`；
- `boundflow/runtime/rvir_v4_native_optimizer.py`；
- `boundflow/runtime/asplos27_s2_crown_pipeline.py`；
- `boundflow/runtime/asplos27_s3_optimizer_pipeline.py`；
- `boundflow/backends/tvm/r31_bounded_arena.py`及相关R31B1/B2、D1C/D2B文件；
- `scripts/run_asplos27_s3_optimizer_artifact_v2.py`；
- `scripts/probe_asplos27_s3_optimizer_v2_tamper.py`；
- RVIR whole-core/five-fresh/live-return/KFSB artifact scripts。

### pinned external

- `/home/lee/Codes/alpha-beta-CROWN@e5c7e17`；
- `/home/lee/Codes/alpha-beta-CROWN/auto_LiRPA@5a098e8`；
- `complete_verifier/activation_split/update_bounds_phases.py`；
- `auto_LiRPA/auto_LiRPA/optimized_bounds.py`。

外审报告不得包含这些绝对路径作为artifact建议；它们只是当前机器的source定位。

## 6. 已执行的设计验证

本轮不是实现测试，但executor已核对设计事实：

- S4-0 snapshot/live反例：两个distinct view共享live storage，但snapshot按object id生成两个alias group并clone成
  两个storage；`OwnedProductionTensorV4`没有live version/storage字段；
- S4-0跨阶段反例：把全部source换成same-content clone后canonical projection/hash仍exact相等
  (`75d3252e...3c9f`)，但raw object/nonempty storage/empty object identity全部不等；因此receipt不能替代strong-ref lease；
- S4-0 V4源码审计：existing live helper接受`MutableMapping/Mapping`并调用`.get()`；pinned provider真实层级为
  built-in `dict/list/Tensor`，因此strict extractor可行且不应改变历史helper；
- S4-0 V4 construction model冻结5入口/18 validation stage/8 token order/12新detail code，formal算术=
  `6/12/12/8502/34008/2/24/68016/56`，canonical hash=
  `471424594fb4b6d017feac936a6005eb9d0451fd5579d026204ec952d0995239`；
- S4-1A plan-order=`17/19/23/25/28/31`，parameter/gradient=`4254/4254`、candidate storage=`16/34,080 B`；
- S4-1A CUDA owner probe为16/16 storage/view exact、source overlap 0；本机allocated delta=`39,936 B`只作诊断；
- DLPack collision probe复现old key equal、full key unequal、TVM direct noncontiguous reject，但old cache静默复用错误stride/content；
- 第一次cleanup只clear containers残留`1,024 B`；single owner+local nulling+raise outside except在retained exception下
  parameter/buffer/view三阶段allocated delta 0且`__context__ is None`；
- S4-1A V5 model冻结3 input/17 phase/20 detail、formal=`5+7=12`、negative=68、D2H=
  `32/85,056 B`、cumulative=`56/153,072 B`，hash=
  `8ad25c2abf1eb98c3b1097bf7acb46aba227f7e94f0c7c03169f39e8da409a9d`；
- lease guard probe：same-content clone、same-storage view和empty clone均以`LIVE_SOURCE_OBJECT_REPLACED`拒绝，in-place
  mutation以`LIVE_TENSOR_VERSION_MISMATCH`拒绝；
- pinned PyTorch/CUDA primitives probe：view共享storage `_cdata`/storage pointer但Tensor pointer受offset影响；empty pointer
  均为0而storage identity各异；弱引用在外部owner删除后失效；copy/deepcopy/pickle/asdict门禁均按冻结结果拒绝；
- version bypass probe：普通in-place增加`_version`，`.data`和DLPack alias写入改变content但原Tensor version不变，
  same-object `set_`保持object却更换storage；
- stable guard-order probe逐项得到object/storage/layout/version/content/admission/transfer/close/serialization预期detail；
- S4-1A formal owner probe：6 α+1 active β leaf、5 empty token，parameter/gradient=`4254/4254 elements`与
  `17016/17016 B`，base DLPack=`16/16`，storage独立；one-step Adam后pointer稳定且source hash/version不变；
- lease retention probe：12 source/8,502 elements/34,008 logical B，incremental allocation=0；外部owner删除后lease
  维持storage，close后allocated回baseline；
- prepare failure probe在parameter/buffer/view三点注入：3/3 `FAILED_CLOSED`、candidate refs=0、allocated delta=0、
  source hash/version不变、device/stream恢复、retry被拒且未调用empty-cache；
- β exact-width exploit：active β width从1扩为2并全量重签、history仍为1时existing snapshot validator接受，证明S4-0
  `beta_width == history_width`是必要门禁；
- formal snapshot mutable=`12`（6 α+6 β value）、source device metadata=`cuda:0`；snapshot/mapping/R31 plan hash分别为
  `2a775b...a256`/`cfcebf...f8df`/`39d617...910f`，plan source hash绑定mapping而非snapshot；
- live B0 terminal probe：α=`6 tensors/33,984 B`、intermediate=`12/299,712 B`、all-node lA=
  `18/471,984 B`，logical total=`805,680 B`、unique storage=`756,528 B`；两组lA shared-storage alias，六α、
  12 intermediate与六export lA return均为共享source storage/data-pointer的新view/object；
- live B0 solver-return：provider KFSB留下batch-24 scratch，连β unique=`2,829,600 B`；live current R：core-entry到
  solver-return保持batch-12 stale scratch，unique=`1,414,752 B`、provider-net β inventory=`0`；
- S4-0/S4-3相关production-state/pre-state/R31/whole-core/live-return/KFSB/commit/terminal targeted：
  `49 passed in 9.13s`；
- S4-4参考artifact相关targeted：历史`19 passed`；本次S3 v1/v2 artifact基础设施`4 passed`；
- CUDA探针：tensor content restore后`_version=0→1→2`；
- S3 v2 raw=`18 rows / 20,747,422 bytes`；
- old RVIR five-fresh=`10 .pt / 16,975,355 bytes`；
- S4-4 historical tamper inventory=`1..71`已由施工包96-case registry取代；order/positive/fault/total=
  `6/18/15/33`不变；
- normal S3 worker loaded snapshot=`101 BoundFlow core + 4 repo scripts + 559 TVM/TVM-FFI Python + 3 repo native`，
  canonical inventory hash=`421ce0b7...74c83`；逐call profile超过120秒后人工中止，禁止进入formal observer；
- legacy B0/R-C indexed raw-binary codec诊断=`225,621/206,180 B`；旧whole-core numeric raw规划区间
  `37.8557—47.5839 MiB`已被S4-4施工重算取代，不再作为active预算；
- Ainput exact class=`positive 8,689 / negative 9,137 / zero 606`；三元endpoint六dα overall max=
  `1.63912773132e-07`、active dβ max=`1.1920928955078125e-07`、sign mismatch均0；
- 第一版nonfinite CUDA/TIR探针FAIL：浮点`x==x && abs(x)!=Inf`把NaN误归zero；该失败已保留而非删除；
- 改用IEEE-754 float32 exponent位检查后边界探针PASS：`+0/-0`均为zero、正负subnormal保留符号、
  NaN/±Inf=`-128`、invalid输出canonical NaN；2 launch、5/5 DLPack pointer exact、workspace=0；
- 本轮独立16-element SM89 TIR得到unscheduled/scheduled/device source hash=`1bbd8e...c394/19a068...c5b4/
  b94c7f...c2eb`，只作diagnostic，不得复制进production receipt；
- fresh allocator probe得到selector/selected/combined allocated delta=`18,432/73,728/92,160 B`，reserved
  delta=`2,097,152 B`且两storage distinct；reserved不进入logical ledger；
- S4-1B0 construction model冻结backend无新IR、20 reason、16 test和future `5+1+5` workers，hash=
  `5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a`；
- formal in-memory CUDA/TIR pack/select逐位PASS，old binary误编码606 zero；selected hash=
  `7e95e075...39b652`，本次真实lower/upper derived-center hash=`2a3b69e1...5f003`，extra center tensor/view=0；
- old R31B2 module hash before/after均为`3871bf0e...be575`，S4 module/cache key均与旧binary隔离；
- max-finite与min-subnormal两组midpoint重结合反例均出现bit difference；
- S4-1B0相关历史S2/R31B2回归：`14 passed in 15.75s`；production code diff=`0`；
- 真实D2B coefficient pass抽取A18/A20/A24/A26/A29/Ainput，六组nonfinite=0；A26/A20由两个existing
  residual scratch直接承载，A18/A20/A24/Ainput与old bitmap逐位一致；
- seven-symbol gradient TIR corrected reference逐位max diff=0、sign exact、float64 max diff=
  `2.3575648810947314e-07`；7 launch、53 arguments、46/46 emitter view、workspace/alloc_buffer/dense output=0；
- 第一次gradient diagnostic因reference upstream广播成`[D,D,W]`而FAIL，已保留；显式reshape为`[D,1,1]`
  后才形成PASS；
- A=NaN、V=Inf、lower>upper、α越界、upstream NaN、index越界六类poison probe均产生qNaN；
- V/lA arena六slot interval不重叠、单storage且与四个coefficient/scratch storage分离；reverse same-stream
  read/copy/reuse simulation中gradient与terminal pretransform A均exact，dynamic allocation=0；
- S4-1B/1C相关R3 residual/D2B/R31B2/B4-B2/terminal回归：`29 passed in 23.99s`；
- S4-1D read-only admission/state-machine设计枚举=`14 cases`，model hash=
  `8942bb5970f268f47314265e0a1683947e7d5cddf6d421d3fd80cd778a9627eb`；
- 旧S4-1D ledger CUDA probe=`438,726/448,000/2,097,152 B`、36 buffers已被降为over-allocation diagnostic：
  它真的独立分配两个scratch，不能验证production arena-slice reuse；
- 源码与GPU storage probe确认residual11/6 storage `_cdata`分别等于arena1/0、offset=`6144/12288`、physical
  storage count=`2`、additional scratch bytes=`0`；
- descriptor机械集合重算=`base16/selected49/overlap5/passA30/S4-1B90/emitter46/base-overlap14/
  flat-bound-overlap12/emitter-new20/full110`；
- corrected ledger=`S4-1D 389,574 / S4-2 491,774 / S4-3 559,838 B`；selected alias失败时S4-1D=`463,302 B`；
- S2 selected graph after D2B prepare的Torch allocated/CUDA free prepare delta=`24,576/-25,165,824 B`，warm均0；
  该数只作本机诊断，证明future formal不能只看Torch allocator；
- S4-1B construction package=`847 lines`，canonical hash=
  `a9b1d90df3cd122eb43491d327432ded52f957928d77e1dbcf2e7286bc4a317d`；
- S4-1C construction package=`775 lines`，canonical hash=
  `ad8ea91c39419cbfef0cf3eaa8db7fc339e54798daecf67ca6d97254a9755b93`；
- Pass C mechanical state validator：nonterminal/terminal=`17/23` actions且正例0 error；copy31-before-dβ、
  ReLU28-before-copy和missing-dβ三类tamper均稳定拒绝；
- native terminal fixture复核lower=`[6,1]`、plan order=`17,19,23,25,28,31`、total lA=`37,464`；
- RTX 4060 synthetic arena lifecycle：six slot/one storage、interval non-overlap/full-cover、6/6 DLPack pointer exact，
  site31 copy-before-dβ数值反例成立，warm allocated/reserved=`0/0`；
- `tvm_ffi.use_torch_stream`内non-default Torch/FFI stream exact，退出后FFI stream恢复；
- S4-1C construction相关五文件回归=`23 passed in 17.68s`；production code diff=`0`；
- 旧S4-1D candidate-only raw预算：nonterminal/terminal/5+5=`17,040/166,896/919,680 B`，旧hash=
  `1e2aab39...e2f0d6`，现已明确降为历史估算；
- S4-1D construction package=`983 lines`，canonical hash=
  `76da18648d874dfec6e867deaf26122f093f8157c68967ef2d06afe362243cd1`；
- 修正formal topology=`12 workers`：nonterminal/terminal各覆盖A/B/C六全排列；candidate output=
  `1,103,616 B`、three-way output=`3,310,848 B`、terminal V sidecar=`899,136 B`、minimum numeric raw=
  `4,209,984 B`；
- capability model=`9 states/9 events/14 legal/67 invalid`，hash=`963e723f...599d`；5条完整close trace通过；
- component→execution→artifact seal DAG=`15 nodes`且无环，hash=`444a98d8...bca9`；
- raw Tensor escape反例：lease property close后拒绝，但先前escaped Tensor仍可读sum=`28.0`，证明public getter
  不能支持revocable claim；
- 本次S4-1D construction依赖回归：`37 passed in 20.65s`；production code diff=`0`；
- 本次S4-2 readiness依赖回归：`41 passed in 26.63s`；production code diff=`0`；
- S4-2/S4-3 ledger检查：`491,774/559,838 B`、checkpoint `[0,6,7,8,9]`、10-step patience `>10`
  不可达；本版34份必读文档全部存在；
- S4-2 construction package=`1184 lines`、file hash=
  `b473e31bd00df48499288f60b4f92b8230a69cc5a22ba6762972e5c8391524e3`；production code diff=`0`；
- policy run机械模型=`16 states/16 events/32 legal/224 invalid`，canonical hash=
  `75e0c1b7aa4fc9bd439d15af41f7c1b86c8c4c7f732ca6bb55108488fa743279`；
- S4-2 formal修正为A/B与B/C各6对、正反各3，共24 fresh subprocess；A/B/C mandatory tensor floor=
  `2,837,288/2,871,296/1,511,936 B`，`6A+12B+6C=60,550,896 B`；
- result policy消费改为opaque exact path；10个one-shot evaluator generation、9次controlled re-arm、
  evaluation-input/optimizer-mutation/storage-commit三轴和28项commit cursor进入施工合同；
- terminal-best/lA门禁要求六domain均为ordinal9 best；earlier-best synthetic只验证restore，不能把ordinal9 lA交S4-3；
- `491,774 B`降为known tensor/base lower bound，完整policy storage和peak必须在implementation receipt中补齐；
- 本次S4-2 construction依赖回归=`36 passed, 1 deselected`；被排除的历史S2 artifact strict replay仅为
  `4.245381964572069 vs 4.24538196457207`末位ULP，不改写冻结raw；
- true B3-C live assembly：intermediate container六key从entry到worker return保持；working-β allocator
  delta/peak=`1,024/2,048 B`，assembly delta=`1,536 B`；12 intermediate和6 working α均exact alias source/candidate；
- true B3-C official post=`1`，query-total domain add=`2`、candidate-post add=`1`；post CUDA allocated delta=`0`；
- first-copy fault后current v1 copy seam=`13`，version delta为一条`+2`、11条untouched `+1`，证明prefix-only
  restore与poison状态是必要修正；
- S4-3旧readiness粗粒度transition hash=`833e8a9b...6ccaf5`已被施工包取代；旧live诊断事实仍保留；
- S4-3 construction package=`1134 lines`、file hash=
  `0a2a9612dbe401fd5c1afb23646eb3ad11c6958dc0c95f7634bf9ff3b63644a6`；production code diff=`0`；
- whole-core机械模型=`23 states/22 events/40 legal/466 invalid`，canonical hash=
  `6ed3d2fd946aaa0f6342f637a4754cc50eeec96e24392ed3b42adbbf92a3388a`；
- logical commit=`12 device + host + container = 14`；36项provider scratch finalization和queue partial mutation均进入
  poison合同；
- S4-3 formal修正为R/C 6对、`RC/CR=3/3`、12 fresh；downstream/R/C per-run tensor occurrence=
  `1,025,952/3,897,248/2,537,888 B`，总计`38,610,816 B`；
- `559,838 B`降为known tensor/base lower bound，不再称total-new或完整footprint；
- 本次S4-3 construction依赖回归=`63 passed in 8.96s`；production code diff=`0`；
- S4-4 construction package=`1133 lines`、file hash=
  `13eede1ca40931cefa0500ca8079fd9aac2210f507ff354ab5d8d1efbd171bc3`；production code diff=`0`；
- 15-fault registry hash=`4b69d50391ff84d42a0d6ea5fb8c43d7b6f8040db4de5cd43d56cc2848256330`；
- 96-tamper registry hash=`5fdfa8bcbc41516807f7eef220ede181253ade7b0c42fa31a4620dcdf37f7d05`，
  layer counts=`13/5/77/1`，95 reject + 1 `OFFLINE_UNATTESTABLE`；
- final evidence DAG=`16 nodes/36 edges`、acyclic、hash=
  `01e179ea504f94c3e9720d5f63b318e34e912738d30c21d690f283b857ac491c`；
- B0/R/C per-run=`3,829,232/3,897,248/2,537,888 B`，positive/fault/all=
  `61,586,208/24,209,400/85,795,608 B`，均为semantic occurrence而非physical size；
- 本次S4-4 construction依赖回归=`34 passed in 8.58s`；registry/DAG/bytes由stdlib独立重算；
- existing live-return/device-commit targeted：`12 passed in 6.58s`；production code diff=`0`；
- S4-0 construction依赖回归：`32 passed in 6.93s`；canonical JSON stdlib重算PASS；production code diff=`0`；
- S4-1A construction依赖回归：`48 passed in 189.07s`；canonical JSON/bytes stdlib重算PASS；production code diff=`0`；
- `git diff --check`、DocOps lint：PASS；S3 exchange状态仍为`ready_for_audit/r001`。

这些只证明设计输入和历史基础设施仍存在，不证明S4-0—S4-4实现通过。

## 7. 建议外审操作

```bash
git diff --stat ebf45cc72438141d8f0b35dadfd5cf774d7e753f..c750fafdde8435f56146294faf509221a780057d
git diff --check ebf45cc72438141d8f0b35dadfd5cf774d7e753f..c750fafdde8435f56146294faf509221a780057d

source env.sh
/home/lee/Codes/alpha-beta-CROWN/.venv/bin/python -m pytest -q \
  tests/test_rvir_v4_live_return.py \
  tests/test_rvir_v4_native_kfsb.py \
  tests/test_fsg4_b3_device_atomic_commit.py \
  tests/test_fsg4_b4a_terminal_lower_adjoint_handoff.py \
  tests/test_rvir_v4_native_optimizer.py \
  tests/test_rvir_v4_optimizer_step_source_parity.py \
  tests/test_r3_compiled_p_alpha_vjp.py \
  tests/test_r3_full_lower_forward_tir.py \
  tests/test_asplos27_s2_crown_pipeline.py
```

另外请用自己的短脚本：

- 重算mutable inventory和memory ledger；
- 从S4-2施工包重建16-state/16-event transition表，确认32 legal、224 invalid及canonical hash；
- 独立重算A/B/C per-run raw floor与`6A+12B+6C=60,550,896 B`，检查unprojected shadow/final restore未漏；
- 攻击5-pair 3/2顺序不平衡与跨A/B、B/C复用B worker，确认24-worker inventory拒绝；
- 构造earlier-best domain，确认policy restore可以测试但ordinal9 terminal lA不能进入S4-3；
- 在28-item commit的parameter/m/v/step各区段注入fault，确认scheduler不提前commit且run poison；
- 试图从opaque policy consume返回/保留raw Tensor或跨generation re-arm，确认exact type/retention gate拒绝；
- 从S4-1B0施工包提取canonical JSON并重算`5056d3...cc2a`；确认backend-local dataclass不是新IR；
- fresh分配18,432 int8和18,432 float32，独立核对logical/allocated/reserved并区分S4-1A base view；
- 构造coefficient arena live reader/旧DLPack descriptor，攻击pass A→B→C alias proof；失败时要求ledger加73,728 B；
- 构造cache entry：改变mutable hit count应保持module identity；篡改stored device source必须在hit revalidation拒绝；
- 检查warm receipt没有tensor content hash/class count或隐式`.cpu()`同步；
- 从S4-1A施工包提取完整canonical JSON并重算`8ad25c...09a9d`；
- 构造same-pointer/shape different-stride view，先复现old key静默碰撞，再验证V5 lookup-before-cache拒绝；
- 在parameter/gradient/output/view/roundtrip/receipt/adoption七点故障，并保留stable exception+traceback，检查allocated=entry；
- 重算`24+8=32`、`68,016+17,040=85,056`及与S4-0累计`56/153,072`；
- 从S4-0施工包提取完整canonical JSON，以`sort_keys=True,separators=(',', ':')`重算construction hash；
- 亲读existing live helper和pinned provider alpha/beta state容器，攻击dict/list/Tensor subclass并确认读取前拒绝；
- 在两次capture间分别替换object/storage、改变version/content、切stream，确认稳定`LIVE_SOURCE_READ_RACE`或对应首要reason；
- 重算`12 tensors × 2`和`8,502 × 4 × 2`，检查receipt D2H accounting不能被全重签篡改；
- 亲读`OwnedProductionTensorV4.own`/`ProductionStateBuilderV4`，自建distinct-view shared-storage反例，核对
  snapshot alias/object/storage/version能力边界；
- 重建R31 plan，确认`source_state_hash==dense mapping hash!=snapshot hash`，检查plan/snapshot projection设计；
- 构造β width大于history但前缀相同的tamper，确认V2要求exact width；
- 独立区分`BatchedlA.from_net(get_splittable_activations)`与`gc_lA_from_net(net.nodes())`，核对当前fixture为6 export/
  18 all-node path而不是6/6；
- 独立重算terminal logical/unique storage与两个lA alias group，确认empty `data_ptr=0`不算alias；
- 检查B0 post-KFSB batch-24 residue、current R batch-12 stale和PlanV2 R/C normalization的phase/owner逻辑；
- 独立复现旧二元endpoint的site19反例：约`1.156e-3/9`；再仅改zero→center，复核site19
  `4.2375177145e-08/0`及六site overall `<=1.63912773132e-07/0`；
- 从formal Ainput独立统计positive/negative/zero，并亲读provider `center-abs*radius`与PyTorch zero-subgradient；
- 复核新增center tensor是否确属多余；验证S4新symbol而非修改v1能保持S2/S3 artifact identity；
- 独立复现浮点`x==x` classifier失败，再以uint32 exponent mask重编译；不要只采信最终PASS；
- 用max-finite与min-subnormal复现midpoint重结合位差，检查TIR attr/cache/receipt是否绑定operation order；
- 自建cache collision：用旧binary key、删policy字段或改threads后尝试命中新ternary module，要求miss/拒绝；
- 独立核对pack/select为2 launch、5 unique view、6 argument occurrence，且无physical center和workspace；
- 对五张binary selector分别注入NaN/Inf，确认pack=`-128`且consumer输出qNaN，不得静默选择upper/lower；
- 用invalid index/location攻击gradient TIR，确认先safe-read再poison而非OOB；同时确认safe clamp不等于admit；
- 独立重算7 launch/53 arguments/emitter46/base16/local48，并确认48只属于base+emitter局部scope；
- 从S4-1B施工包提取canonical JSON并重算`a9b1d9...317d`；
- 用集合而非加法猜测重算base16/selected49/overlap5/passA30/S4-1B90/emitter-new20/full110；
- 亲读D1C scratch slice定义并用storage identity/offset证明additional bytes=0；再攻击把slice改成clone，要求ledger变化；
- 独立重算`389,574/491,774/559,838`及alias失败`463,302`，检查model/fixed/workspace排除边界；
- 攻击Ainput pack/concretize/rebind顺序、stale descriptor generation和VM live reader；
- 分别观测Torch allocated/reserved与CUDA driver free，确认prepared VM/cuDNN footprint没有被warm=0隐去；
- 复现diagnostic reference的`[D,1]`广播错误，再以`[D,1,1]`修正，禁止把第一次FAIL归咎candidate；
- 检查β sign int8、metadata 2,862 B及六slot reverse read→copy→transform phase；
- 从S4-1C施工包提取canonical JSON并重算`ad8ea9...5b93`，独立生成17/23 action表；
- 将site31 copy与dβ交换、将ReLU28与copy交换、删除dβ31，确认phase validator分别拒绝；
- fresh CUDA arena重建六interval，核对one-storage/non-overlap/full-cover、emitter/terminal view pointer和warm 0/0；
- 在`use_torch_stream`前/中/后读取FFI stream，确认中间exact、退出恢复；
- 独立重算110 argument DLPack、6 result ordinary views、13 module symbols三种互不混淆的口径；
- 检查runtime receipt没有V/lA content hash或D2H，formal sidecar在V覆盖前与lA copy后分别绑定raw；
- 从S4-1D施工包提取canonical JSON并重算`76da18...3cd1`；独立重算12-worker raw四项算术；
- 枚举9 states×9 events，确认只有14 legal transition，其余67项reject且state不变；分别跑embedded close、
  child-first、parent-first、nonterminal和post-failure五条完整trace；
- 构造naive lease raw Tensor getter，close wrapper后验证escaped Tensor仍可访问；据此审查API是否完全禁止getter/
  tuple/dict/DLPack/generic callback；
- 亲读future sealed policy/KFSB/formal consumer边界，攻击subclass、duck type、consumer字段保留与raw Tensor return；
- 独立拓扑排序15-node seal DAG，检查result↔receipt、raw↔summary与manifest self-hash均无循环；
- 建立12 worker protocol，确认两fixture各恰含ABC/ACB/BAC/BCA/CAB/CBA且A/B/C mutable storage独立；
- 重算candidate=`1,103,616`、three-way=`3,310,848`、V-sidecar=`899,136`、minimum=`4,209,984 B`；
- 检查historical S4-1D tamper编号1—68，并另从S4-4施工包独立提取T01—T96；
- 用CUDA tensor验证commit+restore后的`_version`；
- 亲读production checkpoint条件，独立重算fixed ordinal `[0,6,7,8,9]`和patience reachability；
- 以相同state/defaults独立比较functional Adam和live Adam，不采信summary的63/63；
- 重算S4-2 current/keep-best/shadow与CPU step ledger，检查fixed intermediate clone是否被安全移除；
- 从S4-3施工包重建23-state/22-event模型，确认40 legal、466 invalid及`6ed3d2fd...a3388a`；
- 在terminal claim后、KFSB、scratch finalization、core staging、device/host/container、post、add、check-worst各点注入
  fault，确认只在claim前可clean，其余分别进入staging/commit/post/queue poison；
- 对36项provider scratch finalization逐ordinal fault，确认partial sentinel state不触发provider reentry或retry；
- 对12条device copy逐ordinal fault，检查只逆序恢复successful prefix、untouched suffix write=0且`_version`不可逆；
- 亲读`BatchedDomainList.add`并建立mutation-unit before/after清单；add中途与check-worst故障后均不得second add；
- 独立重算whole-core/post历史semantic occurrence=`821,976/50,976 B`、transaction=`102,024 B`、queue=
  `50,976 B`和总downstream=`1,025,952 B`；再核R/C 6-pair总计`38,610,816 B`；
- 检查artifact同时披露semantic occurrence、unique content与physical file bytes，不把content-addressed去重冒充漏raw；
- 从S4-4施工包独立生成15-fault registry，按冻结canonical JSON重算`4b69d503...256330`；
- 独立提取96-case registry，核对13/5/77/1分层与`5fdfa8bc...7d05`，确认T19只形成
  `OFFLINE_UNATTESTABLE`；
- 重建16-node/36-edge seal DAG、做topological sort并重算`01e179ea...491c`；
- 独立重算B0/R/C per-run、18 positive、15 fault及总semantic occurrence，禁止采信summary数字；
- 亲读provider core/post确认clear/prune/post顺序；
- 搜索S4文档所有`claimed/open/validated`词，核对没有implementation或performance漂移。

## 8. 外审必须回答的问题

1. 是否存在blocker/major，使S4-0在S3批准后仍不能开工？
2. 三元box endpoint是否完整解释旧site19反例；derived-center且不新增tensor是否为正确最小ABI？
3. IEEE exponent classifier、canonical NaN、operation-order绑定和独立cache key是否关闭了TIR ABI歧义？
4. two-launch/5-view设计应保持独立到correctness closure，还是有充分理由在第一版就与相邻kernel融合？
5. 五binary selector sentinel、safe-index poison和finite poison是否足以关闭silent-zero/OOB边界？
6. emitter46/base+emitter local48、β int8 metadata和七launch物理账是否准确？
7. six-site V→gradient→terminal-lA alias是否有无法由phase state解决的lifetime冲突？
8. net scratch是否必须成为第13+条production数值path，还是应保持为独立phase-aware lifetime/finalization transaction？
9. commit/post/queue failure分别poison且prefix-only restore是否是可实现的最强安全语义？
10. B0/R/C 18 positive加15 isolated fault是否足够证明reference/candidate/failure，不依赖历史`.pt`？
11. indexed-binary stdlib raw schema是否缺dtype、negative-zero、NaN payload、alias或view metadata？
12. loaded snapshot + declared critical source + native/compiled receipt是否是低扰动且足够保守的source closure？
13. snapshot semantic truth、瞬时live observation和S4-1A prepared owner三段边界是否仍遗漏live alias/version race？
14. 96类tamper还缺哪类可全重签semantic attack；13/5/77/1分层是否合理？
15. S4-2是否仍遗漏optimizer/policy state、可达分支、失败owner或raw字段？
16. prepared working-β共享immutable location/sign到post结束是否安全，还是必须为72 B metadata建persistent copy？
17. query-total add=2与candidate-post add=1的receipt边界是否足以避免queue claim混淆？
18. artifact self-consistency与external authenticity anchor的分层是否正确；还有哪类替换不能被anchor绑定？
19. semantic-root→summary→tamper→stdout→manifest→anchor的seal DAG是否仍有循环或未绑定产物？
20. derive/self-check/anchored-check三模式和四类enforcement layer是否避免把digest拒绝伪称semantic拒绝？
21. `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT`是否是生成端唯一合法PASS状态？
22. 是否同意当前唯一执行顺序，不开放S4-P timing？
23. `exact_call_id`的receipt-hash/private-raw分层是否足以支持S4-0 phase identity且不泄漏query字符串？
24. strict built-in provider extraction是否兼容pinned source，并能在读取custom Mapping方法前fail closed？
25. 双capture加后续S4-1A/S4-3 revalidation是否关闭正确窗口；还需要哪种锁或generation owner？
26. 24条logical D2H/`68,016 B`是否为正确语义账，零candidate kernel/allocation措辞是否避免claim漂移？
27. 56类negative和envelope+residual reason normalization是否足以机械稳定，缺哪类复合攻击？
28. 是否同意S4-0只claim local single-transfer，process-global exclusivity必须等S4-3 latch？
29. S4-1A移出evaluator/Adam/terminal后是否仍完整，还是存在必须在buffer prepare阶段拥有的真实事务？
30. caller不再提供device/stream是否正确；runtime观察与ticket绑定是否仍可被伪造或漂移？
31. V5 DLPack full key、contiguous-before-lookup和roundtrip全metadata检查是否关闭stride collision？
32. S4-1A `32/85,056 B`与累计`56/153,072 B`是否正确；还有哪次content validation被漏计或重复？
33. retained traceback清理方案是否足够；是否还需weakref/finalizer或独立resource frame？
34. single resource-owner adoption是否真正避免double/no-owner；adoption fault怎样独立注入最可信？
35. uninitialized gradient/lower的read-forbidden状态是否需要poison fill，若需要其kernel/bytes应归哪个阶段？
36. 68 negative和12-process formal是否适当；是否同意S4-1A仍不得claim provider mapping/global exclusivity？
37. 是否同意S4-1B0不新增endpoint IR，只用backend-local build/schedule metadata和existing compiler Schedule owner？
38. isolated 5 view与S4-1A 16 base view是否零重叠，`92,160 B` output账是否正确？
39. 是否同意`389,574 B`是conditional ledger；S4-1B phase alias未证时必须加回`73,728 B`至`463,302 B`？
40. pass A selector capture→B selected E0→C coefficient recompute是否足以证明arena reuse，缺哪类reader/event门禁？
41. lookup/module/cache-observation/formal-sidecar四层拆分是否正确，device source为何不能进入首次lookup key？
42. warm路径删除content hash/class count是否会削弱fail closed；descriptor/generation/device poison/final-finite是否足够？
43. 20 reason、16 test和future 5 positive+1 cache+5 fault是否足以关闭S4-1B0？
44. 90个S4-1B与110个full A/B/C argument descriptor重算是否准确；旧48是否已被正确降为局部scope？
45. pass A 19-action和A29/A26/A24/A20/A18/Ainput capture位置是否有遗漏或错误先后？
46. selected graph 42-read+7-write、active-α和six caller-owned slot ABI是否能直接由现有S2扩展？
47. residual scratch与coefficient arena的storage/offset复用是否安全，旧49,152 B是否确属重复计账？
48. `389,574/491,774/559,838 B`与alias-failure `463,302 B`是否逐项成立，还有哪项workspace/metadata漏算？
49. coefficient→selected-input→coefficient generation转换还缺什么reader/event/descriptor revocation门禁？
50. immutable module receipt、mutable cache observation和VM result token owner是否足以避免warm view/object泄漏？
51. 55类S4-1B negative和five-fresh formal是否足够，外审还能构造哪类全重签攻击？
52. nonterminal/terminal Pass C=`17/23` action重算是否准确，是否漏掉必要producer、transform或finalize动作？
53. site31双reader顺序能否由逐reader phase稳定保证，copy-before-dβ反例是否足以证明单布尔位不安全？
54. 7 gradient+6 terminal-copy=`13` symbols的首版correctness拆分是否合理，还是copy应作为runtime primitive而非symbol？
55. terminal copy新增descriptor/storage=`0`与single 37,464-element arena alias是否完整，有无隐藏consumer迫使独立arena？
56. full argument DLPack=`110`、result extra ordinary Torch view=`6`是否准确，site31 view reuse会否与lease phase冲突？
57. runtime O(1) receipt/formal pre-overwrite V sidecar/post-copy lA sidecar能否同时保证性能边界与独立审计？
58. native six-clone handoff迁移为one-arena lease时，plan order、spec axis、KFSB消费与post lifetime还有何遗漏？
59. 62类S4-1C negative和5+5 fresh formal是否充分；外审还能构造哪类phase/stream/storage全重签攻击？
60. 是否同意旧`919,680 B`只能称5+5 candidate-only估算，A/B/C独立复核必须保存三方完整output？
61. 两fixture各六全排列、12 fresh是否为合理最小formal拓扑；是否仍需隔离A/B/C到独立subprocess？
62. `4,209,984 B` minimum numeric raw是否完整；terminal V sidecar只存6个terminal worker是否足够依赖前序component closure？
63. raw Tensor escape反例是否否定public view getter；opaque exact sealed-consumer capability还有何可实现的更强防逃逸机制？
64. exact class/implementation hash与post-consume retention audit是否足够，consumer内部异常或CUDA异步引用怎样关闭？
65. 9-state/14-transition模型是否遗漏transfer后parent仍open、child-first、embedded close或并发close状态？
66. 15-node seal DAG是否无循环；execution semantic root、result runtime ref、raw/summary/manifest边界是否正确？
67. 86类S4-1D negative与至少12类fully re-signed tamper是否足够，外审还能构造哪类capability/raw/DAG攻击？
68. 是否同意S4-1D success不回READY，10次re-arm只能由S4-2新合同拥有，而不是提前塞入evaluator？
69. run-level evaluator family是否在不复活S4-1D one-shot的前提下正确保留module/arena/view owner并发行10代？
70. opaque exact consume是否禁止raw Tensor escape；terminal consume原子转移child时如何审计consumer不保留source storage？
71. evaluation-input、optimizer mutation与storage commit三轴是否完整；earlier-best restore需要怎样的version receipt？
72. 28项commit cursor与scheduler-after-tensor顺序是否足够；哪个failure point仍可能被误写为clean reject？
73. terminal-best/lA同轮次门禁是否必须；未来支持earlier-best应选择lA checkpoint还是restore后rerun？
74. 16-state/32-transition模型与224 invalid组合是否完整；terminal handoff失败进入poison是否过强或不足？
75. 24-worker 6-pair balanced topology是否合理；A/B/B/C的B worker必须独立重跑是否正确？
76. `60,550,896 B` mandatory transition-tensor floor是否逐项成立；policy projection、source、receipt和完整storage
    还应强制加入哪些raw/ledger字段？
77. whole-core transaction从terminal claim开始是否必要；claim后尚未device mutation的failure为何不能clean retry？
78. 36项scratch finalization的attribute inventory、partial failure和provider β=0合同是否完整？
79. 23-state/22-event/40-legal/466-invalid模型是否遗漏scratch、post、check-worst或owner close转换？
80. 14-step commit是否正确；host/container mutation失败后best-effort内容恢复为何仍必须commit poison？
81. prefix-only inverse restore、untouched suffix write=0与逐path version receipt是否充分关闭blanket rollback错误？
82. fixed `BatchedDomainList.add` mutation-unit inventory是否完整；生产O(1)与formal/fault全快照分层是否合理？
83. add成功但check-worst失败时，queue已改变且query必须终止的判断是否正确？
84. official post/candidate add/query total add/check-worst=`1/1/2/1`四个counter是否足以防止summary混淆？
85. R/C 6对/12 fresh、`RC/CR=3/3`是否是S4-3 correctness的合理最小拓扑？
86. `1,025,952` downstream、R/C per-run `3,897,248/2,537,888`与总计`38,610,816 B`是否逐项成立？
87. `559,838 B`作为known base lower bound的排除项是否完整；实现receipt还需冻结哪些physical peak测点？
88. 15-fault registry是否覆盖正确terminal state；F01 clean与F02—F15 poison的边界是否有误？
89. B0/R/C per-run=`3,829,232/3,897,248/2,537,888 B`是否逐项成立，B0是否正确排除不存在的candidate snapshot？
90. positive/fault/all=`61,586,208/24,209,400/85,795,608 B`是否完整，是否仍有double count或漏项？
91. semantic occurrence、unique content和physical artifact bytes是否已充分分层，content dedup是否可能掩盖缺raw？
92. 16-node/36-edge seal DAG按冻结canonical编码是否重算为`01e179ea...491c`且无环？
93. semantic root、summary、tamper、stdout、manifest、external anchor、anchored record的seal顺序是否正确？
94. 96-case registry是否编号连续、无重复、层数恰为13/5/77/1，hash是否为`5fdfa8bc...7d05`？
95. T19返回`OFFLINE_UNATTESTABLE`是否是唯一诚实结论；现场外审怎样补足fresh-process evidence？
96. 除T19外95类fully re-signed攻击是否都有确定拒绝层；至少再构造3类未列攻击验证覆盖性。
97. stdlib replay是否能独立重建三个version轴、23-state whole-core、fault terminal和所有summary，不复用production validator？
98. 当前protocol structure projection是否明确不等于future full protocol hash；实现时还必须绑定哪些source/input/numeric/schema值？
99. 是否同意S4-4施工合同仍是design-only，S3批准前不开放S4代码/formal/timing/performance？

## 9. 输出格式

请输出：

- verdict：approve / approve-with-minor / request-changes；
- blocker / major / minor / info数量；
- AC1—AC8逐项PASS/FAIL及证据；
- 每项finding稳定ID、严重度、源码/文档位置、可复现实验和建议修订；
- 独立重算的state/memory/counter数字；
- 至少3个外审自建攻击及预期拒绝层；
- 明确是否同意：**S3批准后只开放S4-0，实现仍必须逐级关闭；S4-P继续关闭**。

不要输出“性能提升已验证”或“ASPLOS-ready”；本轮没有这种证据。
