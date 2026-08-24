# gemini_doc 导引（BoundFlow 工程文档索引）

R3-1b2 compiled P-alpha VJP已正式关闭：source=`12402da`，lower/dα max diff=
`3.81470e-6/6.14673e-8`、sign exact；2 scratch、saved dense A=0、warm allocation=0，12/12
全重签tamper拒绝。当前只开放b3 five-fresh correctness/memory；R3-1仍未admit且不计时。见
`BOUNDFLOW_R3_1B2_COMPILED_P_ALPHA_VJP_FORMAL_CLOSURE_2026_08_25.md`。

R3-1b2 compiled P-alpha VJP已实现待formal：单worker lower/dα max diff=
`3.93391e-6/6.14673e-8`、sign exact，2 coefficient scratch、4 sign bitmap、saved dense A=0、
warm allocation=0。当前只允许clean-source artifact/replay/tamper；尚未关闭b2或开放b3。见
`BOUNDFLOW_R3_1B2_COMPILED_P_ALPHA_VJP_IMPLEMENTATION_2026_08_25.md`。

R3-1b2 P-alpha VJP 数学归约已通过：相对native autograd max diff=`4.47035e-8`、sign exact、
nonzero=`281/281`，且公式不要求跨forward/backward保存dense A。当前只开放checkpoint/sign TIR与
mandatory custom backward；尚无compiled VJP、memory或性能claim。见
`BOUNDFLOW_R3_1B2_P_ALPHA_VJP_MATH_REDUCTION_2026_08_25.md`。

R3-1b1 compiled full-lower已正式关闭：fresh source=`bdfa53d`、lower max diff=`3.8147e-6`、
15 launches、2 scratch×73,728 B、70/70 DLPack、warm allocation=0，10/10全重签tamper拒绝。
只开放b2 compiled P-alpha VJP；R3-1仍未admit且不计时。见
`BOUNDFLOW_R3_1B1_COMPILED_FULL_LOWER_FORMAL_CLOSURE_2026_08_25.md`。

R3-1b0 exact trace/liveness已正式关闭：12-step、2 residual fused region、2 scratch×73,728 B，
clean-source replay与6/6 tamper通过。该“只开放b1”状态已由上方b1 closure取代。见
`BOUNDFLOW_R3_1B0_TRACE_LIVENESS_FORMAL_CLOSURE_2026_08_25.md`。

R3-1 M0 Python rematerialization 已正式 NO-GO：5对独立进程语义/结构全过，lower/dα最大差=
`4.7684e-7/2.3283e-10`，但peak allocated=`1.1181179x`且compiled bounded-arena=0/5；reserved=
`1.0x`。R3-1 admission=false、R3-2A关闭，无性能claim。下一只允许预注册R3-1b compiled
recurrence。见`BOUNDFLOW_R3_1_M0_PYTHON_REMATERIALIZATION_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。
R3-1b bounded-arena compiled recurrence 已预注册；该旧“只开放b0”状态已由上方b0/b1 closure取代。见
`BOUNDFLOW_R3_1B_BOUNDED_ARENA_COMPILED_RECURRENCE_PLAN_2026_08_25.md`。

R3-0 compressed-alpha v2已正式关闭：source=`8941e66`，P-anchor alpha binding exact=
`[2,1,6,86]`，saved logical/unique=`207888/109584 B`；replay逐字节一致、12/12全重签tamper拒绝。
v1只保留通用合同机制历史证据。当前只开放R3-1一个evaluation mandatory custom backward
correctness，不计时。见`BOUNDFLOW_R3_0_COMPRESSED_ALPHA_V2_FORMAL_CLOSURE_2026_08_25.md`。

R3-0 v1复核发现formal fixture的alpha binding是dense native而非P-anchor production compressed
`[2,1,6,86]`。验证器机制保留，但R3-1 admission暂时撤回；修正已实现，下一步生成v2
artifact/replay/tamper。该待v2状态已由上方正式closure取代。见
`BOUNDFLOW_R3_0_COMPRESSED_ALPHA_FIX_CHANGELOG_2026_08_25.md`。

R3-0 已正式关闭为`VALIDATED-R3-0-CONTRACT`：clean-source artifact/replay通过，8 nodes/8 edges、
2 scratch、saved coefficient/dense escape/context=`0/0/0`，12/12 fully re-signed tamper拒绝。
这仍不是production/custom-backward/performance。下一只开放R3-1 `25/Conv_8`一个evaluation的
mandatory custom-backward correctness，不计时。见
`BOUNDFLOW_R3_0_STRUCTURED_OWNER_FORMAL_CLOSURE_2026_08_25.md`。

以下R3-0 source-only状态为历史，已由上方formal closure取代。R3-0 structured-owner contract source 已实现：typed DAG、Template/Instance、closure/fanout/BiasSplit、
两scratch liveness、dense-escape/context/saved-state validators及artifact/replay/tamper runner，40 tests
通过。当前只到`IMPLEMENTED-R3-0-PENDING-CLEAN-SOURCE-FORMAL`；R3-1和production/timing仍关闭。
见`BOUNDFLOW_R3_0_STRUCTURED_OWNER_CONTRACT_IMPLEMENTATION_CHANGELOG_2026_08_25.md`。

CIBC R1-A 已于2026-08-25按冻结协议正式关闭为
`VALIDATED-NO-GO-R1A-ATTRIBUTION`：六组Nsight profile均完整重建42 graph nodes/4400 owner events，
但profile perturbation=`1.1838—1.1859x`、`0/6`通过`[0.95,1.05]`，clock receipt仅`3/6`。
因此R1-B/R1-C/R1-D/R2关闭，不形成op/query share或性能claim。当前下一动作只开放R3-0合同和
静态验证器，不接production、不计时。正式结果、artifact、replay/tamper与claim边界见
`BOUNDFLOW_CIBC_R1_A_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。

R0审计卫生已于2026-08-25关闭；以下R1预注册状态为历史，已由上方正式NO-GO取代。此前唯一
动作是实现R1-0时钟校准、topology/schema与negative tests，再在clean source上做runner smoke。
R1强制按same-solver op type记录`q_B3,k`，并从exact production signature重测`G_query,k`；独立
ResNet2B IBP graph的`2.45631x`不得代填。主计划与修改记录见
`BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`、
`BOUNDFLOW_R0_HYGIENE_R1_PREREGISTRATION_CHANGELOG_2026_08_25.md`。

当前权威执行顺序已按外部建议修正：R0审计卫生→R1协议/目标冻结→CIBC-G1只读归因→same-solver
eligible-IBP share与benchmark admission→数学可达的R2→B0/B3/cumulative candidate三方formal。
query qualification/research与queue research分别冻结为B0-relative `1.00/1.15/1.20x`，任何graph/
query/queue share不得跨scope代入；R1新增CUPTI↔host/NVTX时钟校准receipt。R3-1必须有backward，
原R3-2拆为2A轨迹正确性和2B wrapper timing；R3设计评审可并行但R3-0实现仍关闭。修订总账见
`BOUNDFLOW_RECOVERY_PLAN_TARGET_SCOPE_R3_STAGE_CORRECTION_CHANGELOG_2026_08_24.md`。

α-CROWN 的未来恢复路线已完成独立重设计预注册，状态=
`PREREGISTERED-DESIGN-REVIEW-ONLY-R3-SO-CVJP`。它采用 closed lower region 的 first-class DAG owner
与 region-level single custom VJP：Python/IR/autograd 边界不传 dense A，backward 默认重算，
kernel 只允许 bounded transient scratch；旧 B4-C2、逐层 Function 与 `ctx.executor` 均禁止。
当前没有实现或新性能 claim；R0与R1 protocol/target freeze已完成，executable next是R1-0
instrumentation/schema tests。R3主设计、修改记录和可直接转发的 GitHub
外审 Prompt 见 `BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`、
`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_CHANGELOG_2026_08_24.md` 与
`BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`。父恢复计划的第一轮独立评审
原文保存在 `external_review_failed_gates_recovery_plan_2026_08_24.md`。

CIBC-IBP水平融合已通过Round 1独立外审并由executor关闭exchange，最终=
`EXTERNALLY-APPROVED-VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`：6 Conv operator
geomean/worst=`12.7951/9.1423x`，完整ResNet2B IBP graph geomean/worst=
`2.45631/2.45091x`。外审确认baseline公平、输入copy计入、两侧CUDA Graph、float64 oracle、
replay与13类篡改均成立；同时指出3条新mypy错误、steady-state/1 ULP披露等minor/info。
当前唯一研究动作是按已冻结R1协议实现candidate-only optimized-graph attribution的
calibration/topology/schema基础；不得直接扩TIR或复活B4-C2。失败门禁总账、R0—R5恢复路线与外部评审prompt见
`BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`、
`BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`与
`external_audit_cibc_ibp_horizontal_2026_08_24.md`。

FSG4/B4-C0累计core正式结果=`VALIDATED-NO-GO-B4-C0-NATIVE-VALUE-BRIDGE`：6 fresh/180 groups
geomean/lower/worst=`0.94034x/0.93778x/0.93418x`，语义sign exact、max diff=`7.15e-7`，root
replay与8/8 tamper通过。当前只开放B4-C1 provider-owned lower path rewrite。见
`BOUNDFLOW_FSG4_B4C0_CUMULATIVE_CORE_FORMAL_CLOSURE_2026_08_24.md`。

FSG4/B4-B3 CIBC exact-call已由5 fresh正式关闭：terminal lower与全部α/β allclose/sign exact，
max diff=`3.57628e-07`；provider/forward/backward=`50/50/45`，fallback/eager/materialization=0，
root replay与8/8 tamper通过。当前只开放预热交错的累计core timing，native-value bridge仍显式
保留。见`BOUNDFLOW_FSG4_B4B3_CIBC_FIVE_FRESH_FORMAL_CLOSURE_2026_08_24.md`。

FSG4/B4-B3 CIBC exact-call已实现并等待5 fresh：P-anchor完整native α TIR接入10/9 optimizer，
forward/backward=`10/9`、fallback/eager/materialization=0，terminal lower与全部α/β smoke
allclose/sign exact。当前明确保留10次native-value bridge以稳定Adam轨迹，尚无core/query性能
claim。见`BOUNDFLOW_FSG4_B4B3_CIBC_EXACT_CALL_IMPLEMENTATION_CHANGELOG_2026_08_24.md`。
该待five-fresh状态已由上方正式关闭状态取代。

FSG4/B4-B2 v2 manual TVM TIR已正式关闭：P-anchor CIBC式横向融合真实CUDA exact `1+1`
kernels、workspace=0，5 correctness与6-worker三方formal通过；PyTorch/TIR geomean/lower/worst=
`4.89834x/4.73771x/4.68601x`，Triton/TIR=`1.68273x/1.60695x/1.56888x`。状态=
`VALIDATED-B4-B2-V2-MANUAL-TIR`，当前只开放B4-B3 exact-call integration，core/query claim仍关闭。
见`BOUNDFLOW_FSG4_B4B2_V2_MANUAL_TIR_FORMAL_CLOSURE_2026_08_24.md`。

FSG4/B4-B2 B2-4 P-anchor sparse-source Conv已内部关闭：P0 five raw与12项bounded candidate
共68 metrics/217,770元素通过，ledger冻结且无timing/winner/performance claim。当前只开放B2-4
外审，B2-5/B4-B3关闭。见
`BOUNDFLOW_FSG4_B4B2_B2_4_SPARSE_CONV_TIR_CHANGELOG_2026_08_23.md`。

FSG4/B4-B2 B2-3 P-anchor dense Conv TIR已外审批准：最终=
`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS`。当前只开放B2-4
P-anchor sparse-source schedule，timing/B2-5/B4-B3关闭。见
`BOUNDFLOW_FSG4_B4B2_B2_3_EXTERNAL_AUDIT_CLOSURE_CHANGELOG_2026_08_23.md`。

FSG4/B4-B2 B2-3 P-anchor dense Conv TIR已内部关闭：5 raw、20 metrics、92,190元素
allclose/sign exact，max diff=`2.384185791015625e-06`，beta gradient absent，结构化workspace
门禁通过；状态=`VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。
下一步只开放B2-3外审，B2-4/B2-5/timing/B4-B3关闭。见
`BOUNDFLOW_FSG4_B4B2_B2_3_DENSE_CONV_TIR_CHANGELOG_2026_08_23.md`。

本目录用于存放“由大模型协助生成/维护”的工程文档与变更记录（changelog-style notes），目标是：

- 让每次 PR/阶段推进都有可审计的文字记录；
- 让别人（或未来的你）能快速定位：某个决策/某个口径/某个脚本是“为什么这样做”；
- 让论文/AE 的证据链（claim → 命令 → 产物 → 字段）能闭环。

FSG4/B4-B2 B2-2 S-anchor sparse-source Linear TIR已外审批准：独立float64重算、
GPU runner、workspace结构与全量回归均通过。最终=
`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS`；只开放
B2-3 P-anchor Conv dense correctness，timing/B2-4/B2-5/B4-B3继续关闭。见
`BOUNDFLOW_FSG4_B4B2_B2_2_SPARSE_LINEAR_TIR_CHANGELOG_2026_08_23.md`与
`BOUNDFLOW_FSG4_B4B2_B2_2_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`、
`external_audit_b4b2_b2_2_sparse_linear_tir_2026_08_23.md`与
`BOUNDFLOW_FSG4_B4B2_B2_2_EXTERNAL_AUDIT_CLOSURE_CHANGELOG_2026_08_23.md`。

FSG4/B4-B2 B2-1 S-anchor dense Linear TIR已外审批准：独立float64重算36,750元素
max diff=`6.988e-07`，GPU现场复跑逐位复现原始receipt/results，full=
`1437 passed, 3 skipped`。最终状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS`；只开放B2-2
S-anchor sparse-source fused forward/backward，timing/P-anchor/B2-4/B2-5/B4-B3继续关闭。见
`BOUNDFLOW_FSG4_B4B2_B2_1_DENSE_LINEAR_TIR_CHANGELOG_2026_08_23.md`与
`BOUNDFLOW_FSG4_B4B2_B2_1_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`、
`external_audit_b4b2_b2_1_dense_linear_tir_2026_08_23.md`与
`BOUNDFLOW_FSG4_B4B2_B2_1_EXTERNAL_AUDIT_CLOSURE_CHANGELOG_2026_08_23.md`。

FSG4/B4-B2 B2-0 ABI probe已外审批准，状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`：first-class
compiler/schedule/module/launch IR、identity CUDA/TIR双symbol、一阶custom autograd、DLPack/current
stream/cache/alias门禁已由auditor在RTX 4060现场复现，full=`1426 passed, 3 skipped`。下一步仅B2-1 S-anchor
dense correctness；尚无region融合或性能claim。见
`BOUNDFLOW_FSG4_B4B2_B2_0_IDENTITY_TIR_CHANGELOG_2026_08_23.md`与
`external_audit_b4b2_b2_0_identity_tir_probe_2026_08_23.md`、
`BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`。该“下一步B2-1”历史指令
已由上方B2-1外审关闭状态取代。

FSG4/B4-B1已在Round 2独立外审批准并由executor关闭exchange，最终状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`。F1/F2 CLOSED、AC1—AC6
全PASS、findings=0；receipt负例20/20、execution-policy 6/6、完整性负例2/2。下一步只开放
另行预注册B4-B2，不直接实现/计时TIR。该动作已由上方预注册完成状态取代；见
`change_2026-08-23_fsg4_b4b1_round2_external_closure.md`。

> 约定：每次工程改动都应在 `gemini_doc/` 新增一份 `change_YYYY-MM-DD_*.md` 记录，并在 `docs/change_log.md` 追加一条总账。

---

FSG4/B4-B1 Round 1外审已`request_changes`：F1为receipt exact metric/gradient target inventory
未绑定，F2为PyTorch deterministic warn/debug mode未原样恢复。两项修复已进入clean source
`e711e99`；v3 root replay与2/2完整性负例通过，RTX 4060 full=`1414 passed, 3 skipped`，
当前重交Round 2。B4-B2/TIR/
performance仍关闭。该历史状态已由上方Round 2批准取代；见
`change_2026-08-23_fsg4_b4b1_round1_f1_f2_fix.md`。

FSG4/B4-B1 typed pure-PyTorch reference已内部关闭，状态=
`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。v2从5 fresh raw重编译
S/P静态IR与10个instance，60 metrics/196,380 elements、max diff=`6.109476089477539e-07`、
allclose/sign exact；2/2 all-run bias/adjoint全链重签由数值语义拒绝；related=`131 passed`、full=
`1405 passed, 3 skipped, 6 warnings`。v1因未冻结PyTorch执行策略被明确superseded/fail-closed。
下一唯一动作是外审；获批前B4-B2/TIR/performance仍关闭。见
`change_2026-08-18_fsg4_b4b1_typed_reference_internal_closure.md`。

FSG4/B4-B1a five-fresh capture sufficiency已内部关闭，状态=
`VALIDATED-B4-B1A-FIVE-FRESH-CAPTURE-SUFFICIENCY`。source=`4a17423`；5 fresh/10 captures、
90 tensors/63,645 elements、max diff 0、sign exact；8/8完整性负例与full 1382/3 skip/6 warnings
通过。下一步是typed IR/pure-PyTorch reference，B4-B2/TIR/performance仍关闭。见
`change_2026-08-18_fsg4_b4b1a_five_fresh_internal_closure.md`。

FSG4/B4-B1a five-fresh runner已实现，状态=
`IMPLEMENTED-B4-B1A-FIVE-FRESH-RUNNER-PENDING-FORMAL`。临时5-process pilot比较90 tensors/
63,645 elements，max diff=0、sign exact；正式artifact尚未生成，协调动态bias/adjoint改写限制
留待numerical reference关闭。下一步是clean-source formal run。见
`change_2026-08-18_fsg4_b4b1a_five_fresh_runner_candidate.md`。

FSG4/B4-B1a capture sufficiency contract已实现，状态=
`IMPLEMENTED-B4-B1A-CAPTURE-CONTRACT-PENDING-FIVE-FRESH`。显式observer与新payload已捕获并
绑定bias、region output adjoints及sparse layout raw，单次real CUDA replay通过；下一步是
5-fresh formal artifact，typed IR/reference/TIR仍未实现。见
`change_2026-08-18_fsg4_b4b1a_capture_sufficiency_contract.md`。

FSG4/B4-B1 typed pure-PyTorch reference已预注册。调研确认B4-B0 capture虽可精确重建output A，
但缺incoming/operator bias与region output adjoints，不能自包含重建bias/production gradient。
下一步先做B4-B1a read-only capture amendment，B4-B2/TIR/performance仍关闭。见
`BOUNDFLOW_FSG4_B4B1_TYPED_PYTORCH_REFERENCE_PREREGISTRATION_2026_08_18.md`与
`change_2026-08-18_fsg4_b4b1_preregistration.md`。

FSG4/B4-B0已通过Round 2独立外审并由executor关闭exchange，最终状态=
`VALIDATED-B4-B0-EXTERNALLY-APPROVED`。0 blocker/major/minor/info；Round 1 F1已关闭；审计方
自行构造的all-run topology/lineage全链重签两案均被拒绝。下一步只开放另行预注册B4-B1 typed
pure-PyTorch reference，B4-B2/TIR/performance/memory/ASPLOS-ready仍关闭。见
`change_2026-08-18_fsg4_b4b0_round2_external_closure.md`。

FSG4/B4-B0 v2已内部关闭，状态=`VALIDATED-B4-B0-V2-PENDING-ROUND2-EXTERNAL-AUDIT`。
source=`422a3ee`；5 fresh/10 captures、108 tensors/664,744 elements、max diff=`1.1920928955078125e-07`、
sign exact；绝对source/topology/lineage身份绑定与11/11完整性负例通过。下一步只允许回复F1并提交
Round 2；定向=`24 passed`、全量=`1376 passed, 3 skipped, 6 warnings`；B4-B1/B4-B2/TIR/
performance仍关闭。变更见
`change_2026-08-18_fsg4_b4b0_v2_internal_closure.md`。

FSG4/B4-B0外审Round 1=`changes_requested`：协调一致改写全部run的topology/lineage可绕过原相对
一致性校验。F1 major的v2绝对身份绑定已实现，状态=
`IMPLEMENTED-B4-B0-R1-F1-IDENTITY-BINDING-PENDING-V2`；合法v1 replay保持，11/11本地完整性
负向用例拒绝。下一步生成clean-source v2并重交Round 2；B4-B1/TIR仍关闭。见
`gemini_doc/change_2026-08-18_fsg4_b4b0_round1_identity_binding_fix.md`。

FSG4/B4-B0 five-fresh已内部关闭，状态=
`VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`。source=`1dbb2de`；5个独立CUDA进程、
10份capture、108组tensor/664,744元素root replay，max diff=`1.1920928955078125e-07`、
sign exact；九类outer-resigned tamper `9/9 rejected`；全量=`1372 passed, 3 skipped, 6 warnings`。
只关闭capture correctness/ownership；外审前B4-B1/TIR仍关闭。见
`gemini_doc/change_2026-08-18_fsg4_b4b0_five_fresh_internal_closure.md`。

FSG4/B4-B0 five-fresh runner已实现，状态=
`IMPLEMENTED-B4-B0-FIVE-FRESH-RUNNER-PENDING-FORMAL-RUN`。typed capture现绑定α-index/lookup、
β-location/sign、round-trip、CUDA default-stream与alias；worker/runner从raw重建10个capture并逐tensor
比较，另有9类outer-resigned tamper。当前只完成单fresh CUDA与synthetic summary验证，下一步提交
runner后执行formal 5-fresh；B4-B1/TIR仍关闭。见
`gemini_doc/change_2026-08-18_fsg4_b4b0_five_fresh_runner_candidate.md`。

FSG4/B4-B0 evaluation-0 live observer已实现，状态=
`IMPLEMENTED-B4-B0-LIVE-OBSERVER-PENDING-FIVE-FRESH`。observer仅显式opt-in，在两个锚点上使参与
backward的同一lower-A诊断性materialize/retain-grad；默认B3/B4-A不变。CPU production-state
和真实CUDA smoke均证明S-anchor有active-beta gradient、P-anchor empty-beta无pre-add/gradient，且
evaluation-0 payload在首次optimizer step前冻结。related=`53 passed`，full=
`1369 passed, 3 skipped, 6 warnings`，静态门禁通过。
下一步是5-fresh artifact/replay/tamper，TIR仍关闭。见
`gemini_doc/change_2026-08-18_fsg4_b4b0_live_observer_candidate.md`。

FSG4/B4-B0 typed production-region capture contract已实现，状态=
`IMPLEMENTED-B4-B0-CAPTURE-CONTRACT-PENDING-LIVE-HOOK`。合同明确分离production compressed α/β
源状态、native dense α/β/`relu_pre_add_coeff_l`算子输入及native gradients，并对
evaluation-0、CUDA identity、hash、Conv attrs、provider/fallback实施fail closed。新测试10 passed，
fixed related 46 passed，full=`1366 passed, 3 skipped`，静态门禁通过。尚未接live solver，
无correctness/performance claim，
TIR仍关闭。见`gemini_doc/change_2026-08-18_fsg4_b4b_capture_contract_candidate.md`。

FSG4/B4-B differentiable CUDA/TIR v1已预注册，状态=`PREREGISTERED-B4-B-V1-NOT-IMPLEMENTED`。
计划同时冻结`node31/Gemm_14`的active-beta语义锚点和`node25/Conv_8`候选性能锚点，
先在gradient-active optimizer evaluation 0做5 fresh read-only exact-call capture；该门禁未过前不改TIR。
PR-12 plain-CROWN capability不放宽，单shape局部加速不得外推whole-core/query。见
`gemini_doc/BOUNDFLOW_FSG4_B4B_DIFFERENTIABLE_CUDA_TIR_V1_PLAN_2026_08_18.md`。

FSG4/B4-A正式计时已通过Round 1独立外审并由executor关闭exchange，最终状态=
`EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`。外审独立重算AC1—AC7，确认
core=`1.018995x < 1.03x`、query worst=`0.996947x >= 0.98x`，replay与14/14 tamper通过。
B4-A只保留correctness/mechanism，约1.9%不计入累计performance baseline。下一唯一动作是
单独预注册B4-B differentiable CUDA/TIR。见
`gemini_doc/change_2026-08-18_fsg4_b4a_external_audit_closure.md`。

FSG4/B4-A正式计时已在source=`46a8493`内部关闭：v5 24/24 worker、6/6 semantic pair、activation/
environment/profile、root replay和14/14 tamper全部通过；core wall geomean=`1.018995x < 1.03x`，
query worst=`0.996947x >= 0.98x`，故为`VALIDATED-NO-GO-B4-A-PERFORMANCE-PENDING-EXTERNAL-AUDIT`。
fixed related=`73 passed`、full=`1356 passed, 3 skipped`。该pending状态已由上方Round 1外审批准
取代。见
`gemini_doc/change_2026-08-18_fsg4_b4a_formal_timing_internal_closure.md`。

FSG4/B4-A正式计时v4在source=`03043a3`得到19个admitted worker，run 19 raw返回后因旧环境投影按累计
绝对值比较thermal/power counter而被拒绝；该worker两个counter已有`54579 µs`历史偏移，但区间增量
严格同为`2062477 µs`。门禁已改为interval delta exact，formal replay从raw重算，tamper扩为14类；
v4不形成ratio，下一步验证、提交clean source并从0生成v5。见
`gemini_doc/change_2026-08-18_fsg4_b4a_environment_interval_coupling_fix.md`。

FSG4/B4-A正式计时v3在source=`be2fa96`完成20个worker后，worker 20因执行期software thermal counter
独立增长被environment门禁拒绝；correctness/activation/profile结构完整，但v3不进入ratio。runner现
冻结`nvidia-powerd=inactive`与`enforced.power.limit=55.0 W`，逐worker及replay验证，tamper扩为13类；
其v4指令已被上方失败处置与v5指令取代。见
`gemini_doc/change_2026-08-18_fsg4_b4a_power_policy_binding.md`。

FSG4/B4-A正式计时v2在source=`ee73bc2`完成5个worker后，worker 5因独立software thermal slowdown
被environment门禁拒绝；v2不进入ratio。preflight已加固为每worker前GPU `<=45°C`且software thermal
完全inactive，下一步clean-source v3从position 0重跑。见
`gemini_doc/change_2026-08-18_fsg4_b4a_strict_preflight_hardening.md`。

FSG4/B4-A正式计时v1在source=`292a035`的worker 3因B4-A显式计数器alias覆盖缺口fail closed；不完整
raw不进入结论。计数覆盖修复后的live diagnostic恢复forward=4、bound eval=10、optimizer=`1/10/9`、
handoff/rerun=`1/0`。下一步clean source后从position 0生成v2；当前无性能claim。见
`gemini_doc/change_2026-08-18_fsg4_b4a_profile_counter_coverage_fix.md`。

FSG4/B4-A独立正式计时runner、raw-first/resume、root replay及14类outer-resigned tamper probe已实现，
固定related=`70 passed`，Black/Mypy/Pylint及全量`1353 passed, 3 skipped`通过。状态=
`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`；
下一步提交clean source并运行24个fresh GPU process，当前无性能claim，B4-B/TIR保持关闭。见
`gemini_doc/change_2026-08-18_fsg4_b4a_formal_timing_runner_candidate.md`。

FSG4/B4-A five-fresh correctness已内部关闭为
`INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`：source=`43d4117`，10/10 fresh worker、5/5 pair、
每pair 19个terminal export tensor raw比较、sign/discrete/lineage/counter与root replay全过；最大差=
`6.109476e-06`。本轮不使用latency，`performance_claimed=false`。下一唯一动作是独立正式B3/B4-A
timing；B4-B/TIR仍关闭。见
`gemini_doc/change_2026-08-16_fsg4_b4a_five_fresh_correctness_closure.md`。

FSG4/B4-A typed producer/handoff/no-rerun assembly与same-solver opt-in已实现，状态=
`IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。runtime热路径不做content D2H hash；完整handoff、
lineage、terminal export digest及raw float32 payload均在query后排除计时audit绑定。独立GPU smoke已确认
handoff=1/rerun=0/provider-fallback=0与冻结语义容差，但单pair性能不是claim。下一唯一动作是提交clean
source并生成5 fresh B3/B4-A correctness artifact；B4-B/TIR仍关闭。见
`gemini_doc/change_2026-08-16_fsg4_b4a_terminal_handoff_implementation.md`。

FSG4/B4-A terminal lower/lA handoff已完成实现前预注册，状态=`PREREGISTERED-B4-A-NOT-IMPLEMENTED`。
候选只允许在optimizer第10次、无update evaluation同时产出terminal lower与六层lA，并让export做
typed assembly、CROWN rerun=`0`；typed lineage绑定state/graph/split/topology、producer op ordinal/name、
shape/dtype/device/layout/content hash。先过5 fresh correctness，才允许检验B3/B4-A core `>=1.03x`与
query worst pair `>=0.98x`。B4-B/TIR及B5—B7保持关闭。见
`gemini_doc/BOUNDFLOW_FSG4_B4A_TERMINAL_LOWER_ADJOINT_HANDOFF_PLAN_2026_08_16.md`。

FSG4/B4-0 attribution已通过Round 1外审并由executor关闭exchange，最终状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。typed raw event、control/profile独立worker、
14-call marker、CUDA annotation/kernel区分、correlation/temporal归因、确定性gzip及
operator/kernel/materialization聚合、B3冻结semantic comparator与9类outer-resigned tamper已进入代码；
全量`1329 passed, 3 skipped`；formal raw含270609 events、35367/35367 kernel closure，replay与9/9
outer-resigned tamper通过。B3 raw表明
optimizer-only无限加速仍无法追回B0；B4因此冻结覆盖
10次optimizer、1次terminal export与3次KFSB child的14-call lower-only CROWN主线，依次执行B4-0
kernel/materialization attribution、B4-A terminal export fusion、B4-B differentiable lower-only TIR、
B4-C cumulative coverage与B4-D formal timing。当前下一唯一动作是B4-A预注册与terminal lower/lA
handoff；仍无B4 performance/B0 parity claim，B4-B不得混入，B5—B7继续关闭。见
`gemini_doc/change_2026-08-16_fsg4_b4_0_external_audit_closure.md`。

FSG4/B3正式计时已于2026-08-15通过Round 2独立外审并关闭为
`EXTERNALLY-APPROVED-VALIDATED-REDUCED-B3`：外审从36个raw worker独立重算44项检查，AC1—AC7
全部PASS，无blocker/major/minor；exchange已由executor关闭为`closed/approved`。当前只开放以B3为
直接累计基线的B4 operator/cross-stage CUDA/TIR fusion candidate；B5—B7和最终system gate继续关闭。
见`gemini_doc/change_2026-08-15_fsg4_b3_external_audit_closure.md`与
`.docops/exchange/fsg4-b3-formal-timing-20260814/r002/audit_report_full.md`。

FSG4/B3正式36-process计时已以`VALIDATED-REDUCED-B3`形成内部关闭：source=`36e9069`，六个
B0/B2/B3全排列、36/36 fresh worker、correctness/environment/measurement/activation、root replay与
10/10 tamper全部通过。B2/B3 core/query=`1.071617x/1.006623x`，但B0/B3 query=`0.910001x`，因此
B3只相对B2取得reduced收益，仍未快于原始B0。frozen=`6 passed`、targeted=`114 passed`、full=
`1314 passed, 3 skipped`。该“外审待完成”历史状态已由上方Round 2批准取代；详见
`gemini_doc/change_2026-08-14_fsg4_b3_formal_timing_closure.md`和
`gemini_doc/fsg4_b3_formal_timing_external_audit_handoff_2026_08_14.md`。

FSG4/B3 36-process正式计时runner已实现为
`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`：六个B0/B2/B3全排列、control/profile分离、直接B3
activation receipts、raw-first/resume、root replay与十类outer-resigned tamper probe均已进入代码；
targeted=`108 passed`、full=`1308 passed, 3 skipped`、Black/mypy/Pylint通过。该段是正式运行前历史
状态，现已由上方`VALIDATED-REDUCED-B3`结果取代。实现记录见
`gemini_doc/change_2026-08-14_fsg4_b3_formal_timing_runner_candidate.md`，冻结协议见
`gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`。

FSG4/B3五组fresh correctness已以`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`关闭：source=`75dfd81`，
固定交替顺序下10/10独立GPU worker、5/5 direct semantic pair、environment/provider/counter/audit、root
replay与7/7 tamper全部通过；定向=`56 passed`、全量=`1289 passed, 3 skipped`。这只开放36-process
B0/B2/B3正式计时，仍没有timing/speedup claim，B4—B7关闭。计划与关闭记录见
`gemini_doc/BOUNDFLOW_FSG4_B3_FIVE_FRESH_CORRECTNESS_PLAN_2026_08_14.md`和
`gemini_doc/change_2026-08-14_fsg4_b3_five_fresh_correctness_closure.md`，外审入口见
`gemini_doc/fsg4_b3_five_fresh_correctness_external_audit_handoff_2026_08_14.md`。

FSG4/B3-C已以`VALIDATED-B3-C-COUNTERS`关闭：source=`72bec5e`，fresh GPU artifact=
`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1/`。1484条event确认12个device
candidate/commit/backup/copy、timed candidate D2H=`0`，B3-B其余固定结构和六个B2 control语义保持；
headline digest=`0`，24个GPU content hash全部在query同步后的audit；replay与6/6 tamper通过。定向
`54 passed`、全量`1279 passed, 3 skipped`。这仍不是timing/speedup；该下一动作现已由上方Five-Fresh
关闭取代，B4—B7保持关闭。关闭记录见
`gemini_doc/change_2026-08-14_fsg4_b3c_device_atomic_commit_closure.md`，外审入口见
`gemini_doc/fsg4_b3c_device_commit_external_audit_handoff_2026_08_14.md`。

FSG4/B3-B已以`VALIDATED-B3-B-COUNTERS`关闭：source=`42df2dc`，fresh GPU artifact=
`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1/`。5157条event确认full step snapshots=`0`、
forward builds=`4`，且B3-A template/module/scope、optimizer 10/9、KFSB、D2H/commit等结构保持；冻结语义、
replay和6/6 tamper通过。定向`45 passed`、全量`1265 passed, 3 skipped`。这不是timing/speedup，完整
B3在该时点仍需B3-C与5 fresh pair；该历史下一动作已由上方Five-Fresh关闭取代。关闭记录见
`gemini_doc/change_2026-08-14_fsg4_b3b_terminal_schedule_closure.md`，外审入口见
`gemini_doc/fsg4_b3b_terminal_schedule_external_audit_handoff_2026_08_14.md`。

FSG4/B3-A已以`VALIDATED-B3-A-COUNTERS`关闭：source=`c7851c8`，fresh GPU artifact=
`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1/`。5157条event确认template compile/hit=
`1/1`、module move=`0`、scope=`1`，其余optimizer/forward/KFSB/D2H/commit固定结构与B2一致；六个冻结
B2 control语义、replay和6/6 outer-resigned tamper通过。定向`34 passed`、全量
`1257 passed, 3 skipped`。这只证明单次fresh correctness与B3-A物理激活，不是timing/speedup，也未替代
完整B3计时前的5 fresh pair门禁。该“下一动作”已被上方B3-B正式关闭取代。关闭记录见
`gemini_doc/change_2026-08-14_fsg4_b3a_prepared_core_closure.md`，外审入口见
`gemini_doc/fsg4_b3a_prepared_core_external_audit_handoff_2026_08_14.md`。

FSG4/B3-0已以`VALIDATED-B2-COUNTERS`关闭：source=`4195361`，正式artifact=
`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1/`，4625条event、固定counter全中、FSG3 v5
六个B2 control语义锚定与6/6 outer-resigned tamper通过。它证明B2重复工作真实存在，但没有B3 speedup；
`diagnostic_timing_claimed=false/performance_claimed=false`。该“下一动作”现已由上方B3-A正式关闭取代。
实现记录见
`gemini_doc/change_2026-08-14_fsg4_b3_explicit_counter_diagnostic_implementation.md`。

FSG4/B3当前入口为
`gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`。B3严格拆为
PreparedCoreTemplate、terminal-only optimizer Schedule与device-resident AtomicCommitPlan，只允许
IR/graph/Plan/Schedule复用；TIR fusion、JIT/CUDA Graph、runtime streams与arena仍分别留给B4—B7。
该段是实现前预注册历史状态；B3-0现已由上方正式关闭段取代，下一动作是B3-A。预注册变更记录
见`gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_CHANGELOG_2026_08_14.md`；该历史
“下一动作B3-A”现也已被上方B3-A/B/C与Five-Fresh关闭取代。B3-0关闭
外审交接见`gemini_doc/fsg4_b3_0_counter_external_audit_handoff_2026_08_14.md`，预注册历史外审见
`gemini_doc/fsg4_b3_preregistration_external_audit_handoff_2026_08_14.md`。

FSG3已由正式`resnet2b-prop0-v5`关闭：source=`a4ee291`，六个全排列block共36个fresh GPU进程，
correctness/environment/measurement/replay全部通过，summary hash=`df852590d…1318e`。B1 query wall
geomean=`0.995657x`；当前B2 whole-call reference的query/core分别为`0.908400x/0.516767x`
（均为B0/candidate），所以B2分类为`MEASURED-B2-SLOWER`，不是speedup；显存无变化。B2 profile显示
optimizer/atomic commit/KFSB/typed pre-state约占core `44.0%/24.7%/16.7%/10.7%`。正式状态为
`VALIDATED-FSG3-B0-B1-B2-BASELINE`，下一门禁为FSG4/B3 IR/graph/Plan/Schedule复用；B4 TIR fusion、
B5 JIT、B6 runtime、B7 arena尚未实现或计时。关闭记录见
`gemini_doc/change_2026-08-14_fsg3_same_solver_formal_baseline.md`，外部审计入口见
`gemini_doc/fsg3_same_solver_external_audit_handoff_2026_08_14.md`。下文所有“FSG3尚未执行”均为历史
时点说明，已被本段取代。

RVIR-v4当前入口：`gemini_doc/rvir_v4_optimizer_mutation_plan_2026_08_13.md`。V4-1已关闭
post-state独立复算；V4-2预注册冻结10 evaluation/9 update、双学习率、逐step parity与atomic
copy-out门禁。系统重启后V4-2B正式GPU step artifact已生成，10 evaluation/9 observed Adam update、
每步24项state、call/state/result交叉绑定、原始replay与5类同步重签名tamper全部通过，状态为
`VALIDATED-PRODUCTION-TRACE`。配套记录为
`gemini_doc/change_2026-08-13_rvir_v4_optimizer_mutation_preregistration.md`与
`gemini_doc/change_2026-08-13_rvir_v4_optimizer_step_trace.md`。GPU阻塞期间补做的exact policy与
call/trace cross-binding见
`gemini_doc/change_2026-08-13_rvir_v4_optimizer_trace_cross_binding.md`；正式关闭证据见
`gemini_doc/change_2026-08-13_rvir_v4_optimizer_step_formal_closure.md`。V4-2C随后以正式artifact证明
6组native α/β/split初始化和12/12 round-trip exact，并通过6类provenance+semantic双层重签名攻击，
状态为`VALIDATED-PRE-STATE-INITIALIZER`；见
`gemini_doc/change_2026-08-13_rvir_v4_pre_state_initializer.md`。V4-2D/E及总体关闭状态见下文。

V4-2D关闭证据见`gemini_doc/change_2026-08-13_rvir_v4_native_optimizer_parity.md`：formal native
executor在零provider callback下完成10 evaluations/9 updates，10/10 step lower/α/β allclose/sign exact，
最大误差均低于`2e-4`，并通过6类双层完全重签攻击，状态为
`VALIDATED-NATIVE-STEP-PARITY`。

V4-2E实现记录见`gemini_doc/change_2026-08-13_rvir_v4_atomic_copy_out.md`，正式关闭证据见
`gemini_doc/change_2026-08-13_rvir_v4_atomic_copy_out_formal_closure.md`：12-path private stage与atomic
commit、stale/NaN拒绝、mid-copy rollback、formal replay及6类完全重签攻击全部通过。V4-2E状态为
`VALIDATED-ATOMIC-COPY-OUT`，V4-2整体为`VALIDATED-OPTIMIZER-REPLACEMENT`。它尚未替换whole
`update_bounds_core`；“B2关闭、下一动作V4-3”是该时点历史门禁，现已由下文V4-3E closure取代。

V4-3预注册入口为`gemini_doc/rvir_v4_whole_core_replacement_plan_2026_08_13.md`：先冻结whole-core
truth，再实现native lA/intermediate、KFSB child evaluation、live return assembly和5次fresh correctness；
provider core/compute_bounds/update_bounds必须为`0/0/0`。V4-3A现已以
`VALIDATED-WHOLE-CORE-TRUTH`关闭：451 tensors/213,060 signs fresh replay与六类同步重签攻击通过；见
`gemini_doc/change_2026-08-13_rvir_v4_whole_core_truth_formal_closure.md`。V4-3B/C/D现也已依序关闭；
“只启动V4-3E、不启动B2”是该时点历史门禁，现已由下文V4-3E closure取代。

V4-3B实现记录为`gemini_doc/change_2026-08-13_rvir_v4_native_backward_export.md`：六层native lA与
12个shared-input intermediate tensors现已由正式artifact关闭，状态为
`VALIDATED-NATIVE-BACKWARD-EXPORT`；formal closure见
`gemini_doc/change_2026-08-13_rvir_v4_native_backward_export_formal_closure.md`。其中“下一动作V4-3C”是
该时点历史门禁，现已由后续V4-3C/D closure取代。

V4-3C实现记录为`gemini_doc/change_2026-08-13_rvir_v4_native_kfsb.md`，formal closure见
`gemini_doc/change_2026-08-13_rvir_v4_native_kfsb_formal_closure.md`：六层mask exact、三组top-3候选
共36项与final decision exact，72个child lower sign exact且最大差`3.0994e-06`，八类同步重签攻击
全部拒绝，状态为`VALIDATED-NATIVE-KFSB`。其中“下一动作V4-3D”是该时点历史门禁，现已由下段
V4-3D closure取代。

V4-3D实现记录见`gemini_doc/change_2026-08-13_rvir_v4_live_return_assembly.md`，formal closure见
`gemini_doc/change_2026-08-13_rvir_v4_live_return_formal_closure.md`：BoundFlow whole core已在RTX 4060
真实GPU进程中以provider callback=`0/0/0`接入未修改的official post/queue，完整core/post语义比较
最大差`1.0669e-05`且decision exact，fresh replay与8类完全重签攻击通过。状态为
`VALIDATED-LIVE-RETURN`；“下一门禁只允许V4-3E、B2继续关闭”是该时点历史门禁，现已由下段
V4-3E closure取代。

V4-3E harness实现记录为`gemini_doc/change_2026-08-13_rvir_v4_five_fresh_correctness.md`，formal
closure见`gemini_doc/change_2026-08-13_rvir_v4_five_fresh_formal_closure.md`：
`O,C,C,O,C,O,O,C,O,C`十个fresh进程、五对完整semantic/queue/termination与六类tamper全部通过。
V4-3整体=`VALIDATED-WHOLE-CORE-REPLACEMENT`；“B2 timing已准入但未执行”是该时点历史状态，现已
由上方FSG3正式关闭取代。

FSG3/B2正式计时预注册见
`gemini_doc/fsg3_b2_same_solver_timing_preregistration_2026_08_13.md`：冻结B0 original、B1 typed
passthrough、B2 whole-call reference replacement及六个全排列block，共36个fresh control/profile进程；
预注册时状态为`PREREGISTERED-NOT-RUN`，其门禁未按结果改写；正式结果现以上方FSG3关闭段为准。
schema/replay、真实
B0/B1/B2 worker、profile spans与36-process orchestrator均已实现；v1因旧thermal admission在7个位置后
整轮中止且不形成性能主张。schema v3已对本机严格镜像的SW power/thermal raw telemetry作最窄修正，
单worker及六路block-0 smoke均通过，完整回归为`1227 passed, 3 skipped`。诊断见
`gemini_doc/change_2026-08-13_fsg3_formal_v1_environment_abort.md`与
`gemini_doc/change_2026-08-13_fsg3_coupled_power_thermal_telemetry.md`。首个v3正式尝试32/36准入后暴露
父180秒timeout短于worker 900秒preflight的合同冲突，整轮无主张中止；见
`gemini_doc/change_2026-08-14_fsg3_formal_v3_parent_timeout_abort.md`。下一动作是在clean commit上以
1080秒父timeout从position 0生成完整36-process v4 attempt（这是当时动作，现已完成）。该attempt的首个worker
又以176次raw sample证明post-init 45°C在CUDA-initialized idle状态不可达，0/36无主张中止；schema v4
遂将绝对温度门禁修正为inclusive 50°C而保留全部独立thermal门禁。见
`gemini_doc/change_2026-08-14_fsg3_post_init_temperature_feasibility.md`；下一动作是在clean commit上从
position 0生成`resnet2b-prop0-v5`完整36-process artifact（现已完成）。

---

当前权威研究入口：
`gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`。FSG0 的作用域
纠正、全栈 schema、feature activation ledger 与 replay 合同已以20项定向测试和
`1079 passed, 3 skipped`全量回归关闭；外部审计三项minor亦已修复。FSG1又完成official B0
control full-stack trace：两个workload各5 fresh pair、10/10 attribution closure、semantic replay通过。
FSG2历史阶段以`VALIDATED-REDUCED initial-only`关闭：真实ResNet initial-CROWN可由BoundFlow native
backend在original/fallback=`0/0`下替换，但production inventory的24 calls显示alpha为嵌套
start-node keyed state，11个beta/split call前后又没有显式可own的beta tensor。故完整B2
在该时点`NO-GO/not admitted`，FSG3—FSG5按依赖门禁未运行；这不等于算子、图IR、JIT、调度或内存层
各自已被证伪。该历史ownership blocker现已由上文RVIR-v4 V4-3 whole-core replacement关闭，B2
same-solver timing已经准入但尚未执行；当前仍无BoundFlow全栈GPU speedup claim。FSG2历史关闭记录见
`gemini_doc/change_2026-08-06_fsg2_replacement_boundary_and_downstream_gate.md`。

NRIR49A 的正式数据与 artifact 继续有效，但其 `VALIDATED-NO-GO` 仅关闭 selected-CROWN-only
增量路线：fixed ResNet2B clauses 2/3 的五个 fresh GPU worker测得 selected-CROWN queue/complete share
中位=`7.0986%/7.0523%`；`1/(1-0.070986)=1.0764x` 只是假设该 region 变为零耗时的
deletion-only 单区域 Amdahl 上限，不是 BoundFlow 从算子、图、JIT、调度、并行到内存管理的累计
全栈上限。最大reserved仅1.353%物理显存，memory path=`N/A`；因此只停止 selected-CROWN 专属
G2/G3/TIR。正式 artifact 的 `next_route=gpu-winner-reselection` 是冻结历史机器输出，当前路线已由
上述 full-stack plan 取代。NRIR49A 正式入口为
`gemini_doc/BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_PLAN_2026_08_06.md`，summary hash=
`7eefe6a7…ab50`，不是 speedup claim；closure变更记录为
`gemini_doc/change_2026-08-06_nrir49a_g1_gpu_attribution_nogo.md`。

历史 GPU research-only 计划为
`gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md`；其 selected-CROWN
单主线路线已被 full-stack plan 取代。配套变更记录为
`gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_CHANGELOG_2026_08_05.md`；修订前外部审计为
`gemini_doc/external_audit_gpu_compiler_plan_v1_2026_08_05.md`。

G0 关闭入口：
`gemini_doc/BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md`。post-reboot v2
的 NVIDIA、双方 Torch、TVM CUDA TIR、TVM-FFI stream与跨环境digest六项均 PASS，状态
`ready_for_g1`；该结论只解除基础设施阻塞，不包含性能 claim。

历史前序关闭入口：
`gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`（NRIR48；
frozen NRIR45 clauses 2/3 的 6/6 attribution exact；child refinement execute 在两条 3/3 dominant，
selected-CROWN 占 parent=`71.7725%/72.7291%`，故 attribution `VALIDATED-REDUCED`；不是 speedup，
当时准入的 NRIR49 selected-CROWN execution 已由 NRIR49A 完成，不是当前路线）。其前序
`gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`（NRIR47；
typed receipt 已把每 child compile selector/reselection 收敛为 `30/0`，60 个动态 target ledger 不共享，
显式 full replay 仍重选；但 compiler ratio=`0.936003 > 0.85`、两条 queue ratio=
`1.011205/1.019338 > 0.97`，故 Phase A `VALIDATED-NO-GO`、Phase B gated off）。下一门禁为 top-2
production execution math/queue attribution。其前序 NRIR46 已由 PR #57 合入 `main@ca0bcf3`。

已由用户豁免执行、但保留的外部审计材料：
`gemini_doc/BOUNDFLOW_NRIR45_EXTERNAL_AUDIT_HANDOFF_2026_08_05.md`（PR #56，包含 AC1—AC6、
独立 artifact replay/tamper/回归命令与 claim boundary；未声称获得外部批准）。

## 1) “我应该从哪读起？”

按目的给四条阅读路径：

### A. 论文/AE 视角（最推荐）

当前最新关闭入口：
`gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`（NRIR48；
NRIR45 default production top-2 queue execution-cost attribution 已把 dominant 缩小到
selected-CROWN execution；formal hash=`571c2e47…d177a4`）。其前序
`gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`（NRIR47；
single-pass exact target admission receipt correctness/ownership 已实现并通过；Phase A compiler/queue
timing 未过，formal hash=`a7561e51…042ce`，状态 `VALIDATED-NO-GO`，下一步做 execution-cost
attribution）。其前序为
`gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_PLAN_2026_08_05.md`（NRIR46；
三 fresh process Phase 0 保持 60/60 exact replay，但 strict static topology median 仅
`1.071197 s`，低于预注册 `1.5 s` 门槛，故 `VALIDATED-NO-GO`，未启动 Phase A/B；PR #57 已合入
`main@ca0bcf3`）。其直接 production 基线为
`gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_PLAN_2026_08_05.md`
（NRIR-45 已 `VALIDATED-REDUCED`：typed prepare-once capsule/receipt 将每条 queue 的 target selection
`246→98`、full validation `186→38`；clauses 2/3 median ratio=`0.727519/0.736603`。Phase B
whole trace=`31.262521/31.319772/31.470078 s`、measured median ratio vs NRIR-44=`0.615738`，
每轮 `[31,31]` nodes 与 60/60 full replay exact；final 仍 unknown，ASPLOS-ready=NO）。其直接
前序为 `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_PLAN_2026_08_05.md`
（NRIR-44 已 `VALIDATED-REDUCED`：typed consumer/liveness contract 将 ranking floor 的
`9×n31d4` 投影为 `9×n1d0`，Phase A evaluation `279→9`、floor median ratio=`0.407530`；Phase B
whole=`43.571040/44.144990/44.095736 s`，相对 NRIR-42 median ratio=`0.764254`，两条 production
queue 仍各 31 nodes。该结论只适用于固定 ResNet2B property 0 CPU8，final unknown，ASPLOS-ready
仍为 NO；功能/证据提交 `437680e` 已由 PR #55 合入 `main@f194034`）。其直接前序为
`gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_PLAN_2026_08_05.md`
（NRIR-43 已 `VALIDATED-NO-GO`：typed ragged batch 保持 6/6 exact 并将每条 scorer launch
`31→16`，但 clauses 2/3 median ratio=`1.051134/1.044573`，CPU 墙钟退化；Phase B 按门禁未启动，
formal hash=`692b9e27…30390`）。
NRIR-43 提交 `00b82c2` 已由 PR #54 合入 `main@2d245d6`。
其直接前序为 `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_PLAN_2026_08_05.md`
（NRIR-42 已 `VALIDATED-REDUCED`：typed validated capsule 把每条 31-node queue 的 candidate
enumeration 从 341 次收敛到 compile-only 31 次；Phase A new/old median ratio=
`0.706888/0.698486`，Phase B 三轮均完成 clauses 2/3 的 `[31,31]` nodes，whole=
`57.175184/57.697757/58.114412 s`。该结论仅为固定 ResNet2B property 0 CPU production admission，
ASPLOS-ready 仍为 NO；功能提交 `264365f` 已由 PR #53 合入 `main@8969064`）。其直接前序为
`gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`。
其直接前序为 `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1_PLAN_2026_08_05.md`、
`gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SHARED_EVALUATOR_V1_PLAN_2026_08_05.md`、
`gemini_doc/BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1_PLAN_2026_08_05.md`、
`gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_05.md`、
`gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_TYPED_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_WALL_CLOCK_PARAMETRIC_BAB_SCALING_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_PARAMETRIC_DYNAMIC_BATCH_COMPILER_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_PRODUCTION_PREPARED_VERIFIER_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_TYPED_MULTIPASS_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_DYNAMIC_ANCESTRAL_REFINEMENT_BUDGET_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_EXTERNAL_SEEDED_DEPTH_NODE_CONVERGENCE_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_EXTERNAL_SEEDED_ANCESTRAL_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_ANCESTRAL_CONSTRAINT_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_PLAN_2026_08_04.md`、
`gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_PLAN_2026_08_04.md` 和
`gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_PLAN_2026_08_04.md`。

1. `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_PLAN_2026_08_04.md`
   （当前最新：九子句 query、candidate search、sound aggregation、deadline 与 9/9 unknown blocker）
2. `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_PLAN_2026_08_04.md`
   （verified/unsafe/unknown soundness、concrete witness replay 与 ResNet unknown）
3. `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_PLAN_2026_08_04.md`
   （optimizer Schedule × ReLU-split queue、parent warm-only与 selected native execution）
4. `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_PLAN_2026_08_04.md`
   （fixed-step optimizer Plan/Task/Schedule、action hash chain与固定 ResNet replay）
5. `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_PLAN_2026_08_04.md`
   （frozen alpha/beta state、beta lower-dual execution、warm-start validity与五层 IR）
6. `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_PLAN_2026_08_04.md`
   （first-class ReLU split IR、best-first bounded queue、真实 ResNet packed/serial node stacks）
7. `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_PLAN_2026_08_04.md`
   （8 个不同 input-box leaves、exact child state、domain Plan/Schedule 与 serial reference）
8. `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_PLAN_2026_08_04.md`
   （9 条真实 property query 的 packed/serial execution、exact cache 与 lineage）
9. `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_PLAN_2026_08_04.md`
   （同一 ResNet template/selector 的 dense/structured × full/sliced 联合执行）
10. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_PLAN_2026_08_04.md`
   （真实 ResNet spec BatchDecision → exact Schedule ranges → child stacks → aggregation）
11. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_PLAN_2026_08_04.md`
   （Plan/Schedule representation → execution Bound/Task/Launch 语义绑定与 ResNet replay）
12. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_PLAN_2026_08_03.md`
   （双 storage fresh CUDA 测量协议与 fail-closed unavailable evidence）
13. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`（固定
   ResNet 双 storage、预算切换、Schedule arena 与 runtime last-use release）
14. `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`（固定 ResNet
   native Bound/Plan/Task/Schedule correctness 与 NRIR-1 边界）
15. `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`（production
   Schedule ownership/memory 门禁、NO-GO 与 native real-network 准入条件）
16. `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`（真实 verifier
   correctness/integration 关闭审计与不可升级边界）
17. `gemini_doc/rvir_external_audit_handoff_2026_08_03.md`（可直接交给其他模型的自包含
   RVIR 审计请求、证据和复核顺序）
18. `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`（RVIR 所有权与门禁）
19. `gemini_doc/pr14_external_model_audit_handoff_2026_07_19.md`（从项目起点到 PR-14
   No-Go 的自包含外部审计入口）
20. `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`（PR-14
   后 IR-first 架构重置、对象协议与实施门禁）
21. `gemini_doc/current_status_after_pr13.md`（全项目当前状态）
22. `gemini_doc/asplos_claims_map.md`（论文主张→代码→实验→工件证据）
23. `gemini_doc/asplos_execution_memo_v1_0.md`（唯一历史顺序与门禁）
24. `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`（ASPLOS 顶层计划）

### B. 全流程总览（从 claims 到工程到 AE）

- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
- `gemini_doc/boundflow_full_pipeline_director_view.md`（指挥视角的系统路线）

### C. 研发协作流程（人与大模型怎么配合）

- `gemini_doc/boundflow_build_and_run_workflow.md`（按源码类型编译、运行和测试）
- `gemini_doc/llm_collaboration_workflow.md`（输入计划→修正测试→总结→下一步计划）
- `/home/lee/.codex/skills/boundflow-workflow/SKILL.md`（上述工作流的本机 skill 入口）

### D. 研发演化/接手视角

1. `gemini_doc/current_status_after_pr13.md`（当前冻结状态与下一门禁）
2. `gemini_doc/project_evolution_overview.md`（项目目标、阶段推进、代码落点、未来路线）
3. `docs/change_log.md`（按时间看每一批修改做了什么）
4. `gemini_doc/phase6_summary.md`（当前方法族与 E2E 工件链的阶段总览）

---

## 2) 本目录文件分类

### 2.1 关键交付文档（“长期有效”）

- `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_PLAN_2026_08_05.md`：
  prepare-once validation/hash ownership、typed capsule/receipt、Phase A/B 门禁与 fixed ResNet
  `VALIDATED-REDUCED` 边界
- `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_CHANGELOG_2026_08_05.md`：
  `246→98` target-selection、`186→38` full-validation、31.3 秒 whole trace、artifacts/replay 与限制
- `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_05.md`：
  NRIR-28 template/instance/cache × NRIR-34 ancestral sibling evaluator 的静动态边界、first-class
  Plan/Batch/Task/Schedule、parity/top-2/三重复门禁与下一 tightness attribution 路线
- `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_05.md`：
  phase profile、实现/负向测试、真实 parity、`[31,31]×3` formal artifact/replay 与 claim boundary
- `gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_PLAN_2026_08_04.md`：
  root-lower priority、top-2 selection、dynamic equal-remaining slice、single global clock、三重复门禁与
  atomic-pair coverage NO-GO、shared parametric evaluator 下一路线
- `gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_CHANGELOG_2026_08_04.md`：
  feasibility、first-class IR/runtime、formal artifact/replay/tamper、精确三重复数字与 claim boundary
- `gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_PLAN_2026_08_04.md`：
  NRIR-31 floor + guarded NRIR-34 escalation 的 Plan/Decision/Task/Schedule、单 global deadline、
  monotone aggregate、三重复门禁与 multi-clause priority 下一路线
- `gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_04.md`：
  feasibility、runtime/tests、formal artifact/replay、精确三重复数字与 claim boundary
- `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_PLAN_2026_08_04.md`：
  sibling-group IR、atomic deadline、三重复 coverage 门禁与 cross-clause 下一路线
- `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_CHANGELOG_2026_08_04.md`：
  profiler/runtime/tests/formal/full-query artifact 的精确结果与 claim boundary
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_PLAN_2026_08_04.md`：
  five-cap Policy/Decision、90% retention gate、cap-only NO-GO 与 sibling-packed 下一路线
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_CHANGELOG_2026_08_04.md`：
  additive IR/runtime、fresh-process pilot/replay、精确 cap 曲线与 claim boundary
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`：
  typed root admission、dynamic child refinement、committed queue Task/Schedule、三重复 frontier
  tightness 与下一 child-budget 门禁
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`：
  feasibility、IR/runtime/tests、正式 artifact/replay、精确下界与 claim boundary
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_PLAN_2026_08_04.md`：
  shared-source per-clause objective Plan/Task/Schedule、whole deadline、三重复 tightness 门禁与
  objective-ancestral 下一路线
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`：
  runtime/tests/pilot/artifact/replay、ResNet 九子句 root delta 与 claim boundary
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_PLAN_2026_08_04.md`：
  single-clause objective influence、score/identity/dependency IR、same-budget ResNet artifact 与
  per-child 下一门禁
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`：
  objective policy/runtime/tests、root-tightness 数值、replay 与 claim boundary
- `gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_PLAN_2026_08_04.md`：三种
  VNN-COMP 拓扑、VNNLIB Query IR、21-task/6-worker protocol 与 competitor claim 边界
- `gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_CHANGELOG_2026_08_04.md`：
  parser parity、fresh CPU 矩阵、artifact/replay/tests 与 intermediate-bound 下一门禁
- `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_PLAN_2026_08_04.md`：exact
  prepared optimizer/root-query capsule、cold/warm 公平协议与 identity 门禁
- `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_CHANGELOG_2026_08_04.md`：三组
  59.078 s→110.950 ms 内部 overhead 诊断、cold/payload、artifact/replay/tests 与限制
- `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_PLAN_2026_08_04.md`：clauses
  0/2/4 objective-bound-impact branching、first-class score schedule 与完整队列门禁
- `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_CHANGELOG_2026_08_04.md`：widest
  失效诊断、batched strong-branch feasibility probe 与后续验证记录
- `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_PLAN_2026_08_04.md`：
  external-intermediate optimizer/queue/query bridge、三组 phase/tightness 诊断协议与下一
  prepared-production 门禁
- `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_CHANGELOG_2026_08_04.md`：
  fixed ResNet 6/9 结果、约 6.7 s audit queue 归因、artifact/replay/tests 与限制
- `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_PLAN_2026_08_04.md`：multi-clause
  conjunction、candidate search、sound aggregation/short-circuit、cooperative deadline、固定
  ResNet 9/9 unknown 与下一 tightness/performance baseline 门禁
- `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_CHANGELOG_2026_08_04.md`：differentiable
  primal executor、search/query runtime、scale-aware trace 修复、artifact/tests 与验证数字
- `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_PLAN_2026_08_04.md`：
  verified/unsafe/unknown soundness、concrete witness replay、fixed ResNet explicit unknown 与
  complete-verifier/performance 后续边界
- `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_CHANGELOG_2026_08_04.md`：
  concrete executor、verdict/proof/witness runtime、artifact/tests 与验证数字
- `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_PLAN_2026_08_04.md`：
  optimizer Schedule × ReLU-split queue、parent warm-only、per-node selected state、fixed ResNet
  packed/serial artifact 与 sound verdict 下一门禁
- `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_CHANGELOG_2026_08_04.md`：
  optimized queue runtime/trace/state comparison/artifact/tests 与数值披露
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_PLAN_2026_08_04.md`：
  fixed-step optimizer Plan/Task/Schedule、action transition trace、fixed ResNet legacy/native
  equivalence 与下一 queue-integration 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_CHANGELOG_2026_08_04.md`：
  optimizer IR/runtime/artifact/tests、验证数字与未关闭边界
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_PLAN_2026_08_04.md`：
  frozen alpha/beta typed state、beta lower-dual execution、scope/warm-start gate、固定 ResNet
  五层 IR artifact 与 runtime-owned optimizer hard limitation
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_CHANGELOG_2026_08_04.md`：
  optimized ReLU schema/interpreter、state runtime、artifact/tests 与下一 optimizer-control 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_PLAN_2026_08_04.md`：
  first-class ReLU split Bound/Plan/Task/Schedule、best-first bounded queue、parent exact-state
  prohibition、真实 ResNet packed/serial artifact 与 α/β/performance hard limitations
- `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_CHANGELOG_2026_08_04.md`：
  split schema/compiler/interpreter、queue runtime、artifact/tests、验证与下一 α/β state 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_PLAN_2026_08_04.md`：
  8-leaf input-box tree、parent warm-start-only、exact child state、domain Plan/Schedule、
  packed/full/serial execution 与 full-BaB/performance hard limitations
- `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_CHANGELOG_2026_08_04.md`：
  domain variants/runtime/traces、fixed ResNet artifact/tests 与下一 ReLU-split queue 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_PLAN_2026_08_04.md`：
  真实 property-query contract、packed/serial same-policy execution、exact cache key、per-query
  restore 与 BaB domain/performance hard limitations
- `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_CHANGELOG_2026_08_04.md`：
  query runtime/cache/traces、artifact/tests 与下一 parent/child domain-state 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_PLAN_2026_08_04.md`：
  representation/storage × spec-batch single-template joint selection、required child policy、四组合
  ResNet execution/replay 与跨 query/domain/performance hard limitations
- `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_CHANGELOG_2026_08_04.md`：
  required-storage selector、joint binding/runtime、artifact/tests 与下一 real query-stream 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_PLAN_2026_08_04.md`：
  spec-axis Plan selection、精确 Schedule ranges、child compiler stacks、aggregation、真实 ResNet
  artifact/replay 与 domain/sample/joint-policy hard limitations
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_CHANGELOG_2026_08_04.md`：
  selector/Schedule/compiler/runtime/runner/tests 与下一 representation × batch 联合门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_PLAN_2026_08_04.md`：
  dense/structured-affine 双 policy、Plan/Schedule/Bound transition 一一绑定、独立 execution
  IR stack、真实 ResNet artifact/replay 与 dense-equivalent hard limitation
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_CHANGELOG_2026_08_04.md`：
  binder、selector prefix pruning、execution template、tests、artifact 与下一 sliced-batch 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_PLAN_2026_08_03.md`：
  retain/reuse 的 fresh-process CUDA allocator/lower-only timing 协议、预注册门禁、fail-closed
  environment evidence 与 representation semantic bridge 下一路线
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_CHANGELOG_2026_08_03.md`：
  prepared storage capsule、runner/schema/replay/tests、unavailable probe 与不可升级边界记录
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_PLAN_2026_08_04.md`：同一
  real ResNet Bound IR/PlanTemplate 的 retain-all/lifetime-reuse storage plans、预算选择、
  Schedule arena、runtime last-use trace、artifact 与不可升级边界
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_CHANGELOG_2026_08_04.md`：
  上述 allocator、runtime hook、runner、tests、artifact 与下一 physical-memory 门禁记录
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`：固定
  VNN-COMP ResNet2B 的 native 21-region Bound/Plan/Task/Schedule correctness、portable
  external-bound payload、五层 hash 与 NRIR-2 multi-plan 门禁
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_CHANGELOG_2026_08_04.md`：上述
  compiler、capture payload、runner、artifact、验证和不可升级边界的变更记录
- `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`：RVIR 后
  production Schedule ownership、materialization/storage 与 multi-budget memory 门禁；
  `NO_GO` 后转向 native real-network Bound IR
- `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_CHANGELOG_2026_08_04.md`：上述
  audit module、runner、artifact、验证与权威文档同步记录
- `gemini_doc/pr14_external_model_audit_handoff_2026_07_19.md`：项目起点、ASPLOS 路线、
  PR-14A/B 证据/限制、外部复核命令与下一步的自包含审计交接
- `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`：复审当前
  Bound/Plan/Task/Schedule/runtime 实现后冻结的一等 IR 分层、动态规划边界与逐阶段门禁
- `gemini_doc/change_2026-07-20_ir_planner_schedule_runtime_contract.md`：上述架构重置、
  Claims Map 降级和权威文档修订记录
- `gemini_doc/change_2026-07-28_bound_ir_v1_schema_foundation.md`：IR-1A typed Bound IR
  schema、verifier、deterministic dump/hash、兼容性与未完成 builder/interpreter 边界
- `gemini_doc/change_2026-07-28_bound_ir_v1_plain_crown_lowering.md`：IR-1B plain-CROWN
  Task/trace lowering、显式 affine/fanout 语义、独立 dense interpreter 与 final-bound 对齐边界
- `gemini_doc/change_2026-07-28_bound_ir_v1_representation_rewrite.md`：IR-1C 显式
  dense/structured cast、materialization rewrite、structured reference 执行与 IR-1 closure
- `gemini_doc/change_2026-07-28_plan_ir_v1_schema_and_legacy_migration.md`：IR-2A typed
  PlanTemplate/PlanInstance、跨决策 verifier、instance replay 与 PR-11/12 迁移边界
- `gemini_doc/change_2026-07-28_plan_ir_v1_reference_builder_selector.md`：IR-2B typed
  evidence→PlanTemplate builder、预算/deadline selector 与不可变 selection artifact
- `gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`：IR-2C query-time state
  validity、legacy atomic assembly、raw-record absence audit 与 IR-2 validated-reduced closure
- `gemini_doc/change_2026-07-28_schedule_ir_v1_schema_lowering.md`：IR-3A typed
  ScheduleModule/action schema、PlanInstance lowering、memory/use-def/query verifier foundation
- `gemini_doc/change_2026-07-28_schedule_ir_v1_control_executor.md`：IR-3B batch/event/state/
  retry/replan actions、reference executor、canonical trace 与 fresh-process artifact
- `gemini_doc/change_2026-07-28_task_ir_v1_foundation.md`：IR-3C typed TaskIRModule/Unit、
  Plan region lowering、Task↔Schedule launch linkage 与 dispatch trace
- `gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`：IR-3D stateful
  Bound stepping、逐 Task 数值执行、shape/transfer 契约补项、artifact v2 与 closure audit
- `gemini_doc/change_2026-07-28_ir4a_typed_backend_dispatch.md`：IR-4A typed backend
  dispatch key、PyTorch reference adapter、prepared-task cache 与 stale/capability rejection
- `gemini_doc/change_2026-07-28_ir4b_pytorch_backend_registry.md`：IR-4B backend-specific
  Task identity、fused ReLU→Affine stepping 与真实 dense/structured/chunked registry
- `gemini_doc/change_2026-07-28_ir4c_tvm_backend_cache_fallback.md`：IR-4C typed TVM
  fused/unfused、dispatch-namespaced disk cache、fresh-process replay 与 semantic OOM fallback
- `gemini_doc/change_2026-07-28_ir4d_compiler_query_state_runtime.md`：IR-4D
  capability-gated typed query、Plan/Task cache、exact-version state load/store/task skip、
  fresh-process artifact 与 PR-14 No-Go 保留
- `gemini_doc/change_2026-07-28_ir4e_pr13_query_migration_closure.md`：PR-13
  DynamicBatchManager→typed compiler adapter、legacy α/β historical opt-in、artifact v2 与
  IR-4 validated-reduced closure
- `gemini_doc/change_2026-07-28_ir5a_adaptive_plan_context.md`：IR-5A query-time
  memory/deadline/cache/distribution context、compile amortization 与 per-query plan cache
- `gemini_doc/change_2026-07-28_ir5b_fair_policy_evaluator.md`：IR-5B
  fixed/local/global/oracle 统一 observation evaluator、tail/TTV/peak/regret 与 synthetic
  contract artifact
- `gemini_doc/change_2026-07-28_ir5c0_typed_measured_workload_foundation.md`：IR-5C0
  正式 typed MLP benchmark workload、候选 Plan/Schedule 构造与 predicted/measured compile
  防泄漏
- `gemini_doc/change_2026-07-28_ir5c1_leakage_free_measurement_runner.md`：IR-5C1
  calibration-only 预测、CUDA cold/warm/peak/TVM phase 测量、冻结 split/resource context
  与目录级 manifest/semantic replay
- `gemini_doc/change_2026-07-28_ir5c2_cuda_heldout_partial.md`：IR-5C2 fresh CUDA
  typed MLP artifact、四策略 regret/TTV/tail/peak、低内存切换及 workload-family/batching
  未闭环的 PARTIAL 判定
- `gemini_doc/change_2026-07-28_ir5c3a_independent_cnn_family.md`：IR-5C3A
  deterministic chain-CNN typed workload、跨 architecture calibration feature 与
  PyTorch/TVM fused CUDA semantic probe
- `gemini_doc/change_2026-07-28_ir5c3b_fair_batching_contract.md`：IR-5C3B
  fixed-single/ordinary-batching/batched-original 公平 evaluator、physical-batch 归一化、
  MLP→CNN runner 与 batch 语义门禁
- `gemini_doc/change_2026-07-28_ir5c3c_family_fair_nogo.md`：IR-5C3C
  architecture-held-out CUDA fair artifact、Global p90 regret 70.263×、host
  validate/hash hot-path 归因与 IR-5 v1 VALIDATED-NO-GO
- `gemini_doc/change_2026-07-28_ir5d_prepared_execution_capsule.md`：IR-5D
  prepared Bound/Task execution、production/audit trace 分离、from-forward-trace 公平基线
  与 calibration-only CUDA remediation 诊断；该文档当时尚未执行的 residual final
  已由 IR-5E—H 完成并判定最终 No-Go
- `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`：IR-5 No-Go 后独立的
  真实 verifier correctness/integration 路线、所有权与 RVIR-1—4 门禁
- `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`：external
  intermediate bounds + adaptive slope 的 ResNet initial-CROWN 语义修复
- `gemini_doc/change_2026-08-03_rvir2_typed_external_calls.md`：activation-BaB external exact
  call 的 Bound/Plan/Task/Schedule 类型化、真实调度与 lineage closure
- `gemini_doc/change_2026-08-03_rvir3_cpu_correctness_artifact.md`：394 个历史 activation
  typed admission、377 次真实 CPU exact dispatch、ResNet 等价与自包含 replay artifact
- `gemini_doc/change_2026-08-03_rvir_online_raw_replay_v2.md`：外部审计 M4 后续，冻结
  377 条在线 query/typed-record 原文并在 fresh replay 中重算 lineage、accounting 与 IR hash
- `gemini_doc/change_2026-08-03_rvir_resnet_raw_rerun.md`：外部审计 F5/M5 后续，在固定
  αβ-CROWN 与 VNN-COMP commit 上连续两次重跑 ResNet 原始数值，核对冻结摘要与 tensor digest
- `gemini_doc/change_2026-08-03_rvir_post_hardening_audit_handoff.md`：PR #5—#8
  审计后加固的 AC1—AC6、独立复核命令、claim boundary 与新 DocOps exchange 交接
- `gemini_doc/change_2026-08-04_rvir_post_hardening_audit_closure.md`：外部复审 approve、
  AC1—AC6/F1—F5 关闭、正式 exchange closure 与完整审计附件的 Git 固定记录
- `gemini_doc/change_2026-07-28_ir5e_residual_final_protocol_freeze.md`：IR-5E
  residual-CNN typed workload、chain-CNN calibration→residual final v2 冻结 split、
  from-trace 公平协议及 p90/Pareto 一次性门禁
- `gemini_doc/change_2026-07-28_ir5f_residual_final_v2_protocol_invalid.md`：IR-5F
  v2 首次运行在 fixed-single 输入身份门禁失败、shape-dependent RNG 根因与
  `7401/7402` 永久退役记录
- `gemini_doc/change_2026-07-28_ir5g_exact_input_slice_v3_freeze.md`：IR-5G
  fixed-single 显式切片 batched input 的方法学修复、v3 schema 与未执行
  `7501/7502` final freeze
- `gemini_doc/change_2026-07-28_ir5h_residual_final_v3_nogo.md`：IR-5H
  fresh residual final v3 完整 artifact、Global p90 1.26160×、gray Pareto 缺失与
  当前 ASPLOS system-performance 路线最终 No-Go
- `gemini_doc/change_2026-08-03_ir5_route_closure_and_publish.md`：IR-5 路线封存、
  权威状态去过期、外部 replay 命令与真实 Verifier IR 新路线准入条件
- `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`：真实 verifier
  intermediate-bound semantics、relaxation policy、activation external-call IR 与门禁
- `gemini_doc/change_2026-08-03_start_real_verifier_ir_integration.md`：新 correctness/
  integration 路线启动与 ResNet 根因复核记录
- `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`：external ReLU
  intermediate bounds/adaptive policy 入 IR 与 ResNet CPU correctness closure
- `gemini_doc/asplos_execution_memo_v1_0.md`：ASPLOS 研发的短执行入口与门禁
- `gemini_doc/current_status_after_pr13.md`：PR-13 closure 后的真实状态、证据边界与当前缺口
- `gemini_doc/pr14_execution_plan.md`：真实 verifier workload coverage/execution 的切片、门禁与止损
- `gemini_doc/change_2026-07-19_pr14a_abcrown_query_profile_adapter.md`：外部 αβ-CROWN
  `compute_bounds` → PR-13 `BoundQuery` → coverage profile 的可撤销接入边界
- `gemini_doc/pr14a_real_query_coverage_2026_07_19.md`：MLP/CNN/VNN-COMP ResNet-2B 的真实
  method/phase/backend coverage、observer baseline 与 PR-14B 窄化判定
- `gemini_doc/change_2026-07-19_pr14a_real_query_traces.md`：真实 trace 生成与 fail-closed
  frontend 审计的变更记录
- `gemini_doc/change_2026-07-19_tvm_ffi_library_search_path.md`：新环境中新版 tvm-ffi
  动态库发现与 Conda hooks 的修复记录
- `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`：exact-box MLP/ResNet fixed
  replay、requested-output/bound-equivalence 门禁与 PR-14/C3 最终 No-Go
- `gemini_doc/change_2026-07-19_pr14b_initial_crown_fixed_replay.md`：PR-14B 代码、contract、
  ignored artifacts 与验证记录
- `gemini_doc/change_2026-07-19_fresh_clone_test_split_fixtures.md`：完整测试从代码冻结 split
  重建临时 fixture，不再依赖新环境中不存在的 ignored PR-12 artifacts
- `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`：ASPLOS 总体研发、论文与 artifact 执行计划
- `gemini_doc/asplos_claims_map.md`：ASPLOS 三项贡献的动态证据映射
- `gemini_doc/materialization_trace_schema_v1.md`：PR-10 trace JSONL 与内存口径
- `gemini_doc/pr10_materialization_profile_summary_2026_07_12.md`：PR-10 第一版 clean GPU profile 与 claim 边界
- `gemini_doc/pr10_dense_structured_comparison_2026_07_12.md`：PR-10 双模式 guardrail 与最终判定
- `gemini_doc/pr10_review_integration_2026_07_12.md`：PR-10 外部评审意见、PR-11 收敛与本次修改记录
- `gemini_doc/pr11_materialization_planner_start_2026_07_12.md`：PR-11 第一实现切片、测试证据与剩余 blocker
- `gemini_doc/materialization_plan_schema_v1.md`：PR-11 plan/context/candidate JSON schema 与 capability 语义
- `gemini_doc/pr11_heldout_eval_2026_07_12.md`：PR-11 cost model、final held-out 结果与未通过项
- `gemini_doc/pr11_multi_barrier_placement_start_2026_07_12.md`：非退化 multi-barrier Global placement 与 mixed runtime foundation
- `gemini_doc/pr11_barrier_evaluator_and_retry_2026_07_12.md`：measured Oracle、Global Retry held-out 结果与真实 OOM handling 边界
- `gemini_doc/change_2026-07-12_pr11_bounded_stratified_retry.md`：有界分层 retry、双规模 held-out 与真实 CUDA OOM 证据
- `gemini_doc/change_2026-07-12_pr11_independent_topology_nogo.md`：并行残差 held-out 失败、部署特征审计与 PR-11 No-Go
- `gemini_doc/change_2026-07-12_pr11_static_topology_cost.md`：静态 topology/liveness feature、LOO retry calibration 与三组 final held-out
- `gemini_doc/pr11_closure_audit_2026_07_12.md`：PR-11 逐项 closure、replicated final evidence 与 PR-12/PR-13 边界
- `gemini_doc/pr11_regret_attribution_2026_07_13.md`：高 regret case 的候选覆盖/后端假设归因
- `gemini_doc/pr12_fused_crown_task_plan_2026_07_13.md`：PR-12 收敛范围、接口、门禁与证据版本
- `gemini_doc/backend_candidate_schema_v1.md`：PR-12 placement/backend 二维候选与 capability 合同
- `gemini_doc/change_2026-07-13_pr12_start_and_fused_linear.md`：PR-12 起点、held-out 与 Linear TIR 第一切片
- `gemini_doc/change_2026-07-13_pr12_fused_conv2d.md`：Conv stride-1/2、codegen 与 latency sanity
- `gemini_doc/change_2026-07-13_pr12_e2e_crown_integration.md`：显式 fused region schedule、TVM/Torch executor、网络级 final-bound 与 zero-copy 门禁
- `gemini_doc/change_2026-07-13_pr12d_correctness_closure.md`：fanout soundness、完整 step contract、pre-materialization fallback 与 TVM-FFI custom-stream closure
- `gemini_doc/change_2026-07-13_pr12ef_runtime_pareto_heldout.md`：正式 runtime/memory Pareto、calibration-only Planner、frozen held-out 与性能 No-Go
- `gemini_doc/change_2026-07-13_pr12g_multibackend_planner.md`：chunked-r512 候选、全新 held-out-v2、多后端 Planner 与 canonical 工件
- `gemini_doc/pr12_mid_long_term_completion_plan.md`：PR-12H–N baseline、摊销、profile、Planner 与 closure 执行路线
- `gemini_doc/pr12_execution_status.md`：PR-12 跨会话唯一恢复入口与当前门禁
- `gemini_doc/change_2026-07-14_pr12h_benchmark_contract.md`：三层 benchmark contract、历史证据披露与 PR-12G freeze
- `gemini_doc/change_2026-07-14_pr12i_fair_baselines.md`：structured/TVM-unfused 公平 baseline、条件 torch.compile probe 与正式 Pareto
- `gemini_doc/change_2026-07-14_pr12j_compile_amortization.md`：compile/load/cache 阶段拆分、跨进程 disk hit 与 Q-sweep 摊销
- `gemini_doc/change_2026-07-14_pr12k_cupti_profile.md`：CUPTI activity profile、硬件 counter 权限边界与停止孤立 TIR 调优判定
- `gemini_doc/change_2026-07-14_pr12l_stop_tir_optimization.md`：冻结停止孤立 TIR 调优、未选分支与 PR-12M 接口约束
- `gemini_doc/change_2026-07-14_pr12m_compile_aware_planner.md`：compile/cache/reuse Planner、v3 split、多预算 held-out 与 regret
- `gemini_doc/pr12_closure_audit_2026_07_14.md`：PR-12N 最终判定、H–M 证据/限制与 PR-13 Go/No-Go
- `gemini_doc/pr12_artifact_appendix_2026_07_14.md`：PR-12 reduced artifact 依赖、工作流、expected outputs 与 claims
- `gemini_doc/pr13_execution_status.md`：PR-13 五切片跨会话状态、冻结边界与恢复命令
- `gemini_doc/change_2026-07-14_pr13a_query_contract_fixed_replay.md`：state-versioned query contract、真实 BaB 固定流 replay 与 PR-13B 门禁
- `gemini_doc/change_2026-07-14_pr13b_dynamic_batch_manager.md`：兼容分桶、预算/deadline、OOM 拆批、physical αβ batching 与 PR-13C 门禁
- `gemini_doc/change_2026-07-14_pr13c_same_solver_adapter.md`：原 host solver 仅替换 bound-call path 的对照与 PR-13D 门禁
- `gemini_doc/change_2026-07-14_pr13d_fixed_e2e_gpu.md`：RTX 4060 fixed/E2E reduced 评估、batched-original 归因与负收益
- `gemini_doc/pr13_closure_audit_2026_07_14.md`：PR-13 `VALIDATED-REDUCED` 逐项 closure 与未成立主张
- `gemini_doc/pr13_artifact_appendix_2026_07_14.md`：PR-13 reduced artifact 命令、expected outputs 与证据链
- `gemini_doc/artifact_claims_phase5d.md`：Phase 5D artifact claims（证据链/口径映射）
- `gemini_doc/artifact_appendix_phase5d.md`：Phase 5D artifact appendix（复现说明）
- `gemini_doc/project_evolution_overview.md`：研发脉络总览（目标、阶段推进、代码落点、未来路线）
- `gemini_doc/codex_superpowers_global_install.md`：Codex Superpowers 全局安装说明（主机级安装、跨主机复用、自动检测 skills 目录）
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`：全流程总览（从研究主张到工程到 AE）
- `gemini_doc/boundflow_full_pipeline_director_view.md`：指挥视角的工程主线
- `gemini_doc/why_boundflow_not_auto_lirpa_or_tvm.md`：论文辩护要点（为何不端到端用 auto_LiRPA / 为何不直接用 TVM）
- `gemini_doc/phase0_summary.md`：Phase 0 总结（工程止血：可编辑安装 + 包结构清理 + 最小 smoke）
- `gemini_doc/phase1_summary.md`：Phase 1 总结（工程止血 + Primal IR 加固）
- `gemini_doc/phase2_summary.md`：Phase 2 总结（TorchFrontend：torch.export → Primal IR + 最小 normalize）
- `gemini_doc/phase3_summary.md`：Phase 3 总结（IBP reference + auto_LiRPA 对齐：MLP/CNN）
- `gemini_doc/perturbation_support_design.md`：设计文档（支持 L∞/L2/L1/L0 输入扰动与线性算子统一公式）
- `gemini_doc/bound_methods_and_solvers_design.md`：设计文档（IBP/CROWN/IBP-CROWN/αβ-CROWN/BaB 的三轴解耦与落地路线）
- `gemini_doc/phase4_summary.md`：Phase 4 总结（Task/Planner/Executor/Spec/TVM/ONNX 的闭环与对齐口径）
- `gemini_doc/phase5_summary.md`：Phase 5 总结（bench→JSONL→postprocess→artifact 产线 + schema_version=1.0 冻结）
- `gemini_doc/phase6_summary.md`：Phase 6 总结（语义闭环 + 系统收益归因 + AE/论文工件链）
- `gemini_doc/quick_restart_ibp.md`：Quick Restart（像 auto_LiRPA 一样跑 IBP 边界）
- `gemini_doc/tvm_backend_optimization_memo.md`：TVM/Relax 后端优化备忘
- `gemini_doc/llm_collaboration_workflow.md`：与大模型协作工作流模板

### 2.2 变更记录（`change_YYYY-MM-DD_*.md`）

这些文件按时间记录“当时做了什么/为什么做/怎么验证”，适合：

- 回溯某个接口/口径的由来
- 追踪阶段推进（Phase 4 → Phase 5）

常见命名模式：

- `change_YYYY-MM-DD_phase5*_pr*_*.md`：按阶段/PR 编号
- `change_YYYY-MM-DD_*_memo.md`：备忘或总结

---

## 3) Phase 5（现状）的一句话索引

如果你只想知道“Phase 5 到底做完了什么”：

- 完成声明：`docs/phase5_done.md`
- 口径冻结：`docs/bench_jsonl_schema.md`（`schema_version=1.0`）
- 一键产线：`scripts/run_phase5d_artifact.py`（产 `results.jsonl/table_main.csv/figures/MANIFEST/CLAIMS/APPENDIX`）
- 证据链：`gemini_doc/artifact_claims_phase5d.md`

---

## 4) 维护规则（防止目录继续膨胀失控）

1. **不要移动/改名历史 `change_*.md`**（避免破坏已有引用）。
2. 新增文档时优先选择：
   - `docs/`：面向用户/读者的稳定说明（安装、schema、完成声明）
   - `gemini_doc/`：面向研发/演进的过程记录（变更记录、备忘、决策）
3. 任何影响口径的变更都要同时更新：
   - `docs/bench_jsonl_schema.md`
   - 对应的 contract tests / postprocess tests
4. 运行产物目录 `artifacts/`、`out/` 不进入 git（已在 `.gitignore` 忽略）。
FSG4/B4-C1 provider-owned lower已实现：P-anchor不再native+TIR双算，plan buffer与DLPack view已复用；
单worker语义exact但累计core约`0.95x`。局部`4.90x`的native分母包含observer强制`to_dense()`，不是
真实production materialization frontier。当前下一动作是6 fresh正式关闭，随后若NO-GO转B4-C2大
region累计覆盖。见`BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_CHANGELOG_2026_08_24.md`。
FSG4/B4-C1正式以`VALIDATED-NO-GO-B4-C1-MATERIALIZATION-FRONTIER`关闭：6 fresh/180 groups，
core geomean=`0.94815x`、worst=`0.94547x`，语义max diff=`7.153e-7`且8/8 tamper拒绝。
当前只开放B4-C2真实materialization frontier大区域融合与14-call coverage。见
`BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_FORMAL_CLOSURE_2026_08_24.md`。
FSG4/B4-C2接管6个真实materialization sites后，3 fresh speedup仅`0.337—0.349x`且显存增加34%，
虽语义exact但触发kill gate；纵向alpha-CROWN B4整体以NO-GO关闭，B4-D不开放。下一路线转向CIBC
论文真正的IBP/forward-bound水平融合与autotuning。见
`BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`。
CIBC-IBP水平融合已实现：一个manual TIR kernel同时完成center/deviation与lower/upper，取代4次
PyTorch Conv；另有plan-owned DLPack/runtime和公平CUDA-graph整图路径。当前诊断为真实Conv约
`7.72x`、ResNet2B完整IBP graph约`2.70x`，正式artifact前不形成claim。见
`BOUNDFLOW_CIBC_IBP_HORIZONTAL_FUSION_IMPLEMENTATION_CHANGELOG_2026_08_24.md`。
正式协议已冻结：3个独立operator schedule workers选择64/128/256之一，再用6 fresh CUDA Graph
whole-model workers验证；输入copy计入、6 Conv coverage与全中间interval语义门禁由raw replay重算。
见`BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_PROTOCOL_CHANGELOG_2026_08_24.md`。
CIBC-IBP正式结果已关闭：真实6 Conv算子层geomean=`12.795x`，完整ResNet2B IBP图6-fresh
geomean=`2.4563x`，语义和10/10全重签tamper门禁通过。见
`BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md`；原B4计划、NO-GO根因与CIBC路线关系见
`BOUNDFLOW_B4_ORIGINAL_PLAN_AND_CIBC_FINAL_STATUS_2026_08_24.md`。
