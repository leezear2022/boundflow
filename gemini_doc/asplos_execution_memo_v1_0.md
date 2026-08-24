# BoundFlow ASPLOS 执行备忘录 v1.0

> **2026-08-24 CIBC外审关闭与路线复审指令**：Round 1独立外审`APPROVE`，exchange已
> `closed/approved`；最终=
> `EXTERNALLY-APPROVED-VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`。6 Conv operator
> geomean/worst=`12.7951/9.1423x`，完整ResNet2B IBP graph=`2.45631/2.45091x`；claim仍只限
> RTX 4060/sm_89、ResNet2B prop0、steady-state IBP、相对BoundFlow四-Conv baseline。失败门禁
> 总复盘确认：B4-B2 v2 local TIR成功，B4-C0/C1/C2失败在production ownership/materialization/
> autograd lifetime，B5—B7及complete solve尚未运行。当前先做R0外审卫生，再只允许预注册
> CIBC-G1 candidate-only NVTX/CUPTI/CUDA-Graph attribution；归因前不得选Linear/Conv/elementwise/
> runtime实现分支，也不得复活B4-C2。权威路线见
> `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`。

> **2026-08-24 FSG4/B4-C0正式NO-GO指令**：source=`d1db31e`完成6 fresh/180 groups；
> bridge候选geomean/lower/worst=`0.94034x/0.93778x/0.93418x`，memory allocated/reserved=
> `1.04818/1.0`；semantic max diff=`7.15256e-07`且sign exact，root replay与8/8 tamper通过。最终=
> `VALIDATED-NO-GO-B4-C0-NATIVE-VALUE-BRIDGE`。只开放B4-C1 provider-owned lower path rewrite，
> 禁止以bridge形成performance claim。

> **2026-08-24 FSG4/B4-C0 artifact指令**：6 fresh BC/CB、180 raw groups、root median/
> geomean/bootstrap/worst/memory/semantic/receipt重算已实现。冻结no-regression=
> geomean≥1/lower>1/worst≥0.98/memory≤1.05，research=1.05。当前唯一动作=提交clean source并formal；
> 若bridge候选NO-GO，只开放provider-owned lower path rewrite。
> 该待运行指令已由上方正式NO-GO指令取代。

> **2026-08-24 FSG4/B4-C0累计core runner指令**：双方3 warmups、30 interleaved groups、
> per-group 13项semantic parity与peak memory已实现；timing关闭correctness capture并由receipt显式
> 记录。单worker pilot warmed B3/candidate=`81.918/87.519 ms`，ratio=`0.9360x`，只作runner验证。
> 当前唯一动作=提交clean source并执行6 fresh BC/CB formal；结果前无performance claim。

> **2026-08-24 FSG4/B4-C0累计core计时准备指令**：已把per-evaluation finite `.item()`及
> correctness-only local parity同步移出timing hot path；结构/stream门禁保持fail closed，correctness
> 默认行为不变。当前唯一动作=实现并运行双方预热、BC/CB交错、30 groups、6 fresh cumulative core
> timing；结果前无performance claim。

> **2026-08-24 FSG4/B4-B3 five-fresh正式关闭指令**：source=`1d06aab`完成5 independent
> BC/CB pairs；terminal lower与全部α/β allclose/sign exact，max diff=`3.57628e-07`；provider/
> forward/backward=`50/50/45`，fallback/eager/materialization=0，native-value bridge=50；root replay
> 与8/8 outer-resigned tamper通过。最终=`VALIDATED-B4-B3-CIBC-EXACT-CALL`，只开放预热交错的
> 累计core timing。raw首调用ratio受warmup支配，明确不是性能证据。

> **2026-08-24 FSG4/B4-B3 five-fresh runner指令**：已实现5 independent workers、
> `BC/CB/BC/CB/BC`顺序、13项terminal state metrics、local parity、exact receipt、code/source/
> model/reference identity及root semantic replay。timing字段只作diagnostic。当前唯一动作=提交clean
> source并生成正式artifact；通过后才开放累计core timing。
> 该待运行指令已由上方five-fresh正式关闭指令取代。

> **2026-08-24 FSG4/B4-B3 exact-call实现指令**：P-anchor dense-native-α manual TIR已接入
> production 10/9 optimizer，receipt=`10 forward/9 backward`，fallback/eager/materialization=0；
> S-anchor显式unsupported并走B3。接线证伪compressed-86 α可替代完整native α，并修复了
> incoming-bias identity gradient所有权。smoke terminal lower及全部α/β allclose/sign exact，max
> diff=`3.57628e-07`。当前以`native value + candidate gradient` bridge维持Adam float32轨迹，故
> 只开放5 fresh correctness/replay；bridge移除前不得形成core/query加速claim。

> **2026-08-24 FSG4/B4-B2-v2 manual TIR正式关闭指令**：clean source=`5b2c9ba`
> 完成5 correctness、6-worker `BTR/BRT/TBR/TRB/RBT/RTB`三方计时、root replay与10/10
> outer-resigned tamper。exact kernels=`1+1`，workspace=0，max diff=`1.90735e-06`且sign exact；
> PyTorch/TIR geomean/lower/worst=`4.89834x/4.73771x/4.68601x`，Triton/TIR=
> `1.68273x/1.60695x/1.56888x`，allocated/reserved ratio=`0.450886/1.0`。最终=
> `VALIDATED-B4-B2-V2-MANUAL-TIR`，现在只开放B4-B3 exact-call integration；core/query claim仍关闭。

> **2026-08-24 FSG4/B4-B2-v2 manual TIR实现指令**：manual TVM TIR已等价实现exact
> `1 forward + 1 backward`、workspace=0、5 raw sign exact；PlanInstance常驻PackedFunc/DLPack/
> combined buffers并把stream admission移出双launch hot path。非正式三方probe约PyTorch/Triton/TIR=
> `0.500/0.153/0.093 ms`，不形成claim。当前唯一动作=提交clean source并执行5 correctness+
> 6-worker三方formal timing；正式达到Triton 0.90x且对PyTorch≥1.20x才开放B4-B3。
> 该待运行指令已由上方manual TIR正式关闭指令取代。

> **2026-08-24 FSG4/B4-B2-v2 Triton正式关闭指令**：clean source=`77a15eb`完成
> 12 fresh calibration、5 correctness、6-worker AB/BA timing、root semantic replay+独立重编译与
> 10/10 outer-resigned tamper。winner=1；exact kernels=`1+1`，workspace=0，max diff=
> `1.90735e-06`且sign exact；geomean/lower/worst=`2.83772x/2.78575x/2.74000x`，allocated/
> reserved ratio=`0.363337/1.0`。最终=`VALIDATED-B4-B2-V2-TRITON-PHYSICS`，只开放manual
> TVM TIR等价port；TIR达到Triton 0.90x且对PyTorch≥1.20x前B4-B3关闭。
> 该待实现指令已由上方manual TIR实现指令取代。

> **2026-08-24 FSG4/B4-B2-v2 CIBC-parity实现指令**：Triton horizontal fused
> forward/backward已实现，12/12 config及5/5 raw均对public-PyTorch oracle allclose/sign exact；
> profiler确认exact `1 forward + 1 backward`真实CUDA kernel、global intermediate workspace=0。
> 非正式probe约`2–3.7x`，不形成claim。当前唯一动作=绑定clean source生成12 fresh calibration、
> 5 correctness、6 AB/BA timing的formal artifact/replay；门禁通过才开放manual TIR port，B4-B3
> 仍关闭。
> 该待运行指令已由上方Triton正式关闭指令取代。

> **2026-08-24 FSG4/B4-B2-v2 CIBC-parity预注册指令**：以B2-5 v1 NO-GO为直接输入，冻结
> `1 forward + 1 backward` horizontal fused ABI、零global intermediate workspace、12项Triton
> autotune space与`1.20x` minimum/`2.00x` research target。先以Triton作为CUDA融合/schedule oracle；
> oracle通过后才等价下沉manual TIR并独立重复门禁。当前唯一动作=v2 fused kernel correctness；
> B4-B3/C/D仍关闭。
> 该待实现指令已由上方v2实现指令取代。

> **2026-08-24 FSG4/B4-B2 B2-5正式关闭指令**：clean source=`bf1c8b7`完成12项
> calibration、S/P各5 independent correctness、6-worker AB/BA timing、root replay与8/8
> outer-resigned tamper。winner=11；geomean/bootstrap-lower/worst=
> `0.424842x/0.403157x/0.377693x`，allocated/reserved ratio=`0.474638/1.0`；真实CUDA kernels=
> `3 forward + 3 backward`。最终=`VALIDATED-NO-GO-B4-B2-V1-PHYSICS`，B4-B3关闭。
> 该结论只关闭当前6-kernel/12-schedule v1；下一动作=B4-B2-v2 CIBC-parity horizontal fusion/
> autotuning预注册与实现，B4-C/D继续关闭。

> **2026-08-24 FSG4/B4-B2 B2-5实现候选指令**：wrapper-inclusive public-PyTorch baseline、
> 真实CUDA kernel inventory、12项calibration、S/P各5 independent correctness、6-worker AB/BA
> timing、artifact/replay与8类outer-resigned tamper已实现。开发探针确认当前TIR为3 forward +
> 3 backward kernels，历史`launch=1/1`仅表示module call。下一唯一动作=提交clean source并生成
> formal artifact；结果前无performance claim，B4-B3仍关闭。
> 该待运行指令已由上方B2-5正式关闭指令取代。

> **2026-08-24 FSG4/B4-B2 B2-4外审关闭指令**：Round 1 `APPROVE`，0 blocker/
> major/minor；exchange已由executor关闭为`closed/approved`。独立float64闭合公式、现场GPU、
> 12项ledger与篡改门禁全部通过。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-4-SPARSE-CONV-P0-AND-BOUNDED-LEDGER-CORRECTNESS`。
> 现在只开放B2-5 formal independent-process correctness/artifact/replay/AB-BA timing；必须复用冻结
> 12项ledger，不得追加第13项。B4-B3继续关闭，且B2-4本身不形成performance claim。

> **2026-08-23 FSG4/B4-B2 B2-4内部关闭指令**：P-anchor sparse-source Conv P0与12项
> bounded schedule ledger全部compile/correct，68 metrics/217,770元素通过；无timing、winner或
> performance claim。当前只开放B2-4最终外审；B2-5/B4-B3关闭。
> 该待审指令已由上方B2-4外审关闭指令取代。

> **2026-08-23 FSG4/B4-B2 B2-3外审关闭指令**：Round 1 `APPROVE`，0 blocker/
> major/minor；最终=`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS`。
> 唯一工程动作=B2-4 P-anchor sparse-source schedule；timing/B2-5/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-3内部关闭指令**：P-anchor dense Conv TIR 5/5 raw、
> 20/20 metrics、92,190元素通过，max diff=`2.384185791015625e-06`且sign exact；beta gradient
> absent，结构化workspace门禁通过。当前=
> `VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。唯一下一动作=
> B2-3外审；B2-4/B2-5/timing/B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-2外审关闭指令**：`APPROVE`，0 blocker/major/minor；
> float64独立重算max diff≤`6.99e-07`，现场GPU、workspace结构与hash全部复现。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS`。唯一下一动作=
> B2-3 P-anchor Conv dense correctness；timing、B2-4/B2-5、B4-B3关闭。

> **2026-08-23 FSG4/B4-B2 B2-2内部关闭指令**：S-anchor已直接从compressed alpha
> `[6,27]`/beta`[6,1]`执行TIR forward/backward并返回compressed gradients；5 raw/
> 20 metrics/31,590 elements，max diff=`8.642673492431641e-07`，targeted/related/full=
> `34/88/1448 passed`，3 skipped。禁止dense-state workspace count=`0`，当前=
> `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。下一唯一动作=
> B2-2外审；P-anchor、timing、B2-3/B2-4/B2-5、B4-B3关闭。
> 该待审指令已由上方B2-2外审关闭指令取代。

> **2026-08-23 FSG4/B4-B2 B2-1外审关闭指令**：独立外审`APPROVE`，0 blocker/
> 0 major；float64独立重算36,750元素最大差=`6.988e-07`，现场GPU逐位复现
> runner与三项receipt hash；targeted/related/full=`23/77/1437 passed`，3 skipped。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS`。唯一下一动作=
> B2-2 S-anchor sparse-source fused forward/backward；timing、P-anchor、B2-4/B2-5、B4-B3关闭。
> 该“下一动作B2-2”指令已由上方B2-2内部关闭指令取代。

> **2026-08-23 FSG4/B4-B2 B2-1内部关闭指令**：S-anchor dense Linear TIR完成5 raw、
> 20 metrics/36,750元素，max diff=`8.642673492431641e-07`且sign exact；full=
> `1437 passed, 3 skipped`。当前=
> `VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。下一唯一动作=外审，
> 批准前不得实现B2-2或启动timing。
> 该待审指令已由上方外审关闭指令取代。

> **2026-08-23 FSG4/B4-B2 B2-0外审关闭指令**：verdict=`APPROVE`，0 blocker/0 major；
> auditor现场GPU复跑逐位复现三项receipt hash。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`。下一唯一动作=B2-1 S-anchor dense
> correctness，先处理真实fallback计数与异常退出状态恢复；不得启动timing或B2-2。

> **2026-08-23 FSG4/B4-B2 B2-0关闭指令**：状态=
> `VALIDATED-B4-B2-B2-0-ABI-PROBE`。first-class编译/执行receipt、identity TIR双symbol、DLPack
> 零拷贝、显式current stream、cache miss→hit与一阶custom backward已在RTX 4060通过；full=
> `1426 passed, 3 skipped`。下一唯一动作=B2-1 S-anchor dense correctness。不得计时identity、
> 不得主张region融合/性能，不得提前进入B2-2/P-anchor/B4-B3。

> **2026-08-23 FSG4/B4-B2预注册指令**：状态=
> `PREREGISTERED-B4-B2-TYPED-CUDA-TIR-NOT-IMPLEMENTED`。已冻结dense→sparse-source两级ABI、
> first-class Template/Schedule/Module/Launch receipts、custom-autograd/stream/alias/cache、5-fresh与
> 6-worker物理kill gate。下一唯一动作=B2-0 identity-TIR ABI probe；通过前不得实现region TIR。
> 该待实施指令已由上方B2-0通过状态取代。

> **2026-08-23 FSG4/B4-B1 Round 2外审关闭指令**：exchange已`closed/approved`，F1/F2
> CLOSED，AC1—AC6全PASS且findings=0。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`。下一唯一动作是另行预注册
> B4-B2 typed CUDA/TIR candidate；该指令已由上方预注册完成状态取代。

> **2026-08-23 FSG4/B4-B1 Round 1修复指令**：外审以F1 receipt inventory/target未精确
> fail-closed、F2 deterministic warn/debug mode未原样恢复两个major判定`request_changes`。
> 两项修复已进入clean source=`e711e99`，旧v2被新协议拒绝；v3已完成root replay与2/2完整性
> 负例，RTX 4060全量=`1414 passed, 3 skipped`。下一唯一动作是提交Round 2。外审批准前
> 不得实现B4-B2 CUDA/TIR或升级
> 任何性能类claim。该待审指令已由上方Round 2批准取代。

> **2026-08-18 FSG4/B4-B1内部关闭指令**：deterministic v2以5 fresh/10 captures重编译typed
> IR/instance并独立执行pure-PyTorch forward/VJP；60 metrics/196,380 elements、max diff=
> `6.109476089477539e-07`、sign exact，2/2协调all-run全链重签由数值语义拒绝；related 131、
> full 1405/3 skip/6 warnings通过。状态=
> `VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。下一唯一动作是外审；
> 批准前不得实现B4-B2 CUDA/TIR、计时或升级performance/memory/ASPLOS-ready claim。

> **2026-08-18 FSG4/B4-B1a内部关闭指令**：formal 5-fresh capture sufficiency、8-case完整性
> 与full regression已通过。下一唯一动作是实现B4-B1 typed IR与独立pure-PyTorch reference，
> 从raw重算forward/VJP并关闭coordinated动态改写限制；未过five-fresh reference与外审前不得进TIR。

> **2026-08-18 FSG4/B4-B1a runner指令**：worker/runner/probe与5-process pilot已通过，但pilot
> 非正式证据。下一唯一动作是先提交冻结代码，再生成formal artifact、8-case报告与full regression；
> 之后才可实现typed IR/reference，B4-B2/TIR继续关闭。

> **2026-08-18 FSG4/B4-B1a capture合同指令**：bias/output-adjoint/sparse-layout amendment已
> 实现并通过单次real CUDA replay。下一唯一动作是独立worker/runner的5-fresh formal artifact、
> root replay与bias/adjoint/layout完整性负例；关闭前不得实现typed reference或B4-B2/TIR。

> **2026-08-18 FSG4/B4-B1预注册指令**：先实现B4-B1a capture sufficiency amendment，补齐
> incoming bias、operator bias、region output adjoints与sparse layout raw；再实现typed IR与独立
> pure-PyTorch reference。不得从target倒推输入；five-fresh与外审关闭前不得进入B4-B2/TIR。

> **2026-08-18 FSG4/B4-B0 Round 2外审关闭指令**：exchange已`closed/approved`，Round 1 F1
> 已关闭。下一唯一动作是另行预注册B4-B1 typed pure-PyTorch reference及其correctness/gradient
> 门禁；不得直接实现B4-B2 CUDA/TIR，不得启动performance/memory计时或升级ASPLOS-ready claim。

> **2026-08-18 FSG4/B4-B0 v2内部关闭指令**：source=`422a3ee`的绝对身份绑定v2 artifact
> 已通过5 fresh/10 captures、max diff=`1.1920928955078125e-07`、sign exact与`11/11`
> 完整性负例；定向=`24 passed`、全量=`1376 passed, 3 skipped, 6 warnings`。下一唯一动作
> 是回复F1并提交Round 2独立外审；获批前不得启动B4-B1/B4-B2、
> TIR实现或性能计时。

> **2026-08-18 FSG4/B4-B0 Round 1纠正指令**：外审`changes_requested`，F1 major确认原replay
> 未绑定绝对source/topology/lineage身份。v2修复已实现并在旧raw上拒绝11/11负例；下一唯一动作
> 是冻结提交后生成v2、replay、11类完整性负例与回归，再提交Round 2。批准前B4-B1/TIR关闭。

> **2026-08-18 FSG4/B4-B0 five-fresh内部关闭指令**：source=`1dbb2de`的formal artifact已通过
> 5 fresh/10 capture raw replay、108 tensor/664,744元素、max diff=`1.192e-7`、sign exact与
> 9/9 outer-resigned tamper。状态=`VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`。
> 下一唯一动作是外审；批准后只开放B4-B1 typed IR/reference，B4-B2 TIR仍关闭。

> **2026-08-18 FSG4/B4-B0 five-fresh runner指令**：runner、raw typed replay与9类tamper probe
> 已实现但formal artifact未运行。下一唯一动作是提交冻结runner，然后执行5个fresh CUDA进程、
> root replay与outer-resigned tamper；全部关闭前不得进入B4-B1或TIR。

> **2026-08-18 FSG4/B4-B0 live observer指令**：evaluation-0 observer与CUDA smoke已通过。下一唯一
> 动作是实现独立进程、raw-first的5-fresh B4-B0 artifact、root replay与outer-resigned tamper。
> artifact关闭前不得实现TIR，不得用单次CUDA smoke宣称correctness或performance。

> **2026-08-18 FSG4/B4-B0 capture合同指令**：typed schema和10项测试已实现，但尚未接live
> solver。下一唯一动作是在optimizer evaluation 0的显式opt-in路径捕获两锚点的
> compressed源状态、native dense输入、outputs和native gradients；未live 5 fresh前不得实现TIR。

> **2026-08-18 FSG4/B4-B v1执行指令**：预注册已完成，但未实现。下一唯一动作是
> B4-B0：在optimizer evaluation 0对`node31/Gemm_14`的active-beta语义锚点与
> `node25/Conv_8`候选性能锚点生成read-only production exact-call capture。两锚点5 fresh
> 语义/replay/tamper关闭前不得改TIR；不得放宽PR-12或把B4-A累计进baseline。

> **2026-08-18 FSG4/B4-A外审关闭指令**：Round 1独立外审AC1—AC7全部PASS，exchange已
> `closed/approved`，最终状态=`EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`。
> B4-A的1.018995x只是已审计的NO-GO数字，不得计入B4累计baseline。下一唯一动作
> 是单独预注册B4-B可微 lower-only CUDA/TIR双锚点实验；预注册前不得改TIR。

> **2026-08-18 FSG4/B4-A正式计时内部关闭指令**：source=`46a8493`的v5 24/24完成，correctness/
> environment/activation/profile/replay与14/14 tamper全过；core wall geomean=`1.018995x`未过`1.03x`，
> query worst=`0.996947x`过`0.98x`。下一唯一动作是外审；不得调低阈值或重跑挑样，B4-A不得累计为
> performance candidate。该“下一步外审”指令已由上方Round 1外审关闭取代。

> **2026-08-18 FSG4/B4-A正式计时v4环境处置**：source=`03043a3`的v4有19个worker admitted，
> run 19因旧门禁比较thermal/power累计绝对值而非worker区间增量被拒绝；其区间增量严格同为
> `2062477 µs`。已改为delta-exact投影并由replay从raw重算，tamper共14类。v4不得续跑或形成ratio；
> 下一步验证并以clean source从0运行v5，仍无performance claim。

> **2026-08-18 FSG4/B4-A正式计时v3环境处置**：source=`be2fa96`的v3完成20个worker后因执行期
> software thermal counter独立增长fail closed；v3不得续跑或形成ratio。正式功耗策略冻结为
> `nvidia-powerd=inactive`与`enforced.power.limit=55.0 W`，逐worker/replay重验，tamper共13类。下一步
> 该v4指令已被上方v4失败处置与v5指令取代。

> **2026-08-18 FSG4/B4-A正式计时v2环境处置**：source=`ee73bc2`的v2因worker 5独立software
> thermal slowdown fail closed；不得续跑或挑样本。preflight加固为`<=45°C`且software thermal完全
> inactive；下一步只允许clean-source v3从position 0重跑，仍无performance claim。

> **2026-08-18 FSG4/B4-A正式计时v1失败处置**：source=`292a035`的v1在worker 3因B4-A profiler
> alias覆盖缺口fail closed；v1不完整raw不得续跑或形成ratio。已修复并live验证物理计数；下一步只允许
> clean-source v2从position 0重跑，仍保持`performance_claimed=false`与B4-B/TIR关闭。

> **2026-08-18 FSG4/B4-A正式计时Runner指令**：24-process runner、raw-first/resume、root replay及
> 14类outer-resigned tamper probe已实现。下一唯一动作是提交clean source并运行正式GPU artifact；
> 在replay/tamper与外审前保持`performance_claimed=false`，不得启动B4-B/TIR。

> **2026-08-18 FSG4/B4-A正式计时指令**：只实现并运行冻结的24-process B3/B4-A协议；control检验
> core>=1.03x/query worst>=0.98x，profile只归因。不得复用correctness latency或启动B4-B/TIR。

> **2026-08-16 FSG4/B4-A five-fresh关闭指令**：correctness已以10/10 worker、5/5 pair、19 tensor/pair
> 和root replay内部关闭。下一步只实现独立B3/B4-A正式计时，检验core>=1.03x/query worst>=0.98x；
> 不得复用correctness latency或启动B4-B/TIR。

> **2026-08-16 FSG4/B4-A实现候选指令**：实现已到clean-source five-fresh前，下一步只生成5组独立
> B3/B4-A correctness artifact并root replay；未通过前不测正式性能，不启动B4-B/TIR。

> **2026-08-16 FSG4/B4-A预注册指令**：只实现第10次optimizer evaluation的terminal lower/六层lA
> typed handoff与zero-rerun export assembly；先验证state/topology/shape lineage、10/9/4/3物理结构及
> five-fresh correctness，再允许B3/B4-A计时。不得混入B4-B/TIR、JIT、runtime或allocator改动。

> **2026-08-16 FSG4/B4-0外审关闭指令**：Round 1审计AC1—AC7全部PASS，exchange已
> `closed/approved`，最终状态=`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。下一唯一动作是
> B4-A预注册与terminal lower/lA handoff；先解决shape-lineage与数值复用正确性，不得启动B4-B/TIR。

> **2026-08-16 FSG4/B4-0内部关闭指令**：source=`66154e4`正式artifact、root replay、9/9 tamper及
> opportunity门禁内部通过，状态=`INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`。
> 当前下一唯一动作是外部审计；批准后只启动B4-A terminal lower/lA handoff，不得把B4-B/TIR混入
> 同一变量，B4-C/D和B5—B7继续关闭。

> **2026-08-16 FSG4/B4-0 Runner指令**：schema/runner已实现为
> `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`，但正式fresh artifact和opportunity closure尚未
> 执行。下一唯一动作是提交clean source后生成control/profile raw、root replay与tamper；在B4-0门禁
> 关闭前不得实现TIR，B4-A/B/C/D和B5—B7不得提前混入。

> **2026-08-16 FSG4/B4预注册指令**：B4已冻结为14-call production lower-only CROWN累计路线，状态=
> `PREREGISTERED-NOT-IMPLEMENTED`。B3仍是直接基线、B0仍是累计对照；optimizer-only无限加速不足以
> 回到B0。下一唯一动作是B4-0 read-only kernel/materialization attribution；B4-0关闭前不得实现TIR，
> B4-A/B/C/D和B5—B7不得提前混入。计划入口为
> `gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`。

> **2026-08-15 FSG4/B3外审关闭指令**：Round 2独立外审AC1—AC7全部PASS，无blocker/major/minor，
> exchange已`closed/approved`。`VALIDATED-REDUCED-B3`正式关闭；下一唯一工程主线是以B3为直接对照的
> B4 operator/cross-stage CUDA/TIR fusion预注册与B4-0 qualification。B5—B7及最终system gate不得启动。

> **2026-08-14 FSG4/B3正式计时内部关闭指令**：source `36e9069`完成36/36 fresh worker、全部
> correctness/environment/measurement/activation、root replay与10/10 tamper。B2/B3 core/query=
> `1.071617x/1.006623x`，B0/B3 query=`0.910001x`，因此状态=`VALIDATED-REDUCED-B3`，不是full B3
> 或相对B0 speedup。该“external audit待完成”历史指令已由上方Round 2批准取代。

> **2026-08-14 FSG4/B3正式计时Runner指令**：冻结的B0/B2/B3六全排列36-process runner、direct
> activation receipts、raw-first/replay与十类tamper probe已实现并通过108项定向、1308项全量回归，
> 状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`。下一唯一动作是提交为clean source，从
> position 0运行完整artifact，再执行root replay与tamper；该历史动作现已由上方正式结果取代。

> **2026-08-14 FSG4/B3 Five-Fresh关闭指令**：source `75dfd81`按固定交替顺序完成10/10独立
> fresh GPU worker，5/5 B2/B3-C direct semantic pairs、environment、provider/fallback、physical
> counter、post-query audit和root replay全部通过，7/7 outer-resigned tamper拒绝；定向=`56 passed`，
> full=`1289 passed, 3 skipped`。状态=`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`，只将
> `b3_timing_admitted=true`。五组artifact仍为`timing_admitted=false/performance_claimed=false`。下一
> 唯一动作是实现并验证冻结的B0/B2/B3六全排列36-process正式计时；B4—B7不得混入。

> **2026-08-14 B3-C关闭指令**：source `72bec5e`的fresh GPU artifact已验证12个CUDA
> candidate/commit/backup/copy、candidate D2H=`0`、post-query audit、冻结语义、replay和6/6 tamper，
> 状态=`VALIDATED-B3-C-COUNTERS`，无timing/speedup。下一唯一动作是5组fresh B2/B3 correctness
> pairs；未5/5通过前不得启动36-process计时，B4—B7不得混入。该门禁现已由上方Five-Fresh关闭。

> **2026-08-14 FSG4/B3-B关闭指令**：source `42df2dc`的fresh GPU artifact已验证full step
> snapshots=`0`、forward builds=`4`，冻结语义、replay和6/6 tamper通过，状态=
> `VALIDATED-B3-B-COUNTERS`，无timing/speedup。下一唯一动作是B3-C device-resident AtomicCommitPlan；
> B4—B7不得提前混入。

> **2026-08-14 FSG4/B3-A关闭指令**：source `c7851c8`的fresh GPU artifact已验证template
> compile/hit=`1/1`、module move=`0`、scope=`1`，冻结语义、replay和6/6 tamper通过，状态=
> `VALIDATED-B3-A-COUNTERS`，无timing/speedup。下一唯一动作是B3-B terminal-only optimizer Schedule；
> B3-C和B4—B7不得提前混入。该“下一动作”已被上方B3-B关闭取代。

> **2026-08-14 FSG4/B3-0关闭指令**：B2显式counter已由source `4195361`正式关闭为
> `VALIDATED-B2-COUNTERS`；counter、六个冻结语义锚定、replay、6/6 tamper和全量回归均通过。下一唯一
> 动作是B3-A PreparedCoreTemplate/CorePlanInstance；该指令现已被上方B3-A关闭取代。

> **2026-08-14 FSG4/B3当前指令**：只启动
> `gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`的B3-0显式counter
> 诊断，然后依次B3-A PreparedCoreTemplate、B3-B terminal Schedule、B3-C AtomicCommitPlan。不得
> 并行混入TIR/JIT/streams/arena。当前无B3性能claim；cProfile冲突尝试已fail closed且不产生数字。

> **2026-08-14 FSG3当前指令**：正式`resnet2b-prop0-v5`已在source `a4ee291`完成六个全排列block、
> 36个fresh GPU进程，correctness/environment/measurement/replay全过，summary hash=
> `df852590d…1318e`。B1 query wall=`0.995657x`；当前B2 query/core=`0.908400x/0.516767x`
> （B0/candidate），显存ratio=`1.0`，故FSG3以`VALIDATED-FSG3-B0-B1-B2-BASELINE`关闭，B2分类为
> `MEASURED-B2-SLOWER`而非speedup。B2 core主要为optimizer/atomic commit/KFSB/typed pre-state
> `44.0%/24.7%/16.7%/10.7%`。下一唯一门禁为FSG4/B3 IR/graph/Plan/Schedule复用；不得把B4 TIR
> fusion、B5 JIT、B6 runtime或B7 arena提前混入B3，也不得把当前B2外推为全栈NO-GO。

> 生效日期：2026-07-12
> 当前 integration base：`331086d`（NRIR-39 merge）；历史 closure tag：`pr13-validated-reduced`、
> `ir5-final-validated-nogo`
> 当前研发分支：`feat/rvir-v4-production-state-ownership-v1`。FSG2历史 revisions=
> `aa31eae`/`8bf6981`；FSG0、FSG1均已验证；FSG2 initial-only validated-reduced与完整B2未准入的
> 结论只描述当时门禁，现已被后文RVIR-v4 V4-3 whole-core replacement证据取代。RVIR-v4现已关闭
> V4-1 frozen-state evaluation，V4-2A只关闭双LR/10-9 loop
> 子合同；重启后V4-2B正式GPU step trace的original replay与5类同步重签名tamper通过，以
> `VALIDATED-PRODUCTION-TRACE`关闭。它不是mutation replacement。V4-2C又从该正式capture独立重建
> 6组native α/β/split与真实scope，12/12 mutable round-trip exact；original semantic replay及6类
> 双层重签名tamper通过，以`VALIDATED-PRE-STATE-INITIALIZER`关闭。V4-2D formal native loop又独立
> 执行10 evaluations/9 updates，逐step lower/α/β全过`2e-4`门禁，original replay与6类双层完全重签
> 攻击通过，以`VALIDATED-NATIVE-STEP-PARITY`关闭。V4-2E又完成12-path private stage、原子commit与
> rollback，正式artifact original replay及6类完全重签攻击通过；V4-2整体以
> `VALIDATED-OPTIMIZER-REPLACEMENT`关闭。它还不是whole `update_bounds_core` live replacement；B2与
> 性能claim继续关闭。V4-3A现又冻结完整original core/post truth、pre-KFSB六层lA、六层intermediate、
> 三组candidate child lower与最终decision；fresh replay覆盖451 tensors/213,060 signs，六类同步重签
> 攻击全部拒绝，以`VALIDATED-WHOLE-CORE-TRUTH`关闭。它仍不是replacement；下一门禁只允许V4-3B
> native lA/intermediate export。V4-3B现又以零provider callback导出六层lA、12个shared-input
> intermediate tensors与final lower，最大差均不超过`6.08e-06`，五类同步重签攻击拒绝，以
> `VALIDATED-NATIVE-BACKWARD-EXPORT`关闭。V4-3C随后独立推导六层mask、复现三组top-3候选并执行
> 72个native child lower；candidate/final decision exact、child lower最大差`3.0994e-06`，八类同步
> 重签攻击拒绝，以`VALIDATED-NATIVE-KFSB`关闭。V4-3D随后在RTX 4060真实GPU进程完成
> whole-core→未修改official post/queue接通，
> provider core/compute/update/fallback=`0/0/0/0`，完整core/post最大差`1.0669e-05`且decision exact；
> fresh replay与8类完全重签攻击通过，以`VALIDATED-LIVE-RETURN`关闭。下一门禁只允许V4-3E
> five-fresh correctness。V4-3E现又按`O,C,C,O,C,O,O,C,O,C`启动10个fresh GPU进程，5/5 pairs
> 的完整core/post/state/branch/queue/termination全部通过，六类重签攻击拒绝，以
> `VALIDATED-FIVE-FRESH-CORRECTNESS`关闭；V4-3整体=`VALIDATED-WHOLE-CORE-REPLACEMENT`，B2
> same-solver timing在该时点准入但尚未执行。其后FSG3已按冻结协议关闭，当前结论以上方
> “2026-08-14 FSG3当前指令”为准；本段只保留历史门禁，不得再作为当前next action。
> PR-10—14 为历史执行顺序；当前 IR-first 顺序已推进到 **NRIR-15 E2E diagnosis（完成）→
> NRIR-16 prepared path（完成）→ NRIR-17 objective branching（完成）→ NRIR-18 multiworkload
> competitor E2E（完成）→ native intermediate-bound refinement（完成）→ objective-directed
> intermediate target selection（完成）→ per-child intermediate refinement（NO-GO）→
> ancestral-constraint carry-forward refinement（完成）→ external-seeded hard-clause convergence
>（完成）→ dynamic ancestral refinement budget（完成）→ typed multi-pass refinement（NO-GO）→
> production prepared verifier（完成）→ parametric compiler（完成）→ wall-clock scaling（完成）→
> typed hard-clause escalation（完成）→ objective-directed hard-clause escalation（完成）→
> objective-ancestral queue（完成）→ child-budget Pareto（NO-GO）→ sibling-packed evaluator（完成）→
> cross-clause anytime evaluator（完成）→ multi-clause anytime priority（NO-GO）→ shared parametric
> objective evaluator（完成）→ full-frontier tightness attribution（NO-GO）→ objective-branch shared
> evaluator（完成）→ NRIR48 CPU execution attribution（完成）→ NRIR49A GPU selected-CROWN-only
> opportunity attribution（NO-GO）→ FSG0 full-stack scope/schema/replay（完成）→ FSG1 official control
> full-stack trace（完成）→ FSG2 RVIR-v3 initial replacement（VALIDATED-REDUCED）→历史完整B2
> alpha/beta/split replacement（NO-GO/not admitted）→RVIR-v4 V4-1—V4-3（完成；取代历史ownership
> blocker）→FSG3/B2 same-solver timing（完成；B2较慢）→FSG4/B3 IR/graph/Plan/Schedule复用
> （当前）**。
> 禁止同时启动性能调优与 verifier control-flow 两条主线。

> **2026-07-20 路线修订**：PR-14 No-Go 后对代码进行 IR-first 复审，确认现有
> `runtime/linear_operator.py`、`PlanBundle` 和拓扑执行循环不能分别等同于完整 Bound IR、
> Plan IR 和 Schedule IR。第 10 节原定的纯 `docs/asplos-c1-c2-story-freeze` 不再是下一工程
> 主线；后续按第 11 节和
> `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md` 推进。

> **2026-08-03 最终状态**：IR-1—IR-4 narrow plain-CROWN compiler/runtime 已
> validated-reduced；IR-5D prepared execution remediation 已完成。fresh residual-v3
> final correctness/replay 全过，但 Global p90 1.26160×、gray 无 Pareto、无预算切换。
> IR-5 最终 VALIDATED-NO-GO；当前 ASPLOS system-performance 路线停止，IR-6 不启动。

> **2026-08-03 correctness 后续**：独立 RVIR 路线已以 CPU VALIDATED-REDUCED 关闭。
> ResNet external-semantics initial-CROWN 等价恢复；activation external exact call 已进入
> Bound/Plan/Task/Schedule typed stack。该结果不撤销 IR-5 No-Go，也不构成性能 claim；详见
> 第 12 节。

> **2026-08-04 P0 路线选择**：真实 production Schedule-memory 准入门禁为 `NO_GO`。
> Reduced residual path 有 arena/launch ownership，但没有 materialization、storage choice 或
> budget-driven decision switch；真实 ResNet 仍是单 external opaque call。不得直接重开 IR-5/
> IR-6，下一分支是 `feat/native-real-network-bound-ir-v1`，详见第 13 节。

> **2026-08-04 NRIR-1 结果**：固定 ResNet2B 的 main initial-CROWN backward 已从 external
> opaque wrapper 变为 21 个 native Bound/Task regions 和 21 次 Schedule launch；五层 hash
> 绑定 external-bound payload，CPU lower max diff `7.15256e-7`、sign 9/9。关闭等级只为
> correctness/compiler ownership VALIDATED-REDUCED；下一步是 NRIR-2 多计划/memory decision，
> 不是直接宣布性能结果。详见第 14 节。

> **2026-08-04 NRIR-2 结果**：同一 real ResNet Bound IR/PlanTemplate 已加入 retain-all 与
> lifetime-reuse 两个 storage plan；预算会切换 PlanInstance/Schedule，runtime 按 selected
> last-use 提前释放值。logical/observed peak 为 `1,860,912`/`442,656` bytes，两计划
> bitwise equal。该 closure 仍为 CPU mechanism/correctness，不是 CUDA memory/performance。
> 详见第 15 节。

> **2026-08-04 NRIR-5 结果**：真实 ResNet 的 spec BatchDecision 已驱动 source Schedule
> `[0,3)/[3,6)/[6,9)` 与三个各 21-op 的 child Bound/Plan/Task/Schedule stack；full/sliced
> lower max diff `1.90735e-6`、external sign 9/9，artifact generate/replay 已通过。全量
> `508 passed, 37 skipped`，状态为 correctness/integration VALIDATED-REDUCED；domain/sample、
> representation × batch 联合执行与任何性能/内存主张仍 pending。详见第 18 节。

> **2026-08-04 NRIR-6 结果**：同一 source PlanTemplate 已联合 representation/storage 与
> spec-batch 两轴；budget × spec limit 选择 dense/structured × full/sliced 四组合，child
> op/task/launch 为 `21/63/49/147`，且显式继承 source policy。四路径对 external lower
> max diff 均 ≤`1.90735e-6`、sign 9/9，artifact replay 与全量 `522 passed, 37 skipped`。
> 只升级 joint compiler ownership；跨 query/domain batching、cache 和性能仍 pending。详见第 19 节。

> **2026-08-04 NRIR-7 结果**：固定 ResNet 的 9 个不同 property objectives 已建模为 9 条
> query；packed size-3 执行 3 个 child，same-policy serial 执行 9 个，9/9 lineage 恢复。
> 首次 compile miss、第二次 exact hit，objective/order/state 均进入 cache key。packed/serial
> max diff `3.21865e-6`、external sign 9/9，全量 `540 passed, 37 skipped`。只证明 query
> formation/cache/packing correctness；BaB domain state 与性能仍 pending。详见第 20 节。

> **2026-08-04 NRIR-8 结果**：固定 ResNet 原 input box 已确定性三层二分为 8 个不同 leaf
> domains。每个 leaf 独立重算 IBP intermediate state；parent state 只标记
> `warm_start_only`，从未作为 child exact execution input。Plan 的 domain-size-4 candidate
> 实际驱动两个 child IR stacks；full-size-8 为一个，same-policy serial 为八个；三路径
> lower/upper bitwise 一致。该结果只关闭 input-box domain formation/state validity/packing，
> 不是 ReLU/β branch-and-bound、pruning、终止或性能证据。详见第 21 节。

> **2026-08-04 NRIR-14 结果**：九子句 conjunction、deterministic PGD candidate search、
> concrete witness replay、unsafe short-circuit 与 cooperative deadline 已形成可执行 query
> contract。toy verified/unsafe/unknown/deadline 均闭环；固定 ResNet 九子句全部执行，但
> native scalarized lower bounds 过松，9/9 unresolved，整体仍 unknown。只关闭 correctness/control
> VALIDATED-REDUCED；下一步必须先建立端到端 phase/tightness baseline，不能直接宣称性能。

> **2026-08-04 NRIR-18 结果**：三种 VNN-COMP 真实拓扑已进入 native VNNLIB Query IR 与
> workload Plan/Task/Schedule。BoundFlow 为 unknown×3，固定 αβ-CROWN 为
> verified/unknown/verified；单次 CPU E2E 只作诊断。ResNet native local root lower 仍达
> `-543.717/-789.331`，故下一门禁是 intermediate-bound refinement，而不是 GPU timing 或
> 继续加深同一 branching tree。详见第 31 节。

> **2026-08-04 NRIR-20 结果**：当前 scalar clause 的 CROWN coefficient influence 已成为
> refinement Plan/Task/Schedule 的显式输入；objective hash 与每个 target 的
> influence×width score 均冻结。固定 ResNet 前两个 hard clauses 在相同 96-target 预算下，相对
> width policy 的 root lower 再改善 `+55.928741/+26.228943`，但仍为
> `-417.292480/-602.551392`。该阶段只关闭 objective-directed IR/control 与 fixed root
> tightness `VALIDATED-REDUCED`；下一路线为 per-child refinement，不是性能或 closure claim。

> **2026-08-04 NRIR-35 结果**：NRIR-31 all-clause floor 与 NRIR-34 clause-0 packed queue 已由
> 一等 Plan/Decision/6-stage Task/Schedule 串接，并消费同一 global start。三轮 floor 均完成
> `[0..8]`，余量内 packed nodes=`[7,7,9]`；final 仍为 9/9 unresolved。该结果只关闭 cross-clause
> control/original-ordinal preservation `VALIDATED-REDUCED`，不形成 property/performance claim。
> 下一门禁为 multi-clause anytime priority/time slicing。详见第 48 节。

> **2026-08-05 NRIR-36/37 结果**：NRIR-36 的 typed top-2/equal-remaining control 三轮 coverage=
> `[[3,3],[3,3],[3,1]]`，因一轮第二条未提交 atomic pair 而 NO-GO。NRIR-37 随后保持控制、预算、
> cap 与 workload 不变，把逐 batch optimizer compile + selected-native audit replay 替换为 query-shared
> parametric Template/Instance。真实 parity 通过；三 fresh repeats 均 selected `[2,3]`、packed
> `[31,31]`、每轮仅一次 compile、whole `51.93—52.27 s`。只关闭 compiler ownership/fixed-deadline
> coverage `VALIDATED-REDUCED`；final 仍 9/9 unresolved，无 performance/ASPLOS-ready 升级。详见第 50 节。

## 1. 锁定的论文命题

BoundFlow 是面向神经网络验证中相关边界查询的 query- and memory-aware compiler/runtime。
它不重新发明 CROWN/αβ-CROWN/BaB，而是暴露 eager tensor execution 隐藏的结构、物化、
显存和跨查询复用决策。

三项正式贡献为：

1. **Structured Bound-Operator IR with Explicit Materialization Semantics**：保留可组合结构，
   显式表示 barrier、reason、bytes 与 lifetime；dense path 是参考语义，不承诺永不物化。
2. **Method-, Autograd- and Memory-Aware Materialization Planner**：在 bound method、
   differentiation/optimization stage、query workload、硬件 capability 和显存预算下选择
   物化、partition、fusion、batch、cache、recompute 与 storage/schedule。
3. **BaB-Oriented Repeated-Query Runtime for Multi-Spec and Domain Batches**：只把 multi-spec
   和 BaB domain batch 作为首篇主线；certified training 是第二客户端，其余场景为未来工作。

## 2. C2 的正式问题边界

输入为 `(G, Q, H, B, R)`：operator DAG、query 集合/分布、硬件 profile、显存预算和参考
bound 配置。计划为 `P=(m, π, f, b, c, r, s)`：materialization、partition、fusion、batch
layout、cache、recompute、storage/scheduling。

优化目标包含 amortized compile、execute、queue、transfer 和 peak-memory cost，约束为
`M_peak(P) <= B`，并要求 planned path 在相同浮点语义下保持 dense reference computation。
实现采用 candidate generation → staged cost-aware heuristic，不承诺精确联合求解；评价包含
fixed、local greedy、global heuristic 与 small-graph exhaustive oracle。

## 3. 状态有效性规则

Runtime 中的缓存对象必须标记为以下之一：

- `EXACT_REUSE`
- `CONDITIONAL_REUSE`
- `WARM_START_ONLY`
- `INVALIDATE`

| 对象 | Multi-spec | BaB 父→子 | 参数更新后 |
|---|---|---|---|
| 图结构 | EXACT_REUSE | EXACT_REUSE | EXACT_REUSE |
| Planner 模板 | EXACT_REUSE | EXACT_REUSE | CONDITIONAL_REUSE |
| 编译 kernel | EXACT_REUSE | EXACT_REUSE | shape/dtype 不变时 CONDITIONAL_REUSE |
| 参数相关常量折叠 | EXACT_REUSE | EXACT_REUSE | INVALIDATE |
| intermediate bounds | CONDITIONAL_REUSE | WARM_START_ONLY 或 INVALIDATE | INVALIDATE |
| α 参数 | CONDITIONAL_REUSE | WARM_START_ONLY | 通常 INVALIDATE |
| β/split state | INVALIDATE | 子节点专属 | INVALIDATE |
| 输出 bounds | INVALIDATE | INVALIDATE | INVALIDATE |

禁止把父节点 intermediate bounds 直接描述成子节点的有效精确结果。

## 4. Correctness/Soundness 术语

1. **数学 soundness**：由 CROWN/IBP/αβ transformer 与 solver 保证。
2. **编译变换语义保持**：dense/operator/planned/fused path 在相同浮点语义下保持参考计算。
3. **实现验证**：dense reference、allclose、gradient comparison、auto_LiRPA comparison、
   sampled concrete sanity 和 deterministic replay。

没有 outward rounding、误差 envelope 或 proof checker 时，不宣称 GPU FP32 对实数语义具有
严格 numerical soundness。论文统一使用：

> preserving the reference bound computation under the same floating-point semantics

## 5. 立即执行的 Gate 0

- 将当前环境迁移、TVM/tvm-ffi ABI、Conda hooks 和 PyTorch 2.12 reshape 兼容整理为独立边界；
- 去除 `crown_ibp.py` 全文件 Black 噪声；
- 建立统一 build/run workflow；
- 运行激活/反激活、nvcc、TVM CUDA、TVM↔Triton、auto_LiRPA 与全量测试；
- 将 MLP/CNN baseline 从单次 quick 升级为多次 reduced evidence；
- 只在 Gate 0 干净后启动 PR-10 instrumentation。

## 6. PR-10 的成功标准

- dense reference 数值等价；
- α gradient 等价；
- CROWN、α、αβ、BaB、CNN、DAG 回归通过；
- 主 coefficient 不永久退化为 dense；
- fallback/materialization reason、count、bytes 可追踪；
- materialization count/bytes 下降；
- Python lazy path 不强制当场加速，端到端性能门槛属于 PR-12；
- 不接受无法解释的严重 runtime 或显存退化。

PR-10 的第一步必须是 instrumentation，再改 ReLU operator。

## 6.1 PR-10 最终判定

- 表示、正确性与研究机会门禁：PASS；
- structured 统一默认策略：被证据否定；
- 默认保持 dense，structured 仅作为 feature-gated memory escape capability；
- plain CROWN 代表点 peak 降约 29.8%，但慢约 9.17×；α/αβ structured 出现显存恶化与
  6 个 OOM；
- 不再打磨 Python structured 特例，唯一主线转为 PR-11。

## 6.2 PR-11 最小执行边界

- 显式输入 `bound_method`、`requires_grad`、`optimization_stage`、alpha/beta/split state、
  spec/domain batch、operator summary、memory budget/available memory、reuse 与 target；
- v1 action 仅为 dense、structured、reduce-batch；capability filter 禁止当前 α/αβ optimize
  选择 structured；
- 先满足安全显存预算，再在可行候选中最小化 latency；不可行时确定性缩 batch 并 re-plan；
- 基线为 Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy、
  Global Planner、Oracle；
- 按 workload family held-out，不随机拆分相邻 shape。

当前进度（2026-07-12）：context/capability/action/plan dump、真实 CROWN shape summary、
CROWN/α/αβ runtime guard、per-case Oracle、architecture-family cost-model split 与 final held-out
matrix 已落地；全量 200 passed、1 skipped。mini-ResNet held-out 上 Global 为 239/239 feasible、
0 unexpected、median/p90 regret 1.0，但与 Memory-Threshold 决策完全相同，p99/max 仍为
5.44×/9.17×。因此下一 blocker 是 multi-barrier global placement，不是继续调单一 query
threshold；scheduler 自动缩 batch 仍未完成，PR-11/C2 状态保持 partial。

第三切片已新增 multi-barrier placement：Local 独立选最快可能超预算，Global 可联合选择 mixed
dense/structured 组合并由 runtime 按 ReLU source value 执行；合成机制与两 ReLU 数值/trace
门禁通过，全量增至 207 passed、1 skipped。该机制尚缺真实 barrier-level cost profile 与
mini-ResNet held-out Oracle，不得用合成结果替代论文证据。

Barrier evaluator 与 Global Retry 已进一步落地：在一个 final mini-ResNet `spec=32/domain=8` 的
8-budget reduced matrix 上，Global Retry 为 7/7 feasible、0 unexpected、median/p90 regret
1.159×/1.562×；Always Structured median 为 5.486×，Memory Threshold 为 2.668×。host runtime
已有真实 CUDA OOM catch/blacklist/retry 状态机，但当前结果仍是 measured budget-rejection replay，
随后真实受控 OOM 也已完成：380 MiB process-local cap、mini-ResNet s128/d32，all-dense 真实 OOM
后 all-structured 成功，3/3 独立重复稳定。BaB 长生命周期 scheduler、timeout 与状态泄漏验证仍
未完成。

随后补齐了 `latency_rank_stratified_v1` 有界候选序列：两个最快候选、80%/90% latency-rank
候选和最低 predicted-peak fallback，总尝试数上限为 5。mini-ResNet s32/d8 与 s128/d8 两组
held-out 均为 7/7 feasible、0 unexpected，median regret 为 1.159×/1.171×，p90 为
1.722×/1.221×，最大尝试数为 3/5；真实 380 MiB OOM 实验也已改走同一 bounded runtime 入口。
当前下一门禁是独立 workload/query 与 BaB scheduler，而不是继续调这两个点的候选分位数。
本切片收尾为全量 216 passed、1 skipped，Mypy/Pylint/diff check 均通过。

独立并行 branched-ResNet held-out 随后给出 No-Go：128/128 combinations 正确，有界 retry
9/9 feasible、0 unexpected，但 median/p90 regret 为 1.976×/4.494×。审计还确认现 evaluator
读取 held-out candidate 的 trace logical bytes，故只能称 profile-guided replay。PR-11 下一唯一
主线改为从 IR shape/fanout/live interval 静态生成 topology/liveness-aware barrier cost；完成前不进入
PR-12。
独立 topology 切片收尾为全量 217 passed、1 skipped，profiler Mypy/Pylint/diff check 通过。

Static-v3 随后移除了 evaluator 对 candidate trace 的 feature 依赖：Task IR + forward shape 静态
生成 shape/FLOPs/bytes/reuse/batch axes 以及 fanout/live-span/depth/merge/path summary。所有 profile
执行 3 次独立 shuffled order 并按 pattern 聚合 median；6-family/36-budget LOO 联合冻结
ridge=.001、retry factor=1.30。三组 final held-out 共 23/23 feasible、0 unexpected，median regret
为 1.000×/1.194×/1.880×，p90 为 1.747×/1.194×/2.377×，最坏 max 3.160×。
StaticPlacementQuery→model load→candidate generator→plain-CROWN runtime 已连通并通过真实 OOM
3/3。PR-11 closure audit 判定为 validated-reduced；统一 QueryState/BaB wiring 按原计划保留到
PR-13，当前可在独立提交冻结后进入 PR-12，不把 reduced 证据扩大成论文级 C2 complete。

冻结前高-regret 归因进一步表明：9 个 `regret >= 1.5` case 全部首先属于 bounded candidate
set 未包含 measured oracle，而非已有候选的 cost-model misrank；7 个 backend-gap flag 仅是
PR-12 待验证假设。PR-12 因此收敛为无梯度 plain CROWN 的 ReLU+Linear/Conv fused TIR，
不得回写 PR-11 profile，也不得包装成 Planner 修复。

PR-12 kernel foundation 已进一步覆盖 Linear 与 Conv 1×1/3×3、stride 1/2。Conv 使用显式
DSCOHW/OIHW/DSCIHW layout、原始 input-shape/output-padding contract 与 output-centric gather；
CUDA matrix 四项输出对齐，三个代表 codegen 点为 0 stack/spill/local-memory 指令。calibration
sanity 中前三点快于 PyTorch dense eager，但 stride-2 medium 仍慢 1.717×。因此当前状态只能是
kernel-level correctness/mechanism PASS。

PR-12D 已将 dense-boundary fused region 接入真实 plain-CROWN backward：显式 execution step
消费 Affine→ReLU，后端无关 executor 可在 Torch dense reference 与 TVM fused TIR 间切换；
Linear chain、stride-1/2 chain CNN、residual 与 stride-2 downsample mini-ResNet-like block 的
最终 bounds 对齐，DLPack storage alias 成立。随后复审发现 fanout contribution 丢失与
TVM-FFI custom-stream race；修复后 v1 只 fuse single-consumer Affine→ReLU，fanout/stale plan
确定性 fallback，并以 `tvm_ffi.use_torch_stream` 桥接 stream。multi-block mini-ResNet、fanout
soundness 与 adversarial custom-stream 回归通过，全量为 299 passed、1 skipped。PR-12D
correctness closure 现为 PASS。随后 PR-12E/F 建立 calibration-only backend Planner 与
default/custom-stream runtime Pareto：calibration 12/12、held-out 24/24 candidate rows 正确，
5/5 held-out 预算可行、0 unsafe，median/p90 regret 为 1.000×/1.262×。fused 在所有 held-out
降低 peak，但 memory-sensitive Linear 慢 4.21×，unseen Conv/mini-ResNet 也发生 latency reversal；
仅 3/5 选择更快或为预算唯一可行。故证据链 PASS、性能门禁 FAIL、Planner quality 仅
guarded/partial，PR-12 overall 继续 IN PROGRESS，PR-13 继续阻塞。

PR-12G 随后没有回写 v1 held-out，而是先从 TIR source/schedule 归因 Linear 长 reduction，增加
`pytorch_chunked_r512` 预算型候选，再冻结全新 multibackend-v2 split。authoritative v2 证据为
calibration 48/48、held-out 36/36 candidate rows 正确；offline calibration-only Planner 在 5 个
held-out 上 5/5 预算可行、0 unsafe，exact Oracle 3/5，median/p90 regret 1.000×/1.054×，并分别
选择 eager/chunked/TIR 1/2/2 次。selected geomean 相对 eager 为 1.081×，memory-sensitive
Linear 同时满足 64 MiB 预算并比 eager 快 1.481×。这使 reduced 多后端 Planner quality 通过，
但不能替代 structured-eager/TVM-unfused baseline、真实 profiler 与 2× headline 门禁；PR-12
overall 和 PR-13 状态不变。

PR-12H 已切换到证据闭环阶段：`44f87ae` 以本地 tag `pr12g-validated-reduced` 冻结；kernel、
region-runtime、end-to-end final-bound 三层 benchmark contract 有机器可读 schema。审计确认旧
fused-sanity 的 allocation contract 不公平，旧 PR-12E/G candidate timing 又不包含 timed
region matching/Planner，故统一保留为 `compliant=false` historical evidence，不重写旧数值。
下一唯一工程阶段是 PR-12I structured eager/TVM-unfused 公平 baseline，仍禁止启动 PR-13。

PR-12I 已在新合同下补齐 structured eager 与显式 scaled-A workspace 的 TVM-unfused 对照：
正式 v2 为 72 rows（54 ok、18 N/A、0 correctness failure）。complete final-bound 中 TVM fused
geomean 仅为 eager 的 0.546×，但 median peak ratio 为 0.512 且 3/3 Pareto；TVM-unfused 为
0.481×、0/3 Pareto，说明 fusion 的主要已验证价值是消除中间物化而非普遍加速。条件
`torch.compile(fullgraph=True)` 在三类 workload、两种 stream 上均因 `ContextVar.set` 无法
capture，已保留结构化 N/A，未为迎合 baseline 改写 workload。下一唯一阶段为 PR-12J
compile/load/cache amortization；PR-12 overall 与 PR-13 状态不变。

PR-12J 已把 TIR generation、schedule、compile、serialization、module load、memory hit 与独立
进程 disk hit 分离。authoritative v4 为 3/3 correct、0 hidden recompile。Linear/Conv 因 fused
warm 本身较慢而不可摊销；mini-ResNet 对 eager 的 fresh/disk-first/process break-even 为
4668/1062/4450 queries，均超出 Q≤1024，且对 chunked 仍不可摊销。v1 的 Conv tuple/list cache
验证 bug 和 v2 的 warm-path SHA 污染均保留为失败证据。下一唯一阶段为 PR-12K profiler；不得
以 module load 仅 0.17–0.60 ms 掩盖 process first query 约 350–419 ms 的事实。

PR-12K 在不改 schedule 的前提下完成 6 workload×5 backend 的 complete final-bound CUPTI
activity profile，30/30 correctness 通过。Nsight Compute 2026.1.1 实测因
`RmProfilingAdminOnly=1` 返回 `ERR_NVGPUCTRPERM`，因此只报告 kernel/activity time 与 launch，
禁止 bandwidth/cache、occupancy、stall 等硬件 counter claim。fusion 对 TVM-unfused 最大整体
launch 降幅仅 1.96%；按 5% 阈值为 3/6 device-time 退化、1/6 改善、2/6 中性。PR-12L 唯一
选择分支 E：停止继续手工调孤立 TIR，保留 fused 作为 Planner 候选；下一工程阶段是全新 split、
多预算和 expected-reuse 驱动的 PR-12M compile-aware Planner。PR-13 继续阻塞。

PR-12L 已将该结论冻结为唯一分支 `E_STOP_OPTIMIZING_TIR`，且没有 TIR/schedule/runtime 代码
变化。Linear tiled reduction、CUDA Graph/dispatch、chunk-size family 与 Conv capability 扩展均
不进入本次 closure；它们若未来重启，必须使用新假设和新 split。PR-12M 只能推进
capability→budget→risk→amortized latency Planner，并一次性消费全新 final held-out。

PR-12M 已完成上述 Planner 与全新 v3 held-out。calibration/final candidate 均 25/25 correct，
fit 前 manifest 明确 final 未消费且 fit/replay model SHA 一致。16/32/64/128 MiB/unbounded ×
Q1/Q32/Q1024 共 75 decisions；72 个存在实测可行 candidate 的机会全部选到可行 backend，
0 unsafe，feasible median/p90/max regret 为 1.000×/1.000×/1.016×。计划随 reuse/budget 在
eager/chunked/structured/fused 间变化；3 个 16 MiB capacity failure 单列。下一唯一阶段为
PR-12N closure audit，仍禁止启动 PR-13。

PR-12N 最终判定为 `VALIDATED-REDUCED`，closure tag `pr12-validated-reduced`。它不满足 full
`VALIDATED`，因为 Q≤1024 compile 摊销为 0/3、硬件 counter 不可用、收益只在部分 regime，
且尚无真实 BaB/VNN-COMP；但 non-toy E2E Pareto、预算价值、自动选择与独立 held-out 足以避免
`MECHANISM-ONLY`。PR-13 硬门禁因此 GO/READY，但本 closure 不启动 PR-13；后续只允许推进
真实 multi-domain/BaB query runtime，不回到 PR-12 TIR 试参。

PR-13A 随后正式建立 state-versioned `BoundQuery`、完整 compatibility key、四级 state-validity
规则和 BaB recorder。现有 host solver 生成的 8-query two-ReLU smoke 固定流为 8/8 replay、
max abs diff 0、0 loss/duplicate。该结果只关闭 contract/replay foundation；PR-13B dynamic
BatchManager、same-solver multi-backend、non-toy TTV 与 tail latency 均未完成。

PR-13B 现已补齐 exact-key dynamic buckets、budget first-fit、fill/timeout/deadline、OOM 二分重试、
结果顺序恢复和 queue/fill/latency counters，并通过现有 αβ dense executor 做真实 physical pack/
unpack。8-query smoke 动态 3 batches 为 8/8、0 loss；OOM fault 8→4+4→四个 2 后仍 8/8。
当前仍只称 foundation；下一阶段是 PR-13C same-solver adapter。

PR-13C 已把 query runtime 作为 optional bound-call adapter 接回同一 `solve_bab_mlp`。αβ
steps=3/batch=4 smoke 中 original/runtime query ID、bounds、branch、αβ state 与 solver
status/node counters 全部一致（7/7、0 loss）；forged plain capability 在 executor 0 调用时拒绝。
单次 wall time 不具权威性。

PR-13D/E 随后在 RTX 4060 上完成 5-repeat fixed/E2E reduced 评估并以
`VALIDATED-REDUCED` 关闭：fixed runtime 相对 per-node 96.52×、相对 batched original 1.024×；
hard 16-node E2E 分别为 9.93×/0.980×，status/node count 一致。结果证明 runtime 能保留 batching
收益，但不证明超越普通 batching。αβ/split 对 PR-12 compiled Planner 不兼容，non-toy/VNN-COMP、
真实 OOM 和完整 TTV 未完成；ASPLOS-ready 仍为 NO。

## 7. 投稿门禁

- **7 月 26 日**：PR-10 与真实 materialization profile；
- **8 月 5 日**：第一次硬 Go/No-Go；必须已有非平凡 Planner、held-out 非 toy workload、
  首个 latency–memory Pareto、不同预算下不同计划、0 unexpected OOM，并报告相对 Oracle
  的 median/p90 latency regret；
- **8 月 14 日**：fused task 与 headline v0；
- **8 月 15 日**：BaB prototype 与前两页初稿；
- **8 月 20 日**：主实验冻结；
- **8 月 24 日**：最终投稿决定；
- **8 月 25 日后**：禁止新增技术功能。

8 月 5 日任一核心条件缺失，立即切换 ASPLOS 2028。

## 8. 公平 baseline

- PyTorch eager、`torch.compile`/TorchInductor、TVM default Relax/TIR；
- BoundFlow dense、always-lazy、fixed barrier、local planner、global planner；
- auto_LiRPA、α,β-CROWN、条件允许时 Luna；
- 最重要的端到端对照是：**相同 host solver，只替换 original executor 与 BoundFlow executor**。

每个后续 PR 必须回答：消除什么瓶颈、改善哪个北极星指标、为哪项论文贡献增加证据、如何
验证参考语义、原始 JSONL/表图/manifest 在哪里。

## 9. PR-13 后批准路线：PR-14 Verification-Aware Execution on Real Verification Workloads

PR-13 已以 `VALIDATED-REDUCED` 关闭。其 fixed/E2E 大幅逐节点 speedup 主要来自普通物理
batching；相对公平 batched original 没有稳定净加速。因此下一阶段不得回到 PR-10B.2、继续
孤立 TIR 调优或重新设计 BaB 算法。

PR-14 的唯一目标是量化并验证已有 `BoundQuery`、Planner、multi-backend execution 和 same-solver
adapter 在真实 complete-verification workload 中的 coverage 与作用：

1. PR-14A：真实 verifier/workload adapter、query distribution 与 backend eligibility coverage；
2. PR-14B：固定真实 query replay、backend eligibility 与公平 original-batched 对照；
3. PR-14C：只在 Go 后运行 CIFAR CNN、multi-block ResNet、VNN-COMP 代表实例的完整评估。

PR-14 不重新实现 query recorder；PR-13A 的 state-versioned contract、split lineage 和 fixed replay
是唯一基础。完整门禁见 `gemini_doc/pr14_execution_plan.md`。在真实 workload、0 query loss、
same-solver correctness 和相对 batched-original 的可归因证据成立前，ASPLOS-ready 继续为 NO，
C3 不得描述成“更快的 BaB runtime”。

## 10. PR-14A/B 最终判定：VALIDATED-NO-GO

PR-14A observer 在官方 MLP/CNN 与 VNN-COMP ResNet-2B 上记录 540 个真实 bound calls；
initial phase 有 143/146 个 query 含 capability-legal region，但 activation-BaB 为 0/394。
因此 PR-14B 只允许 replay initial plain-CROWN，不新增 α/β/split kernel。

PR-14B 使用真实 `x_L/x_U/C` 和 exact per-element box。simple MLP 的 external replay 与
BoundFlow eager/chunked/TVM lower 完全对齐，但 external 请求 lower-only，而当前 BoundFlow
总是 lower+upper，故不产生公平性能 claim。VNN-COMP ResNet-2B nominal forward 与 ONNX 对齐到
`1.67e-6`，但 whole-query lower 对 external max diff `796.765`，符号仅 3/9；same-solver
替换会改变 incomplete-verifier decision。

硬决策：

1. PR-14C 不启动，不用 full E2E 绕过 bound-equivalence gate；
2. 不继续调 TIR，不新增 α/β/split kernel，不重写 verifier 算法；
3. C3 降级为支撑 C1/C2 的 query/state/capability infrastructure；
4. 原判定的下一分支为 `docs/asplos-c1-c2-story-freeze`；该项已被第 11 节的 IR-first 复审
   取代。若未来研究 external-semantics-preserving region adapter，仍必须另立新假设。

最终证据见 `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`。ASPLOS-ready 继续为
NO，直到 C1+C2 paper-level story 独立通过评审门禁。

## 11. 2026-07-20 IR-first 路线纠正

PR-14 后复审发现，C1/C2 不能只靠整理已有 story 达到 paper level：

1. `boundflow/ir/bound.py` 仍是占位骨架，结构化系数语义主要存在于 runtime Python 对象中；
2. `PlanBundle` 及 PR-11/12 的局部计划对象尚未汇合为带统一引用、合法性和 replay 的 Plan IR；
3. 现有 scheduler 只是 TaskGraph 拓扑执行，项目不存在一等 Schedule IR；
4. PR-13/14 的 query/runtime 结果不能弥补上述编译器核心缺口。

因此下一工程分支修订为 `feat/compiler-ir-stack-v1`，顺序冻结为：

```text
Bound IR v1
  -> Plan IR v1
  -> Task IR + Schedule IR lowering
  -> reference/backend runtime migration
  -> adaptive PlanInstance evaluation
```

在三层 IR 的 typed schema、verifier、deterministic dump/hash 和最小端到端闭环完成前：

- C1 只能称 runtime mechanism foundation；
- C2 只能称局部 planner/backend mechanism validated-reduced；
- C3 保持降级，不以普通 batching 或计划中的 JIT 重新包装；
- 不新增 α/β/split kernel，不重写 BaB，不继续孤立 TIR 调优。

完整对象边界、迁移关系、JIT/状态有效性门禁和逐阶段 DoD 见新的架构契约文档。

2026-07-28 状态追加：

- IR-1 typed Bound IR、plain-CROWN lowering、dense/structured interpreter/rewrite 已通过；
- IR-2 typed PlanTemplate/PlanInstance、builder/selector、state-validity、legacy assembly 与
  deterministic artifact replay 已通过 reference closure；
- 当前 `artifacts/` 不含 PR-11/12 raw planner records，因此不声称历史逐记录迁移；
- IR-3 typed Task/Schedule schema、逐 Task reference semantics、control trace 与 artifact v2
  已通过 synchronous reference closure；
- IR-3 关闭时冻结的下一步曾为 IR-4 production backend/runtime migration；该动作现已由
  下方 IR-4A—E 完成记录取代，C1/C2 在 IR-5 公平证据前仍不得升级为 paper-level complete。
- IR-4A 已新增跨 Bound/Plan/Task/backend 的 typed dispatch key 和 PyTorch reference
  prepared-task adapter；这只是迁移入口，chunked/structured/TVM/query runtime 仍为 pending。
- IR-4B 已把 PyTorch dense/structured/chunked 接入 typed registry；chunked fused Task 在 CUDA
  上真实调用原 executor。下一门禁是 TVM typed compile/cache，不是孤立 kernel 调优。
- IR-4C 已完成 TVM fused/unfused typed dispatch、dispatch-key cache v2、跨进程 disk replay
  与 Schedule semantic OOM fallback；
- IR-4D 已完成 capability-gated typed compiler query、Plan/Task cache、exact-version dense
  state payload、真实 load/store/task skip 与 fresh-process artifact；PR-13 α/β 请求保持
  external No-Go，不降级为 plain CROWN；
- IR-4E 已新增 `plain_crown_typed_ir` BoundQuery capability，并让 PR-13
  DynamicBatchManager 只通过正式 adapter 调用 typed compiler；旧 `SameSolverQueryRuntime`
  默认关闭，仅 PR-13 历史回归显式 opt-in，且错误/审计保留 PR-14 No-Go；
- IR-4 已以 validated-reduced 关闭。下一步进入 IR-5 adaptive PlanInstance；不得提前启动
  IR-6 cached specialization，也不得把 compiler closure 升级成 α/β external integration。
- IR-5A 已新增 query-time memory/deadline/cache/distribution context，并按 uncached
  compile/setup 在 expected query count 上摊销选择；cold/repeated/warm context 可产生不同
  typed PlanInstance；在 IR-5A 时点仍需 fixed/local/global/oracle 与 held-out 系统证据。
- IR-5B 已冻结四策略共享 observation/context 的公平 evaluator，输出 tail/TTV/peak/regret；
  当前 artifact 明确为 synthetic contract，不得写成 held-out 性能结果。
- IR-5C1/C2 已冻结 calibration-only CUDA runner 和资源 context，并在 fresh typed MLP
  artifact 上得到 Global 8/8 feasible、p50/p90 regret 1.000×/1.00766×；高内存选择
  PyTorch dense，低内存选择 TVM fused。该时点结果仅为 PARTIAL：同-family split、
  ordinary batching/fair batched-original 与 non-toy workload 尚缺，随后由 IR-5C3 补测。
- IR-5C3 已用 MLP calibration→chain-CNN held-out 和 fair batched-original 补齐关键口径；
  correctness/feasibility 全通过，但 Global p50/p90 regret 为 68.065×/70.263×，
  64/512 MiB 均选 chunked且无 Pareto。当前 IR-5 v1 VALIDATED-NO-GO，IR-6 blocked。
  profile 指向 query hot path 重复 validate/hash。
- IR-5D 已把静态 validate/hash/dispatch key 移入 prepared capsule，并分离 audit/production
  trace；在旧 CNN 上使用 from-forward-trace 公平计时的 calibration median 比值最快为
  0.880×/0.896×。该诊断不撤销 No-Go；其后按预注册门禁执行了一次 residual final。
- IR-5E 已新增 residual fanout/add typed workload，并冻结 chain-CNN calibration →
  residual-CNN final v2、from-forward-trace baseline、p90≤1.20 与 Pareto 判定字段。
  `7401/7402` 随后首次生成时因输入身份协议错误失效并永久退役。
- IR-5F 首次 v2 生成在 semantic gate 中止：同 seed 不保证不同 batch shape 的随机输入
  具有前缀关系。参数一致但 input max diff 为 3.735/2.167；无 summary/manifest，不能作
  性能结论。只允许显式 slice batched input、升级 schema 并旋转 fresh identities。
- IR-5G 已实现上述唯一修复：single 输入 exact clone batched query zero，并在 bound
  comparison 前做 tensor identity gate；v3 `7501/7502` 随后按协议运行一次并冻结。
- IR-5H v3 final correctness/integrity/semantic replay 全过，但 Global p90 regret
  `1.26160× > 1.20×`，gray compiler frontier 只有单点且无 multi-budget switch。
  按冻结止损规则，IR-5 保持 VALIDATED-NO-GO，禁止继续旋转 final 或启动 IR-6。

## 12. 真实 Verifier IR correctness 路线关闭

IR-5 No-Go 后另立的 `feat/real-verifier-ir-integration-v1` 不继续性能调参，只修复并审计
PR-14 暴露的两个 correctness 缺口：

1. ResNet initial-CROWN 通过显式 external intermediate bounds 与 adaptive ReLU slope，
   lower max diff 从历史 `796.765` 降为 `3.09944e-6`，sign 从 3/9 恢复为 9/9；
2. activation-BaB 作为 provider-owned external exact operation 进入 Bound/Plan/Task/Schedule
   stack。历史 394/394 query 可生成五层 IR hash；当前 CPU 真实运行 377/377 dispatch 完成，
   observer on/off 均访问 380 domains 且 final lower 一致。

范围必须按三条口径分开：

- fused BoundFlow kernel replacement 仍为历史 `0/394`；
- typed external-call admission 为 `394/394`，但历史 v1 identity 有明确 limitation；
- 当前 adapter v2 exact execution 为 `377/377`，external αβ-CROWN 继续拥有算法和 termination。

全量回归 `452 passed, 37 skipped`，artifact fresh-process replay 通过。因本机 CUDA 不可用且
external lower-only 公平性能合同未建立，关闭等级为 correctness/integration
VALIDATED-REDUCED，ASPLOS system-performance 总判定仍为 NO。

## 13. Production Schedule IR + Memory P0 门禁

RVIR closure 后没有凭对象名称直接宣布 Schedule IR 已成为论文主线，而是对当前 production
控制面做了独立、可重放的 P0 audit：

1. residual-final-v3 的 8 个 workload/backend case 均由 Schedule IR 覆盖完整 10-op Bound
   graph，并显式拥有 budget check、arena allocate/free、batch loop 与 launch；
2. 这些 case 没有 `MaterializeAction`，且每个 template 只有一个 batch/storage candidate；
3. 64/512 MiB 下 PlanInstance hash 会变化，但 region/representation/backend/batch/storage/state
   决策均不变化；冻结 artifact 同样没有 multi-budget switch，双 workload Pareto 失败；
4. VNN-COMP ResNet 的 51 个 activation call 五层 IR hash 可逐条复算，但每条主图仍只是一个
   provider-owned `EXTERNAL_VERIFIER_CALL` 和一个 launch；
5. 当前没有 production OOM-rescue artifact。

因此 `feat/production-schedule-memory-v1` 不准入。下一唯一工程问题改为：能否把一个冻结真实
residual network 的 main compute lower 为非 opaque、multi-region native Bound IR，并先通过
external-semantics correctness。只有此后存在至少两个合法 storage/batch plan、预算触发真实
决策切换，且出现 baseline OOM rescue 或可重现 memory Pareto，才允许重开 Schedule-memory
性能路线。P0 artifact 位于
`artifacts/schedule-p0/production-schedule-memory-p0-20260804/`。

## 14. Native Real-Network IR v1 与下一门禁

NRIR-1 固定 VNN-COMP 2021 `resnet_2b.onnx`、prop0 VNNLIB、αβ-CROWN commit 和 6 组逐 ReLU
external preactivation bounds。新的 portable payload 对 identity/tensor tamper fail closed，并让
aggregate digest 进入 ReLU state version 和 Plan provenance。

执行结果：17 个 Primal ops lower 为 21 个 native Bound ops；PlanInstance 选择 21 个 singleton
reference regions；Task IR 与 Schedule IR 分别拥有 21 units/launches；Bound/Task external-call
count 均为 0。五层 hash fresh replay 一致，final lower 对 external oracle max diff
`7.152557373046875e-07`、sign 9/9。

这只证明真实主 backward 已进入编译器 IR。external intermediate bounds 仍由 αβ-CROWN 提供，
NRIR-1 冻结时 Plan 只有一个 dense storage/full batch、没有 materialization alternative，也没有
GPU/timing。storage-axis 后续已由 NRIR-2 完成；历史获准顺序修订为：

```text
NRIR-2 real-graph storage alternatives + runtime last-use (completed)
  -> fresh CUDA physical-memory/OOM protocol, if device is available
  -> representation semantic binding + real materialization
  -> sliced batch execution
  -> only then reconsider Schedule-memory/performance claim
```

artifact 位于 `artifacts/native-real-network-ir/vnncomp21-resnet2b-prop0-cpu-v1/`；实现与复现命令
见 `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_PLAN_2026_08_04.md`。

## 15. Native Real-Network Memory Plans v1 与下一门禁

NRIR-2 没有改变 NRIR-1 的 Bound semantics。它从原 dense storage baseline 派生：

1. retain-all：独占对齐 byte ranges，所有 value 保留到 final op；
2. lifetime-reuse：使用 verified exact last-use，只让 lifetime 不重叠的值复用 arena range，
   Task runtime 在消费完成后删除对应 tensor/operator reference。

固定真实 ResNet 上，二者共享 Bound hash `16e27f31...80fb` 和 PlanTemplate hash
`359ee68f...43f3`。高预算选择 retain-all（`1,860,912` bytes）；预算降至 `442,656` 选择
lifetime-reuse；再减 1 byte 以 `memory_budget_exceeded` 拒绝。低内存计划有 386 对合法 alias、
85 个 final-task 前释放。两计划 final lower/upper bitwise equal，对 external lower max diff
`7.152557373046875e-07`、sign 9/9。

该结果关闭 real-graph storage decision mechanism，但不关闭 performance：

- `442,656` 是 Plan/Schedule arena 与 runtime live-value ledger，不是 CUDA allocator counter；
- runtime release 会删除引用，但当前没有 `torch.cuda.max_memory_allocated/reserved` 或 OOM rescue；
- `0.001 ms` policy cost 只用于稳定排序，标注 `policy_cost_not_benchmarked`；
- Plan representation decision 仍未绑定 Bound rewrite/backend semantics，Schedule
  `MaterializeAction` 仍只记账；full-query batch 也尚未被 slice execution 消费。

下一动作优先尝试 fresh CUDA physical-memory protocol。只有实际 device measurement 同时通过
correctness、重复运行和 baseline OOM/Pareto 门禁，才可进入性能主张；若 CUDA 不可用，则转向
representation semantic binding bridge，不等待或伪造设备结果。artifact 位于
`artifacts/native-real-network-memory-plans/vnncomp21-resnet2b-prop0-cpu-v1/`。

## 16. Native CUDA Physical-Memory Protocol v1 与环境边界

NRIR-3 已在任何正式 CUDA 结果产生前冻结并实现双 storage 的设备测量协议：

- 5 个 repeat，每个 plan/repeat 独立 fresh process；偶数 retain→reuse、奇数反向；
- 每 worker 5 warmup、20 measured，计时只覆盖 prepared lower-only native CROWN execution；
- 同步采集 baseline/peak allocated 与 reserved delta，保留全部 latency samples；
- 模型、intermediate-bound、环境、worker PID、Bound/PlanTemplate、result hash 和 raw→summary
  派生关系全部 fail closed/replay；
- 只有 reuse median allocated delta 至少降低 20%，且 median latency 不超过 retain 1.20×，
  才允许 `performance_claimed=true`；reserved 只报告，无实际 OOM 不声明 rescue。

当前主机的 PyTorch CUDA build 为 13.2，但 driver/device 不可用。`probe` 已以 exit 2 生成
digest-protected `environment_unavailable` artifact，`generate` 在输出目录和 measured row 产生前
exit 2。因此本阶段只关闭 protocol implementation，不产生 performance No-Go/Go。全量回归
`484 passed, 37 skipped`，Mypy/Pylint/Black/diff check 均通过。

冻结顺序继续为：

```text
CUDA protocol implemented; device run pending external availability
  -> representation semantic binding + executable MaterializeAction (completed)
  -> sliced batch execution
  -> only then reconsider Schedule-memory/performance claim
```

下一分支不得只新增 representation metadata/hash；至少一个 Plan decision 必须驱动真实 Bound
rewrite/backend conversion，并在固定 ResNet 上以 dense reference/external oracle 双重校验。

## 17. Native Representation Semantic Binding v1 与下一门禁

NRIR-4 已让 source Plan representation decision 真实决定 execution Bound program，而不是只改变
candidate ID：dense policy 执行原 21-op graph；structured-affine policy 由 binder 生成另一份
49-op graph，其中 14 cast + 14 materialize 全部与 selected transition、source Schedule action、
Task 和 Launch 一一对应。execution graph 使用独立 Plan/Task/Schedule identity。

固定 ResNet fresh replay 中，高预算选择 dense/retain-all，`442,656` bytes 选择
structured-affine/lifetime-reuse，再减 1 byte fail closed。dense/structured lower 最大差
`9.5367431640625e-07`，二者均对 external oracle allclose、sign 9/9。artifact 位于
`artifacts/native-real-network-representation-binding/vnncomp21-resnet2b-prop0-cpu-v1/`。

这只关闭 C1/C2 的 representation semantic binding mechanism：当前 structured operator 仍包装
dense tensor，storage 仍按 dense-equivalent bytes 记账；policy 与 storage 绑定也不能归因为
structured compression。`performance_claimed=false`，禁止 memory/latency/CUDA/OOM/Pareto/speedup
表述。

当前唯一代码顺序修订为：

```text
NRIR-4 representation binding (completed)
  -> real-network sliced batch execution
  -> execute frozen NRIR-3 CUDA protocol when a device becomes available
  -> only with physical evidence reconsider Schedule-memory/performance claim
```

下一分支至少要让一个 domain/spec/sample batch decision 改变实际 Task/Schedule slice 和 query
accounting；仅新增 batch metadata、hash 或 synthetic loop 不算完成。

## 18. Native Real-Network Sliced Batch Execution v1

NRIR-5 已实现一个真实、可重放的 spec-axis batch execution slice：同一 source BoundModule 与
PlanTemplate 同时包含 full-query 和 spec-size-3 BatchCandidate；query-time
`max_spec_batch_size` 进入 selection provenance，并分别生成 full/sliced PlanInstance 与 Schedule。
sliced source Schedule 明确拥有 `[0,3)`、`[3,6)`、`[6,9)` 三个 objective ranges，每个 range
编译为独立 21-op Bound、21 Task、21 Launch 的 child stack，执行结果按 spec 轴拼接。

固定 ResNet 上，full path 为 1 个 child，sliced path 为 3 个 child、合计 63 Task/Launch；二者
共享 source Bound/PlanTemplate，但 source PlanInstance/Schedule identity 不同。full/sliced lower
最大差 `1.9073486328125e-06`；full/external 为 `7.152557373046875e-07`；sliced/external 为
`1.9073486328125e-06`，均 allclose、sign 9/9。artifact 位于
`artifacts/native-real-network-sliced-batch/vnncomp21-resnet2b-prop0-cpu-v1/`，包含 binding、
child IR、execution trace、digest 与同步重哈希后的结构篡改门禁。

本阶段只关闭 spec-axis actual-execution ownership，不能升级为 batching performance：三个 child
当前顺序执行，source storage 仍为完整 query ledger，未测 allocator/latency。domain/sample 尚未
实现；NRIR-4 representation 与 NRIR-5 batch 各自成立，但其联合 cross-product 尚未执行。

后续唯一代码顺序为：

```text
NRIR-5 spec sliced execution (completed)
  -> representation × batch policy composition (completed)
  -> real repeated-query/domain batching and cache/accounting
  -> execute frozen NRIR-3 protocol when CUDA is available
  -> only with physical end-to-end evidence reconsider ASPLOS performance claim
```

## 19. Native Representation × Batch Composition v1

NRIR-6 把 NRIR-4/5 的两个独立 mechanism 放入同一 source template 和 selector。高/低 memory
budget 决定 dense-retain/structured-reuse；query-time max spec 9/3 决定 full/sliced。因此四个
组合不是四条硬编码入口，而是同一候选空间的四个可验证 PlanInstance/Schedule。

source selected storage/representation 通过 `required_storage_candidate_id` 显式传播到每个 child；
即使切片改变 child shape，也不能重新打分并偷偷换 policy。四组合共享 source Bound/PlanTemplate，
有四个不同 PlanInstance/Schedule。真实 ResNet child op/task/launch 为 `21/63/49/147`；structured
两条路径继续保留 source 28 transitions/49-op execution ownership，sliced 两条路径继续拥有
`[0,3)/[3,6)/[6,9)` ranges。

对 frozen external lower 的最大差依次为 `7.152557373046875e-07`、
`1.9073486328125e-06`、`9.5367431640625e-07`、`1.6689300537109375e-06`，均 allclose、sign
9/9。artifact 位于
`artifacts/native-real-network-joint-policy/vnncomp21-resnet2b-prop0-cpu-v1/`；聚焦
`103 passed`，全量 `522 passed, 37 skipped`，静态门禁全过。

该结果只关闭 cross-axis compiler/runtime ownership。structured 仍 dense-equivalent，spec child
仍顺序执行，没有 physical memory/latency/CUDA 证据。下一工程门禁必须进入真实 query stream：
跨 query/domain batch formation、plan/code cache reuse、per-query lineage/结果恢复和公平 baseline。

## 20. Native Repeated-Query Batching and Cache v1

NRIR-7 将 frozen ResNet property 的 9 行 linear objectives 显式命名为 9 条 query，每条 query
具有 objective digest 与 `[start, stop)` lineage。packed runtime 将其组成 9-spec source，按
size-3 执行 3 个 native child stacks；serial reference 在相同 dense/retain policy 下分别编译/
执行 9 条 query，再按 query ID逐项比较。

compile cache key 覆盖 workload、input、intermediate-bound/state、ordered query content、budget、
policy 与 batch configuration；首次 lookup miss，完全相同第二次 hit 并返回同一 validated
compilation。objective 内容、query 顺序或 state identity 变化各产生不同 key/miss。packed
aggregate 恢复 9/9 query results；packed/cache hit bitwise 相同，packed/serial lower max diff
`3.2186508178710938e-06`，packed/external `1.9073486328125e-06`，serial/external
`3.2186508178710938e-06`，均 allclose、sign 9/9。

artifact 位于
`artifacts/native-real-network-repeated-query/vnncomp21-resnet2b-prop0-cpu-v1/`；聚焦
`121 passed`，全量 `540 passed, 37 skipped`，静态门禁全过。

本阶段只证明同一 input domain 上的 property-query formation、physical spec packing、exact
in-process compile cache 和结果 lineage；3 vs 9 是 child-count mechanism，不是 speedup。下一
工程门禁是 BaB parent/child domain stream：不同 input boxes、state validity/invalidation、
domain-axis packing/restore 与 same-solver baseline。

## 21. Native BaB Input-Domain Batching v1

NRIR-8 将 frozen ResNet 原始 VNNLIB box 的前三个正宽坐标逐层二分，形成 8 个确定性 leaf
query。每个 leaf 都保留 parent query/box identity，并独立运行 forward IBP；exact state hash
绑定 child box、完整 interval environment 和 ReLU preactivation，8 个 leaf hash 全部不同。
parent state 另行计算但只能是 `warm_start_only`，typed trace、compiler validation 与 runtime
trace 都要求 `parent_state_consumed_as_exact=false`。

source PlanTemplate 同时包含 full-domain 与 domain-size-4 candidates。query-time max domain=4
选择两个连续四域 Schedule slices，并为每个 slice 编译/执行独立 representation-bound
Bound/Plan/Task/Schedule child；max domain=8 选择一个 full child。same-policy serial reference
分别编译/执行八个单域 child。固定 ResNet artifact 中 packed/full/serial 的 8×1 lower/upper
全部 bitwise 一致，8/8 query/parent/result lineage 恢复。artifact 位于
`artifacts/native-real-network-domain-batch/vnncomp21-resnet2b-prop0-cpu-v1/`。

聚焦 runtime/artifact/tamper tests 为 `19 passed`，全量为 `559 passed, 37 skipped`；
Black/Mypy/Pylint/diff 与 fresh semantic replay 全过。

该关闭边界是 input-box domain formation、exact child-state ownership、domain Plan/Schedule
packing 与 restore `VALIDATED-REDUCED`。2 vs 8 只是 child-stack mechanism count，不是 timing。
它也不是完整 BaB：尚无 ReLU split decision、β/split state、priority queue、bound-based prune、
termination 或 property verdict。下一工程门禁是 native ReLU-split BaB queue/state v1；CUDA
physical protocol 仍只在设备可用时执行。

## 22. Native ReLU-Split BaB Queue v1

NRIR-9 将 legacy `ReluSplitState` 提升为 native plain-CROWN 的一等 IR 输入。固定 ResNet 的 6 个
ReLU 各有显式 `int8{-1,0,+1}` domain-batched split value；value/content hash、ReLU attrs 与
state version 进入 BoundModule，Plan workload/capability 声明 split ownership，Task/Schedule 对
实际 split-aware Bound program 执行。local split-constrained IBP provenance 不再伪装成 external
verifier bounds。

runtime 实现 deterministic widest-ambiguous-ReLU branch 与 best-first bounded queue。typed trace
冻结 node/parent/depth/branch、priority、prune/expand/terminal reason、exact state 与 native 五层 IR
hash。child 只继承离散 split constraints；每个 child batch 都重新计算 forward IBP 并编译/执行
新的 representation-bound Bound/Plan/Task/Schedule stack，parent exact state 永不作为 child exact
input。

toy complete queue 执行 15 个节点，packed/serial stacks 为 5/15，bounds/branch/queue identity
一致。固定 ResNet 有界运行执行 7 个节点、3 次 expand，保留 4 个已计算 frontier nodes并明确
`budget_exhausted`、`property_status=not_claimed`。packed-4/serial-1 使用 3/7 个 native stacks；
lower/upper max diff 为 `1.8310546875e-04/1.220703125e-04`，在 `2e-4` 绝对/相对容差内
allclose，logical queue 与 split identity 一致。artifact 位于
`artifacts/native-real-network-relu-split-bab/vnncomp21-resnet2b-prop0-cpu-v1/`。
聚焦回归为 `68 passed`，全量为 `577 passed, 37 skipped`；artifact generate/replay、Black、
Mypy、Pylint 10.00/10 与 diff check 全过。

该阶段只把 first-class ReLU split ownership、bounded queue/control flow 与实际 node batching
升为 `VALIDATED-REDUCED`。它仍是 plain CROWN：没有 α/β optimization、beta constraint、完整
搜索/verified verdict 或公平 timing。3 vs 7 只是 native-stack mechanism count，不是 latency、
memory、CUDA 或 speedup。下一工程门禁是 native α/β optimization state 与 warm-start validity v1。

## 23. Native Alpha/Beta Optimization State v1

NRIR-10 将 legacy `AlphaState`/`BetaState` 的冻结结果提升为 native state contract。每个 ReLU
BoundOp 现在显式消费 split、alpha、beta 三类 graph inputs；固定 ResNet 的 6 个 ReLU 对应 18 个
state inputs，加 objective 共 19 个。alpha 只替换 ambiguous lower slope，beta 只以
`-beta * split` 进入 lower dual coefficient。state/scope hash 绑定 primal graph、input region、
objective、local intermediate bounds、split、optimizer policy 和所有 tensor payload。

warm-start classifier 区分 exact、monotonic split refinement 和 rejected。exact same scope 才允许
exact state reuse；parent zero-split 到 child active-split 只允许 alpha/beta initialization，明确
`exact_state_reuse_allowed=false`。split reversal/removal、key/schema 或 model/input/objective/policy
漂移均 fail closed。

固定 ResNet 首个 widest branch 为 ReLU input `31`/neuron `93`。native 与 legacy αβ oracle 的
lower/upper max diff 均为 `0.0`；beta sum `0.04999999701976776`，相对 zero-beta lower 提升
`0.34039306640625`。source/execution optimized ReLU ops 均 6，Task/Launch/trace event 均 21；
10 个 compiler-layer hash 全部随 beta payload 改变。artifact 位于
`artifacts/native-alpha-beta-optimization-state/vnncomp21-resnet2b-prop0-cpu-v1/`，generate/replay
hash 为 `302f536685885e75248582698589d49f667d7709ca3258c043310e02278e6884`。聚焦
`50 passed`，全量 `591 passed, 37 skipped`，静态门禁全过。

该阶段只把 frozen optimized-state ownership、beta constraint execution 和 warm-start validity
升为 `VALIDATED-REDUCED`。Adam iteration/gradient/update 仍由 runtime adapter 控制，不是 compiled
optimizer；也没有完整 BaB/property verdict 或性能证据。下一工程门禁是 native alpha/beta
optimizer-step Task/Schedule control v1。

## 24. Native Alpha/Beta Optimizer-Step Schedule v1

NRIR-11 将 NRIR-10 的 runtime-owned optimizer loop lower 为 first-class control IR。Optimizer Plan
绑定 10 个 NRIR-10 source compiler hash、initial state/scope、policy、ReLU keys、warm-start kind 与
固定 step budget；Task/Schedule 静态展开 evaluate、metric reduction、backward、Adam update、
projection 和 select-best。execution trace 对每个 action 的输入/输出 hash、gradient、projection、
evaluation 与 best iteration 建立可重放链。

2-step toy 生成 13 个 Task/Action，并与 legacy optimizer 的 bounds/alpha/beta 逐张量一致。固定
ResNet 1-step child 生成 8 个 Task/Action、2 次 evaluation、1 次 backward/Adam/project；alpha/beta
gradient L1 为 `169.23175295069814/12.862210273742676`。Schedule 对 legacy optimizer、最终 selected
state 的 native Bound/Plan/Task/Schedule re-execution 的 lower/upper max diff 全为 `0.0`。

artifact 位于
`artifacts/native-alpha-beta-optimizer-schedule/vnncomp21-resnet2b-prop0-cpu-v1/`，generate/replay
hash 为 `31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`。聚焦
`35 passed`，全量 `612 passed, 37 skipped`，静态门禁全过。

该阶段只关闭 fixed-step optimizer control ownership `VALIDATED-REDUCED`。dynamic early stop、
multi-node BaB integration、complete termination/property verdict 与 CUDA/performance 都未关闭。
下一工程门禁是将 optimizer Schedule 接回 native ReLU-split BaB queue 的逐节点 evaluation，同时
维持 parent→child initialization-only 与最终 state native execution。

## 25. Native Optimized ReLU-Split BaB v1

NRIR-12 将 NRIR-9 queue、NRIR-11 optimizer Schedule 和 NRIR-10 selected-state native compiler
连接为连续执行链。每个 node batch 先运行固定 1-step 的 8 个 optimizer Task/Action，再把 selected
alpha/beta state 编译并执行为 21-task native Bound stack。child 的 parent state 只经重建 batch scope
后作为 monotonic-refinement initialization，`parent_state_consumed_as_exact=false`。

toy complete tree 执行 15 nodes，packed/serial stacks 为 5/15，queue/branch/bounds/selected-state hash
一致。固定 ResNet bounded run 执行 7 nodes、3 expands、4 frontier，packed/serial stacks 为 3/7；
lower/upper max diff=`1.220703125e-04/1.8310546875e-04`，alpha/beta tensor max diff=
`4.172325134277344e-07/7.450580596923828e-09`。exact state hash 因 batch-layout intermediate 数值
不同而不相等，已明确披露。所有 active child stacks 有非零 beta gradient；selected-state native
re-execution max diff 为 0。

artifact 位于
`artifacts/native-optimized-relu-split-bab/vnncomp21-resnet2b-prop0-cpu-v1/`，replay hash 为
`e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`。聚焦 `18 passed`，
全量 `630 passed, 37 skipped`，静态门禁全过。

该阶段只关闭 optimized queue integration/control ownership `VALIDATED-REDUCED`。固定 ResNet 仍是
`budget_exhausted/property_status=not_claimed`；没有 complete termination/verdict 或性能证据。下一
工程门禁是 sound property termination/verdict v1，任何未闭合 frontier 必须保持 unknown。

## 26. Native Property Termination and Verdict v1

NRIR-13 在不修改 NRIR-12 queue schema/hash 的前提下新增独立证明层。性质语义为单标量
`C f(x) >= threshold`：`verified` 必须 queue complete、frontier 为空、且所有 leaf 都有
`lower >= threshold` 的 sound prune；任何 budget/depth/unproven terminal 均进入 unknown。

新 concrete Task IR executor 独立执行 linear/conv2d/ReLU/residual/reshape 等 primal ops 并保留
intermediate value trace。`unsafe` 只能由 concrete input 产生：input box、node ReLU split path、
primal output 与严格 `objective < threshold` 全部重执行通过，并将 tensor/value-trace hash
绑入 counterexample trace。toy matrix 独立覆盖 verified/unsafe/unknown；同步重哈希后的
verdict/witness/claim 篡改均 fail closed。

固定 ResNet 仍执行 7 nodes/3 expands/4 frontier，显式输出
`unknown/node_budget_frontier_open`。其中心点经完整 primal Task IR 重执行的 objective 为
`0.8564349412918091`，不是 counterexample。artifact 位于
`artifacts/native-property-verdict/vnncomp21-resnet2b-prop0-cpu-v1/`，generate/replay hash 为
`9e3dceed23c8759c910938ba7c9f84caaeb949c8f19b72fab104ce4e1b733405`；聚焦 `19 passed`，
全量 `649 passed, 37 skipped`，静态门禁全过。

该阶段关闭 three-state verdict soundness/control ownership `VALIDATED-REDUCED`，但仍不是完整
verifier：counterexample discovery 仍由 caller 提供，只支持单标量性质，timeout/dynamic
optimizer early-stop 未接入，固定 ResNet 也未闭合。这里冻结的 complete verifier query 下一路线
已由第 27 节执行；NRIR-13 本身仍不得升级为端到端或性能 claim。

## 27. Complete Verifier Query v1

NRIR-14 新增 typed candidate-search 与 multi-clause query control。性质固定为 conjunction：
全部 clause 的 sound verdict 为 verified 才返回 verified；任何 concrete-replayed violation 立即
unsafe，并把后续 clauses 标为 skipped；其余 unresolved/pending 均返回 unknown。candidate
search 使用 deterministic center-start sign-gradient descent 与 exact box projection，明确
`proof_claimed=false`；deadline 在 clause/search/queue stage 边界 cooperative 检查，不声称
可抢占 active kernel。

toy evidence 覆盖两子句 verified、第二子句 unsafe short-circuit、attack-not-found unknown 与
deadline pending unknown。固定 ResNet 使用九个真实 objectives；九个 candidate best objective
均大于 0，但 native lower bounds 约为 `-408.01` 至 `-863.19`，因此九个 clauses 全部 unresolved，
总体是 sound unknown。该数值差距明确表明下一 blocker 是 bound tightness/optimizer/branching，
不是 query API 或 trace 包装。

artifact 位于 `artifacts/complete-verifier-query/vnncomp21-resnet2b-prop0-cpu-v1/`，replay hash=
`d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`；相关 `39 passed`，
全量 `670 passed, 37 skipped`，Black/Mypy/Pylint 全过。

该阶段只关闭 complete-query correctness/control `VALIDATED-REDUCED`，不关闭 fixed real property，
也没有 latency/memory/CUDA/speedup claim。下一阶段先冻结 end-to-end phase/tightness baseline：
分别测 candidate、bound optimization、queue、verdict，记录 proof gap、nodes、batch/cache 和
same-solver/竞品口径；随后才允许按证据选择 dynamic optimizer、branching/tightness 或执行优化。

## 28. End-to-End Tightness and Performance Baseline v1

NRIR-15 将 NRIR-1/RVIR 的 frozen external intermediate semantics 接入 optimizer Schedule、
selected-state native compiler、optimized queue child batches 与 complete query。external bounds 与
typed `EXTERNAL_VERIFIER` provenance 必须成对出现；child 的 root external interval 会与 node
active/inactive split 相交，parent state 仍仅允许 monotonic-refinement initialization。adaptive α
初始化与 frozen initial-CROWN lower-slope policy 对齐，同时默认 constant policy payload/hash 不变。

固定 ResNet 九子句的 local NRIR-14 reference 为 0/9；external-adaptive 1-step query 直接关闭
clauses `1/3/5/6/7/8`，变为 6/9 verified、`0/2/4` unknown。九个 lower 相对 frozen external
initial 没有退化，最大改善 `0.0072252750`、sign `9/9`。artifact fresh replay hash 为
`14c3b9dc2e5376156be1f33f3e8804ec21f60e11096bd3bdc95225b7e1474376`。

三组轮换 CPU 诊断显示 clause 0 的 audit queue median：local-constant `6.7178 s`、
external-constant `6.7969 s`、external-adaptive `6.7317 s`；candidate 与 verdict 只有约
`3.6/3.9 ms`。因此 fixed compile/hash/selected-native re-execution 是当前 wall-time blocker，
不是搜索或 verdict。该结果是 `VALIDATED-REDUCED` 的单 workload CPU diagnosis，不是 production
latency、CUDA 或竞品 speedup。下一工程门禁是 prepared production fast path；之后再对三个 hard
clauses 推进 branching/stronger-bound，不得把 6/9 写成完整 verifier closure。

## 29. Prepared Production Fast Path v1

NRIR-16 把 exact optimizer program validation、compiler/hash construction 与 dynamic execution
分离。preparation 为九个 root objectives 各自冻结 optimizer Plan/Task/Schedule、native source
compiler hashes、input/objective/intermediate/split/policy scope；steady-state 仍逐 action 执行
evaluate/reduce/backward/Adam/project/select-best，但不构造 audit tensor hash chain，也不做
selected-native validation re-execution。任何 program/module/input/objective/source/scope 漂移均拒绝。

fixed ResNet 三组轮换中，audit complete-query raw=`58.713/59.078/59.587 s`，prepared warm
raw=`111.166/110.262/110.950 ms`；median `59.078 s` 对 `110.950 ms`，内部 audit-overhead
diagnostic ratio=`532.47×`。cold preparation=`14.724 s`、首次执行=`1.415 s`，合计
`16.139 s`，相对 audit median=`3.660×`；retained prepared tensor payload=`2,076,372 B`。

production lower 对 audit max diff=`1.90735e-6`、candidate exact、status exact，仍为 6/9
verified、clauses `0/2/4` unknown。artifact fresh replay hash=
`e14fcd62b322c0bc60d45c726cf94a7aa6cfb8d7aa3212662d08996db169b6b2`，全量
`698 passed, 37 skipped`。该结果只关闭 root-only repeated-query prepared mechanism 与单 workload
CPU internal-overhead diagnosis `VALIDATED-REDUCED`；不是 competitor speedup、CUDA 或完整
verification。下一工程门禁转为三个 hard clauses 的 branching/stronger-bound。

## 30. Hard-Clause Objective Branching v1

NRIR-17 将 objective-aware ReLU branch scoring lower 为独立 Plan/Task/Schedule IR。每个节点的
candidate enumeration、双子域 materialization、fixed selected-state child-bound evaluation、
worst-child reduction 与 deterministic selection 均有 exact hash；objective、split、selected
alpha/beta state、semantic scope、policy 与 candidate identity 任一漂移 fail closed。原 widest
路径与 NRIR-15/16 frozen replay 保持不变。

fixed ResNet clauses `0/2/4` 使用相同 7-node/depth-2、25-step adaptive optimizer 预算。widest
worst leaf 为 `-0.440550/-0.498173/-0.562577`；objective path 为
`-0.319799/-0.426609/-0.504676`，分别改善 `0.120752/0.071564/0.057901`。三棵 objective
tree 的所有 terminal leaves 仍为负，因此 property status 保持 unknown，6/9 总体语义不升级。

artifact fresh replay hash=
`1193bee8817e4acc9ec33f8ddadc00a671d0ac3c9411f14f62978eb5ab1a95bd`，全量
`707 passed, 37 skipped`。该阶段只关闭 branch IR/control 与单 workload fixed-budget tightness
`VALIDATED-REDUCED`；单次 audit timing 不是 performance claim。下一路线为多 workload/设备/
竞品 E2E 协议与 stronger-bound，而不是把 bounded-tree improvement 写成完整 verifier closure。

## 31. Multiworkload Competitor E2E Baseline v1

NRIR-18 将 VNNLIB input box 与 unsafe output DNF 编译为 immutable Query IR；v1 只接受每个
unsafe disjunct 恰含一条线性 inequality，缺界、重复界、非连续变量、非线性或多 inequality
均 fail closed。三份真实 property 的 lower/upper/C/rhs 与固定 αβ-CROWN parser 逐字段一致。
顶层 workload Plan/Task/Schedule 冻结三项 CSV selection、21 tasks 与 6 个 fresh-process native/
competitor execution action，并绑定 model/property/CSV/query/compiler/policy/device/timeout hash。

正式 CPU 矩阵结果如下：MNISTFC 的 BoundFlow/αβ-CROWN 为 `unknown/verified`，OVAL21 为
`unknown/verified`，ResNet2B 为 `unknown/unknown`。BoundFlow 在 MNISTFC 留 3/9 unresolved，
OVAL21 留 1/9 unresolved；ResNet 只在 deadline 前完成 2/9，root lower=
`-543.717/-789.331`。fresh-process E2E 为 `38.644/4.312 s`、`31.498/4.527 s` 和
`66.910/64.198 s`，但算法完整性和运行路径不同且只有单次 CPU observation，禁止计算 speedup。

artifact fresh replay hash=
`473b287bb88e4c52426b405aeb4164aa72a98d7b1bbd74c00471fe1d1451deb0`，全量
`723 passed, 37 skipped`。该结果只关闭 ingest、typed IR/control 和真实 workload coverage
`VALIDATED-REDUCED`；GPU/performance 与 ASPLOS-ready 仍为 NO。下一单一工程门禁是 native
intermediate-bound refinement Plan/Task/Schedule：先缩小三 workload 的 root/closed-clause gap，
再讨论 selective policy、prepared execution 或可用 CUDA 主机上的冻结矩阵。

## 32. Native Intermediate-Bound Refinement v1

NRIR-19 把任意中间张量 selected-row plain CROWN、top-width target selection、分块 backward、
intersection 和 forward propagation lower 为可哈希 Plan/Task/Schedule。`native_refined` provenance
独立于 `external_verifier`，并进入 optimizer/Bound IR/BaB child path；source、input、split、初始
bounds、policy、target、action trace 任一漂移均拒绝。

正式 CPU same-policy fresh-process 结果：MNISTFC 关闭 clauses `3/7`，unresolved `3→1`、nodes
`31→21`；OVAL21 关闭 clause `8`，`unknown→verified`、nodes `15→11`；ResNet 仍 unknown，但
两个 root lower 改善 `+70.496/+160.551`。refinement 本身约 `21.8/114.3/32.1 ms`，只用于
方法成本诊断，不是性能 claim。

artifact replay hash=
`f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；全量
`732 passed, 37 skipped`。该阶段以 native refinement IR/control + multiworkload tightness
`VALIDATED-REDUCED` 关闭；只 1/3 complete verified，ASPLOS-ready 仍为 NO。下一单一路线是
objective-directed intermediate target selection，优先解决 ResNet，而不是扩大 tree budget、
先做 CUDA timing 或把单次 CPU E2E 写成 speedup。该历史下一路线已由第 33 节 NRIR-20 完成。

## 33. Objective-Directed Intermediate Refinement v1

NRIR-20 新增 `objective_influence_width_per_relu_v1`。当前单个 scalar property clause 的 plain
CROWN backward coefficients 在每个 ReLU 处转为 `max(abs(A_u), abs(A_l))` influence，并与
ambiguous pre-activation width 相乘排序。Plan 冻结 objective hash 和 target score；Task/Schedule
显式消费 `refine.objective_influence`，多子句 objective、shape/dtype/device/finite 漂移均 fail
closed。排序只影响计算预算，soundness 仍由 selected CROWN 与 interval intersection 保证。

固定 VNN-COMP 2021 ResNet2B property 0 的 clauses `0/1` 上，width 与 objective policy 都选
`96` targets。目标重合仅 `16/96`、`27/96`；root lower 从
`-473.221222/-628.780334` 改为 `-417.292480/-602.551392`，改善
`+55.928741/+26.228943`。32/64-target 和第二 pass 的开发敏感性探针仍保持负 lower，说明
单纯扩大 root shortlist 有收益但不足以闭合，不纳入冻结 performance claim。

artifact 位于
`artifacts/objective-directed-intermediate-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`，
fresh source-to-IR semantic replay hash=
`8fce1c7c3e5c63adb14a7ab5b9f23407e4a7a1406353750e4f150ee745b4e88e`；focused
`16 passed`、全量 `739 passed, 37 skipped`。本阶段以 fixed-root tightness
`VALIDATED-REDUCED` 关闭；没有 complete closure、CUDA、重复性能或 ASPLOS-ready claim。下一
单一路线是让 ReLU-split child 依据其 exact split state 重算 clause-sensitive refinement，禁止把
parent refined bounds 当作 child exact state。

## 34. Per-Child Objective Refinement v1

NRIR-21 将每个 optimized ReLU-split queue node 的 exact split state 编译成独立 refinement
Plan/Task/Schedule：逐 node 重跑 split-forward IBP、clause-sensitive influence、target selection、
selected CROWN、intersection 与 propagation，再把 child-specific bounds 拼成 optimizer batch。
queue trace 一一绑定 node split、三层 refinement IR hash、去 timing semantic trace、initial/final
intermediate hash 与 target count；parent alpha/beta 仅作 monotonic warm initialization，parent
refined bounds 从未当作 child exact result。默认关闭时旧 queue payload 不增加字段。

固定 ResNet2B property 0 clauses `0/1` 使用同一 96-target policy、5-step optimizer、7-node/
depth-2 tree。root-global 与 per-child root lower 完全一致，分别为
`-417.292480/-602.551392`；但 per-child 最差 depth-limit leaf lower 为
`-414.587006/-592.880920`，弱于 root-global 的 `-413.739044/-591.944275`，delta=
`-0.847961/-0.936646`。因此该策略按预设门禁以 `VALIDATED-NO-GO` 关闭，不升级 tightness、
property、performance 或 ASPLOS-ready claim。

下一单一路线是 ancestral-constraint carry-forward：child 必须从 exact split-forward 与祖先已证明
refined constraints 的单调交集出发再重选 targets，避免当前 per-child recomputation 丢失 root
selected-CROWN tightening；该方法仍须进入一等 Plan/Task/Schedule 后才能比较。

## 35. Ancestral-Constraint Refinement v1

NRIR-22 不接受裸 intermediate mapping，只接受已通过自身 IR/trace 验证的 parent refinement
execution。child Plan 同时绑定 parent final bounds、parent Plan 和去 timing semantic trace；
materialize-forward Task/Schedule 显式消费 source constraints。运行时重算 local exact-split forward，
与 source 单调交集/propagation 后再执行 child influence/selection/CROWN，source consumption 只标记
为 `sound_constraint_only`，从未升级为 child exact reuse。

固定 ResNet clauses `0/1`、同 96-target/5-step、7-node/depth-2 预算下，ancestral carry worst leaf
为 `-340.971832/-517.858826`；相对 independent 提升 `+73.615173/+75.022095`，相对
root-global 提升 `+72.767212/+74.085449`，root lower 完全不变。该阶段以 IR/control + fixed
bounded-tree tightness `VALIDATED-REDUCED` 关闭；叶 lower 仍负，不能形成 complete property 或
ASPLOS-ready/performance claim。

下一路线从“再改 refinement plumbing”切到 hard-clause convergence expansion：固定更多 hard
clauses 与 depth/node budget 曲线，判断 ancestral carry 是否能推动 complete closure；CPU timing
继续只作诊断，公平 E2E 必须等待可用 CUDA 环境并与相同算法能力竞品重测。

## 36. External-Seeded Ancestral Refinement v1

NRIR-23 连接了此前分离的 external intermediate semantics 与 native ancestral refinement。
`ExternalIntermediateConstraintSeedIR` 保留 `semantics_owner=external_verifier`，同时绑定 primal/input、
external ordered digest、source artifact/model/property/objective-set 和 local-intersection constraint
hash。raw external 先与 local forward 求可行交集；refinement Plan/Task/Schedule/action trace 显式消费
effective seed，且 external seed 与 native parent execution 严格互斥。

queue 的 `external_seeded_ancestral_carry_v1` 只允许 root 消费 typed seed；六个 non-root 节点逐一
消费已验证 parent refinement final/Plan/semantic trace，alpha/beta 仍为 warm-only。固定 ResNet
clauses `0/2/4`、objective branch、25-step optimizer、16 targets/ReLU、7-node/depth-2 下，ancestral
worst leaf 为 `-0.318287/-0.425477/-0.504142`；相对 external baseline 改善
`+0.001512/+0.001133/+0.000534`，相对 seeded root-global 为
`+0.000823/+0.000004/0`。

artifact semantic replay hash=
`9f52b99a74dab448626061f5b8f060f3b8c43b6c03f6deb0899d9fe91883d9f7`；全量
`766 passed, 37 skipped`。该阶段只关闭 typed seed/control/lineage 与 fixed-budget tightness
`VALIDATED-REDUCED`；所有 terminal leaves 仍负，无 complete property、performance、CUDA、
multi-workload 或 ASPLOS-ready claim。下一门禁为 external-seeded depth/node convergence curve。

## 37. External-Seeded Depth/Node Convergence v1

NRIR-24 在不改变 source、typed seed、objective branch、25-step optimizer、16-target/ReLU 单 pass
refinement 或 batching 的前提下，只把完整树预算从 `7/depth2` 扩为 `15/depth3` 和
`31/depth4`。每个 clause/budget 由 fresh Python process 独立执行并原子写 shard；九个 shard
全部完成 semantic replay。

clauses `0/2/4` 的 worst terminal lower 曲线分别为：

- `-0.318287 → -0.299506 → -0.282360`；
- `-0.425477 → -0.413456 → -0.401845`；
- `-0.504142 → -0.479104 → -0.459939`。

三条均单调严格改善，15→31 nodes delta 为 `+0.017146/+0.011611/+0.019165`，未触发冻结的
`1e-6` 饱和门禁；但三条仍为负，无 fixed bounded-tree closure。best-first 跨预算的 node/batch
序号不是稳定逻辑身份；artifact 以 `split_state_hash` 校验 `7⊂15⊂31` logical domains、parent
split lineage、branch selection 与去执行序号的 refinement semantics，公共域最大 lower 漂移
`1.13249e-6`，低于 runtime `1e-5` tolerance。

最终 semantic replay hash=
`db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`。本阶段只关闭
external-seeded fixed-hard-clause convergence trend `VALIDATED-REDUCED`；不声明 complete
property、performance、CUDA、multi-workload、competitor parity 或 ASPLOS-ready。proof deficit
仍为 `0.282360/0.401845/0.459939`，下一门禁转向 dynamic ancestral refinement budget/multi-pass，
而不是继续盲目增加固定树深。

## 38. Dynamic Ancestral Refinement Budget v1

NRIR-25 新增 first-class refinement-budget policy/decision。每个 evaluation-generated group 依据
parent lower 分配 24/8 targets/ReLU；root、single-parent 或 `1e-6` tie 使用 base 16。decision
绑定 policy/group/node/split/depth/parent 与 exact conservation totals，assigned cap 派生为实际
refinement Plan policy，Task/Schedule/execution/queue trace 逐层交叉校验；旧 fixed16 路径条件兼容。

固定 ResNet clauses `0/2/4`、31 nodes/depth 4、external seed、ancestral carry、objective branch、
25-step optimizer 与单 pass 下，fixed16→dynamic8_24 的 worst terminal lower 为：

- `-0.2823597193 → -0.2819737196`（delta `+0.0003859997`）；
- `-0.4018449783 → -0.4016119838`（delta `+0.0002329946`）；
- `-0.4599394798 → -0.4596676826`（delta `+0.0002717972`）。

两 mode 的 planned cap 都是 `496`，实际 selected targets 都是 `2976`；三条 dynamic 均不弱且严格
改善，因此按预注册门禁为 `VALIDATED-REDUCED`。artifact evidence hash=
`85d9f274c6e17614bcbf318bdbfea18219b03876024be16aea3329ee4d3c56bd`。三条树仍 unknown，不能
升级 complete property、performance、CUDA、multi-workload、competitor 或 ASPLOS-ready。
下一门禁为 typed multi-pass refinement/termination 与 pass-to-pass lineage，不回到盲目固定树扩展。

## 39. Typed Multi-Pass Refinement v1

NRIR-26 修正了历史 `passes=2` 仅重复相同 targets、未拥有 pass control IR 的问题。新 policy 将每
node 的 dynamic assigned total cap 等分两 pass；每个 pass 的 enumerate、updated-width selection、
prior-target exclusion/ledger、continue/stop、backward、intersection 与 propagation 都是一等
Task/Schedule action。pass decision 绑定 Plan/policy、input bounds、ledger、target、cap 与 termination；
无 unseen target 时执行 sound passthrough。legacy lowering/hash 条件兼容。

固定 ResNet clauses `0/2/4`、31 nodes/depth 4 下，single-pass 与 split-two-pass 的 worst terminal
lower 都分别为 `-0.2819737196/-0.4016119838/-0.4596676826`，三条 delta=`0.0`；logical tree
均 `31/31` 重合。每 mode planned total cap=`496`、actual targets=`2976`；split 没有 stopped pass，
证明第二 pass 确实执行并选满，但没有改善 worst domain。

因此按预注册门禁以 `VALIDATED-NO-GO` 关闭；artifact evidence hash=
`38992cace70214ffcbd670f03dcfca182e0925bee31eb4df885dab4dab03494d`。first-class multi-pass
IR/control 可保留，但不能形成 tightness/property/performance/ASPLOS-ready claim。停止继续同一静态
influence 拆 pass；后续必须先验证 pass-local influence recomputation 或 branch/cut 新信息能改变
target/critical domain，再立正式路线。

## 40. Production Prepared Verifier v1

NRIR-27 把已验证的 prepared optimizer、ReLU-split queue、property verdict 与 conjunction query
接成 production complete-verifier 路径。每个动态 node batch 均拥有 first-class
Plan/Task/Schedule，并按 validate program→execute optimizer→materialize results→commit queue 的
真实次序执行；production 模式明确不生成 audit tensor hash chain，也不重复 selected-native
compiler/oracle execution。旧 audit 默认行为、payload 与 hash 条件兼容。

MNISTFC、CIFAR10 ResNet2B、OVAL21 各执行三组交替次序的 fresh-process clause-0 对照；相同算法
audit→production median 为 `4.510→3.301 s`、`22.509→9.104 s`、`5.192→3.578 s`，内部
speedup 分别 `1.3663×/2.4723×/1.4511×`，semantic parity 全过。full production median 为
`14.834/60.754/11.964 s`，三类 query 仍为 unknown；ResNet 三次均完成 `9/9` clauses，而历史
deadline-bound audit 只完成 `2/9`。

artifact evidence hash=
`7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`；全量
`800 passed, 37 skipped`。本阶段以 production runtime/internal CPU overhead
`VALIDATED-REDUCED` 关闭；历史 αβ-CROWN 数字只作不同完整性协议的单次诊断，不是竞品 speedup。
full-query execution 仍有约 `59%–65%` 位于四个 production action 之外，下一门禁为 parametric
dynamic-batch PlanTemplate/PlanInstance 与 compile cache；GPU、complete property、公平 competitor
与 ASPLOS-ready 均保持 pending。

## 41. Parametric Dynamic Batch Compiler v1

NRIR-28 将逐 dynamic batch 的 optimizer 编译拆为静态 PlanTemplate 与 exact PlanInstance。
graph、input/objective contract、ReLU layout、policy、provenance 和 reusable Task/Schedule 只编译
一次；input/objective/split/intermediate/parent warm-state content 全部进入 instance hash。query-local
cache 只接受 exact contract hit，任何 contract、event、instance 或 runtime tensor 漂移 fail closed。
NRIR-27 frozen 路径保持未修改并继续 artifact replay。

三类真实拓扑各三组交替 fresh-process full-query production-v1→parametric-v2：MNISTFC median
`14.807→3.456 s`（`4.2849×`）、ResNet2B `61.239→6.209 s`（`9.8630×`）、OVAL21
`13.021→3.718 s`（`3.5024×`）。每次 query 只有一个 template miss；其余
`18/26/10` 个 instances 均 exact hit。所有 solver status、clause accounting、logical queue、node
coverage、selected-state hash 和 root bounds 保持一致。

artifact evidence hash=
`117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`；全量
`818 passed, 37 skipped`。本阶段以 same-algorithm full-query internal CPU performance
`VALIDATED-REDUCED` 关闭。三类 query 仍为 unknown；无 CUDA、external competitor speedup、
complete-property 或 ASPLOS-ready claim。下一门禁为 fixed-wall-clock typed BaB search scaling，
判断 compiler/runtime 收益能否转化为更多节点、更深覆盖与 property closure。

## 42. Wall-Clock Parametric BaB Scaling v1

NRIR-29 把 `7/depth2`、`31/depth4`、`127/depth6` 三档搜索预算和三真实 workload × 三 fresh
repeats 编译成一等 Plan/Task/Schedule。budget 之外的 model/property、parametric runtime、5-step
optimizer、4-step candidate search、batching、threads 与 60 秒 query deadline 全部固定；budget
顺序按 repeat 轮转。worker 保存 logical split-state domains、leaf verdict、compiler cache/instance
和 raw timing，replay 重建 source-to-experiment IR 并重算门禁。

27/27 worker 完成 `9/9` clauses、无 pending；同预算三次 semantic signature 一致，三 workload
全部满足 `7⊂31⊂127` logical domain nesting，公共 lower 漂移为 `0.0`。MNISTFC verified 从
`6/9` 提升到 `8/9`，31/127 nodes 相同；ResNet 三档均 `0/9`，OVAL21 三档均 `8/9`。127-node
median execution 为 `2.515/58.566/2.287 s`，ResNet p90=`58.939 s`；这是 fixed-protocol
resource/coverage 曲线，不是跨预算 speedup。

artifact evidence hash=
`e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`。按预注册门禁以
search-coverage `VALIDATED-REDUCED` 关闭；完整 query 仍全部 unknown，无 GPU、competitor、完整
property 或 ASPLOS-ready claim。单轴扩大同一搜索已出现明确饱和：下一门禁必须是 typed
hard-clause escalation，在固定总 deadline 下只为 unresolved clauses 编译更强 native
intermediate-refinement/branch Plan，并验证 sound fallback 与新增 closure。

## 43. Typed Hard-Clause Escalation v1

NRIR-30 把 NRIR-29 饱和结果编译成 staged verifier：先运行 local-forward parametric `7/depth2`
baseline，Decision 只 admit exact unresolved original ordinals；随后共享一份 1-pass、128-target/ReLU、
chunk32 native selected-CROWN refinement，把 hard objective/threshold 双射投影到 `31/depth4`
parametric query，最后恢复原 ordinal 聚合。八类 Task/Schedule action 和 60 秒 whole deadline 均由
Plan 所有；deadline/refinement/escalation 失败只能保留 baseline verdict。

三 fresh repeats 中，MNISTFC 都从 6/9 提升到 8/9；ResNet2B 都保持 0/9；OVAL21 都只 admit
clause 8 并从 8/9 unknown 变为 9/9 query verified。median whole-stage execution 分别为
`2.974/20.146/2.208 s`（MNIST/ResNet/OVAL），全部 `fallback=none`；timing 只作 deadline
accounting，`performance_claimed=false`。

artifact evidence hash=
`df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`。本阶段以 first-class
staged control + property coverage `VALIDATED-REDUCED` 关闭；无 GPU、competitor、完整 suite 或
ASPLOS-ready claim。下一门禁在相同 admission/budget/deadline 下把 shared top-width refinement
替换为 per-clause objective-influence Plan，隔离检验 MNIST clause 8 与 ResNet hard clauses。

## 44. Objective-Directed Hard-Clause Escalation v1

NRIR-31 保留 NRIR-30 的 baseline、exact admission、shared selected-CROWN source、31/depth4 budget、
batching 和 60 秒 whole deadline，只为每个 admitted original clause 新增 objective-influence
128-target/ReLU、chunk32 refinement。九子句 workload 被 lower 为 33-task 静态 TaskModule；每条
objective child 显式绑定 shared execution 的 Plan hash、semantic trace hash、scalar objective hash
与 original ordinal，未 admitted 或 deadline 后任务走 guarded skip。

单次 pilot 因 ResNet 9/9 common root 全部严格改善而通过预注册 gate，随后才运行三 fresh repeats。
MNIST 三次保持 8/9，OVAL 三次保持 9/9；ResNet 仍 0/9，但九条 root lower delta 三轮逐值一致，
范围 `+81.522583—+179.970459`。9/9 run 都 `fallback=none`；median execution 为
`3.143/24.188/2.255 s`，仅用于 deadline accounting。

artifact evidence hash=
`fb9e503bdf93cb9ce56f52915f1965f1f542e092945d4d7d77d8b8c4bd91764a`。本阶段以 per-clause
objective root tightness `VALIDATED-REDUCED` 关闭；没有新增 closure，不声明 performance、GPU、
competitor、完整 suite 或 ASPLOS-ready。下一门禁为 objective-ancestral hard-clause escalation：
把 objective root execution 作为动态 child refinement 的 typed source，使已证实的 root tightening
进入 frontier；禁止继续追加 root-only pass。

## 45. Objective-Ancestral Hard-Clause Escalation v1

NRIR-32 先在固定 ResNet2B property 0 clause 0 上执行 two-child feasibility：root-global 与
ancestral 使用 exact root、branch、split、optimizer 与 serial evaluator，只改变 child 是否消费
parent refinement execution。两个 child lower 分别改善 `+59.367462/+59.253479`，达到预注册 gate，
随后才新增 first-class objective-ancestral Plan/Task/Schedule 与 cooperative-deadline queue runtime。

正式 `31 nodes/depth 4/60 s` 三 fresh repeats 中，typed queue 每次均提交 7 nodes、24 tasks、到
depth 2；root 与 root-global 对照 exact parity=`-204.17315673828125`。ancestral worst active lower
三次均为 `-104.76541137695312`，31-node root-global 为 `-200.46539306640625`，严格改善
`+95.69998168945312`。committed queue trace、Task IR、node-refinement hash 三轮分别一致；late
child evaluation 只作为 discarded diagnostic，不进入 proof identity。

artifact evidence hash=
`8fba8deca18dcbf0b4b258aa390c1dd48d250c71ea1a48ddb991388765411bfc`；全量
`846 passed, 37 skipped`。本阶段以 typed lineage + committed-frontier tightness
`VALIDATED-REDUCED` 关闭；没有 property closure、performance、GPU、competitor、multi-clause 或
ASPLOS-ready claim。下一门禁为固定 60 秒下预注册 child-refinement cap/resource Pareto，把 tighter
bound 转化为更多 committed nodes；不直接延长 deadline。

## 46. Objective-Ancestral Child Budget Pareto v1

NRIR-33 在不修改 frozen NRIR-32 engine/artifact 的前提下，新增 child-budget Policy、Calibration、
Decision 与 Plan IR；five-cap `[8,16,32,64,128]`、pilot order 和“选择最小且保留 cap128 至少 90%
gain 的 cap”在运行前冻结。每个 candidate 由 fresh process 重建 root，并执行相同 31/depth4/60 s
queue 与独立 root-global 对照。

五档 accepted nodes 全部为 7、max depth 全部为 2；worst active lower 依次为
`-173.078613/-162.253326/-148.134460/-126.962929/-104.765411`，而 root-global 为
`-200.465393`。cap128 gain=`+95.699982`；90% retention rule 只能选择 cap128。较小 cap 同时降低
tightness 且没有增加 coverage，说明当前瓶颈在 serial child evaluator/optimizer，不在 target cap。

pilot hash=`db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`。
本路线以 cap-only coverage `VALIDATED-NO-GO` 关闭；timing 只作诊断，无 performance/property/GPU/
competitor/ASPLOS-ready claim。下一门禁固定为 cap128 sibling packed refinement/evaluation 与
parametric evaluator，目标是在同一 60 秒内严格增加 committed nodes。

## 47. Sibling-Packed Objective-Ancestral Evaluator v1

NRIR-34 保持 cap128 objective-ancestral refinement、31/depth4 和 60 秒 cooperative deadline，新增
source `(1,1,10)`→evaluator `(1,10)` typed projection与 same-parent `(-1,+1)` SiblingGroup IR。
两个 child 的 refinement Plan/execution/final bounds 仍独立；optimizer 与 selected-native compiler
execution 合并为 domain-batch 2。Task/Schedule 显式包含 root admission/projection/evaluation、parent
transition、两条 child refinement、packed compile/execute 与 emit；late complete group 原子丢弃。

first-pair feasibility 的 serial/packed child elapsed 为 `13.291550/7.018038 s`，optimizer/native
execution group 都由 `2→1`，bounds exact。随后三 fresh-process 交替 formal repeat 中，serial
accepted nodes=`[7,7,7]`，packed=`[15,15,15]`，minimum gain=`+8`；common 7 domains 的
lower/upper max diff 均为 `7.62939453125e-06`，split/branch/final refinement exact，alpha/beta max diff
为 `1.0728836e-04/8.9406967e-08`。packed 到 depth 3，worst active lower 从 serial
`-104.765411` 改善到 `-76.077194`。formal hash=
`9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`。

九子句 global-60s adapter 保持 search、sound verdict 与 original ordinal accounting；一次 integration
完成 clause 0 的 13 nodes/6 groups 后，总体 `unknown`、unresolved=`[0]`、pending=`[1..8]`。artifact
hash=`dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`。本阶段以 single-hard-clause
same-algorithm deadline coverage `VALIDATED-REDUCED` 关闭；cooperative atomic completion wall time
约 `64.5—66.2 s`，不是 60 秒硬实时或 wall-clock speedup。无 property closure、GPU、competitor、
multi-workload 或 ASPLOS-ready claim。下一门禁为 NRIR-35 cross-clause objective/root/compiler sharing
与 anytime budget，必须在同一全局 60 秒内增加 completed original clauses，不得给每 clause 独立延长
deadline。

## 48. Cross-Clause Anytime Objective Evaluator v1

NRIR-35 先执行 frozen NRIR-31 objective-hard-clause program，使固定 ResNet property 0 的九个
original clauses 都获得 sound floor；只有 floor completed `[0..8]`、final unknown、clause 0
unresolved 且 exact accepted child source 存在时，Decision 才 admit NRIR-34 packed queue。static
Task/Schedule 固定为 floor、decision、guarded packed compile、guarded packed execute、monotone
original-ordinal aggregate、emit 六阶段；source Plan/semantic/final-bound、objective/threshold、policy
与单一 global deadline 全部 hash-bound。

单次 feasibility 先以 floor `22.180303 s`、packed 7 nodes/3 groups 通过。正式一等 runtime pilot
同样通过；兼容性修复后重生成的三 fresh repeats，floor elapsed 为
`22.227251/21.622773/21.834220 s`，每轮都 completed/unresolved=`[0..8]`；packed accepted
nodes=`[7,7,9]`。whole cooperative elapsed 为 `61.991720/62.598928/68.042604 s`，来自 deadline
前开始的完整 sibling group 原子收尾，不是
60 秒硬实时或 wall-clock speedup。三轮 packed verdict 与最终 query 都仍是 sound `unknown`。

formal hash=`74533c9c211a3007bf5af43c08865febd95c3f9ccf1a268e56738793ec9d14d5`；replay、wrong ordinal/
source、deadline reset、baseline omission、non-monotone aggregate、partial-group 篡改与全量
`874 passed, 37 skipped` 均通过。本阶段以 cross-clause control/original-ordinal preservation
`VALIDATED-REDUCED` 关闭，`performance_claimed=false`；没有 property closure、GPU、competitor、
multi-workload 或 ASPLOS-ready claim。下一门禁为 multi-clause anytime priority/time slicing：在同一
global 60 秒预算内为多个 unresolved clauses 分配 additive work，不得继续让 clause 0 独占余量或
给每个 clause 重置 deadline。

## 49. Multi-Clause Anytime Priority v1

NRIR-36 保留 frozen NRIR-31 九子句 floor 与 NRIR-34 sibling-packed evaluator，新增一等
Policy/Plan/Candidate/Decision/8-task Task/Schedule/Slice/Outcome/Aggregate IR。priority 只消费 floor
sound root lower margin，按降序、ordinal 升序打破平局，固定选 top-2；每次 dispatch 将真实剩余
global budget 等分给尚未执行的 selected clauses。私有 one-shot clock 只在 slice cutoff 首次向 frozen
packed queue 暴露 global expiry，完整 sibling pair 才能原子提交；所有 source Plan/semantic/final-bound、
allocation、packed verdict 与 original ordinal 均 hash-bound。

单次 first-class pilot 通过后，正式三 fresh repeats 的 priority 都为
`[2,3,4,5,0,8,6,7,1]`、selected 都为 `[2,3]`。floor elapsed=
`[21.637124,21.604930,21.871310] s`；packed nodes=`[[3,3],[3,3],[3,1]]`。前两轮 clauses 2/3
各提交 `3 nodes/1 group`；repeat 2 的 clause 3 在 global cutoff 只提交 root，未形成 atomic pair，
worst active lower 仍为 floor `-152.287033`。whole cooperative elapsed=
`[67.213556,66.833706,60.228863] s`；三轮 final 都仍为 sound unknown、9/9 unresolved。

formal hash=`2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；
replay、wrong rank/selection/source、slice inflation、deadline reset、ordinal omission、non-monotone
aggregate、partial group、trace binding 篡改、NRIR-31/34/35 predecessor replay 与全量
`890 passed, 37 skipped` 均通过。由于“两条 selected clauses 三轮均提交 atomic pair”的预注册 gate 失败，本阶段以
multi-clause allocation `VALIDATED-NO-GO` 关闭，`performance_claimed=false`；IR/control 可保留，
没有 property closure、GPU、competitor、multi-workload 或 ASPLOS-ready claim。下一门禁转向 shared
parametric compiler/root/evaluator 与
stronger candidate/bound，先量化 compile/root/child phase 并冻结复用合同，不继续调 top-k/slice 常数。

## 50. Shared Parametric Objective Evaluator v1

NRIR-37 的因果变量只有 evaluator compiler ownership。frozen NRIR-31 exact floor、NRIR-36 priority/
top-2/dynamic equal-remaining slice、NRIR-34 cap128 ancestral refinement/sibling atomic commit、31/depth4
与 global 60 秒全部不变。新增的 shared-parametric Plan/Batch/Task/Schedule 明确区分：

1. template：graph、input non-batch shape、objective shape/dtype/device、ReLU layout、optimizer policy、
   intermediate-bound provenance；
2. instance：objective content、split state、intermediate bounds、warm state、refinement lineage、batch size；
3. cache：一个 query owner，第一次 `miss_compiled`，其余跨 batch/跨 clause 必须 exact hit；
4. production batch：不构造 audit hash chain，不做 selected-native re-execution，root/完整 pair 才 commit。

first-class clause-2 root+pair parity：audit/shared elapsed=`14.073795/1.198798 s`；lower、branch、split、
α、β 与 refinement final-bound hashes exact；upper max diff=`1.52587890625e-5`，既有 relative+absolute
allclose guard 通过。该 timing 只用于内部 phase 归因，`performance_claimed=false`。

单轮 top-2 pilot 已得到 `[31,31]`，随后三 fresh processes 均复现 rank=
`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`、packed nodes=`[31,31]`、cache miss=1。floor elapsed=
`[21.733539,21.941763,21.925033] s`，whole elapsed=
`[51.996191,52.251681,52.695640] s`。pilot/formal hashes 分别为
`c96fff3fa2bc2563b4d46886d69b33f51ac985b19ad80d916309db57fe6cfefa`、
`9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`。

replay、rank/source/allocation/group/cache/event/native-reexecution/compiler-coverage 与 Task/Batch binding
tamper、27 focused tests、全量 `917 passed, 37 skipped` 与静态门禁通过。NRIR-37 以 shared compiler ownership + fixed-deadline
coverage `VALIDATED-REDUCED` 关闭；三轮 final 仍 9/9 unresolved，depth-4 worst active lower 为
clauses 2/3 的 `-37.574287/-35.900215`。下一门禁只做 frontier tightness attribution，再预注册一个
单变量 stronger-bound/candidate 实验；不得继续调 top-k、slice、cache 或把 CPU 内部 timing 升级为
competitor speedup。

## 51. Full Frontier Tightness Attribution v1

NRIR-38 保持 NRIR-37 的 source execution、31/depth4、cap128、widest branch、ancestral refinement、
sibling grouping、parent warm state 与 dtype/device 不变，只预注册 optimizer `steps=5→15`。新增
first-class attribution Plan 与七阶段 Task/Schedule；source 31 nodes 的 depth/path、refinement pass、
alpha/beta state 和 16-node active frontier 全量进入 typed evidence，candidate 按原八个 sibling pair
重放，baseline/candidate 各有独立 exact template cache。

clauses 2/3 baseline replay lower/upper max diff=0，refinement hashes exact。steps15 的 32/32 active
nodes 都严格改善，median delta=`+0.107208/+0.132715`，但 worst-active lower 仅改善
`+0.055496/+0.028557`；depth-4 alpha interior fraction 仅 `2.164%/2.518%`。由于预注册门禁要求两条
clause worst improvement 均至少 `+1.0`，本轴以 `VALIDATED-NO-GO` 关闭，不运行 steps15 full-query
formal，也不补试其他 step 数。

pilot hash=`2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`；13 focused、全量
`930 passed, 37 skipped` 与静态门禁通过。下一阶段只改变 branch candidate：复用仓库已有 objective
branch Plan/Task/Schedule，把 objective-bound-impact selection 接入 shared ancestral evaluator，并与
widest branch 做 exact fixed-tree 对照；不得重新打开 optimizer/cap/multipass/control 常数。

## 52. Objective Branch Shared Evaluator v1

NRIR-39 只改变 branch candidate selection。frozen shared plan、steps5、cap128 ancestral refinement、
parent warm、query-owned cross-clause cache、best-first queue、31/depth4 和 sibling atomic commit 都不变；
candidate 固定使用历史 `top_width_per_relu_v1`（8/ReLU、batch64、cap256）及
`maximize_worst_child_then_mean`。composite Plan/6-task TaskModule/Schedule 绑定底层 shared execution 和
31/31 objective branch Plan/Task/Schedule/score traces。

真实 clauses 2/3 control/candidate 都达到 31 evaluations、16 depth-4 active nodes，root lower exact。
worst-active improvement=`+2.043362/+5.641768`、median delta=`+2.537640/+5.885233`，两条均过预注册
`+1.0` gate。pilot hash=`dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`，
结论为 fixed-budget branch selection `VALIDATED-REDUCED`。

这不构成 wall-clock speedup 或 property closure：pilot 使用 logical fixed-budget clock，objective scoring
会增加实际工作量。下一门禁必须做 three fresh whole-query/global-deadline repeats，同时报告 floor、branch
scoring/queue、committed nodes、cache 与 final 9-clause verdict；不得把本阶段 tightness 直接写成
performance、GPU、competitor、multi-workload 或 ASPLOS-ready。

## 53. Objective Branch Whole Query Formal v1

NRIR-40 将 NRIR-39 frozen objective-bound-impact branch 接入 raw shared production queue，并与
NRIR-36/37 nine-clause anytime runtime 组合。floor、rank/top-2、dynamic equal-remaining allocation、
steps5、cap128 ancestral refinement、query-owned cache、best-first、31/depth4 与 global 60 秒均不变；
objective scoring 在真实 slice/global monotonic deadline 内执行，不额外重放 widest control。新增 runner
冻结 three fresh workers、worker/formal/manifest schema、逐轮 shards/logs，并将 shard 与 formal 内嵌结果
交叉绑定。

三轮 correctness gate 全过：floor 9/9、rank=`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`、branch
executions 与 accepted nodes 一一对应、每轮 cache 恰好 `1 miss`，original ordinals 和 sound aggregate
完整。floor elapsed=`[21.636507,22.057062,22.088135] s`；whole cooperative elapsed=
`[63.357098,63.161128,62.485366] s`。

production gate 三轮均失败：nodes/groups 仅为
`[[29/14,23/11],[29/14,21/10],[29/14,21/10]]`，没有达到 clauses 2/3 各 `31/15`；worst-active
lower 分别为 `[-48.315041,-43.299690]`、`[-48.315041,-44.731468]`、
`[-48.315041,-44.731468]`，相对 frozen widest `-37.574287/-35.900215` 也未达到 `+1.0`。
formal hash=`d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`；原样 replay 与
formal+shard+manifest 同步重哈希 branch-coverage tamper 均通过预期门禁。
focused `8 passed`、predecessor-inclusive `55 passed`、全量 `944 passed, 37 skipped` 与 Black/mypy/
Pylint `10.00/10` 通过。

本阶段以 objective-branch global-budget `VALIDATED-NO-GO` 关闭，`performance_claimed=false`。
NRIR-39 fixed-budget `VALIDATED-REDUCED` 仍只证明在相同 31-node frontier 上 branch policy 可改善 lower，
不能推导真实 deadline 下的 production 收益。下一阶段若继续，必须先做 objective scoring/queue phase
wall-time 与 frontier-order 因果归因，再预注册一个单变量；不得事后调 top-k、slice、node cap、optimizer
或门槛，也不得形成 property/GPU/competitor/multi-workload/ASPLOS-ready claim。

## 54. Objective Branch Production Cost Attribution v1

NRIR-41 不修改 NRIR-39/40 frozen 文件，也不直接优化 policy。它先把 NO-GO 拆成两个可证伪问题：
一是在相同 `21/23/29/31` accepted-node 前缀上 objective branch 的 worst frontier 是否仍不弱于
widest；二是 objective scoring 的真实 queue wall 成本是否足以解释 global deadline 下缺失最后一个
sibling pair。prefix 只从 NRIR-39 frozen evaluations 按 parent lineage 独立重建。

成本实验固定 clauses 2/3、CPU 8 threads、31/depth4、fresh cache，使用 3 fresh paired subprocesses 并
按 `W→O/O→W/W→O` counterbalance；另设 1 个 cProfile diagnostic，profiled timing 不进入 unprofiled
median。新增 attribution Plan/Task/Schedule/Decision 拥有 source admission、prefix reconstruction、paired
execution、phase profile、causal decision 和 emit；`performance_claimed=false`。

只有 `frontier_order_retained`（两 clause 四 prefix 均不弱且 31-node `>=+1.0`）与
`scoring_cost_dominant`（两 clause objective/widest queue median ratio 均 `>=1.20` 且 branch-program
cumulative share `>=20%`）同时成立，下一阶段才允许优化 scorer ownership/复用。前者失败则冻结
objective branch production 路线；前者成立但后者失败则转查 deadline/atomic-tail scheduling，不扫
top-k、slice、node cap、optimizer 或门槛。

正式结果中，same-prefix worst-active improvement 对 clauses 2/3 分别为
`[+2.171364,+2.416264,+2.947929,+2.043362]` 与
`[+4.988102,+6.255299,+6.350922,+5.641768]`，故 frontier gate 成立。三 fresh paired runs 的
widest/objective queue median 为 `10.515292/18.387675 s` 与 `10.619606/18.591097 s`，ratio=
`1.748660/1.750639`；MAD 分别为 `0.020595/0.266792 s` 与 `0.002217/0.242127 s`。
cProfile branch-program share=`21.9371%/21.9139%`，并显示 31 次 branch program 触发 341 次
candidate enumeration。两个方向门禁均成立，Decision 为 `optimize_scorer_ownership`。

formal hash=`fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`；replay 与同步重哈希
prefix tamper、focused `4 passed`、predecessor-inclusive `12 passed`、全量
`948 passed, 37 skipped` 与静态门禁通过。本阶段以 internal causal attribution
`VALIDATED-REDUCED` 关闭，但不构成
system speedup/property/GPU/competitor/multi-workload/ASPLOS-ready claim，也不撤销 NRIR-40
global-budget NO-GO。下一阶段只能消除 scorer ownership/validation 重复并保持 exact branch semantics。

## 55. Objective Branch Scorer Ownership v1

NRIR-42 只改变 scorer candidate-table/validation ownership。新增 typed validated capsule，使每个 node 的
candidate enumeration 由 branch Plan compile 恰好拥有一次，执行和下游验证只消费 immutable
candidate table 与 semantic token；historical scorer 和 NRIR-39/40 frozen 文件不改。objective policy、
optimizer/refinement、31/depth4、queue、cache、slice 与 deadline 全部冻结。

Phase A 必须证明 clauses 2/3 old/new 31-node 的 selected branch、所有 score rows、child lower、queue
lower/upper、split、α/β、refinement exact；enumeration calls 从 `341` 变为 compile=`31`、execute=`0`；
三 fresh counterbalanced new/old queue median ratio 两条均 `<=0.75` 且改善大于 MAD。任一失败即
scorer optimization NO-GO。

只有 Phase A 全过才运行 three fresh whole-query/global-60s Phase B：两条每轮 `31 nodes/15 groups`、
相对 widest worst lower 各 `>=+1.0`、whole `<=70s` 且无 partial/reset/recompile/evidence omission 才能
恢复 objective-branch production `VALIDATED-REDUCED`。否则 NRIR-40 production NO-GO 保持不变；所有
timing 仍为内部准入，`performance_claimed=false`。

Phase A 正式结果：clauses 2/3 三 fresh counterbalanced new/old queue median ratio=
`0.706888/0.698486`，median 节省=`5.468696/5.680614 s`，均严格大于历史/新路径 MAD；每条
31-node queue 都由 historical `341` 次 enumeration 降为 prevalidated compile=`31`、execute=`0`。
六组 old/new selected branch、全部 score、child-lower、queue lower/upper、split、α/β 与 refinement
exact。typed replay 会重新构造 Plan/Task/Schedule/capsule；同步 token/score/call tamper fail closed。
Phase-A formal hash=`0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58`。

Phase B 随后按门禁执行。三 fresh whole-query 都选择 `[2,3]`，两条每轮均提交
`31 nodes/15 groups/31 capsules`，whole elapsed=`[57.175184,57.697757,58.114412] s`；worst active
lower 固定为 `-35.530926/-30.258448`，相对 NRIR-37 widest 改善 `+2.043362/+5.641768`，无
partial/reset/recompile/evidence omission。Phase-B formal hash=
`7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`；targeted `10 passed`，
全量 `958 passed, 37 skipped`，静态门禁通过。

NRIR-42 因此以 fixed ResNet2B property 0、CPU8、global-60s objective-branch production admission
`VALIDATED-REDUCED` 关闭，并在该窄范围内取代 NRIR-40 的 production NO-GO；NRIR-40 frozen 证据本身
不改。final property 仍 unknown，且没有 GPU、multi-workload、fair competitor speedup 或
ASPLOS-ready claim。下一单变量应把当前顺序的 cross-clause/node/candidate work lower 为联合 batch
Schedule，并以 exact semantics + fresh paired timing 判定；不再优化 scorer validation 常数。

发布状态：功能提交 `264365f` 已由 PR #53 合入 `main@8969064`。后续实验必须以该 merge commit
为 integration base，不能在旧 NRIR-41 基线上继续分叉。

## 56. Cross-Axis Verification Batch Schedule v1（预注册）

NRIR-43 的唯一变量是 ready-work Schedule。NRIR-42 每条 selected clause 由 1 个 root 与 15 个
sibling groups 组成；每节点 48 candidates 对应 96 child domains。现实现按 clause 串行，并在每个
sibling group 内按 node 串行 scorer，因此两 clause 合计发射 32 次 optimizer batch 和 62 次 scorer
lower batch。NRIR-43 只把这些已经独立且 ready 的工作沿 clause/node/candidate 轴联合装箱，再用
typed ragged segments 还原到原 queue；不得改变任何算法或数值 policy。

Phase A 先验证单 queue sibling-node scorer pack：逐节点 candidate/score/branch/child lower/queue/
split/α/β/refinement 等价，scorer launch `62→<=32`，三 paired repeats 每条 queue ratio `<=0.85` 且
改善大于 pooled MAD。任一失败即 `VALIDATED-NO-GO`，不进入全局运行。

Phase B 才允许 two-clause ready-set coordinator：两个 queue 的状态与提交顺序互相隔离，只联合
同 round 的 root 或 sibling tensor work。门禁为 optimizer launch `32→<=16`、scorer launch
`62→<=16`、两条均 `31 nodes/15 groups/31 capsules`、three fresh whole 每轮 `<=45 s` 且 median
ratio `<=0.80`。即使通过也仅是 fixed ResNet2B property 0 CPU8 internal admission，
`performance_claimed=false`；公平竞品、多 workload、GPU 与 ASPLOS-ready 仍需后续独立门禁。

Phase A 正式结果：6 个 clause-repeat 组的 queue/branch/48-entry score/child lower/state/split/α/β/
refinement 全部 exact；每条 scorer launches `31→16`。但是 clauses 2/3 的 NRIR-42/cross-axis median=
`12.821506/13.477127 s` 与 `13.004753/13.584418 s`，ratio=`1.051134/1.044573`，两条 timing gate
均失败。formal hash=`692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`；
targeted `10 passed`、全量 `968 passed, 37 skipped` 与静态门禁通过。

因此 NRIR-43 以 `VALIDATED-NO-GO` 关闭，Phase B 按预注册不得启动。下一单变量转 NRIR-44
Root-Projection Floor Schedule：当前 floor 约 21.77 秒，其中 baseline 约 4.82 秒、九条顺序
objective queries 合计约 13.88 秒，而 ranking consumer 只读取每条 root lower；应以 typed consumer
contract 消除非 top-2 的深层 queue work，不继续扩大 CPU domain batch。

发布状态：NRIR-43 提交 `00b82c2` 已由 PR #54 合入 `main@2d245d6`；production 默认仍为 NRIR-42。

## 57. Root-Projection Floor Schedule v1（预注册）

NRIR-44 解决 floor 的 consumer/liveness 冗余。单次分解显示 21.77 秒 floor 中 baseline 约
4.82 秒，9 条顺序 objective queries 合计约 13.88 秒；而 multi-clause ranking 后续只读取每条
accepted child 的 root lower。路线冻结前用同一 objective refinement 做 n1d0 probe，9 条合计
`0.789371 s`，root lower/upper/branch 9/9 与 n31d4 root exact。

唯一变量是将 floor child query 从 `9×n31d4` 投影为 `9×n1d0`。baseline、shared/objective
refinement、search/optimizer、root semantics、rank/tie-break/top-2、NRIR-42 31-node production、dtype
与 global-60s deadline 均冻结。该 specialization 是 sound-but-less-complete：非 top-2 clause 在 floor
阶段不再尝试深层证明，必须显式由 ranking-only consumer contract 启用。

Phase A 要求 baseline/refinement/root/rank/selected exact、objective evaluations `279→9`、three paired
floor 每轮 `<=11 s` 且 median ratio `<=0.50`。只有全过才运行 Phase B，其要求 three fresh whole
每轮 `<=48 s`、ratio `<=0.82`，并保持 selected `[2,3]`、两条 `[31,31]` nodes 与 NRIR-42
branch/score/queue/state/refinement exact。

Phase A 正式结果三轮 exact，old/projected floor elapsed=
`[24.235039,22.859521,24.252771]/[9.739498,10.740998,9.876515] s`，median ratio=`0.407530`，
evaluations=`279→9`，formal hash=`ecb553d88be065054abb0a480b79086ae12cec55a84e5c0ba537572e904ff0fe`。
Phase B 随后按门禁执行，floor=`[8.538814,8.622447,8.648849] s`，whole=
`[43.571040,44.144990,44.095736] s`，相对 frozen NRIR-42 median ratio=`0.764254`；每轮 selected
`[2,3]`、nodes=`[31,31]`、worst active lower=`-35.530926/-30.258448`，formal payload hash=
`2f22d44fe9f57f233c8a853b66f67f404b03a087d097451e10f663ee257272d9`。

NRIR-44 以 fixed ResNet2B property 0 CPU8 ranking-floor + production admission
`VALIDATED-REDUCED` 关闭，`performance_claimed=false`；final 仍 9/9 unknown，没有公平竞品、GPU、
multi-workload、property closure 或 ASPLOS-ready claim；全量 `979 passed, 37 skipped`。下一单变量必须来自剩余约 35 秒 top-2
production queue 的成本归因，不回退 NRIR-43 已否决的 CPU scorer batching。

发布状态：NRIR-44 功能/证据提交 `437680e` 已由 PR #55 合入 `main@f194034`。

## 58. Prepared Intermediate Refinement Capsule v1 判定

NRIR-45 的唯一变量是 per-child intermediate refinement 的 validation ownership。NRIR-44 projected
floor、rank/top-2、refinement target/policy/selected-CROWN、optimizer、objective branch、31/depth4、queue、
dtype、workload 与 global-60s deadline 全部冻结。每个 exact Program/Execution 必须在 prepare 时完整
验证一次并生成 typed immutable capsule；runtime 只消费 capsule/Plan-owned targets，不得用 object-ID
cache、裸 bool 或无条件跳过验证。

路线前 cProfile 显示单条 31-node queue 的 246 次 `_select_targets` 中，186 次来自重复
`Program.validate()`；30 次 compile 与 30 次 runtime 才是当前语义路径。每 exact object 仍完整验证一次的
只读 ceiling probe 将 clause 3 trace 从约 `12.85 s` 降到 `9.761678 s`，31 nodes 和 worst lower exact。
该 probe 只用于选路线。

Phase A 要求 clauses 2/3 three fresh counterbalanced 31-node queues 全语义 exact，prepared/control
median ratio 均 `<=0.80` 且改善超过 pooled MAD，并证明 validation/target-selection ownership 收敛。
只有 Phase A 全过才运行 Phase B；其要求 three fresh whole queries 每轮 execution trace `<=40 s`、
measured wall `<=50 s`，相对 NRIR-44 median ratio 分别 `<=0.90/0.85`。以上是开工前冻结门禁；
预注册时没有正式结果或新 claim。

正式 Phase A 六组 control/prepared 31-node queue 的 branch/score/state/refinement/worst lower exact；
target selection=`246→98`、full Program validation=`186→38`、full hash=`217→39`。clauses 2/3
control/prepared median=`12.981239/9.444103 s` 与 `13.122778/9.666283 s`，ratio=
`0.727519/0.736603`，改善均大于 pooled MAD。formal hash=
`be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`。

Phase B 三 fresh processes 的 floor=`8.625022/8.583826/8.628565 s`，whole trace=
`31.262521/31.319772/31.470078 s`，measured wall=`36.396631/36.513683/36.611709 s`；相对 frozen
NRIR-44 trace/measured median ratio=`0.710268/0.615738`，均大于 pooled MAD。每轮 selected `[2,3]`、
nodes `[31,31]`、worst active lower=`-35.530926/-30.258448`，prepared capsules/full replay=`60/60`。
Phase-B payload hash=`4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`；
两阶段 replay、tamper、全量 `984 passed, 37 skipped` 与静态门禁通过。

NRIR-45 以 fixed ResNet2B property 0 CPU8 internal production admission `VALIDATED-REDUCED` 关闭。
final 仍 9/9 unknown，`performance_claimed=false`，没有公平竞品、GPU、multi-workload、property closure
或 ASPLOS-ready claim。下一步先做最终约 31.3 秒路径的 residual phase attribution，再冻结 NRIR-46
单变量；不重开 NRIR-43 CPU scorer batching，也不事后降低 cap/nodes/depth。

## 59. Intermediate Refinement Template/Instance v1 Phase 0 NO-GO

NRIR-45 raw Phase-B trace 的 residual attribution 已完成：floor action median=`10.818262 s`，两条
packed slice 六样本 median=`9.932808 s`，packed-plan compile/rank median 仅
`0.146457/0.024966 s`。一次 diagnostic repeat0 进一步测得 60 child prepared compile/execute=
`5.300590/5.659414 s`、per-child total=`10.975123 s`、optimizer execute=`1.156098 s`。这些数字只用于
冻结路线，不是新的 formal performance claim。

NRIR-46 的唯一变量原定为 first-class compiler IR 静态/动态分层：PlanTemplate/TaskTemplate/
ScheduleTemplate 拥有 graph、policy、selection recipe 与拓扑；PlanInstance/InstanceSchedule 逐 child
拥有 split、source lineage、objective、bounds 与 exact target ledger。NRIR-46 不引入跨节点数值
batching，也不改 refinement/optimizer/branch/queue/policy/budget/deadline。

Phase 0 三 fresh processes 的 compile total=`5.356892/5.366369/5.452290 s`；strict static topology=
`1.071197/1.062492/1.071704 s`，median=`1.071197 s`；ownership-convertible ceiling=
`2.097255/2.102134/2.109857 s`。预注册 static-shareable gate 要求 median 至少 `1.5 s`，故门禁失败，
NRIR46 `VALIDATED-NO-GO`，Template/Instance 未实现，Phase A/B gated off。

三轮 60 个 target identity/table hash 全部互异，但 primal graph、Task/Schedule topology 各只有一种；
target selection observed/semantic=`124/60`，64 次冗余 selection 估计耗时=
`1.026058/1.039642/1.038153 s`。formal hash=
`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`；replay/tamper 通过，
`performance_claimed=false`。

下一路线不能共享动态 target ledger，而应作为 NRIR47 独立预注册 single-pass exact target admission
receipt：production compile 只选择一次，显式 full replay 仍从源输入重算。它不是公平竞品、10x 或
ASPLOS-ready 终点。

## 60. Single-Pass Target Admission Receipt v1 Phase A NO-GO

PR #57 已将 NRIR46 Phase 0 NO-GO 合入 `main@ca0bcf3`。NRIR47 只处理已观测的 target reselection：
每 child 从 exact bounds/effective policy/objective influence 选择一次并生成 typed receipt；receipt
绑定 ordered target table 及全部语义输入，production validator 消费 receipt 而不重调 selector；
显式 `validate_full` 仍重调 selector并逐项比较。

NRIR46 三轮 observed/semantic target selection=`124/60`，64 次冗余耗时估计 median=`1.038153 s`；
60/60 target ledger 互异。因此 NRIR47 禁止跨 child 缓存 targets，也不恢复已 NO-GO 的
Template/Instance。Phase A compiler ratio 门槛=`0.85`、clauses 2/3 queue ratio=`0.97`；Phase B
trace/measured ratio=`0.98`，所有改善必须大于 pooled MAD。

NRIR47 已实现 typed receipt/Task/Schedule、additive single-pass compiler、prepared capsule binding、
candidate production route 与 explicit full replay。每条 candidate queue compile selector/reselection=
`30/0`、runtime selector=`30`、receipt/full replay=`31/31`，三轮两条 clause 共 replay 186 份 receipt；
correctness/ownership 与 synchronized outer-rehash tamper 门禁通过。

正式 compiler control/candidate median=`2.739226/2.563922 s`，ratio=`0.936003 > 0.85`；clauses 2/3
queue ratio=`1.011205/1.019338 > 0.97`，且 queue 改善未超过 pooled MAD。因此 NRIR47
`VALIDATED-NO-GO`，Phase B gated off。formal hash=
`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；全量
`992 passed, 37 skipped`，Pylint `10.00/10`。receipt candidate 不默认启用，不形成 performance claim；
下一门禁转向 top-2 production execution math/queue phase attribution。

## 61. Top-2 Production Execution Cost Attribution v1 判定

NRIR47 已由 PR #58 合入 `main@1e44949`。NRIR48 不实现优化，只测 frozen NRIR45 default
production route 的 clauses 2/3 31-node queues；NRIR47 candidate 保持禁用。七个互斥顶层类别为
child refinement compile/execute、optimizer prepare/execute、branch bind/score、materialize/commit 与
queue-control residual；child execute 再分 fast validate、runtime target select、selected-CROWN、
propagate-forward 与 hash/trace residual。

正式协议为 clauses 2/3 各 three fresh counterbalanced control/profile。要求 6/6 semantics exact、
category closure error `<=1%`、profile/control median ratio `<=1.05`。只有同一 category 在两条 clause
各 3/3 repeats 排第一、median share 均 `>=20%`、share range `<=10` percentage points 且 median
exclusive ns 大于 MAD，才允许成为 NRIR49 来源。以上门禁均在正式运行前冻结。

正式结果 6/6 control/profile semantic exact，clauses 2/3 profile/control median ratio=
`1.023199/1.020221`，插桩门禁通过。两条 clause 的 3/3 winner 都是
`child_refinement_execute_ns`：median=`3.816002/3.704755 s`，queue share=
`32.1966%/31.1640%`。内部唯一超过 `30%` parent-share 的子类为 selected-CROWN：median=
`2.663321/2.694436 s`，占 child execute=`71.7725%/72.7291%`。

formal hash=`571c2e47c0c8906d2486e5e19e8152eb1ef0d3024b08cf561e25ed4f71d177a4`；
顶层/内部闭合、6 profile rows replay、同步 category tamper 拒绝、全量 `996 passed, 37 skipped`
与静态门禁通过。NRIR48 以 attribution
`VALIDATED-REDUCED` 关闭，只证明 dominant cost 已缩小到 selected-CROWN execution，不是 speedup。
当时准入的下一单变量为 NRIR49 selected-CROWN execution，不再优化 validator、optimizer 或 queue
bookkeeping；该历史动作已由下节完成，不是当前指令。

## 62. NRIR49A G1 GPU Selected-CROWN-only Opportunity Attribution NO-GO

G0 post-reboot六项CUDA门禁通过后，NRIR49A只读测量frozen clauses 2/3的GPU selected-CROWN；不改
production runtime、TIR、kernel、default chunk、solver policy、target ledger或termination。正式协议为
五fresh workers、chunk `8/16/128/32/64` Latin轮转、paired default32 control和一个排除在timing外的
真实child CUPTI profile。

正式5/5 worker通过hash/envelope；60组离散结构exact，raw浮点最大absolute/relative diff=
`2.288818359375e-05/1.710717646052519e-04 <=2e-4`。两条clause profiler/control ratio中位=
`0.999304/1.006747 <=1.05`。selected-CROWN queue/complete share中位仅=
`7.0986%/7.0523%`：低于20%机会门槛，且queue `1.20x`、complete `1.15x`均超过Amdahl无限区域
加速上限。最大allocated/reserved仅物理显存`0.996%/1.353%`，合法batch上限1、无OOM，memory
path=`N/A`。

formal summary/manifest hash=`7eefe6a7…ab50`/`d0272fe4…c81f`；独立replay exit 0、stdout exact、
所有payload与manifest digest重算吻合。NRIR49A G1以
`VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`关闭；只停止selected-CROWN专属
G2/G3/TIR/JIT/融合。`1/(1-0.070986)=1.0764x`只是假设该region变为零耗时的deletion-only单区域
上限，不是BoundFlow从operator、graph/IR、JIT、runtime schedule到allocator/memory的累计全栈上限。

正式artifact中的`next_route=gpu-winner-reselection`为冻结历史机器输出，当前路线已由
`gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`取代。FSG0的
作用域/schema/feature activation/replay合同已以20项定向测试和`1079 passed, 3 skipped`关闭；外部
审计三项minor已修复。当前执行FSG1 official original executor control full-stack trace。FSG1只建立B0
分层基线，不能声称已有BoundFlow
全栈性能。`performance_claimed=false`，公平竞品、multi-workload、solved verdict、memory headline与
ASPLOS-ready仍未成立。
> **2026-08-24 FSG4/B4-C1实现指令**：P-anchor lower ReLU→Conv已改为provider-owned，native
> lower不再双算；10次迭代复用plan buffer/DLPack view。单worker语义误差≤`7.153e-7`且sign exact，
> 但累计core仅约`0.95x`。当前只开放6 fresh正式关闭；若NO-GO，转B4-C2真实materialization
> frontier累计覆盖，不得继续把局部eager-materialization的`4.90x`外推为production speedup。
>
> **2026-08-24 FSG4/B4-C1正式NO-GO指令**：source=`01bb215`完成6 fresh/180 groups；
> provider-owned lower语义max diff=`7.153e-7`且sign exact，但core geomean=`0.94815x`、
> worst=`0.94547x`，8/8 tamper rejected。最终=
> `VALIDATED-NO-GO-B4-C1-MATERIALIZATION-FRONTIER`。只开放B4-C2真实materialization frontier
> operator-tree融合与14-call累计coverage；不得继续调同一P-anchor。
>
> **2026-08-24 FSG4/B4-C2与B4总关闭指令**：真实6-site lower materialization frontier已覆盖
> optimizer 60/60次，三worker语义exact但speedup仅`0.3488/0.3374/0.3460x`，peak allocated
> ratio=`1.3401`。根因是dense autograd中间态跨层保留。最终=
> `VALIDATED-NO-GO-B4-C2-DENSE-RETENTION`且本轮纵向alpha-CROWN B4整体NO-GO；B4-D关闭。
> 下一路线独立启动CIBC论文的IBP/forward-bound水平融合与autotuning，不继承B4 claim。
>
> **2026-08-24 CIBC-IBP水平融合实现指令**：已按论文center/deviation公式实现单kernel lower/upper
> Conv TIR、64/128/256 schedule、plan-owned零拷贝runtime及公平baseline/candidate CUDA graph。
> 诊断上真实Conv约`7.72x`、完整ResNet2B IBP graph约`2.70x`，但仍为
> `IMPLEMENTED-PENDING-FORMAL`。下一唯一动作=6 fresh operator/model formal artifact；不得与已NO-GO
> 的B4 alpha-CROWN claim合并。
>
> **2026-08-24 CIBC-IBP正式协议指令**：operator层固定3个fresh schedule workers、真实6 Conv、
> 每算子30×500次；whole-model固定6 fresh、每组100次×30组。baseline/candidate均用CUDA Graph，
> 输入lower/upper copy计入。门禁为operator geomean/worst≥`2.0/1.2x`、whole-model
> geomean/worst≥`1.5/1.2x`、全部中间interval absolute diff≤`3e-4`且sign exact。协议提交后才运行，
> 运行中不得新增schedule或修改门槛；显存与alpha-CROWN不属于本claim。
>
> **2026-08-24 CIBC-IBP正式关闭指令**：source=`a52b177`的3 operator+6 model fresh artifact通过。
> 128-thread schedule的6 Conv geomean/worst=`12.795/9.142x`；完整ResNet2B IBP CUDA Graph
> geomean/bootstrap lower/worst=`2.4563/2.4539/2.4509x`，输入copy计入；全中间interval max diff=
> `2.4414e-4`且sign exact；10/10 fully-resigned tamper rejected。最终=
> `VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL`。不得升级为auto_LiRPA、alpha-CROWN/BaB/query、
> memory、跨模型或ASPLOS-ready claim。
>
