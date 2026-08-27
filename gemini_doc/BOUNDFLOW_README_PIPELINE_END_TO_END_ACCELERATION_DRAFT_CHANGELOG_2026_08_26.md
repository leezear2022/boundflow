# README 主流水线端到端接通计划变更记录

date: 2026-08-26
updated: 2026-08-27
status: draft-for-user-review
revision: v5-hybrid-capture-import
code-changed: false
performance-claimed: false

## v5 捕获前端、唯一执行IR与prepared publish收束

- 接受用户对“用IR还是别的机制”的再次追问，把设计从“最小Verification Flow + TransactionContract”进一步
  收束为`capture/import + BF/Verification semantic overlay + Relax/TIR + RVIR`混合架构；
- 冻结唯一执行身份：只有Relax是高层production execution IR、TIR是kernel execution IR；runtime trace、
  ExportedProgram/FX/AOT图、BFBoundModule/VerificationGraph source、`VerificationOverlayV1`、publish protocol和RVIR state machine均不
  再被描述成平行compiler execution IR；
- 将runtime/source instrumentation定位为production事实权威，只记录真实call/state/effect/alias/lifetime/
  sync/failure boundary，不从observed trace直接生成可优化程序；
- 将`torch.export`、Dynamo fullgraph与AOTAutograd引入per-region capture bake-off：分别处理IBP、P/S CROWN、
  VJP、optimizer step和10/9组合，不预设一个工具捕获整个solver；
- formal capture新增op/value/output/gradient coverage、graph break/silent fallback、guards/shape constraints、
  Python/static constantization、functionalized mutation、alias/view、saved tensor、host escape、re-export identity
  与import-to-Relax feasibility门禁；
- 可复用formal forward只接受strict export经冻结decomposition得到的functional Core ATen（mutable op=`0`）；
  non-strict export和Dynamo/裸FX只作discovery/baseline，或在全部输入与初始state内容hash绑定时作单实例replay；
- formal backward只接受固定Torch build、AOT API、decomposition、partition、primal/tangent/gradient mapping和
  saved-output ABI的AOT pair，不把raw joint graph当稳定接口；
- M0改为`runtime ground truth → capture bake-off → per-region CaptureDecision`，交付
  `AutoLiRPACaptureEvidenceV1/CaptureBakeoffReportV1/CaptureRegionManifestV1`，三者均非runtime IR；
- M1改为import-first：优先导入获选strict functional ExportedProgram/pinned AOT pair，capture失败的host policy
  形成显式cut；
  BFBound/GC0只补polarity、A representation、soundness、effect/VJP/escape等验证语义，禁止复制ATen DAG；
- M1新增`native vs captured/exported vs imported Relax`三方语义闭合，覆盖lower/sign/gradient、10/9 state、
  mutation/alias/saved-state与terminal owner；禁止trace直接充当canonical tensor program；
- 删除未来`bf.txn.begin/seal/commit/abort` seam，不再预注册transaction opcode或interpreter；
- 将prepared协议拆为`ExecutableArtifactIdentityV1 → multi-entry PreparedExecutionBindingV1 → tagged
  PreparedArtifactEnvelopeV1`；M2 `return-only`不制造空publish合同，只有`rvir-publish`才增加编译前
  `PublishIntentV1`和编译后`PreparedPublishBindingV1`；后者同时绑定compiled tensor字段与runtime-assembled
  host字段，避免pre-rewrite value ID、单entry假设和cache artifact错配；
- runtime状态机补齐pending completion→RVIR await→`DEVICE_READY`，terminal集合冻结为
  `COMMITTED/ABORTED/CONFLICT/ROLLED_BACK_INVALIDATED/POISONED`；prewrite stale先分类为`CONFLICT`、generic
  failure不得重复捕获，compensation失败必须quarantine而非伪装回滚；
- 将TX0—TX3改为PUB0—PUB5：M1只生成schema/verifier，M2—M4生成standalone resolved binding/envelope，M5
  才关闭runtime conformance；若要提升completion/lifetime token仍须重新通过新IR
  门禁，不能由某一个async/speculative优化自动开放；
- 将`PUB1 schema-negative`与`PUB3 runtime-negative`拆开；double terminal、await/release顺序、midpublish fault、
  compensation/poison和动态receipt tamper不得由M1纯verifier虚假关闭；
- 新增L0—L5表达升级梯子与NIR-0—NIR-5硬门禁：具体blocker、现有机制不足、两个workload、两个pass、
  Amdahl可达、单一runtime owner与完整hash/lowering/negative/replay closure缺一不可；
- 明确DPL先于局部e-graph，新Relax Op晚于sidecar/SSA/private function，TIR AST fork默认关闭，standalone
  execution IR/interpreter本计划禁止；
- 重画唯一production架构和对象分类表，冻结`VerificationOverlayV1`只能做Relax stable-ID→facts/rules映射，
  不得持有op/edge/topology/interpreter；BFBound/VerificationGraph在one-shot lifting后只留下hash，消除隐藏第三图；
- API拆成`capture_boundflow`与`compile_boundflow`，compile按exposure mode接收optional `PublishIntentV1`，
  prepared run按entry role返回pending
  completion handle，RVIR执行await/validate/publish；identity链在final executable后才加入binding/envelope；
- 迁移表补入Primal/Torch/ONNX frontend、interval-v0 PlanBundle/Task/scheduler、Python Task/Schedule executor、
  production capture adapter和专用实验IR，冻结“additive新API、closure后再deprecate”策略；
- 候选论文贡献改为production per-region capture + verification semantic lifting、Relax/TIR联合
  representation/lifetime/rematerialization，以及RVIR completion-aware prepared publish；不宣称whole-solver
  transaction IR；
- 外部资料补入torch.export programming model/IR spec、PyTorch custom backend、TVM DPL、MLIR dialect
  definition/conversion、CUDA Graph与egg equality saturation，替代方案边界写入主设计；
- 纠正optimizer阶段：M1只闭合single evaluation/VJP/functional optimizer step；M4默认compiled step后回host
  policy，whole-10/9仅限全内容hash replay或全部predicate显式进入受证bounded state；
- 统一alias/saved门禁：external observable alias/mutation对native exact；internal alias与saved inventory允许随
  functionalization/arena/rematerialization改变，但须完整归因并通过VJP、escape、lifetime与saved-cut legality；
- 仍保留production流程统计、CIBC/B4/R3/RVIR/GC0/NRIR资产、相关NO-GO、memory witness、Amdahl与final
  system gates；退出的是重复执行容器和错误ownership，不是既有kernel、oracle、证据或runtime成果；
- 补齐MR3 single-site `0.979727x`/worst `0.916094x`和NRIR49A selected-CROWN share
  `7.0986%/7.0523%`、无限加速上限约`1.0764x`；把540个DLPack/pointer round-trip明确归于MR5/MR6；
- 把artifact/replay定位为可复用纪律、manifest schema与harness模式，各阶段仍需实现自己的semantic replay；
- 将warm runtime检查精确为`O(N_targets)`、每target `O(1)`guard且无content scan，不再错误写成整体`O(1)`；
- 明确rvir-publish的compiled tensor target与runtime-assembled host target互斥并完整覆盖intent集合；pass图中
  `return-only`明确跳过PublishIntent检查且必须无publish ref；
- 本轮只修改用户审阅计划、配套变更记录和文档索引，不修改compiler/runtime/schema/tests，不处理GC0-1
  异步外审，不形成执行授权或新性能claim。

## v5 当前验证

- `git diff --check` PASS；Markdown code fence共62个、成对闭合；canonical `compile_boundflow`声明恰为1处；
- deterministic文档断言PASS：唯一Relax/TIR执行IR、无第三执行图、strict/pinned capture边界、tagged
  return-only/rvir-publish envelope、multi-entry execution binding、host-field binding、互斥terminal、PUB1/PUB3
  分层、external/internal alias边界及历史NO-GO均存在；旧`PreparedPublishContract/STAGED_READY_TOKEN`等术语
  不再出现在当前主计划；
- 三个独立只读子代理分别复核production capture protocol、仓库资产映射和最终设计一致性；发现的问题均已
  回写，最终设计复核=`0 blocker / 0 major`；
- `dol lint --soft` PASS；
- 本轮是文档设计修订，未修改代码、未运行GPU benchmark或测试套件，不形成新性能claim。

## v4 事务语义审计与最小overlay修订（历史，已由v5 publish protocol拆分取代）

- 先定义事务边界：区分external solver-state transaction、private/function-state transition、arena/event
  resource protocol与artifact/receipt evidence，明确CROWN scratch、10/9内部candidate和best select不是外部事务；
- 从formal artifact与fixture独立重数当前实例：1 core、24个provider call（`12/1/11`）、10 evaluation/9
  mutation、每step 24个α/β state tensor、outer snapshot 62 tensors（read-only/copy-in/mutable=`16/34/12`）、
  terminal publish 6 α + 6 β value、3个host packet字段、36 history entry、12 receipt中7条内容变化；
- 独立枚举已有schema：17 value role、17 op kind、8 effect resource、4 effect access、22 rejection reason；
  production state另有19 tensor role和3 ownership，确认不是从零再造语义；
- 审计现有`DeviceAtomicCommitPlanV1/DeviceAtomicTransactionV1/live return`，把真实流程冻结为private stage→
  complete provider assembly→version/alias/placement validation→12 backup/copy→host publish→post-query audit；
- 纠正“atomic rollback”措辞：当前是exclusive executor假设下的软件补偿/逻辑原子性，不是跨GPU tensor与
  Python host object的硬件atomic instruction；
- 通过代码和本机最小实验确认：mid-commit失败可恢复tensor内容/host packet，但两次`copy_`会继续递增
  PyTorch `_version`；因此rollback不是version rollback，失败transaction/frame/token必须失效，retry要重新
  snapshot；
- 将CUDA `ENQUEUED/DEVICE_READY/PUBLISHED/AUDITED`分开：launch/copy返回不等于完成，event/synchronize可暴露
  asynchronous error；commit、cross-stream use和arena release必须显式依赖completion；
- 纠正whole-BaB transaction外推：provider `pick_out()`在core前逐storage destructive pop，`add()`在core后
  逐storage append，当前均无stage/undo/rollback；RVIR只替换`update_bounds_core`，因此只证明成功路径queue
  accounting，不证明整个pop→bound→push round失败原子性或安全retry；
- 将`SolverTransition`拆为`DomainReservation`（尚未实现）、`CoreStatePublishTxn`（已有12-path逻辑事务）和
  `QueueAppendTransition`（成功路径已验证、失败原子性未实现）；
- 收紧10/9语义：production optimizer还包含keep-best/restore、pruning preserve mask、early-stop/timeout、
  dense-β mask、scheduler和last-iteration policy；当前只允许叫exact-signature specialization，进入M4前必须
  显式表示或用冻结witness fail-closed admission；
- 对MLIR Memory Effects/SCF/Async、StableHLO token、IREE Stream、PyTorch functionalization、TVM Relax/
  `call_tir_inplace`与CUDA event做一手资料能力矩阵：effect、token、alias/liveness分别解决冲突、完成与复用，
  都不自动提供all-or-none/rollback/跨host-device原子性；
- 决定不新增独立Transaction execution IR：在现有`VerificationGraph`/Flow sidecar增加静态
  `VerificationTransactionContractV1`与最小target binding，聚合既有value/region/effect；
- 静态合同新增read/private/staged/live集合、isolation domain、ready token、unique commit authority、
  compensation/version policy、completion policy与receipt schema；动态path/pointer/`_version`/event留在
  `InvocationFrame`和RVIR runtime，不进入通用schema；
- 把现有`COMMIT_TOKEN`拆为compiled module只能产出的`STAGED_READY_TOKEN`与RVIR真实publish后才能产生的
  commit token/receipt，消除compiled module和host双重commit owner；
- 冻结动态状态机：OPEN→LAUNCHED→DEVICE_READY→SEALED→VALIDATED→ASSEMBLED→COMMITTING→COMMITTED；
  precommit失败ABORT，stale进入CONFLICT，mid-commit进入content compensation后
  `ROLLED_BACK_INVALIDATED`；
- 线性`begin/seal/commit/abort`仅作为未来概念seam：TX1先交付静态sidecar/verifier，只有physical arena、
  async overlap、去backup clone或epoch publish等具体优化证明需要时才开放TX2 handle；不改TIR AST、不增加
  Python transaction interpreter；
- 明确IR能开放static-check hoist、provider-layout staged output、统一arena、commit strategy selection、
  multistream/speculative child和多signature泛化，但不能自动加速kernel、制造硬件原子性、删除backup或把动态
  BaB queue编入TIR；
- 保留整体性能纪律：atomic commit子阶段`73.295→22.476 ms`约`3.26x`，但typed pre-state
  `24.412→58.284 ms`，最终core/query仅`1.071617x/1.006623x`且对B0为`0.910001x`；后续必须量
  pre-state+stage+compute+assembly+commit+completion整体，禁止只报commit子阶段；
- M0扩为Flow/Lifetime/Transaction只读捕获；M1增加静态TransactionContract和负向verifier；M4补动态
  optimizer branch witness；M5只claimcore publish rollback；M6只有真实failure/retry/speculation收益时才研究
  DomainList reservation和queue-delta transaction；
- 确定性验证：相关GC0/transaction/live-return/optimizer测试`49 passed`；schema、snapshot、optimizer、call
  topology与rollback-version断言脚本PASS；production αβ-CROWN/auto_LiRPA pin核对PASS；`git diff --check`、
  文档合同检查与`dol lint --soft`均PASS；
- `dol validate`仍因`.docops/ev.jsonl`的重复事件ID失败：既有`ev009180/ev009388/ev010862/ev014888`，以及
  本轮并行只读命令hook产生的`ev015120/ev015187/ev015294`。依照append-only规则不改写历史；这些记录不影响
  本轮代码/文档验证与soft lint结论；
- 本轮仍只修改用户审阅计划和变更记录，不修改compiler/runtime/schema/tests，不处理GC0-1异步外审，不形成
  新性能claim或执行授权。

## v3 深度修订

- 回答“是否存在能表示 auto_LiRPA 主流程的 IR”：区分通用控制流的语法表达能力与 production
  αβ-CROWN 所需的 verification semantics；确认没有发现可直接覆盖 BoundNode DAG、CROWN reverse
  wavefront、固定 optimizer、BaB state/transaction 和 GPU lowering 的单一现成 IR；
- 对 production 实际版本重新定锚：RVIR formal 使用外部 αβ-CROWN `e5c7e17` / auto_LiRPA
  `5a098e8`，vendored `9d100ec` 只作本仓参考；
- 逐段解剖 IBP Python DFS、CROWN degree-driven reverse queue、10 evaluation / 9 mutation optimizer、
  BaB preprocess/solve/postprocess 与 DomainList ownership，明确哪些进入 Relax/TIR、哪些继续由 RVIR/host
  管理；
- 将 v2 的“标准 Relax/TIR + attrs”收紧为**一个最小 verification-aware Flow overlay**：第一版只用
  private Relax function、显式 SSA state、attrs 与 sidecar witness，不新造 AST/runtime interpreter；
- 冻结两个顶层边界：可编译 `BoundRegion` 与 host-owned `SolverTransition`；禁止把动态 queue、timeout、
  LP/MIP 和 rollback 硬塞进 TIR；
- 增加 value-level role/version/representation/axes/effect/lifetime 合同，以及
  `ibp_propagate/relax/coefficient_propagate/fanout_accumulate/beta_inject/concretize/optimizer_step/
  select_best/stage_result` 最小语义集合；
- 深入梳理中间张量：per-node bounds、dense/Patches/sparse/identity A、relaxation、α/β、optimizer
  moments、best snapshots、autograd saved tensors 与 domain CPU/GPU copies；
- 补回 v2 资产表遗漏的 NRIR-2：逻辑 arena `1,860,912→442,656 B`（`-76.213%`）、386 alias
  pairs、85 early releases；明确旧 runtime 只删 Python 引用，未形成物理 CUDA claim；
- 将 NRIR-2 last-use/first-gap/alias verifier 迁移方向冻结为 Relax/TIR lifetime pass + physical prepared
  arena；Plan 只保留算法 oracle，不丢掉既有工作；
- 新增非执行 `MemoryWitness`，要求 produced/peak-live/lifetime-area/saved/remat/arena/HBM/transfer/
  allocation/DLPack/launch/sync/framework crossing 物理账；逻辑 bytes、allocated、reserved 与 HBM 不得互替；
- 冻结 `PreparedArtifact / InvocationFrame / AutogradToken` 三生命周期，禁止 module handle 或 autograd ctx
  间接持有本次 dynamic tensor；
- 把研究创新收窄为 verification-aware Flow formation、A 表示/lifetime/rematerialization 联合规划、lazy
  sparse domain overlay、coarse custom VJP、optimizer state compilation、后期开门的 KFSB batch/hot frontier
  与 RVIR transactional exact-call；
- 将 basic arena/lifetime/rematerialization 从后期 M5 提前到 IBP/CROWN 首个 executable slice，避免先形成
  错误物理 ownership 再补内存；
- 把执行顺序改为 M0真实Flow/Lifetime只读捕获→M1最小overlay/legality→M2 CIBC+physical arena→
  M3 P-anchor coarse CROWN/VJP→M4 10/9+active-β/domain overlay→M5 RVIR same-solver→M6 formal；
- 第一刀改为 `AutoLiRPAFlowTraceV1`：同时覆盖 IBP、P/S CROWN、完整10/9与BaB边界，冻结所有中间
  value 的role/representation/version/escape/bytes/lifetime/clone/materialize/transfer/launch；它是证据
  artifact，不是新production IR；
- 增加 ConstraintFlow、Faith、GPUPoly、TorchLean、ACT、MLIR SCF/Bufferization、PyTorch AOTAutograd等
  prior art，对“首个bound IR/verification compiler/sparse runtime/bound-aware fusion/GPU verifier”明确禁 claim；
- 将候选论文贡献收窄为 production optimized LiRPA 的 stateful differentiable flow、trajectory-constrained
  representation/lifetime/rematerialization、sparse domain overlay 与 transactional prepared runtime；
- 保留并重新定位 CIBC、B4-B2、R3、B4-B3、RVIR-v4、GC0 与 artifact/replay 工作；保留 B4-C2、
  R3细粒度launch、R3-3 active-β、MR5/MR6、IR-5/JIT 等 NO-GO 作为新设计硬约束；
- 本轮只修改用户审阅计划与变更记录，不修改 README、compiler/runtime/tests，不处理 GC0-1 异步外审，
  不形成新性能 claim 或执行授权。

## v3 交叉复核后的收紧

- 纠正 Relax/VM 循环能力：vendored TVM `6248b5db` 的 Relax 无一等structured `for/while`前端，VM
  `Goto`不等于可分析的Relax loop；固定签名10/9先显式展开，一般optimizer保留host/bounded control；
- 禁止把TVM保留的`Composite=boundflow.*`当语义标签；function/PrimFunc使用namespaced attrs，value级
  metadata放稳定value-id sidecar并在rewrite后重建；普通`call_tir`不会自动带来warm allocation=0，
  `call_tir_inplace`只能由验证过的alias/liveness pass插入；
- 将`MemoryWitness`拆成编译期`CompileMemoryWitness`与运行时`ExecutionMemoryReceipt`，receipt单向绑定
  witness hash，canonical projection排除自身hash；
- 修正pass次序为初始liveness→有界representation/fusion/remat候选→rewrite后重推liveness→lower/schedule→
  workspace→final arena→HB/alias复核，避免在concrete buffer出现前错误pack；
- 完善`PreparedArtifact/InvocationFrame/AutogradToken`：exclusive arena slice、stream/event/version、合法compact
  `save_for_backward`、frame lease、GPU event后释放、reject清理与禁止Torch/TVM双份static storage；
- M2拆成instrumented correctness module和final-only-escape formal module，禁止用debug module性能代表正式
  candidate；
- M0 admission扩为三个完整external commit、model/property/config/18 controls/environment/blob manifest；冻结
  24-call topology、initial/alpha/beta phase、KFSB child调用、gradient-active P/S、第10次terminal、branch消费前
  lA、standalone与intermediate IBP；
- M0增加storage identity、alias/version、allocation/free、saved-tensor hook、NVTX/CUPTI correlation、弱引用
  无扰动约束与attributed/unattributed coverage；coverage不过门不形成HBM/Amdahl headline；
- 明确production formal只覆盖固定ResNet2B property 0、1 core/6 domains/depth 1、`max_iterations=1`；第一版
  lower-CROWN only，`bound_upper=True`与未支持cut/output/aux effect fail closed；
- 将lazy sparse overlay降为可证伪COW/override候选，补入unstable/refined/reference/aux/cut/clip来源、density/
  lookup/tightness/latency kill gate和dense fallback；
- 分离rematerialization legality与profitability，补充multistream happens-before、memory-pressure admission、
  `h`与`r_required` Amdahl公式；
- 补齐`SavedStateLedger`与runtime持有检查：entry role、logical/unique bytes、version、pin interval、frame
  lease/release、saved dense-A count/bytes及ctx/module/registry动态tensor检查均可从artifact复核；
- 将candidate选择改为legality prune后逐exact-signature实际rewrite/lower/schedule/arena/build/measure，按实测
  wrapper/device/bytes/launch选择并保留rejected ledger；禁止在未知concrete workspace前拍板；
- 将物理归因门禁扩到kernel/device time、bandwidth、occupancy、compute utilization、vectorization/tensor-core
  等机制，避免把bytes/launch不变但locality或计算利用率改善误判为无物理收益；
- 更新相关工作：PyTorch structured HOP能力、StableHLO while、TorchLean的SSA/AD/IBP/CROWN/optimizer credit、
  DiffAI及Faith的double-bound/weight-pairing/cross-layer fusion，进一步收窄候选贡献。

## v2 历史修订

- 接受用户对“自定义 IR 过多、既有工作没有成为主线”的批评，撤销上一稿
  `Primal→Bound→Verification→Plan→Task→Schedule→TIR` production 必经链；
- 基于 TVM 官方架构，将 canonical representation 收束为一条 mixed `IRModule` lineage：Relax负责静态
  IBP/CROWN图与region组合，TIR负责bound op、fused region和custom VJP；
- 明确“Bound-TIR”不是新AST：第一版只使用标准`PrimFunc`/block attrs、Relax bound ops、
  legalization/fusion/schedule pass；出现可复现硬blocker前不fork TVM核心；
- 保留αβ-CROWN/RVIR作为动态BaB queue、branch、termination、timeout、state commit/rollback owner；
- 修正“没有统一入口”的旧表述：仓库已有`TypedCompilerQueryRuntime`窄原型、Relax/VM、CIBC CUDA
  Graph、R3 arena/custom backward等资产，缺的是汇入同一module/pass pipeline；
- 新增完整既有成果迁移表，明确CIBC、B4-B2、R3-D2、RVIR-v4、GC0、backend dispatch、artifact/
  replay如何直接复用或薄适配；
- 显式保留B4-C2、R3早期细粒度launch、MR3/MR5/MR6、IR-5、旧JIT等NO-GO约束；
- 将Plan/Task/Schedule降为compatibility view、decision record和artifact reader，不再由production runtime
  逐层解释；
- 把R0—R7改为M0—M6：归并→CIBC mixed module→legality/pass→R3-D2 CROWN module→RVIR
  same-solver→按归因扩主流程→formal；
- 第一刀改为现有CIBC `PrimFunc`接入现有Relax interval graph，不写新IR、不重写数学/kernel，但
  新接prepared GPU ABI并重验CUDA Graph资格，不先调新kernel。

## 交叉复核后的收紧

- 冻结RVIR为solver live state唯一commit/rollback owner；Relax/TIR只读versioned state并输出
  `StagedMutation`，`call_tir_inplace`仅限module-private arena；
- 将CIBC-specific最小legality/admission并入M1，M2只负责扩成GC0 generic pass，避免首个production
  candidate越过fail-closed；
- 纠正`CIBCIBPCUDAGraphPlanV1`定位：它是Python/Torch/TIR混合direct baseline，不是现成Relax
  executable；只复用TIR、static buffer、winner、capture和计时协议；
- 冻结M3粗粒度custom-autograd ABI：R3-1b1/1b2、D1、D2B是待组装部件/候选，不虚构现成mixed
  module，也不把`call_tir_with_grad`当成可直接执行runtime；
- 按vendored TVM真实能力拆分零核心改动、轻量Relax Op/intrinsic扩展和TIR AST fork三档；收窄
  `FuseTIR`、compiled mode、TIR `While`与attrs/effect的表述；
- 区分编译输入`IRModule` lineage与其生成的`VMExecutable/runtime.Module`，区分Relax外部调用与TIR
  `call_extern`；不再暗示自定义pass能融合未知opaque PackedFunc，也不把提前编译kernel误称为独立Relax
  AOT executor；
- 修正`tvm_executor.py`的CPU NumPy搬运/静默fallback、typed dispatch two-op-only边界，以及B4-B2
  performance与B2-1—B2-4 correctness scope混用；
- 补入R3-D1C、R3-3 active-β、MR7 profile invalid、MR1 CIBC `0/51`和完整JIT break-even等NO-GO。

## 搜索与仓库证据

- 以production αβ-CROWN `e5c7e17` / auto_LiRPA `5a098e8`源码为主，复核IBP DFS、CROWN reverse
  wavefront、intermediate-bound recursion、optimized-bounds循环、DomainList/KFSB与RVIR exact-call边界；
- 核对formal 24-call phase topology、10/9 optimizer、第10次terminal、3组×24 KFSB child-lower、
  lower-CROWN only与RVIR单迭代/单workload证明范围；
- 搜索并对照ConstraintFlow、Faith、GPUPoly、TorchLean、ACT、DiffAI、multi-GPU auto_LiRPA、PyTorch
  AOTAutograd/HOP、MLIR SCF/Bufferization与StableHLO while；
- 以本仓vendored TVM `6248b5db`源码复核Relax loop、BYOC `Composite`、value attrs、`call_tir` DPS与
  `call_tir_inplace`限制，不直接外推官方滚动文档的新能力；
- 核对NRIR-2逻辑arena、R3-1B3 allocator、R3-D2B wrapper及B4-C2/R3/MR5/MR6失败证据，未重跑性能；
- 核对TVM官方`IRModule`、TensorIR、Relax、`call_tir`、fusion、VM和pass infrastructure文档；
- 对照`docs/CIBC_for_DAC.pdf`的高层BC graph→fused tensor expression→target code三阶段设计，
  将其映射到Relax→TIR而不是删除其高层语义；
- 核对仓库现有Relax builders、TVM executor、CIBC TIR/CUDA Graph、B4-B2、R3-D2、RVIR-v4及
  typed compiler runtime；
- 复核文档引用的历史性能与NO-GO数字，但不重跑benchmark，不形成新性能claim。

## 边界

本轮只修订用户审阅稿和修改记录，不改README源码、compiler/runtime/tests，不处理GC0-1异步外审，
不形成执行授权。

## v3当时验证记录（历史快照，不代表v5当前验证）

- `git diff --check` PASS；
- 文档围栏、代码范围、production commits、24-call/72-child拓扑、witness/receipt/saved-state及关键修订
  断言 PASS；
- 三组只读内部交叉复核的IR/相关工作、auto_LiRPA流程与memory/runtime blocker均已关闭；
- `dol lint --soft` PASS；
- `dol validate` 仅因事件历史中既存的`ev009180/ev009388/ev010862/ev014888`四个重复ID失败；本轮新增
  `ev015092/ev015095`唯一且有效，不改写append-only历史；
- 未重跑GPU benchmark；全部历史数字保留原closure scope，本稿`performance-claimed=false`。
