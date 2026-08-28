---
status: active-documentation-only
date: 2026-08-28
type: change-log
topic: boundflow
slug: asplos27-s4-change-log
stage: s04
performance-claimed: false
---

# ASPLOS'27 S4 修改记录

## 2026-08-29：关闭S4-4 formal evidence实施前六项结构缺口

- 新增631行S4-4 implementation-readiness审计；从S3 v2 runner、RVIR whole-core `.pt`、真实single-worker
  source snapshot和Python stdlib edge behavior独立诊断，不生成S4 production代码或formal结果；
- 证明artifact内部hash只能证明self-consistency，无法阻止攻击者同时替换source/model/raw/replayer并全重签；新增由
  DocOps/Git审计请求持有的external trust anchor，绑定manifest/semantic root/source/protocol/input/replayer identity；
- normal worker低扰动snapshot得到101个BoundFlow core Python、4个repo script、559个TVM/TVM-FFI Python和3个
  repo-local native library，canonical inventory hash=`421ce0...c83`；旧12项`CODE_PATHS`不能声称完整source closure；
- 逐call `sys.setprofile` source probe在超过120秒后仍未完成，而normal约15—17秒；正式worker禁止该高扰动observer；
- 对B0/R/C现有tensor raw实测content duplication，并把inline base64改为per-worker tensor index + content-addressed
  raw-binary gzip sidecar；B0/candidate raw-binary规划载体为`225,621/206,180 B`，只作codec诊断，不形成formal size claim；
- 修正process topology为18 positive + 15 isolated fault=`33`；poisoned process不得复用；
- 冻结无环seal DAG、derive/self-check/anchored-check三种replay、四类tamper enforcement和外审前
  `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT`状态；71类tamper数量不变但不再虚称全部由artifact内部语义拒绝；
- whole-core已知numeric raw规划区间为`37.8557—47.5839 MiB`，只用于资源预案，明确排除metadata/new field和实际
  compression结果；同步主预注册、S4-4蓝图、README主计划与目录导引。

### 验证

- source inventory/codec/stdlib edge probe：完成；高扰动call trace人工中止并作为diagnostic failure披露；
- 文档合同、定向回归、DocOps validation/lint在提交前执行。

## 2026-08-29：刷新S4设计外审交接至whole-core transaction readiness v8

- design-result commit更新为`751fc39580dae82267398d17c1c2bad551eca8f1`，冻结base仍为
  `ebf45cc72438141d8f0b35dadfd5cf774d7e753f`；
- 必读文档增至24份，加入S4-3 implementation-readiness；top-level任务增至18项，要求独立攻击prepared working-β、
  prefix-only rollback、two-stage host packet、lease last-consumer和post/queue counter；
- memory账同步为S4-2/S4-3=`540,926/608,990 B`，tamper minimum同步为71类；
- evidence inventory加入true B3-C assembly/post/queue/fault probe与37项依赖回归；
- 仍只交付设计审计材料，不新建S4 exchange、不开放production/formal/timing/performance。

### 验证

- handoff base/result commit、24份必读文档、18项top-level任务、18项末尾问题及ledger/tamper字段：提交前验证；
- `git diff --check`、DocOps validation/lint在提交前执行。

## 2026-08-29：关闭S4-3 whole-core事务实施前歧义

- 新增488行S4-3 implementation-readiness审计；亲读pinned provider/BoundFlow源码并在真实B3-C same-solver context中
  观察assembly、device commit、official post、queue add与故障rollback；
- 证明current candidate没有清空`pre_result.interm_bounds`，六key从assembly前一直存活到worker结束；host packet则
  经过含`betas/history/depths`的decision阶段后才最终裁为`thresholds/history/depths`；
- 量化current working-β `deepcopy`的CUDA allocator delta=`1,024 B`、peak delta=`2,048 B`；与temporary upper一起使
  assembly allocated delta=`1,536 B`，因此冻结prepared working-β bridge、persistent upper/depths与hot tensor
  allocation/deepcopy=`0`；
- 修正S4-3 known-new logical subtotal为CUDA=`608,910 B`、CPU=`80 B`、总计=`608,990 B`；β location/sign
  `72 B`作为external retained liveness披露而不重复计入new allocation，该账仍不是peak；
- 故障注入证明current v1在第一条copy失败后执行12条blanket restore：第一条version `+2`，11条untouched target也
  `+1`；V2改为committed-prefix-only best-effort restore，但无论content是否恢复都进入`COMMIT_POISONED`；
- live post/queue观察确认official post=`1`、query total domain add=`2`、candidate post domain add=`1`，三者必须
  独立counter；旧execution schema缺post counter，禁止从null/其他counter推断；
- exclusive latch扩成`UNCLAIMED→PREPARED→COMMITTING→CORE_COMMITTED→POSTPROCESSING→POST_READY→QUEUEING→`
  `COMPLETED`，commit/post/queue fault分别poison；14条transition与ledger canonical hash=
  `833e8a9b...6ccaf5`；
- S4-4 tamper从68类扩为71类，加入post/queue counter混写、blanket rollback伪装prefix rollback和failure-state互换；
  同步主预注册、evaluator ABI、S4-3/3A、S4-4与README；S3外审前production/formal/timing仍关闭。

### 验证

- true B3-C live assembly/post/queue/fault probes：完成；首次wrong-import与scope-mismatch失败作为诊断边界保留；
- ledger/state canonical model、文档引用/旧数字/旧状态扫描、定向回归与DocOps validation/lint在提交前执行。

## 2026-08-29：刷新S4设计外审交接至policy-driver readiness v7

- design-result commit更新为`83d27b0f62db88fcf17b00daf36e865846ccc208`，冻结base仍为
  `ebf45cc72438141d8f0b35dadfd5cf774d7e753f`；
- 必读文档增至23份，加入S4-2 implementation-readiness；must-answer新增live policy/functional Adam、checkpoint/
  patience可达性、shadow/poison、20-worker raw与`540,926/608,942 B`账；
- 建议审计命令加入native optimizer/source-parity回归；
- 仍只交付设计审计材料，不新建exchange、不打开S4 production代码/formal/timing/performance。

### 验证

- handoff commit/range、23份必读文档、17个top-level must-answer、16个末尾问题及修正memory/test字段完整性：
  提交前验证；
- `git diff --check`、DocOps validation/exchange validate/lint在提交前执行。

## 2026-08-29：关闭S4-2 policy driver实施前歧义

- 新增S4-2 implementation-readiness审计，亲读pinned production源码并用live observer冻结10个ordinal的
  keep-best/stop/patience/pruning/scheduler轨迹；fixed physical pruning为inactive，checkpoint ordinal精确为
  `0,6,7,8,9`；
- 纠正旧synthetic fixture：`iteration=10`、patience从0开始且源码条件为`patience>10`时该分支不可达；改用不同
  program hash的sealed test-only policy（evaluation limit至少11），不伪装成fixed ResNet事实；
- live optimizer ABI冻结two-group `batch_dim=2/0`、全部Adam defaults、7个CPU float32 step scalar=`28 B`；
  functional Adam对9次×7 parameter的parameter/m/v/step共63组比较全部bit exact；
- production best checkpoint inventory确认full α/β=`34,008 B`、intermediate=`299,712 B`、best lower/`ret_0`
  各`24 B`；candidate只checkpoint active compressed α/β=`17,016 B`，preserved α由immutable lease恢复，fixed
  intermediate bounds以identity/version/content guards替代clone；
- 为validate-before-commit冻结parameter/m/v/step shadow=`51,076 B`；修正S4-2 known logical subtotal为
  `540,926 B`（CUDA `540,870 B`+CPU `56 B`），S4-3 subtotal为`608,942 B`；这些仍不是peak-memory claim；
- 删除S4-2“mutation失败可完整rollback”的错误承诺：pre-begin错误才clean reject，evaluator begin、shadow transition、
  stable copy-commit或scheduler failure均进入`POISONED_NO_RETRY`，因为content copy-back不能恢复PyTorch `_version`
  或hidden state；
- formal口径从含糊five-fresh收紧为A/B 5对10 worker、B/C 5对10 worker，共20 fresh process；candidate full-IEEE
  numeric payload下限=`1,341,776 B/run`，20 worker下限约`25.59 MiB`；
- 同步S4-2/S4-3蓝图、S4-1D readiness、主预注册、evaluator ABI与README；S3外审前S4 production代码/formal/
  timing/performance继续closed。

### 验证

- ledger独立重算：S4-2=`540,926 B`、S4-3=`608,942 B`，PASS；
- branch reachability重算：10-step patience最大10，`>10`不可达；checkpoint=`[0,6,7,8,9]`，PASS；
- live policy/functional Adam probes、定向回归、DocOps validation/lint在提交前执行。

## 2026-08-29：刷新S4设计外审交接至evaluator transaction v6

- design-result commit更新为`52d7bd875466ae539eca34a552b4b5c7957d2437`，审计范围仍从冻结base
  `ebf45cc72438141d8f0b35dadfd5cf774d7e753f`开始；
- 必读顺序加入S4-1D transaction readiness，must-answer新增read-only admission、post-begin poison、composite lease、
  5+5 full-IEEE raw和修正memory ledger；
- 外审known memory账同步为S4-1D/S4-2/S4-3=`438,726/472,758/540,774 B`并显式解释旧账漏项；
- evidence inventory加入14-case状态机hash、CUDA logical/allocator账、`919,680 B` raw预算与`37 passed`回归；
- 本次只刷新用户可交给外部模型的设计审计材料，不新建exchange、不开放S4代码或性能门禁。

### 验证

- handoff commit/range、22份必读文档、16个must-answer、memory/raw/test字段完整性检查：PASS；
- `git diff --check`、DocOps validation/exchange validate/lint在提交前执行。

## 2026-08-28：S4-1D冻结evaluator事务、composite lease与完整IEEE replay

- 对原S4-1D蓝图做implementation-readiness复核，旧`386,712 B`logical ledger漏掉`49,152 B` residual scratch与
  `2,862 B` compressed metadata，共漏`52,014 B`；修正S4-1D/S4-2/S4-3 subtotal为
  `438,726/472,758/540,774 B`；
- CUDA allocation探针实例化36个logical physical buffers：logical=`438,726 B`、allocated delta=`448,000 B`、
  reserved delta=`2,097,152 B`、allocator minus logical=`9,274 B`；existing source lease `34,008 B`只延长
  lifetime且不计新增allocation；这些均不形成peak-memory claim；
- 冻结read-only request admission与顶层状态机：pre-begin reject不消耗generation；进入`EVALUATING`后counter reset、
  pass A/B/C、finite gate、receipt build/validate任一失败均进入`POISONED_NO_RETRY`，generation永久烧毁；
- 14-case独立状态机枚举全部通过，model hash=`8942bb5970f268f47314265e0a1683947e7d5cddf6d421d3fd80cd778a9627eb`；
- lower、六dα、六dβ及terminal lA统一为composite result lease；overlap evaluate拒绝，terminal child只可transfer一次，
  parent可先close但child close才最终释放arena，S4-1D success不隐式回到`READY`；
- prepared ABI冻结base/additional/total view=`16/32/48`，receipt绑定selector nonfinite/safe-index/operation-order及
  live counters；
- formal从含糊“至少five fresh”收紧为5个ordinal0 nonterminal + 5个ordinal9 terminal fresh，每process只执行一次；
- 全部candidate lower/gradient/lA raw只有`919,680 B`（`0.877075 MiB`），冻结为stdlib可解码完整IEEE payload；
  hash/projection只可作附加摘要，不能替代第三方numeric replay；
- 新增S4-1D evaluator transaction implementation-readiness文档并同步蓝图、主ABI、S4-2/S4-3账、README与主预注册；
  S3外审前S4 production代码/formal/timing/performance继续closed。

### 验证

- logical ledger独立重算：`17,016+17,016+55,296+149,856+147,456+49,152+72+2,862=438,726 B`，PASS；
- full raw budget独立重算：nonterminal=`17,040 B`、terminal=`166,896 B`、5+5=`919,680 B`，canonical hash=
  `1e2aab39a7f7049a09371fef6ec1e0a01dc1e2ec6b25ed7c4060b2cf78e2f0d6`，PASS；
- 状态机/CUDA probe和既有定向回归、DocOps validation/lint在提交前执行。

## 2026-08-28：S4-1B/1C冻结六selector sentinel、七gradient TIR与48-view ABI

- 真实D2B staged pass抽取A18/A20/A24/A26/A29/Ainput，normal数据全部finite；A26/A20直接来自两个existing
  stage1 scratch，证明无需新增persistent dense A；
- 把IEEE nonfinite sentinel从Ainput扩展到五张binary selector：legal 0/1，invalid=-128；consumer只显式接受
  0/1，其余输出canonical qNaN，关闭NaN静默选upper的漏洞；
- 修正selector物理账：S4总55,296 B，相对R31B2新增A26/A29共12,288 B；“不新增center”不等于总bytes不变；
- 隔离编译6个dα+1个dβ symbol，corrected PyTorch reference逐位max diff=0、sign exact，float64 max diff=
  `2.3576e-7`；global workspace/alloc_buffer/dense output/saved A均0；
- 第一次reference因upstream `[D,1]`广播成跨domain `[D,D,W]`而FAIL，已保留并以显式`[D,1,1]`修正后重跑；
- 新emitter用IEEE finite poison和safe-index-then-poison：A/V/bound/α/upstream/index六类tamper均产生NaN；
- 冻结7 launch、53 argument occurrence、emitter view 46/46 exact；与S4-1A base重叠14，新增32，总prepared=48；
- β sign保持int8，metadata总量修正为2,862 B；
- 六V/lA slot单storage、interval不重叠且与四个coefficient/scratch storage分离；reverse same-stream
  read→terminal copy→reuse simulation通过；
- 新增S4-1BC实施就绪合同；S3批准前production代码/formal/timing/performance继续closed。

## 2026-08-28：S4-1B0冻结TIR位级nonfinite ABI、缓存隔离与精确物理账

- 首次CUDA探针发现浮点`x==x && abs(x)!=Inf`在当前TVM/CUDA lowering下不能可靠拒绝NaN：NaN被错误
  编成zero，探针FAIL；正式设计改为float32 IEEE-754 exponent mask `0x7f800000`位级检查；
- 修正后NaN/±Inf全部编码`-128`，select对任何非法selector输出bits=`0x7fc00000`的canonical quiet NaN；
  `+0/-0`均归center，正负min-subnormal保留符号，边界探针PASS；
- max-finite与min-subnormal给出两组确定性位级反例，证明`(lower+upper)*0.5`不能重结合成
  `lower*0.5+upper*0.5`；
- 冻结独立pack/select物理账：2 launch、5 unique tensor/prepare view、6 argument occurrence、pointer exact 5/5、
  warm view 0、center tensor/view 0、workspace 0；
- cache key必须绑定schema、symbol、TIR/module、target、shape/dtype/threads及endpoint/midpoint/nonfinite policy，
  不能复用只按compute capability索引的旧R31B2 cache；
- 真实18,432 Ainput重跑仍为positive/negative/zero/invalid=`8,689/9,137/606/0`，selected hash仍为
  `7e95e075...39b652`，606个zero全部被旧binary误编码；legacy module hash before/after相同；
- 新增S4-1B0 TIR ABI/cache/receipt实施就绪文档；S3批准前production代码、formal、timing与performance仍closed。

## 2026-08-28：S4-1A冻结two-phase prepare、private lease retention与failure cleanup

- 审计发现旧蓝图自相矛盾：prepared runtime必须把S4-0 strong-ref lease保留到S4-3，却又以
  `PROVIDER_SOURCE_RETAINED_AFTER_PREPARE`拒绝任何source Tensor；修正为private lease内恰12条source合法且必要，
  lease外source引用、provider container/callback/closure才拒绝；
- formal owner CUDA probe得到6 α leaf、1 active β leaf、5 empty token，parameter/gradient均4,254元素/17,016 B，
  16 base DLPack view全部pointer exact；candidate 16 storage互异且与12 source storage不相交；
- 一次双param-group Adam后parameter/gradient pointer稳定、LR=`0.0098/0.049`，source hash/version不变；
- source lease retention probe得到12 tensors/8,502 elements/34,008 logical B，新增allocated bytes=0；外部owner删除后
  lease合法延长lifetime，close后allocated回到baseline；
- 冻结Phase A零allocation validation→Phase B local staging→Phase C single-transfer adoption；
- parameter/buffer/view三阶段异常注入全部`FAILED_CLOSED`、candidate refs=0、allocated delta=0、source不变、stream/device
  恢复，且retry/fallback/empty-cache=0；
- S4-1A negative最低冻结为36类；S3/S4-0未批准前implementation、TIR、timing继续closed。

## 2026-08-28：S4-0 V3冻结pinned PyTorch/CUDA identity与serialization ABI

- 在`torch 2.12.1+cu132/cuda:0`验证view共享storage `_cdata`与storage起始pointer，但Tensor `data_ptr()`受offset影响；
  empty tensor pointer均为0而storage `_cdata`各异；`detach()`新建object但共享storage/version；
- 弱引用在外部mapping删除后失效，strong-ref lease是防止object/address reuse并维持事务owner的必要条件；
- 实测普通in-place增加`_version`，但`.data.add_()`与DLPack alias写入改变content而原Tensor `_version`不变；因此
  correctness保留content hash，未来S4-P单列同步成本，禁止未证明地降级为version-only；
- 冻结storage token=`device+untyped_storage._cdata+storage.data_ptr+nbytes`，raw字段只在进程内lease使用；
- lease/prepared wrapper固定为非dataclass `__slots__` class，二者同时拒绝copy/deepcopy/pickle；receipt继续frozen dataclass；
- 冻结guard顺序为state→admission→exact dict/coverage→object→storage→physical signature→stride/offset→version→content→alias；
- negative最低由38扩为44类，新增version bypass、same-object storage rebind、detach和custom Mapping门禁；S4 code仍closed。

## 2026-08-28：S4-0 V3以ephemeral strong-ref lease关闭跨阶段object replacement漏洞

- 审计V2“瞬时live mapping→tensor-free receipt→S4-1A重新取得mapping”后发现稳定投影无法证明跨时间对象身份：
  全量same-content clone替换可保持canonical projection/hash完全相同，但Python object、nonempty storage及empty object
  identity全部不同；反例canonical hash=`75d3252e...3c9f`；
- 冻结双层owner：`S4MutableStateAdmissionV1`继续tensor-free/canonical/replayable，新增同模块内不可序列化
  `S4LiveMutableLeaseV1`及prepared wrapper，强引用原始12条Tensor并保存raw object/storage/version私有token；这不是IR；
- lease生命周期改为S4-0 create→S4-1A single transfer→S4-2 candidate-only mutation→S4-3 current-provider rebind/
  precommit revalidate→commit/abort/poisoned→close；same-content clone、same-storage view、empty clone和provider rebind均拒绝；
- 复用existing `live_targets_from_pre_result_v4()`枚举current provider targets，不再新增provider adapter；
- 独立构造active β width `1→2`、history仍为1且全量重签的snapshot，现有validator接受，证明S4-0 width/history exact
  门禁必要；
- S4-0 negative最低由30扩为38类，并加入lease错配、重复transfer、close后使用和serialization/artifact泄漏拒绝；
- S3仍`ready_for_audit`，S4-0 production code、GPU执行、timing与performance全部保持closed。

## 2026-08-28：S4-1B0源码映射与CUDA探针冻结最小实施边界

- 亲读R31/S2/R31B2源码确认static plan只有input lower/upper，没有独立center tensor；pinned provider同样以
  `(x_U+x_L)/2`派生center；
- 决定zero分支在新S4 select TIR内按冻结`add→*0.5`顺序派生center，不新增plan tensor、DLPack view、pointer或
  warm allocation；
- 旧`R31B2_PACK_AINPUT_SYMBOL`、S2 selected-value v1与其source/module hash保持不变；S4必须新增独立schema和
  symbol，防止历史artifact出现同名不同义；
- 内存中CUDA/TIR signed-zero探针通过：`[-1,0,0,1,-1,1,-1,1]`，`+0.0/-0.0`均归zero，subnormal保留符号，
  workspace=0；
- formal真实Ainput探针通过：positive/negative/zero=`8,689/9,137/606`，old binary误编码zero=`606`，新TIR
  selected output与independent PyTorch逐位相等，extra center tensor=0，selector=`18,432 bytes`；
- nonfinite CUDA探针通过：NaN/±Inf pack为reserved `-128`，select传播NaN，合法`-0/+1/-1`仍得到
  midpoint/lower/upper；失败由S4-1D final-finite gate拒绝，正常路径不加status buffer；
- 新增逐文件patch blueprint、16项negative矩阵与nonfinite fail-closed边界；S3外审前仍不写production代码。

### 验证

- formal selected hash=`7e95e075...39b652`；derived-center hash=`d6164a06...f5b003`；
- diagnostic formal TIR module hash=`eb3e7ec6...250fb5`；
- 文档一致性、既有S2/R31定向回归、DocOps lint在提交前执行。

## 2026-08-28：逐层tap修正site19根因，selected-primal以三元endpoint恢复

- 对上一轮site19反例继续做只读逐层tap：exact V18重建pre19为0误差，差异最终定位到input affine；
- provider lower concretization为`A*center-abs(A)*radius+bias`，PyTorch在`A==0`处`abs`导数为0，因此精确
  endpoint必须是`A>0→lower / A<0→upper / A==0→center`，而不是旧`A>=0→lower`；
- formal Ainput inventory独立统计为positive=`8,689`、negative=`9,137`、exact-zero=`606`；
- 用三元规则重算六site：dα最大误差依次为`7.334e-9/4.238e-8/4.063e-8/4.470e-8/8.196e-8/
  1.639e-7`，sign mismatch全0；active dβ最大误差`1.192e-7`、sign mismatch=0；
- 新增S4-1B0权威合同；保留coefficient-program VJP为规范oracle，恢复selected-primal为优化lowering；无需推倒
  S2/R31B2、两块coefficient arena、residual scratch、V/lA arena或compressed emitter；
- Ainput int8 buffer升级为三元selector，其余五张ReLU bitmap保持二元，总存储仍55,296 bytes；新增derived-center、
  zero inventory和binary→ternary tamper门禁；formal tamper总数仍为68；
- 上一节“DAG/fanout根因”显式标为被本节取代，不静默改写历史；S3外审前S4 production代码/timing继续关闭。

### 验证

- six-site ternary projection：六dα + active dβ全PASS，overall dα max=`1.63912773132e-07`；
- Ainput inventory：`8689 + 9137 + 606 = 18432`；
- 文档一致性、目标测试、DocOps validation在提交前执行。

## 2026-08-28（历史v1，已被上节收窄）：六site预检阻止site19错误进入production

- S3 exchange仍为`ready_for_audit/r001`，本轮没有写S4 production代码或开放timing；
- 用冻结ResNet2B production pre-state、同一CUDA/objective/α/β/split做六site只读数学探针：full PyTorch CROWN
  dense α autograd投影production compressed ownership，对比原S4-1B普通selected-primal候选；
- site17/23/25/28/31 max diff分别为`7.334e-9/4.063e-8/4.470e-8/8.196e-8/1.639e-7`且sign exact；
  site19失败：max diff=`0.0011564247542992234`、9个sign mismatch；
- 独立核对generic/compiled input A max diff=`5.029141902923584e-08`且sign exact；当时据此排除endpoint错误，
  但后续证明数值相同并不能排除`A==0`次梯度错误；
- 亲读provider ReLU backward确认terminal `lA`是transform前incoming A；formal handoff必须恢复
  `[D,S,*feature]` view，不能只以当前`S=1`的元素数证明ABI；
- 当时提出完整coefficient-action VJP replay；当前仅保留其规范oracle角色，物理实现改用已闭合的三元endpoint
  selected-primal lowering；
- 两块coefficient arena、D1C residual scratch、55,296-byte sign、149,856-byte V/lA shared arena、compressed
  ABI与terminal handoff继续复用；formal tamper由64扩为68类；
- 新增独立纠正文档并同步主计划、S4 prereg、1B/1C/1D、evaluator、formal与外审交接；不升级correctness/
  performance claim。

### 验证

- 六site compressed投影：5 PASS / site19 FAIL已显式冻结，未把失败包装为validated；
- β31 owned 6：max diff=`1.1920928955078125e-07`、sign mismatch=`0`；
- terminal lA inventory shape现场为17 `[6,1,8,16,16]`、19/23/25/28 `[6,1,16,8,8]`、31 `[6,1,100]`；
- 文档一致性、目标测试与DocOps validation在提交前执行。

## 2026-08-28：S4-0开工前源码审计纠正offline snapshot与live binding边界

- S3 exchange仍为`ready_for_audit/r001`，因此未写S4 production代码；本轮只完成不越级的implementation preflight；
- 亲读`OwnedProductionTensorV4.own`与`ProductionStateBuilderV4`，确认snapshot会CPU clone，alias group按source
  Tensor object `id`而非storage生成，且不保存live `_version`、stride、offset或storage identity；
- 只读反例用同一base的两个distinct view证明：live storage共享，但snapshot生成两个alias group并克隆成两个独立
  storage。由此原`snapshot+topology+plan`三输入函数不能关闭live mutation ownership；
- formal snapshot独立重算仍为12 mutable path（六α+六β value）、source device metadata=`cuda:0`、snapshot hash=
  `2a775b...a256`；这些是capture语义证据，不是当前live storage/version证据；
- 现场重建确认R31 `plan.source_state_hash=cfcebf...f8df`绑定dense mapping，不等于snapshot hash；plan validator本身还
  冻结六layout/domain6/spec1/P-anchor，因此只作为当前formal specialization，不冒充generic plan；
- S4-0 V2入口增加瞬时`Mapping[path, live Tensor]`，只观察object/storage/version/stride/offset/device/content，raw
  pointer不进canonical receipt；snapshot alias与live storage alias分列，不新增IR层；
- topology hash改为plan-order canonical；新增plan/snapshot projection hash、β width/history exact门禁、S4-1A pack前
  lease复核；negative最低由15扩为30类，S4-1A detail reason由18扩为20类，S4-4 formal tamper由56扩为64类；
- 新增S4-0 preflight correction并同步S4-0、主预注册、S4-1A、evaluator ABI、README和设计外审交接；S4代码/
  GPU/timing/performance继续closed。

### 验证

- live-view/snapshot反例：`live_objects_distinct=true`、`live_storage_shared=true`、snapshot alias=`2 unique`、snapshot
  storage=`2 unique`、version/storage字段均不存在；
- formal inventory：mutable=`12`、roles=`6 alpha + 6 beta_value`、source device metadata=`cuda:0`、α leading axes exact；
- fixture identity：snapshot=`2a775b...a256`、mapping=`cfcebf...f8df`、plan=`39d617...910f`，source语义分层成立；
- 文档一致性、目标测试、DocOps validation在提交前执行；无性能claim。

## 2026-08-28：live B0/R phase probe将scratch disposal升级为finalization v2

- 在上一轮36项terminal inventory基础上继续做只读live probe，分别观察original provider B0与现有
  provider-independent RVIR R的`core-entry/terminal-transfer/post-KFSB/core-return/solver-return`对象、storage和指针；
- B0 terminal pre-extract的36项logical=`805,680 B`，unique storage=`756,528 B`；`/37`与`/38`、`/45`与`/46`
  各共享一组lA storage。六α、12 intermediate和六export lA的terminal返回值都是新Python对象，但与net attribute
  共享storage/data pointer，所以attribute替换为sentinel不等于立即释放全部logical bytes；
- B0随后三个provider KFSB child CROWN重新写入batch-24 scratch，solver return仍保留α/intermediate/lA/β unique
  storage=`2,829,600 B`；这是真实provider KFSB residue，不是terminal authoritative state；
- 当前R native KFSB不读写provider net，在core-entry已有batch-12 stale scratch时，36项对象/storage一直原样保留到
  solver return，unique storage=`1,414,752 B`，因此“R已经镜像B0 terminal disposal”不成立；
- 设计升级为`ProviderNetScratchFinalizationPlanV2`：B0只观察`PROVIDER_KFSB_RESIDUE`；R/C在native KFSB后以
  query-scoped exclusive owner为前提，把live枚举36项归一化为sentinel，并要求provider-net β inventory=`0`。B0与R/C
  final scratch差异以`NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`显式准入，不重建无消费者的batch-24 residue；
- formal raw增加phase ordinal、logical/unique bytes、alias group及object/storage/data-pointer lineage；empty tensor
  `data_ptr=0`不算真实alias。fully re-signed tamper最低由48扩为56类；不形成allocated/reserved下降或性能claim；
- 同步S4-3A、S4-3、主预注册、evaluator ABI、S4-4、README和设计外审材料；S3外审仍待返回，S4代码/timing关闭。

### 验证

- live B0：terminal scratch unique=`756,528 B`，连β=`756,624 B`；solver-return batch-24 residue连β=
  `2,829,600 B`；live R：core-entry到solver-return batch-12 stale unique=`1,414,752 B`；
- 36项finalization path、两组lA storage alias、六α/12 intermediate/六export-lA terminal view alias和β owner边界已逐项盘点；
- 本节取代下节“candidate镜像terminal disposal”的设计结论；下节数值保留为terminal phase历史事实。

## 2026-08-28：live reference probe纠正scratch disposal 24→36

- 在pinned ResNet2B/CUDA/reference worker上外包只读provider extraction observer，现场执行一个真实
  `update_bounds_core`，不修改BoundFlow或provider production源码；
- 实测terminal part-scope α为6个tensor/33,984 B，六层intermediate lower/upper为12个tensor/299,712 B；
- 发现初稿把“`BatchedlA`导出的六条split-layer lA”误当成“`gc_lA_from_net`清理的全部lA”。现场实际为
  18个nonempty node lA/471,984 B，全部变为`EmptiedTensor`；第二次GC为18个sentinel/0 tensor；
- sparse β为六个layer container、每个一项，extract前后不变，验证“不纳入disposal但需披露retention”的设计；
- 当前formal fixture disposal静态最低由`6+12+6=24`纠正为`6+12+18=36`，同时保持terminal/export lA=`6`、
  production mutable tensor path=`12`和tamper minimum=`48`不变；generic schema仍必须live枚举，禁止硬编码36；
- 三类被清理tensor logical bytes合计`805,680 B`；该值不是unique-storage、peak allocated/reserved或性能claim，
  临时diagnostic raw已自动清理，S4-4 formal仍需冻结storage/alias/raw；
- 同步S4-3A、S4-3、主预注册、evaluator ABI、S4-4和README；S4代码与timing继续closed。

### 验证

- live reference worker：`core_count=1`，scratch observer event=`6`，solver status=`verified`；
- α `6→0 tensor`、intermediate `12→0 tensor`、lA GC `18→0 tensor`、β container `6×1→6×1`；
- 36项算术、18条lA path、805,680 B逻辑字节、tamper `1..48`及S4 closed flags独立复核：PASS；
- whole-core/live-return/KFSB/device-commit/pre-state/production-state/terminal targeted：`45 passed in 8.43s`；
- 下节原24项记录保留为历史错误并由本节明确取代。

## 2026-08-28：关闭S4-3A provider net scratch consumer/lifetime源码审计

- 亲读pinned provider pre/core/post/domain-storage与auto_LiRPA optimizer源码，确认fixed candidate KFSB、official post、
  queue storage和candidate next-pre不读取net dynamic scratch作数值输入；all-node LP、cuts/clip/BFS/multitree及
  provider reentry仍可能重新读取net，因此继续fail closed；
- 【历史，已由上节live probe纠正】区分production numeric ownership与scratch lifetime：12条α/β tensor commit保持不变，新增
  `ProviderNetScratchDisposalPlanV1`镜像reference的六α、12个intermediate lower/upper和六lA move/gc；formal
  fixture静态最低24个attribute，但generic schema必须live枚举，禁止硬编码24；
- 确认reference不清理sparse β，candidate必须披露stale β retention且证明consumer count=`0`，不能伪写成memory收益；
- 发现`last_update_preserve_mask`只在pruner分支更新、不会每call无条件reset；因此S4-v1冻结query-scoped exclusive
  core-owner latch，candidate首次commit后禁止同query provider reentry、fallback、第二次core call和solver复用；
- attribute reference swap可恢复identity，但任何tensor copy后的失败仍因PyTorch `_version`漂移进入
  `POISONED_NO_RETRY`；post failure仍是`COMMITTED_POST_FAILED_POISONED`；
- formal raw增加scratch inventory/disposal/sentinel/preserve-mask/latch投影，fully re-signed tamper最低由40类扩为48类；
- 新增独立S4-3A诊断文档并同步主预注册、S4-3/S4-4、evaluator ABI、设计外审交接与README；仍无S4代码、
  runtime evidence、performance或same-solver claim。

### 验证

- pinned provider/auto_LiRPA source AST与文本事实、24项静态inventory算术、1—48 tamper编号及跨文档链接独立
  检查：PASS；首次编号脚本把攻击执行步骤1—8误纳入inventory，收窄到A—F分区后确认`1..48`连续完备；
- live-return/native-KFSB/whole-core/device-commit/pre-state/production-state/terminal-handoff targeted：
  `45 passed in 8.48s`；
- `git diff --check`、DocOps change/validation/exchange/lint在提交前执行；
- 保持S3 exchange为`ready_for_audit`，DocOps `next=external-audit-asplos27-s3-optimizer-runtime`不变。

## 2026-08-28：生成S4-0—S4-4设计外审交接

- 以`ebf45cc..1d378eb`为精确设计范围，新增独立external design audit handoff；
- 冻结AC1—AC8，覆盖IR边界、admission、all-state VJP、sealed policy、whole-core transaction、core/post、
  stdlib artifact和tamper/claim；
- 要求外审自建至少3个全重签攻击，并重点核对`_version`、net scratch、post-after-commit与executed-source closure；
- 明确本轮是设计审计，不得因蓝图完整而推断S4实现、GPU correctness或性能已经存在；
- 不创建新的DocOps exchange，不改变仍在等待结果的S3 exchange；该文件供用户下一轮手工交给外部模型。

### 验证

- handoff引用的base/result commit、11份S4设计文档、AC1—AC8及10个must-answer问题完整性在提交前核对；
- `git diff --check`、DocOps change/validation/lint在提交前执行。

## 2026-08-28：冻结S4-4 formal artifact/stdlib replay/tamper关闭蓝图

- 审计S3 v2 JSONL与RVIR five-fresh/whole-core/KFSB/live-return artifacts，确认旧`.pt`不能作为外部stdlib审计的
  唯一raw，历史manifest链也不能证明新S4组合路径真正执行；
- 冻结B0 original provider、R provider-independent RVIR native、C S4 compiled三方六全排列，共18个fresh subprocess；
- 冻结标准库可解码`base64-ieee-bytes` tensor record、deterministic gzip、executed-source inventory与无本机路径泄漏；
- 冻结pre-state、10/9/10 trajectory、terminal six-lA、KFSB 3/3/72、core/transaction/post/solver全层raw；
- 新增stdlib-only semantic replayer，禁止import BoundFlow/PyTorch/TVM/Numpy/αβ-CROWN或复用production validator；
- 新识别official post发生于commit后的failure边界，状态冻结为`COMMITTED_POST_FAILED_POISONED`，禁止伪装clean
  rollback、重试或继续queue；
- 冻结minimum 40类fully re-signed tamper、12项fault injection和外审独立重算要求；
- 新增S4-4实施蓝图并回链S4-3、主预注册、evaluator ABI与README；仍无S4代码/formal运行/性能claim。

### 验证

- 现有artifact结构、raw格式、`.pt`依赖、S3 v2 20 MB JSONL与source/manifest绑定方式已独立盘点；
- stdlib盘点确认S3 v2 raw=`18 rows / 20,747,422 bytes`，旧RVIR five-fresh=`10 .pt / 16,975,355 bytes`；
- S3 v2与RVIR five-fresh/whole-core/live-return/KFSB/atomic artifact targeted：`19 passed in 19.09s`；
- 文档自检首次把tamper执行步骤1—8误纳入攻击编号，修正解析范围后确认攻击inventory=`1..40`、
  order/worker=`6/18`、蓝图=`703 lines`，PASS；该失败属于检查脚本选择范围，不是repo测试回归；
- 交叉链接与`git diff --check`：PASS；DocOps change/validation、exchange validate与lint在提交前执行。

## 2026-08-28：冻结S4-3 whole-core exact-call事务与失败语义

- 再次确认S3 exchange仍为`ready_for_audit/r001`、无audit产物，保持S4代码/correctness/timing关闭；
- 亲读live-return、terminal export、KFSB、device atomic commit与pinned provider core/post路径，恢复完整solver事务；
- 纠正“provider调用为0”的粗口径：bound callback=`0`，但固定路径return constructor=`12`、official post=`1`；
- 补入provider真实副作用：host `d`必须prune到history/depths/thresholds，`pre_result.interm_bounds`必须clear；
- 确认现有device commit v1的mid-commit rollback只能恢复内容，不能恢复PyTorch tensor `_version`；冻结
  precommit `ABORTED_CLEAN`与mid-commit `POISONED_NO_RETRY`两类失败，后者禁止fallback/retry/继续queue；
- 冻结`PreparedWholeCoreTransactionV2`、terminal one-shot handoff、existing KFSB 3×batch-24、provider-compatible
  core return、official post与provider net scratch consumer audit；
- 独立汇总candidate+rollback=`68,016 bytes`；本节当时沿用旧S4-1D账得到`488,760 B`，现已由本文首节纠正为
  `540,774 B`，并继续明确不形成peak-memory claim；
- 冻结five-fresh R/C、minimum 26类fully re-signed tamper和18类negative/fault-injection门禁；仍无S4实现、
  GPU correctness、same-solver或性能claim。

### 验证

- stdlib/AST/source重算：candidate=`34,008 bytes`、candidate+rollback=`68,016 bytes`本身仍成立；旧known subtotal
  `488,760 B`因上游ledger漏项已被首节修正为`540,774 B`；provider return constructor=`4 direct + 8 helper = 12`，PASS；
- CUDA最小探针确认同一tensor先commit再content rollback后值恢复，但`_version=0→1→2`，不能恢复为0；
- pinned provider source确认`interm_bounds.clear()`、`pre_result.interm_bounds.clear()`与post
  `torch.max(ret_l[final_name], lb_last.cpu())`存在；
- live return、atomic copy-out、terminal export、native KFSB、whole-core truth、device commit/live return与B4-A
  terminal handoff targeted：`36 passed in 9.90s`；
- S4-3蓝图和五份回链文档`git diff --check`：PASS；
- DocOps change/validation、exchange validate与lint在提交前执行。

## 2026-08-28：冻结S4-2 sealed production policy driver精确实施蓝图

- 再次确认S3 exchange仍为`ready_for_audit/r001`、无audit产物，因此不越过S4代码门禁；
- 亲读pinned production `auto_LiRPA@5a098e8/optimized_bounds.py`，确认现有RVIR/S3简化10/9循环未完整覆盖
  keep-best、stop、patience、iteration pruning、max-time、restore-best与terminal scheduler call；
- 从production raw独立重算六domain winner均为ordinal 9，terminal post mutable state与step 9为`12/12`路径一致；
- 纠正scheduler计数：production为10 evaluation、9 parameter mutation、10 scheduler call；第10次post LR不再被消费；
- 冻结representation-neutral sealed driver、native dense/compiled compressed两个closed evaluator、policy runtime state、
  functional Adam prepared moments、per-step raw receipt与minimum 23类negative/tamper门禁；
- 本节当时沿用旧S4-1D账汇总加m/v=`420,744 B`；首节补齐scratch/metadata后修正为`472,758 B`。step scalar/
  best checkpoint/pruner/workspace仍需分项测量，不形成memory claim；
- 新增S4-2实施蓝图并回链S4主预注册、evaluator ABI、S4-1D与README；仍无S4实现/GPU执行/性能claim。

### 验证

- production raw重算：`best_iteration_by_domain=[9,9,9,9,9,9]`、terminal-is-best全true、post mutable与step9=
  `12/12`，PASS；
- pinned external source核对：αβ-CROWN=`e5c7e17`、auto_LiRPA=`5a098e8`，terminal iteration后仍有
  `scheduler.step()`，PASS；
- targeted optimizer/policy/terminal tests、文档一致性、DocOps/exchange/lint结果在本批提交前补录。

## 2026-08-28：完成same-solver接入前的production coverage普查

- 正式创建并交付DocOps exchange `asplos27-s3-optimizer-runtime-20260828`，状态=
  `ready_for_audit/r001`，exchange validate PASS；
- 亲读RVIR live exact-call、native optimizer、pre-state、terminal export、KFSB与atomic commit路径；
- 从冻结production optimizer raw独立统计每step为6条α、8,496个α元素，以及1条`[6,1]` active β；
- 核对S3只动态拥有P-anchor `alpha/%2Finput-24/%2F49:[2,1,6,86]`的1,032元素，占α state元素
  `12.1468926554%`，且其P β为`[6,0]`；
- 确认S3只返回terminal P α/lower，不足以直接组装whole-core lA/intermediates/KFSB/12-path commit；
- 冻结S4为all-mutable-state compiled evaluation → existing host production policy → terminal handoff →
  existing KFSB/commit的组合路线；禁止P-only whole-core包装；
- 新增S4-0—S4-4 correctness/replay门禁与21类fail-closed reason；timing另设S4-P，当前关闭；
- 本轮只有文档/流程变更，无S4代码、GPU运行或性能claim。

### 验证

- production raw inventory stdlib解析：10/10 steps、六α、8,496 α元素、P=1,032元素、active β path=
  `beta/%2Finput-28/0/value`，PASS；
- RVIR native optimizer/mutation/live return、FSG4 same-solver合同与S3 pipeline targeted：`44 passed`；
- S4 source mapping与`git diff --check`：PASS；
- DocOps change/validation与lint：最终落账后执行。

## 2026-08-28：完成六路α/active β compiled VJP可行性审计

- 从冻结capture逐site恢复六个logical shape、compressed width、β owner及active β location；
- 对照R3 bounded-arena trace，确认site31/25分别有exact sparse Linear/Conv单siteTIR，其余site不能直接串
  现有B4-B2 wrapper；
- 亲读R31B1、R31B2、D1C、D2B与S2/S3代码，确认整图forward早已消费六α与active β，当前缺口仅为P-only
  gradient output ABI；
- 冻结all-state VJP物理方案：一次完整sign pass、一次六site effective-value pass、一次coefficient重算并在每个
  ReLU即时压缩gradient；
- 复用D1C/D2B residual stage scratch导出site25/site19 incoming coefficient，保持跨层saved dense A=`0`；
- 新增独立可行性文档并把S4-1细分为1A ABI、1B effective values、1C emitters、1D evaluator closure；
- 本轮仍无S4代码/GPU执行/性能claim，S3外审门禁不变。

## 2026-08-28：冻结compressed evaluator ABI与terminal handoff

- 纠正α计数口径：六source为8,496 stored元素，lower-only optimizer-active为4,248，preserved direction为4,248；
  P为1,032 stored/516 active，coverage比例不变；
- 确认RVIR native optimizer使用37,464-element dense α和同shape dense β，仅作为provider-independent oracle更合适；
- 冻结S4 candidate直接优化production compressed lower-α/sparse β，terminal才一次性展开dense state供existing
  KFSB/commit；
- 把optimizer接入从“给existing函数塞callback”修正为“抽出sealed policy driver，只允许native oracle与compiled
  candidate两个exact evaluator”；
- 独立复核B4-A terminal handoff六lA共37,464 float32/149,856 bytes，handoff count=1、rerun=0；
- 冻结effective-value/terminal-lA phase-tagged slot复用方案与10类新增fail-closed reason；
- 明确existing KFSB仍执行3次batch-24 child CROWN、共72 child lower，S4-P必须单列，不能隐藏为host overhead；
- 新增S4 evaluator ABI/terminal handoff实施蓝图；仍无代码、GPU执行或性能claim。

### 验证

- stdlib-only重算：六α`stored/active/preserved=8496/4248/4248`，P=`1032/516`，B4-A terminal lA=
  `37,464 float32 / 149,856 bytes`，handoff/rerun=`1/0`，KFSB=`3 candidates × batch 24 = 72 child lower`，PASS；
- RVIR pre-state、atomic copy-out、native KFSB、B4-A terminal handoff与S3 optimizer targeted：最终
  `24 passed in 11.17s`；
- 验证过程先后暴露两项shell装载问题：`source env.sh; pytest`因当前shell未含conda env bin而找不到命令；
  只用固定Conda解释器则因未加载`env.sh`而有3项`import tvm`失败。最终按项目约定执行
  `source env.sh`后再用`/home/lee/miniconda3/envs/boundflow/bin/python -m pytest`全部通过；均为环境装载问题，
  不是代码回归；
- 主计划、S4预注册、可行性、ABI蓝图与三份权威状态文档的口径统一；`git diff --check`：PASS；
- 本批DocOps change/validation、exchange validate与lint在提交前执行。

## 2026-08-28：完成S4-0 mutable-state admission精确实施蓝图

- 再次核对S3 exchange仍为`ready_for_audit/r001`且无audit产物，保持S4代码/timing门禁关闭；
- 亲读`ProductionStateSnapshotV4`、`ProductionReluTopologyV4`、`R31FullRegionPlanV1`、atomic copy-out与
  GC0 rejection vocabulary，确认S4-0无需新增IR或dense candidate state；
- 冻结唯一新增对象为tensor-free `S4MutableStateAdmissionV1`及六个`S4MutableSlotV1` metadata receipt；
- 冻结snapshot→topology→plan的15步确定性admission算法、15个稳定detail reason到GC0 reason的映射；
- 预注册10项positive/structural与minimum 15类negative/tamper测试；
- 明确S4-0不得分配GPU buffer、调用native dense initializer/TVM/provider、创建optimizer或记录timing；
- 新增S4-0实施蓝图并回链主预注册；本轮仍无S4代码/GPU执行/性能claim。

### 验证

- stdlib/PyTorch独立重算formal inventory：mutable paths=`12`、α/β slots=`6/6`、
  stored/active/preserved=`8496/4248/4248`、active β=`1 slot/6 elements`、mutable alias全唯一，PASS；
- production state、pre-state initializer、GC0 schema与R3 structured owner targeted：`33 passed in 4.37s`；
- S4-0蓝图保持`execution-authority=false/code-change-open=false/performance-claimed=false`；
- `git diff --check`与文档关键字段检索：PASS；
- DocOps change/validation与lint在提交前执行。

## 2026-08-28（历史v1，内存/事务/fresh口径已被上节修正）：完成S4-1D all-state evaluator closure实施蓝图

- 收束S4-0 admission、S4-1A buffers、S4-1B effective graph与S4-1C emitters为唯一prepared evaluator owner；
- 冻结prepare/evaluate/rollback/result lease序列与component receipt hash链；
- 独立汇总correctness logical ledger=`386,712 bytes`，明确排除model/fixed inputs/cuDNN workspace与S4-2 moments；
- 冻结one logical evaluation=`2 coefficient pass + 1 effective graph + 6 α emitter + 1 β emitter`，不得隐藏实际kernel/copy；
- 冻结five-fresh A/B/C三方、lower/gradient/lA容差、raw-first artifact/replay与20类closure negative gate；
- 明确S4-1D不接Adam、不计时，通过只开放S4-2 production 10/9 trajectory；
- 新增S4-1D closure蓝图并同步主ABI/预注册；仍无S4实现、GPU correctness或性能claim。

### 验证

- 独立重算logical ledger：parameters/gradients/signs/effective-or-lA/two-coefficient-arenas/scalars=
  `17,016/17,016/55,296/149,856/147,456/72`，合计`386,712 bytes`，PASS；
- S2/S3 pipeline、bounded-arena compiler、RVIR atomic copy-out与B4-A terminal handoff：
  `24 passed in 16.85s`；
- S4-0/1A/1B/1C/1D五份蓝图均存在且非空，1D implementation/timing/performance/S4-2门禁保持closed；
- `git diff --check`与DocOps前关键字段检查：PASS；
- DocOps change/validation与lint在提交前执行。

## 2026-08-28：完成S4-1C通用compressed gradient emitter实施蓝图

- 逐项核对explicit ReLU与D1C staged residual边界，冻结pass C插入顺序为31→28→25→23→19→17；
- 冻结通用`[D,S,F]→[D,W]` lower-α VJP模板，operator种类不进入emitter schema；
- 冻结site31 sparse β公式、production location/sign与B4-B2/full-autograd/float64三方oracle；
- 明确五empty β为token且launch=0，全部physical gradient输出复用S4-1A 17,016 bytes；
- 冻结coefficient/effective/version、same-stream arena lifetime、六emitter/一β emitter与22类negative gate；
- 冻结ordinal9在gradient消费后复用effective slot写terminal lA，六slot各一次，禁止第三次coefficient/第11次CROWN；
- 新增S4-1C实施蓝图并同步可行性/主预注册；仍无实现、GPU correctness或性能claim。

### 验证

- 独立重算dα/dβ elements=`4248/6`、persistent gradient=`17,016 bytes`、compressed indices=
  `708 int32/2,832 bytes`、active β normalized location/sign=`24/24 bytes`，PASS；
- B4-B2 sparse Linear、R31B2 P VJP、residual11/6 staged与B4-A terminal lA：`26 passed in 19.96s`；
- source核对B4-B2 β公式确为`-adjoint_relu*sign`，residual stage1/2 caller-owned scratch路径存在；
- empty β旧“零宽view”措辞已统一修正为typed token；`git diff --check`：PASS；
- DocOps change/validation与lint在提交前执行。

## 2026-08-28：完成S4-1B六site effective-value graph实施蓝图

- 亲读R31B2 effective-pre17/23/25 TIR、S2 selected Relax graph与closed-form oracle，逐stage恢复selected primal语义；
- 冻结一个37,464-element/149,856-byte persistent value arena及六slot exact offset/shape；
- 确认现有sign bitmap为43,008 bytes，S4补A26/A29各6,144 bytes后合计55,296；不需要A32 bitmap；
- 冻结扩展S2 safe-VM selected Relax graph为首选实现，logical stage 6、persistent output copy第一版允许6次并如实披露；
- 明确A29在ReLU28 transform前pack，A26必须从D1C/D2B residual11 staged scratch导出，不准额外重跑CROWN；
- 冻结active `[D,W]` α TIR ABI、effective-value receipt、三方oracle与18类negative gate；
- 澄清active β应以B4-B2 `-adjoint_relu*split_sign`与full-composition effective value双oracle验证，不能只按类比实现；
- 新增S4-1B实施蓝图并同步可行性/主预注册；仍无实现、GPU correctness或性能claim。

### 验证

- 独立重算six-value offsets=`[0,12288,18432,24576,30720,36864]`、elements/bytes=
  `37,464/149,856`，six-sign=`55,296 bytes`、existing/new=`43,008/12,288`，PASS；
- S2 selected pipeline、P-alpha closed-form oracle、D1C cumulative与D2B staged backward：
  `13 passed in 18.76s`；
- source检索确认existing effective-pre symbols恰为17/23/25，S2 graph内部已有pre17/19/23/25；
- S4-1B蓝图保持implementation/correctness/timing/performance全部closed；`git diff --check`：PASS；
- DocOps change/validation与lint在提交前执行。

## 2026-08-28：完成S4-1A ordered buffer/lease ABI实施蓝图

- 亲读S2/S3 prepared executor、P-only direct VJP、host Adam与R31B2 DLPack/view cache，确认S4 all-state不应
  复用full α source或nonleaf slice作为optimizer参数；
- 冻结6个独立contiguous lower-α leaf buffer、1个active β leaf buffer与5个empty β metadata token；
- preserved α只由immutable host snapshot/commit receipt拥有，不进入candidate GPU optimizer；
- 冻结persistent dα/dβ/lower/upstream、ordered tuple ABI、result lease与0—9/version 0—9状态机；
- 明确compiled VJP直接向sealed driver提供gradient，不使用PyTorch autograd Function、global registry或saved history；
- 冻结prepare-only DLPack/pointer纪律、18类fail-closed reason及positive/negative测试矩阵；
- 用冻结snapshot做一次不入库GPU owner原型，验证7个leaf参数、两Adam groups、persistent grad赋值与step后pointer稳定；
- 新增S4-1A实施蓝图并同步主S4 ABI/预注册；仍无S4实现、correctness closure或性能claim。

### 验证

- GPU owner原型：α/active β/empty β=`6/1/5`，parameter/gradient logical bytes=
  `17,016/17,016`，Adam m+v=`34,032`，leaf all=true，step后pointer stable=true，PASS；
- S3 optimizer、R31B2 compiled P VJP与R31B1 full-lower targeted：`11 passed in 13.66s`；
- 文档交叉检索确认preserved α不进入candidate GPU optimizer、empty β无物理参数、S4-1A仍closed；
- `git diff --check`：PASS；
- DocOps change/validation与lint在提交前执行。
