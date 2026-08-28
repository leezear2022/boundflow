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
- 独立汇总candidate+rollback=`68,016 bytes`，连同S4-1D+S4-2已知账得到S4-3 known logical subtotal=
  `488,760 bytes`，并明确不形成peak-memory claim；
- 冻结five-fresh R/C、minimum 26类fully re-signed tamper和18类negative/fault-injection门禁；仍无S4实现、
  GPU correctness、same-solver或性能claim。

### 验证

- stdlib/AST/source重算：candidate=`34,008 bytes`、candidate+rollback=`68,016 bytes`、known subtotal=
  `488,760 bytes`；provider return constructor=`4 direct + 8 helper = 12`，PASS；
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
- 汇总S4-1D加m/v的known logical subtotal=`420,744 bytes`，并明确step scalar/best checkpoint/pruner/workspace仍需
  分项测量，不形成memory claim；
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

## 2026-08-28：完成S4-1D all-state evaluator closure实施蓝图

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
