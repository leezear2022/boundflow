---
status: ready-for-external-design-audit-v3-python-abi-frozen
date: 2026-08-28
type: external-audit-handoff
topic: boundflow
slug: asplos27-s4-design-audit
audit-kind: preregistration-and-implementation-blueprint
base-commit: ebf45cc72438141d8f0b35dadfd5cf774d7e753f
design-result-commit: 7f9854332f223a703b9a2315f58bac56cf07a677
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
10. S4-4的stdlib raw/replay和68类fully re-signed tamper是否足以支持第三方独立审计；
11. S4-1B0把site19根因收窄到`Ainput==0→center`、恢复selected-primal lowering是否数学成立；
12. 是否同意在S3外审批准后仍按S4-0→1A→1B0→1B→1C→1D→2→3→4顺序实施；
13. 是否发现必须在第一行S4代码开工前修正的blocker/major。

## 1. 审计范围和Git边界

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- 本轮设计base：`ebf45cc72438141d8f0b35dadfd5cf774d7e753f`；
- S4-0 live admission/lease、S4-3A scratch finalization与S4-1B0 ternary endpoint纠正全部设计结果：
  `7f9854332f223a703b9a2315f58bac56cf07a677`；
- 审计范围以`ebf45cc72438141d8f0b35dadfd5cf774d7e753f..7f9854332f223a703b9a2315f58bac56cf07a677`
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
7. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_ORDERED_BUFFER_ABI_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
8. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_PREPARE_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`；
9. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_BOX_ENDPOINT_SUBGRADIENT_CLOSURE_2026_08_28.md`；
10. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_ENDPOINT_IMPLEMENTATION_READINESS_2026_08_28.md`；
11. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1BC_DAG_ADJOINT_PREFLIGHT_CORRECTION_2026_08_28.md`（历史v1）；
12. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B_SIX_SITE_EFFECTIVE_VALUE_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
13. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1C_COMPRESSED_GRADIENT_EMITTER_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`；
14. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_ALL_STATE_EVALUATOR_CLOSURE_BLUEPRINT_2026_08_28.md`；
15. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_SEALED_PRODUCTION_POLICY_DRIVER_BLUEPRINT_2026_08_28.md`；
16. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3_WHOLE_CORE_EXACT_CALL_TRANSACTION_BLUEPRINT_2026_08_28.md`；
17. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_3A_PROVIDER_NET_SCRATCH_CONSUMER_AUDIT_2026_08_28.md`；
18. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_4_FORMAL_ARTIFACT_REPLAY_TAMPER_CLOSURE_BLUEPRINT_2026_08_28.md`；
19. `gemini_doc/BOUNDFLOW_ASPLOS27_S4_CHANGE_LOG_2026_08_28.md`。

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
- formal raw中六domain best ordinal均为9，但设计仍保留keep-best/prune/stop/patience/timeout/restore分支。

请亲读pinned `auto_LiRPA@5a098e8/optimized_bounds.py`确认。

### 3.3 terminal/KFSB/return

- terminal six-lA=`37,464 float32 / 149,856 bytes`；
- handoff必须来自ordinal 9，duplicate CROWN=0；
- KFSB=`3 candidates × batch 24 = 72 child lower`；
- live commit path=`12`；
- fixed candidate provider return constructor=`12`；
- C路径provider bound callback=`0`，official postprocess=`1`；
- host `d`必须prune到history/depths/thresholds；
- `pre_result.interm_bounds`必须clear。

### 3.4 known logical memory账

- S4-1D correctness ledger=`386,712 bytes`；
- 加S4-2 Adam m/v known subtotal=`420,744 bytes`；
- full candidate=`34,008 bytes`；
- candidate+rollback=`68,016 bytes`；
- S4-3 known subtotal=`488,760 bytes`。

这些是design-time logical bytes，不是实测peak allocated/reserved。请检查是否有重复计数、漏项或错误归属。

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

- snapshot/topology/plan/live-source四方key、shape、dtype、device、feature index、β location/sign、object/storage/
  stride/offset/version全覆盖；
- auditor复现distinct views共享storage反例，确认snapshot alias group不能替代live storage group；
- auditor复现全量same-content clone替换，确认canonical group/hash可完全相同，因此receipt不能替代strong-ref lease；
- prepared lease必须single-transfer、不可序列化，并从S4-0保持到S4-3 current-provider precommit；
- live input只接受existing helper返回的exact built-in dict；lease/wrapper必须为非dataclass `__slots__` class并同时拒绝
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
- S4-0无GPU执行、dense initializer、TIR或timing。

### AC3：S4-1 all-state evaluator物理可行性

请从现有R31B1/R31B2/D1C/D2B/B4-B2代码验证：

- S4-1A formal buffer是否为6 α leaf+1 active β leaf+5 empty token、parameter/gradient各4,254元素/17,016 B、
  base DLPack 16/16 exact，且16 candidate storage与12 source storage完全不相交；
- private lease保留12 source/8,502 elements/34,008 logical B是否确为incremental allocation 0，同时确实延长lifetime；
- `PROVIDER_SOURCE_RETAINED_AFTER_PREPARE`删除是否正确，provider container/callback与lease外source引用是否才应拒绝；
- two-phase prepare是否在validation前锁定single-attempt、在local staging后single-transfer；parameter/buffer/view故障时是否
  逆序释放、source不变、allocated回entry、stream/device恢复，且retry/fallback/empty-cache均为0；
- forward实际已消费六α+active β；
- 缺口确实是P-only gradient ABI，而非forward根本不支持其他site；
- 旧二元selected-primal在site19失败是否确由606个Ainput zero错误映射到lower导致；
- 三元`positive→lower / negative→upper / zero→center`是否与provider `abs`零点次梯度严格一致；
- coefficient-action VJP作为规范oracle、selected-primal作为优化lowering的双层owner是否合理；
- pass C按31→28→25→23→19→17即时导出六dα和site31 active dβ；
- site25/site19可从existing residual scratch取incoming coefficient；
- cross-layer saved/persistent dense A可保持0；
- two coefficient arenas足够；
- terminal lA与V slot alias的lifetime门禁充分，且复制发生于ReLU transform前并恢复
  `[D,S,*feature]` spec-axis view；若不足是否应默认独立arena。

请特别找出文档中可能把“数学可行”误写为“已有production implementation”的地方。

### AC4：S4-2 sealed policy等价性

PASS要求：

- native dense oracle与compiled compressed evaluator是两个closed实现，不接受任意callback；
- sealed driver完整拥有live policy，而非无条件`for range(10)`；
- functional Adam/ExponentialLR、clamp、keep-best/prune/stop/patience/timeout/restore均可逐step比较；
- terminal compressed→dense→compressed只在ordinal 9执行一次；
- preserved α不进入optimizer；
- `10/9/10`不被错误简化成`10/9/9`；
- S4-2不执行KFSB/commit/post/timing。

### AC5：S4-3 whole-core transaction与失败语义

请重点攻击：

1. existing device commit v1中commit copy和rollback copy是否确实都会增加PyTorch `_version`；
2. `ABORTED_CLEAN`、`POISONED_NO_RETRY`、`COMMITTED_POST_FAILED_POISONED`是否覆盖全部边界；
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

- 6 B0/R/C全排列×3 subprocess=`18 fresh worker`合理；
- 不从partial output resume；
- `.pt`不是唯一formal raw；
- base64 IEEE tensor record、deterministic gzip和stdlib decode足以独立重算；
- stdlib replayer不import BoundFlow/PyTorch/TVM/Numpy/αβ-CROWN，不复用production validator；
- source inventory覆盖实际executed code closure，而非不完整手写列表；
- summary所有字段均能从protocol/source/raw重建；
- artifact无绝对本机路径/credential泄漏；
- failure artifact与positive worker分离但被同一manifest绑定。
- scratch按`core-entry/terminal-pre/terminal-post-transfer/post-KFSB/post-finalization/solver-return`投影，live
  finalization keys/sentinel、logical/unique bytes、alias与object/storage/data-pointer lineage均可从raw重建；
- B0六β container/96 B nonempty residue与R/C provider-net β inventory=`0`按variant核验；scratch count与production
  12-path count严格分离，且不把sentinel替换升级为即时CUDA memory free。

请评估18-worker设计是否过度或不足，以及B0/R/C在独立进程下如何证明同一个deterministic pre-state。

### AC8：tamper、claim与执行顺序

PASS要求：

- 68类攻击编号/分区完备且全部fully re-signed；
- 攻击同步更新payload/file/summary/manifest，拒绝原因来自semantic invariant而非简单digest；
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
- S4-4参考artifact相关targeted：`19 passed`；
- CUDA探针：tensor content restore后`_version=0→1→2`；
- S3 v2 raw=`18 rows / 20,747,422 bytes`；
- old RVIR five-fresh=`10 .pt / 16,975,355 bytes`；
- S4-4 tamper inventory=`1..68`、order/worker=`6/18`；
- Ainput exact class=`positive 8,689 / negative 9,137 / zero 606`；三元endpoint六dα overall max=
  `1.63912773132e-07`、active dβ max=`1.1920928955078125e-07`、sign mismatch均0；
- formal in-memory CUDA/TIR pack/select逐位PASS，old binary误编码606 zero；selected hash=
  `7e95e075...39b652`，derived-center hash=`d6164a06...f5b003`，extra center tensor=0；
- nonfinite sentinel CUDA/TIR PASS：NaN/±Inf→`-128`→NaN，S4-1D final-finite gate前不发布result；
- existing live-return/device-commit targeted：`12 passed in 6.58s`；production code diff=`0`；
- `git diff --check`、DocOps exchange validate/lint：PASS。

这些只证明设计输入和历史基础设施仍存在，不证明S4-0—S4-4实现通过。

## 7. 建议外审操作

```bash
git diff --stat ebf45cc72438141d8f0b35dadfd5cf774d7e753f..7f9854332f223a703b9a2315f58bac56cf07a677
git diff --check ebf45cc72438141d8f0b35dadfd5cf774d7e753f..7f9854332f223a703b9a2315f58bac56cf07a677

source env.sh
/home/lee/miniconda3/envs/boundflow/bin/python -m pytest -q \
  tests/test_rvir_v4_live_return.py \
  tests/test_rvir_v4_native_kfsb.py \
  tests/test_fsg4_b3_device_atomic_commit.py \
  tests/test_fsg4_b4a_terminal_lower_adjoint_handoff.py
```

另外请用自己的短脚本：

- 重算mutable inventory和memory ledger；
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
- 检查tamper编号1—68；
- 用CUDA tensor验证commit+restore后的`_version`；
- 亲读provider core/post确认clear/prune/post顺序；
- 搜索S4文档所有`claimed/open/validated`词，核对没有implementation或performance漂移。

## 8. 外审必须回答的问题

1. 是否存在blocker/major，使S4-0在S3批准后仍不能开工？
2. 三元box endpoint是否完整解释旧site19反例；derived-center且不新增tensor是否为正确最小ABI？
3. six-site V→gradient→terminal-lA alias是否有无法由phase state解决的lifetime冲突？
4. net scratch是否必须成为第13+条production数值path，还是应保持为独立phase-aware lifetime/finalization transaction？
5. post failure后是否有比`COMMITTED_POST_FAILED_POISONED`更严格、可实现的安全语义？
6. B0/R/C 18 fresh是否足够证明reference和candidate，不依赖历史`.pt`？
7. stdlib raw schema是否缺dtype、negative-zero、NaN payload、alias或view metadata？
8. executed-source inventory是否有更可靠的闭包算法？
9. snapshot semantic truth、瞬时live observation和S4-1A prepared owner三段边界是否仍遗漏live alias/version race？
10. 68类tamper还缺哪类可全重签semantic attack？
11. 是否同意当前唯一执行顺序，不开放S4-P timing？

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
