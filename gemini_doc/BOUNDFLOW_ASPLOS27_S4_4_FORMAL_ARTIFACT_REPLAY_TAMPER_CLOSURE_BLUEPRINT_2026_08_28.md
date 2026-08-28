---
status: diagnostic-complete-code-closed
date: 2026-08-28
type: implementation-blueprint
topic: boundflow
slug: asplos27-s4-4-formal-artifact-replay-tamper
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4-4：formal artifact、stdlib replay与fully re-signed tamper关闭蓝图

## 0. 直接结论

S4-4不是“把S4-0—S4-3测试结果复制进manifest”。它必须从同一个clean source重新执行B0/R/C三条whole-core
路径，冻结原始tensor、trajectory、KFSB、transaction、post和solver结果，再由**不import BoundFlow、PyTorch、TVM或
αβ-CROWN的标准库replayer**独立重建全部summary与门禁。

正式协议固定为：

```text
6个全排列triplet × 3个独立fresh subprocess = 18个worker

B0 = pinned original provider whole-core
R  = provider-independent RVIR native whole-core reference
C  = S4 compiled evaluator + sealed policy + S4-3 transaction
```

选择六全排列而不是只做五个R/C pair，原因是：

1. B0用于防止RVIR reference本身的历史语义漂移；
2. R用于把provider bound callback从candidate correctness oracle中剥离；
3. C才是需要关闭的compiled same-solver路径；
4. 六个排列可完整平衡variant的执行位置和冷/热顺序，即使本轮不形成latency claim也能暴露顺序相关状态污染。

S4-4只关闭correctness/evidence chain。即使18/18通过，状态也只能升级为
`VALIDATED-S4-SAME-SOLVER-CORRECTNESS`；S4-P timing必须另行预注册、另行生成artifact。

## 1. 对现有artifact基础设施的审计结论

### 1.1 可直接复用的资产

| 资产 | 可复用内容 |
|---|---|
| S3 v2 artifact | fresh subprocess、六order、3 replicate、raw-first、code blob绑定、summary semantic replay |
| S3 v2 tamper | 修改raw/protocol/summary后重签file manifest再验证语义拒绝 |
| RVIR five-fresh | B0/candidate独立进程、whole-core tensor inventory、pair mapping |
| RVIR whole-core truth | core/post双层truth投影、solver status/success/visited |
| RVIR native KFSB | 三candidate、72 child lower、final decision与fully re-signed tamper |
| RVIR live return | provider callback/fallback counters、commit/assembly/post receipt |
| B4-A handoff | terminal lower/lA one-shot lease和duplicate-CROWN=0 |
| DocOps exchange | executor交付、auditor独立报告、state machine和deterministic validation |

### 1.2 不能直接沿用的做法

旧RVIR formal主要把完整tensor放在PyTorch `.pt`中。虽然repo replayer使用`torch.load(..., weights_only=True)`，外部
审计者只用Python标准库无法解析tensor raw，只能相信repo代码生成的summary。

S3 v2的20 MB JSONL可由标准库读取，但只覆盖P-anchor optimizer，并没有whole-core所需的：

- 6 α + 6 β完整terminal state；
- terminal six-lA与shared intermediate source；
- KFSB 3 candidate/72 child lower；
- core-return/post-return 451个tensor-derived对象；
- device target `_version`、host packet prune和intermediate container clear；
- provider constructor/postprocess分离计数；
- poisoned failure state。

此外，串联历史artifact manifest只能证明旧source下各子机制曾分别通过，不能证明新S4 source中的组合路径执行过。
S4-4可把旧manifest作为historical-oracle provenance输入，但**formal truth必须由本轮18个新worker产生**。

## 2. artifact目录与文件合同

建议正式目录：

```text
artifacts/asplos27-s4-whole-core/resnet2b-prop0-v1/
  protocol.json
  environment.json
  source_identity.json
  workers.jsonl
  pairs.jsonl
  raw/
    w00-b0/
      worker.json
      tensor_records.jsonl.gz
      trajectory.jsonl.gz
      kfsb.jsonl.gz
      transaction.jsonl.gz
      stdout.txt
      stderr.txt
    ...
    w17-c/
  fault/
    precommit.jsonl
    midcommit.jsonl
    postcommit.jsonl
  summary.json
  replay_stdout.txt
  tamper_report.json
  README.md
  manifest.json
```

每个worker先写入同filesystem临时目录；只有worker自身validation全部通过后才atomic rename到`raw/wXX-*`。parent
process任何失败都把整个输出移动为`.failed-<reason>`或写到另一个non-formal目录，禁止在目标formal目录resume。

### 2.1 formal目录必须从空路径生成

- 目标不存在或为空；
- 任何已有formal文件都导致fail closed；
- 18个worker必须一次完整生成；
- 缺一个worker、stderr出现未允许exception、进程返回非0或atomic rename未完成，均不得生成summary/manifest；
- failed attempt保留供诊断，但manifest中`formal=false`，不能被后续工具“补齐”为formal。

## 3. protocol冻结

### 3.1 variant与顺序

六个triplet固定覆盖：

```text
B0-R-C
B0-C-R
R-B0-C
R-C-B0
C-B0-R
C-R-B0
```

每个triplet内三个variant是三个独立subprocess。`triplet_index/order/position/variant/worker_ordinal`必须同时写入
worker raw，且parent重算后的inventory逐元素exact。

不允许：

- 在同一Python进程依次跑B0/R/C；
- 复用前一个variant的compiled cache或mutable live object；
- 通过fork继承CUDA context；
- 缺失worker后从中间resume；
- 只在summary声称fresh而不保存PID/process-start/CUDA-context evidence。

### 3.2 frozen environment

每个worker必须核对并记录：

- Python、PyTorch、CUDA runtime/driver、cuDNN、TVM/TVM-FFI版本；
- GPU name、UUID、compute capability、total memory；
- dtype/device/deterministic flags/TF32/cudnn benchmark；
- model/property/config digest；
- αβ-CROWN、auto_LiRPA、VNN-COMP external repo commit；
- BoundFlow git HEAD、TVM/tvm-ffi/auto_LiRPA submodule commit；
- relevant environment变量的allowlist投影；
- wall clock只作日志，不能进入semantic hash或headline。

任何绝对本机路径、用户名、SSH信息或credential-like value必须在seal前扫描并拒绝。不能先写入artifact再事后替换。

### 3.3 correctness thresholds

冻结：

```text
lower/state/core/post/KFSB child lower: atol=2e-4, rtol=2e-4
compiled internal gradient:             atol=2e-5, rtol=2e-5
finite sign:                            exact
discrete fields:                        exact
NaN/+Inf/-Inf class and positions:      exact
```

容差必须写在protocol中，生成后不可改变。summary只能引用protocol值，不能自行携带另一套阈值。

### 3.4 claim flags

以下在protocol、每个worker、summary、manifest和tamper report均必须为false：

```text
timing_open
performance_claimed
same_solver_performance_claimed
complete_query_claimed
queue_claimed
tenx_claimed
asplos_ready_claimed
```

本轮允许的唯一positive状态是correctness/evidence-chain closure。

## 4. source identity：绑定真实执行闭包

### 4.1 repository identity

`source_identity.json`至少保存：

- BoundFlow HEAD commit；
- relevant worktree clean=true；
- formal runner、worker、replayer、tamper probe、S4-0—S4-3实现与全部直接import runtime文件的Git blob id和
  SHA256；
- TVM、TVM-FFI、vendored auto_LiRPA submodule commit；
- external αβ-CROWN/auto_LiRPA/VNN-COMP commit；
- pinned provider source文件SHA256；
- model/property/config SHA256；
- compile artifact/TIR/module receipt hash。

不能只绑定手写的5—10个`CODE_PATHS`然后把未列出的transitive implementation当成可信。第一版实现应生成
`executed_source_inventory.jsonl`：结合静态allowlist和运行时import observer，取二者并集后冻结。出现repo内已执行
Python模块却不在inventory时fail closed。

### 4.2 clean-source纪律

formal generation前：

- 所有executed source、protocol、runner、replayer和tamper文件必须由HEAD追踪；
- relevant path不得dirty/untracked；
- HEAD必须与指定remote branch相同，或manifest显式披露未推送commit；
- submodule HEAD必须与superproject gitlink一致；
- external repo不得dirty；
- source identity在第一个worker启动前冻结，最后一个worker完成后再次核对未变化。

## 5. 标准库可审计tensor raw

### 5.1 为什么不使用`.pt`作为formal truth

`.pt`可以保留为开发缓存或repo内部复跑输入，但不得是S4-4唯一raw。外部模型/审计脚本应能只用：

```text
json, gzip, base64, hashlib, struct, math
```

恢复shape、dtype、全部IEEE payload、sign/finite class和numeric differences。

### 5.2 tensor record schema

每个logical tensor一行canonical JSON：

```text
TensorRecordV1:
    variant
    worker_ordinal
    phase
    semantic_path
    ordinal
    dtype
    shape
    byte_order = little
    encoding = base64-ieee-bytes
    payload_base64
    payload_nbytes
    payload_sha256
    source_device
    materialization_reason
```

raw payload先按contiguous CPU logical value导出IEEE bytes；`payload_sha256`绑定解码前原始bytes。正式dtype至少支持：

- bool、int8/16/32/64；
- float16、bfloat16、float32、float64。

stdlib replayer必须自己实现bfloat16→float32解码，不得import numpy。gzip固定`mtime=0`、固定压缩级别和文件名字段，
确保同一raw投影可确定性重建；manifest绑定压缩文件bytes，semantic replay绑定解压后的record/payload hash。

### 5.3 tensor path与重复检测

`(worker_ordinal, phase, semantic_path, ordinal)`是唯一键。重复、乱序、缺失、extra path全部拒绝。path来自typed
owner inventory，不允许根据Python对象遍历顺序临时命名为`tensor_0`。

## 6. 每个worker必须记录的语义层

### 6.1 pre-state

- production snapshot/topology/policy/plan/module identity；
- 12条mutable path的shape/dtype/device/object-group/storage-group/stride/offset/version与content hash；raw
  object id/data pointer/storage handle不进入canonical artifact；
- stored/active/preserved α inventory；
- active/empty β inventory；
- host packet与intermediate container pre hash；
- solver/model/property/config identity。

### 6.2 10/9/10 trajectory

每个ordinal记录：

- lower；
- six α/six β before、gradient、after；
- Adam m/v/step；
- α/β LR before/after；
- scheduler call；
- keep-best、prune、stop、patience、timeout、restore decision；
- evaluator/launch/gradient/handoff counters；
- policy state hash与step hash。

B0、R内部representation可以不同，但必须导出同一个production-visible projection。C额外保存compiled receipt与
compressed internal gradient，用于2e-5门禁。

### 6.3 terminal handoff

- terminal ordinal=9；
- lower和six lA raw；
- lA total=37,464；
- handoff lease create/consume/release=`1/1/1`；
- terminal duplicate CROWN=0；
- shared intermediate source identity/version；
- terminal bridge count及compressed→dense→compressed round-trip。

### 6.4 KFSB

- candidate count=3；
- 每candidate 12个split decision，合计36；
- 每candidate child batch=24；
- child CROWN=3、child lower elements=72；
- unstable mask inventory；
- score/winner/final six decisions；
- provider bound callbacks、fallback/native shadow计数。

### 6.5 core return与transaction

provider scratch的variant policy与36-path binding必须来自`ProviderNetScratchFinalizationPlanV2`，不能由artifact runner
根据expected summary临时拼装。

- core-return所有字段及provider type identity；
- provider return constructor inventory=12；
- 12 target pre/candidate/post value和`_version`；
- candidate/rollback buffer inventory；
- host packet before/candidate/after；
- intermediate container before/after；
- commit order/state/hash；
- content audit与post-query audit；
- failure state必须属于合法state machine。
- provider net scratch必须按`core-entry → terminal-pre-extract → terminal-post-transfer → post-KFSB →
  post-finalization → solver-return`逐phase保存inventory；每个tensor同时记录logical bytes、unique storage bytes、
  object/storage/data-pointer lineage和view/alias group；empty tensor的`data_ptr=0`不得合并为真实alias；
- 当前fixture live枚举`6 α + 12 intermediate + 18 all-node lA = 36`个finalization attributes，同时另记六条
  terminal/export lA；B0只观察terminal transfer和KFSB重新产生的batch-24 residue，R/C在native KFSB后把36项归一化
  为sentinel。production tensor commit仍恰为12，scratch/export/commit count必须分列；
- B0允许保留六个provider β container及96 B nonempty storage；R/C admission要求provider-net β inventory=`0`，active β
  由typed core-result owner持有。B0与R/C final scratch差异只可用
  `NON_AUTHORITATIVE_PROVIDER_KFSB_RESIDUE`准入，不能伪写成数值parity或立即CUDA memory release。

### 6.6 official post与solver result

- provider postprocess call=1；
- core/post lower、upper、α、β、history、depth、threshold projection；
- 451个tensor-derived对象和sign inventory由新raw重算，历史数量不作硬编码通用条件；
- solver status/success/visited；
- queue-visible packet identity；
- verified/unknown verdict及termination reason。

## 7. variant-specific counter合同

| counter | B0 | R | C |
|---|---:|---:|---:|
| provider bound core call | 1 | 0 | 0 |
| provider compute/update callback | provider原生值 | 0 | 0 |
| provider return constructor | provider内部，不作12硬门禁 | 12 | 12 |
| official postprocess | 1 | 1 | 1 |
| compiled S4 evaluation | 0 | 0 | 10 |
| native RVIR evaluation | 0 | 10 | 0 |
| terminal duplicate CROWN | 按provider原生披露 | 0 | 0 |
| KFSB child CROWN | 3 | 3 | 3 |
| provider scratch finalization | observe only | 36 sentinel | 36 sentinel |
| provider-net β inventory | 6 containers / 96 B nonempty | 0 | 0 |
| final scratch policy | batch-24 residue | normalized | normalized |
| fallback/native shadow/eager | 不适用/披露 | 0 | 0 |

不得把B0原生provider调用计入C的`provider_bound_callback_count`，也不得用B0的不同内部constructor结构判C失败。
scratch table描述variant-specific non-authoritative provider state；queue-visible/core-result语义仍必须B0/R/C一致。

## 8. transaction与fault artifact

### 8.1 成功路径状态机

```text
PREPARED
  → COMMITTING
  → COMMITTED
  → POSTPROCESSING
  → COMPLETED
```

### 8.2 失败路径

```text
PREPARED validation/staging/KFSB/assembly failure
  → ABORTED_CLEAN

COMMITTING device/host/container failure
  → content rollback where possible
  → POISONED_NO_RETRY

POSTPROCESSING official post failure
  → COMMITTED_POST_FAILED_POISONED
```

`COMMITTED_POST_FAILED_POISONED`是S4-4审计中新识别的必要状态：post发生在commit后，不能把它误归为clean abort，
也不能自动回滚后重新调用post。current query必须终止，保留commit/post failure raw，禁止fallback和queue继续。

### 8.3 fault injection matrix

至少fresh注入：

1. precommit source/version validation；
2. KFSB后、core assembly前；
3. provider constructor；
4. device copy ordinal 1；
5. device copy ordinal 6；
6. device copy ordinal 12；
7. host packet replace；
8. intermediate container clear；
9. receipt seal前；
10. official post entry；
11. official post materialization中；
12. official post return前。

每案保存pre/post内容、object identity、`_version`、host/container state、fallback/retry/queue counters和最终failure state。
fault raw不混入18个positive worker summary，但必须被同一manifest绑定。

## 9. stdlib semantic replay

### 9.1 独立性要求

建议新增：

```text
scripts/replay_asplos27_s4_whole_core_stdlib.py
```

该文件在import graph中只能使用Python标准库；测试必须通过AST/import probe证明未import：

```text
boundflow, torch, tvm, numpy, scipy, alpha-beta-CROWN
```

replayer不得调用production receipt `.validate()`、artifact runner的`validate_records()`或任何会重新使用expected summary
的helper。

### 9.2 replay顺序

1. canonical验证manifest及manifest hash；
2. 逐文件核对size/SHA256；
3. 验证source identity与protocol hash；
4. 解码18个worker tensor raw；
5. 重建triplet/variant/process inventory；
6. 重建每step trajectory和10/9/10 cardinality；
7. 重建terminal handoff/lA inventory；
8. 重建KFSB 3/3/72与decision；
9. 重建transaction/provider/post counters；
10. 计算B0/R、R/C、B0/C逐tensornumeric/sign/discrete parity；
11. 重建fault state matrix；
12. 重建summary；
13. 与冻结`summary.json`逐canonical byte比较；
14. 输出固定`replay_stdout.txt`投影。

任何summary字段不能仅从summary自身复制；必须能追溯到protocol、source identity或raw。

### 9.3 numeric算法

stdlib replayer按tensor path对齐：

- shape/dtype exact；
- 逐元素finite class；
- finite元素计算absolute/relative diff；
- sign bit按`<0/==0/>0`比较，必要时单列negative zero；
- NaN不参与max，但positions必须exact；
- Inf符号必须exact；
- 每tensor、每phase、全局均输出max diff与argmax path/index；
- 禁止只算digest相同/不同后跳过数值。

## 10. summary最小字段

`summary.json`至少包含：

- schema/status/source/protocol hash；
- worker/triplet/order/variant counts=`18/6/6/3`；
- B0/R、R/C、B0/C的lower/state/gradient/core/post/KFSB max diff；
- sign/discrete exact flags与元素计数；
- 10/9/10 trajectory exact；
- terminal handoff/lA/rerun inventory；
- KFSB 3/3/72和final decision；
- 12-path commit/host prune/container clear；
- provider bound callback/constructor/postprocess counters；
- provider net scratch按phase/variant投影：B0 terminal transfer、B0 KFSB batch-24 residue、R/C finalization；
- logical bytes、unique storage bytes、alias group和object/storage/data-pointer lineage；
- B0 β retention、R/C provider-net β inventory=`0`与exclusive owner counters；
- solver status/success/visited/verdict；
- fault matrix与clean/poisoned状态计数；
- candidate/rollback logical bytes和实测allocated/reserved披露（非performance）；
- replay status；
- tamper report hash；
- 所有claim flags=false；
- `summary_hash`。

状态只有两种：

```text
VALIDATED-S4-SAME-SOLVER-CORRECTNESS
VALIDATED-NO-GO-S4-SAME-SOLVER-CORRECTNESS
```

不能设置“部分通过但gate open”。

## 11. manifest设计

### 11.1 file inventory

manifest为canonical JSON，绑定除自身外的每个文件：

```text
relative_path
size_bytes
sha256
semantic_role
```

并绑定：protocol hash、source identity hash、summary hash、tamper report hash、artifact schema、status和claim flags。
最后对去掉`manifest_hash`的canonical payload计算`manifest_hash`。

禁止绝对路径、symlink、重复normalized path、未列文件和列出但不存在文件。日志也必须绑定，避免只篡改stdout/error掩盖
worker异常。

### 11.2 seal顺序

正确顺序：

```text
raw workers
  → fault raw
  → stdlib replay-derived summary
  → fully re-signed tamper report
  → replay stdout
  → README
  → final manifest seal
  → final replay
```

tamper report必须进入manifest；不能在manifest后生成一个未绑定报告。

## 12. fully re-signed tamper矩阵

S4-4冻结minimum 68类。每案都要：

1. copy完整artifact；
2. 修改semantic raw/protocol/source/summary之一；
3. 同步重算被改tensor payload hash；
4. 同步重算gzip/file digest；
5. 同步重算summary（攻击者可伪造）；
6. 同步重算manifest file inventory和`manifest_hash`；
7. 运行stdlib replayer；
8. 必须由raw-derived invariant拒绝。

### A. source/protocol（8）

1. BoundFlow source commit；
2. executed code blob；
3. TVM submodule commit；
4. external αβ-CROWN commit；
5. model digest；
6. property/config digest；
7. tolerance放宽；
8. claim flag提前true。

### B. worker/process/trajectory（8）

9. 删除worker；
10. 重复worker；
11. 交换variant/order；
12. 伪造fresh process identity；
13. 删除evaluation ordinal；
14. 10/9/10 cardinality；
15. Adam moment；
16. keep-best/prune/stop/restore decision。

### C. mutable/terminal/KFSB（10）

17. missing α path；
18. swap α path；
19. preserved α drift；
20. active β location/sign；
21. compressed gradient；
22. terminal ordinal；
23. terminal lA inventory/value；
24. handoff lease reuse或duplicate CROWN；
25. KFSB child lower；
26. KFSB final decision。

### D. transaction/provider/post（10）

27. committed path count/order；
28. candidate tensor value；
29. tensor `_version`；
30. host packet extra field；
31. intermediate container未clear；
32. provider bound callback；
33. provider constructor count；
34. official postprocess count；
35. official post lower/value；
36. poisoned failure伪写为clean rollback。

### E. artifact/replay（4）

37. tensor payload改值并重签payload/file/manifest；
38. summary status伪升级；
39. tamper report删案后重签；
40. replay stdout伪造PASS。

### F. provider net scratch/finalization（8）

41. 删除一个R/C finalization path；
42. 把R/C lA/intermediate sentinel改回stale tensor；
43. 修改`last_update_preserve_mask` mirror；
44. 伪造exclusive owner latch transition；
45. 把provider reentry count从1改0；
46. 把multi-core count从2改1；
47. 隐藏B0 stale net β retention或伪造R/C provider-net β inventory；
48. 把scratch finalization错误混入production 12-path commit count。

### G. scratch phase/storage/alias（8）

49. 把B0 post-KFSB batch-24 residue伪写为batch-12；
50. 把R/C post-finalization sentinel伪写为未归一化stale tensor；
51. 交换terminal-transfer与post-KFSB phase ordinal；
52. 用logical tensor bytes冒充unique storage bytes；
53. 删除一个lA shared-storage alias group；
54. 把empty tensor的`data_ptr=0`错误合并成真实alias group；
55. 篡改β field/container identity或跨phase residue lineage；
56. 把attribute sentinel替换伪写为立即释放CUDA storage或allocated下降。

### H. S4-0 live mutable admission（8）

57. 用snapshot object alias group替代live storage group；
58. 隐藏两个distinct nonempty view共享同一storage；
59. 修改live Tensor `_version`并全重签外层receipt；
60. 保持shape/dtype但修改stride或storage offset；
61. 把五个empty β的zero pointer伪装成一个共享storage alias；
62. 把R31 dense mapping `source_state_hash`冒充snapshot hash并删除plan binding projection；
63. 在β/history已匹配前缀后追加一个未拥有slot；
64. 把raw object id/data pointer写入canonical receipt，或用topology输入顺序改变canonical hash。

### I. S4-1 DAG adjoint与terminal lA phase（4）

65. 用普通selected-primal输出替换coefficient-program adjoint并全重签；
66. 删除residual fanout/accumulation VJP provenance或交换其action owner；
67. 注入已知site19 reduction反例错值并同步重签gradient/summary/manifest；
68. 把terminal lA改为post-transform coefficient，或删除`[D,S,*feature]`中的spec-axis identity。

必须报告每案稳定reason，不接受“因为文件digest不匹配”作为fully re-signed攻击的唯一拒绝理由。

## 13. tests与静态门禁

建议固定文件集合，不再用`-k`模糊统计：

### 13.1 runner/replayer专项

- protocol/order/source identity；
- stdlib tensor encode/decode；
- float16/bfloat16/NaN/Inf/negative-zero；
- deterministic gzip；
- incomplete/no-resume；
- summary全raw重算；
- manifest seal/order；
- 68类tamper。

### 13.2 S4 whole-core专项

- S4-0 admission；
- S4-1A—1D evaluator；
- S4-2 policy trajectory；
- S4-3 terminal/KFSB/core/post/transaction；
- precommit/midcommit/postcommit faults；
- provider net scratch consumer audit。

### 13.3 repository gate

- targeted全过；
- full pytest；
- black；
- touched-file mypy；
- touched-file pylint 10.00/10；
- `git diff --check`；
- `dol exchange validate`；
- `dol lint --soft`。

任何skip必须逐条列reason；S4 formal GPU测试不能因`CUDA unavailable`被skip后仍关闭S4-4。

## 14. 外部审计最低要求

auditor不能采信executor summary数字，至少独立完成：

1. 用自己的stdlib脚本核对manifest/source/file tree；
2. 解码tensor raw并重算B0/R、R/C parity；
3. 重算10/9/10 trajectory与六α/active β coverage；
4. 重算terminal 37,464 lA、KFSB 3/3/72；
5. 核对12-path commit、host packet prune、container clear；
6. 核对provider bound callback=0、constructor=12、post=1；
7. 现场运行repo stdlib replay；
8. 自建至少3个未预注册fully re-signed攻击；
9. 抽查一个precommit clean、一个midcommit poisoned、一个postcommit poisoned fault；
10. 核对claim flags和S4-P仍关闭。

## 15. 实现切分

只有S3外审批准且S4-0—S4-3依序关闭后才允许：

1. `docs: preregister S4-4 formal closure`；
2. `feat(artifact): add stdlib tensor record codec`；
3. `feat(artifact): add S4 whole-core fresh worker projection`；
4. `feat(artifact): add 18-worker six-permutation runner`；
5. `feat(artifact): add pre/mid/post commit fault records`；
6. `feat(artifact): add independent stdlib semantic replay`；
7. `test(artifact): add 68 fully re-signed attacks`；
8. `artifact: generate S4 whole-core formal v1`；
9. `docs: close S4 correctness and prepare external audit`；
10. `docs: preregister S4-P timing`。

artifact代码、formal raw和closure文档应分提交，避免代码与其第一次结果共享不可审计dirty source。

## 16. GO / STOP

### GO

只有以下全部成立：

- source clean、18 worker完整、六全排列exact；
- B0/R与R/C whole-core/core/post/solver parity全部通过；
- 10/9/10、terminal handoff、KFSB 3/3/72、12-path transaction exact；
- provider C路径bound callback=0、constructor=12、post=1；
- precommit/midcommit/postcommit failure分类正确且禁止非法fallback/retry；
- stdlib replay从raw逐字重建summary；
- 68/68 fully re-signed tamper拒绝；
- targeted/full/static/DocOps全过；
- 外部审计批准；
- 所有性能/complete-query/10x flag仍false。

### STOP

任一情况直接NO-GO：

- 只有`.pt`可用、stdlib无法独立解析；
- B0/R reference漂移；
- C需要native shadow或provider bound callback；
- 只比较summary hash而不比较raw tensor；
- 缺worker后resume；
- post失败被误报为clean rollback；
- tamper只靠外层digest拒绝；
- code inventory遗漏实际执行的implementation；
- 为过门禁放宽容差或删除不一致字段；
- 正确性artifact中夹带performance headline。

## 17. 当前停止点

```text
S3 exchange = ready_for_audit / no audit result
S4-0..S4-3 implementation = closed
S4-4 implementation/formal/artifact = closed
S4-P timing = closed
```

本文只把S4 correctness证据链设计到可实施、可独立审计的程度，不改变当前DocOps
`next=external-audit-asplos27-s3-optimizer-runtime`。
