# S4-1C compressed gradient 与 terminal lA 实现变更记录

date: 2026-08-31
status: implemented-correctness-candidate
external-audit: pending
timing-recorded: false
performance-claimed: false
source-commits: dcbfe80,7110437,82a928a,ef8d704

## 1. 本轮完成了什么

在已经关闭的S4-1A ordered mutable buffer和S4-1B six-site V图之上，实现单次production evaluation的
第三遍Pass C：

```text
S4-1B V17/V19/V23/V25/V28/V31
  + R31/D2B coefficient recompute
  + S4-1A caller-owned α/β gradient buffers
  -> 6 compressed dα + 1 active dβ
  -> optional terminal pre-transform lA in-place handoff
```

本轮严格停在implementation/correctness。没有接Adam、没有10/9 trajectory、没有计时，也没有性能claim。

## 2. TVM/TIR实现

新增`boundflow/backends/tvm/asplos27_s4_compressed_gradient.py`：

- 一个layout-parameterized dα模板，实例化site 17/19/23/25/28/31六次；
- 一个site31 active dβ模板；
- 六个pre-transform terminal lA copy模板；
- 合计13个exported symbol，固定128 threads/block；
- dα ABI固定为A/V/lower/upper/active-alpha/indices/upstream/output八项；
- dβ ABI固定为V/location/sign/upstream/output五项；
- safe index先clamp再read，非法输入统一产生canonical qNaN bits `0x7fc00000`；
- stable/nonambiguous ReLU或A<0合法地产生正零，不把它们误报为非法；
- unscheduled TIR、scheduled TIR、device source都按实际content绑定hash；
- 当前RTX 4060 Laptop / SM89编译身份：template=`5073ba16...5340`、schedule=
  `044533c9...485e`、device source=`41237a51...539e`。

## 3. production runtime实现

新增`boundflow/runtime/asplos27_s4_gradient_emitters.py`：

- nonterminal严格执行17项：10 coefficient action + 6 dα + 1 dβ；
- terminal严格执行23项，在每个site最后一个V reader之后插入六次copy；
- site31顺序强制为`dα31 -> dβ31 -> copy lA31 -> ReLU31`；
- site25/site19强制在residual stage1/stage2之间emit/copy；
- 七个gradient直接写S4-1A已有caller-owned buffer，warm output allocation=`0`；
- terminal lA原位覆盖S4-1B的单一37,464-element V arena，新增physical storage=`0`；
- terminal result恢复`[D,S,*feature]`，并通过one-shot lease交给后续native consumer；
- 46个unique emitter DLPack view只在prepare建立，warm view construction=`0`；
- receipt固定53 argument occurrences、46 unique views、110 full descriptor union，并把所有timing/
  performance/fallback/native-shadow/dense-A标志钉为false/0；
- module/value/buffer/metadata/evaluation/state identity共同形成prepared id，default stream、metadata漂移、
  S4-1A参数内容漂移、重复lease和action篡改均fail closed。

## 4. correctness与负向验证

新增两个测试文件：

- `tests/test_asplos27_s4_compressed_gradient.py`：独立PyTorch公式覆盖六种F/W布局；检查dβ、copy、
  safe-index、alpha endpoint、nonfinite、lower>upper、beta location/sign和compiled identity；
- `tests/test_asplos27_s4_gradient_phase.py`：从冻结ResNet2B pre-state现场建立S4-0/1A/1B/1C链，分别执行
  nonterminal/terminal；terminal run在覆盖V前保留test-only oracle副本，再用closed formula重算全部六dα和
  一dβ；检查23-action、单arena、spec轴、one-shot lease、stream/state/claim负向门禁。

当前结果（源码固定至`ef8d704`）：

- S4-1C新增专项：`11 passed`；
- S4/R3 production联合：`200 passed`；
- 四个交付文件mypy clean；
- 四个交付文件Pylint `10.00/10`；
- Black和`git diff --check`通过；
- 全量：`2093 passed, 3 skipped`；skip为既有TVM重复编译规避1项和冻结VNN-COMP checkout缺失2项，
  均非本批回归。

## 5. 明确边界与遗留

1. S4-1C只验证冻结参数下的single evaluation。coefficient propagation仍调用已批准的R31/D2B kernel，
   prepare时机械核对它读取的原始α/β与S4-1A clone逐内容相等；S4-1D接optimizer时必须把coefficient kernel
   ABI直接改绑到S4-1A active buffers，不能只依赖初始内容相等。
2. 当前没有formal multi-process artifact；外审可现场重跑真实GPU专项，并从源码独立重算公式和动作表。
3. terminal lease尚未接KFSB production consumer；该接线属于S4-1D evaluator closure。
4. 没有10-step/9-mutation、same-solver、timing、memory headline、speedup、10x或ASPLOS-ready结论。

## 6. 当前结论

当前只能写：

`IMPLEMENTED-CORRECTNESS-CANDIDATE-S4-1C-COMPRESSED-GRADIENT`

下一动作是对固定源码提交和本轮验证做一次独立外审。外审批准前不升级`VALIDATED-S4-1C`，也不开放
S4-1D optimizer/evaluator implementation。
