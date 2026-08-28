# BoundFlow ASPLOS’27 production verification compiler 与 10× 端到端加速计划（v10 执行稿）

status: s3-ready-for-external-audit-s4-prereg-draft
date: 2026-08-26
updated: 2026-08-28
revision: v10-s4-compressed-evaluator-abi
supersedes-revision: v6-asplos27-ten-x
supersedes-draft-at: 20f4741
submission-target: ASPLOS-2027-September-cycle
submission-deadline: 2026-09-09-AoE
paper-primary-pillar: programming-languages-and-compilers
paper-secondary-pillar: gpu-runtime-and-accelerator-systems
performance-north-star: 10x-same-solver-complete-query
performance-north-star-status: hypothesis-not-validated
execution-authority: true
code-change-open: false-pending-s3-external-audit
external-audit: deferred-by-user
performance-claimed: false

> **2026-08-28 S4 ABI修正（v10状态，不改10x北极星）**：S3已进入DocOps外审round 1。只读核对
> production optimizer raw与atomic copy-out发现：六个α source共8,496 stored元素，但lower-only optimizer
> 实际只更新`[0,0]`方向的4,248 active元素，另4,248元素保持不变；P-anchor对应1,032 stored/516 active，
> 两种口径下coverage均为`12.1469%`（不是runtime share）。因此S4不能把S3 wrapper直接替换whole
> `update_bounds_core`；必须让compiled evaluator直接消费六个compressed lower-α slot与sparse β，并由
> sealed host policy driver唯一拥有10/9 optimizer policy。RVIR dense state只作独立oracle与terminal一次性bridge；
> terminal lA/KFSB/atomic commit继续沿用现有语义，KFSB的3次batch-24 child CROWN必须在S4-P单列。
> S3独立外审批准前S4仅允许预注册审阅，代码和timing关闭。

> **2026-08-28 执行状态更新（v8）**：S1 standalone IBP compiler pipeline、S2 single-evaluation coarse
> CROWN与S3完整10 evaluation/9 Adam mutation本地wrapper已依次关闭。S3 v2的18 fresh稳健formal达到
> P/native geomean/worst=`3.2438943700x/3.2246091003x`、P/旧D2B=`1.8422164274x`，逐step语义、
> replay和10/10全重签篡改通过；v1 `2.5695746x/0.7595405x` NO-GO并列保留。S3发现并修复了
> selected-value CUDA Graph临时cuDNN workspace与TVM TLS析构所有权问题，当前结构为安全VM调用+
> persistent output TIR。下一只开放S4 RVIR same-solver implementation/correctness与真实share归因；本文
> 10x north star仍为未验证假设，不能由S3局部3.2439x外推。

## A. ASPLOS’27 总控目标：把现有成果收束为一篇论文，而不是继续扩张对象数量

### A.1 直接结论与论文主张

本计划从v6起不再以“把所有历史层都接通”为终点，而以**2026年9月9日AoE提交ASPLOS’27完整论文**为
条件目标。论文的PL/compiler主柱与GPU runtime/accelerator副柱冻结为：

> **BoundFlow把production αβ-CROWN/BaB exact-call中的verification tensor regions导入标准
> Relax/TIR，在不改变optimizer、split/history、branch、termination和publish trajectory的前提下，联合优化
> bound representation、horizontal/vertical fusion、lifetime、rematerialization、custom VJP与prepared GPU
> runtime，并把收益传播到同一solver的complete-query。**

只保留三项候选贡献，避免把capture、schema、Plan、Task、Schedule或receipt分别包装成论文贡献：

1. **verification-semantic lifting**：从production tensor graph向标准Relax/TIR附加足以证明融合、表示切换、
   saved-state与publish边界合法的最小语义；不新造solver execution IR；
2. **joint compiler optimization**：继承CIBC，自lower/upper横向融合扩展到CROWN reverse-wavefront纵向region，
   联合决定coefficient representation、lifetime/rematerialization、custom VJP、physical arena和schedule；
3. **trajectory-preserving system integration**：通过RVIR exact-call接入原αβ-CROWN/BaB host，以一个prepared
   cumulative candidate消除per-site framework crossing、allocation、synchronization和重复optimizer/runtime工作，
   并做B0→final的same-solver端到端评估。

CIBC是2023年AAAI’24未录用稿所代表的既有研究起点；v6不重新发明其BoundConv、horizontal fusion、tensor
program lowering或autotuning，而把它们作为第一项优化资产。BoundFlow相对CIBC的新增量必须在一页
resubmission changes note中清楚列出：production αβ-CROWN/BaB语义、CROWN/optimizer region、representation/
lifetime/rematerialization/custom VJP、RVIR publish，以及same-solver complete-query证据。

### A.2 `10×`的唯一定义与claim边界

用户提出的总体约`10×`被采纳为**研究北极星和headline stretch target**，但在formal完成前不是结果。唯一允许
升级为“总体10×”的口径是：

```text
S_query = native B0 complete-query wall time
          / final BoundFlow complete-query wall time
```

两侧必须使用同一official αβ-CROWN、model/property、seed、dtype/device、branching、iteration/termination、
timeout、α/β初始化和solver trajectory。primary systems mode固定算法工作量；如果另做algorithmic mode并减少
node/iteration，只能形成独立TTV/solved结果，不能与compiler speedup相乘。

必须同时披露：

- warm same-solver complete-query；
- cold capture/compile/prepare/setup与break-even；
- 预注册query mix下包含compile cost的amortized结果；
- fixed-iteration/fixed-node systems-only结果；
- queue吞吐、time-to-verdict、solved/unknown、peak allocated/reserved与worst pair。

历史`1.15× complete-query / 1.20× queue`保留为最低研究资格线，不再是v6北极星；局部operator、kernel、
region、wrapper、fixed-prefix、memory ratio或CUDA Graph结果都不得单独写成“overall 10×”。

### A.3 10×速度预算与物理可达性

把同scope native时间归一化为1，`u`为确认无法优化的时间份额，`h`为candidate新增integration overhead，
其余部分平均加速为`r`：

```text
S = 1 / (u + (1-u)/r + h)
r_required(S=10) = (1-u) / (0.1-u-h)
```

因此`u+h >= 0.10`时10×在数学上不可达。即使`h=0`，也有：

| 不可优化份额 `u` | 无限region加速的系统上限 | 达到10×所需其余全栈平均`r` |
|---:|---:|---:|
| 0% | 无穷 | 10.00× |
| 1% | 100× | 11.00× |
| 2% | 50× | 12.25× |
| 3% | 33.33× | 13.86× |
| 5% | 20× | 19.00× |
| 8% | 12.50× | 46.00× |
| 9% | 11.11× | 91.00× |
| 10% | 10×渐近线 | 有限`r`不可达 |

工程目标不是勉强保持`u+h<10%`，而是在第一轮full-stack归因中把`u+h`压到`<=3%`；否则需要超过约14×
的全栈平均加速，风险极高。每个bucket使用同scope多区域公式：

```text
S = 1 / (sum_i(s_i/r_i) + u + h)
residual_budget_for_10x = sum_i(s_i/r_i) + u + h <= 0.10
```

现有FSG1 B0 raw只能提供**fixed 16-BaB-iteration prefix诊断先验**，不是complete-query/TTV claim：234个
provider call下，tensor/operator execution约占61.3%，solver/runtime control约占38.7%。用历史CIBC graph
`2.45631×`和局部CROWN TIR `4.89834×`做极乐观代入也只有约`1.83×`；即使61%左右的operator全部达到
CIBC operator `12.795×`，若剩余host/runtime不动，总体也仅约`2.3×`。要到10×，在operator约12.8×时，
其余setup/unclassified host/runtime还需约8—9×。这说明目标只能来自**coarse full-stack compilation**，不能
来自再调一个kernel。

上述诊断与FSG4单core/exact-call的约1.29秒query、0.13秒core属于不同protocol，禁止拼表、相乘或互相补齐
分母。S0随后已按A.5.2完成fixed-trajectory explicit transaction raw与预算；solved-query/TTV raw仍必须在
S5用双方都能solve的公开property独立采集，不能由当前fixed-prefix替代。

### A.4 “引入所有优化”的累计候选，而不是速度数字相乘

“所有优化”定义为**所有通过归因、合法性和累计消融门禁的优化**，不等于无条件打开每个技巧。最终只有一个
`BoundFlow-final` candidate；每加入一轴都直接从B0及前一累计candidate重测。

| 优化包 | 继承的已有资产 | v6必须补齐的物理机制 | 主要时间桶 |
|---|---|---|---|
| O1 CIBC tensor fusion | BoundConv horizontal fusion、Conv schedule/autotuning、IBP graph `2.456×` | Linear/ReLU/Add/residual/epilogue完整lowering；cross-op/vertical fusion；shape-specific schedule | IBP/initial bounds |
| O2 coarse CROWN region | B4-B2 P/S Linear/Conv TIR、local `4.898×` | relaxation→β injection→sign select→adjoint→fanout→concretization闭合region，减少中间A和launch | α/β-CROWN operator path |
| O3 representation/VJP/memory | R3 structured owner、saved dense A=0、D2B custom backward | dense/Patches/sparse/factorized选择；lifetime/remat；residual reuse；physical arena和HBM traffic闭合 | CROWN forward/backward与memory |
| O4 optimizer transition | 10 evaluation/9 mutation trajectory、α/β/Adam correctness | static tensor/weight/view hoist；loss/Adam/clamp/best-select fusion；host policy只留必要predicate | optimizer/VJP与setup |
| O5 prepared runtime | RVIR exact-call、atomic publish、DLPack/stream ABI、MR7R host ledger | 234-call/per-site bridge收束；persistent views/cache；同步guard合并；粗粒度submission；可证后CUDA Graph | framework/launch/alloc/sync |
| O6 batching与branching | domain/spec三轴schema、KFSB调用证据 | domain/spec/target batching；branch score与bound共享；GPU hot frontier；等显存吞吐 | solver/runtime control |
| O7并行与specialization | static schedule/cache资产 | shape/signature specialization；有`>=10%`可重叠critical-path证据后才开multistream | residual host/device gap |

下列数字明确禁止相乘：CIBC operator与CIBC graph；B4-B2 local与同region的MR7机会；B2/B3相对值与省略B0
后的新值；memory ratio与latency；CUDA Graph与已被fusion删除的launch；kernel-sum与multistream critical path；
改变batch/节点数的吞吐与same-work complete-query。所有headline只来自最终candidate的direct B0 pair。

### A.5 累计里程碑与kill gates

| Gate | 累计candidate必须证明 | 性能作用 |
|---|---|---|
| S0 ASPLOS feasibility | B0 scope、两个claim mode、`s_i/u/h`覆盖`>=97%`、unclassified`<=3%`、前两页论文骨架 | 判断10×是否数学可达及13天投稿是否现实 |
| S1 compiler path | CIBC整图经唯一Relax/TIR/prepared path，正确性与fallback=0 | 保住IBP `>=2.20×`资格，不外推query |
| S2 CROWN/VJP | multi-layer coarse region、saved dense A=0、physical arena、active β | 同scope CROWN累计目标`>=4×` |
| S3 optimizer/runtime | per-step compiled transition、host policy cut、single prepared invocation、跨site bridge收束 | whole 10/9 wrapper累计目标`>=3×`，exact-call目标`>=5×` |
| S4 solver-wide | batching/branching/runtime residual按新profile打开，所有结果直接对B0 | fixed-prefix/complete-query累计目标`>=8×` |
| S5 final formal | 两模型族、held-out、至少一个双方solved workload、完整消融与cold/amortized | complete-query geomean目标约`10×`，建议worst `>=5×` |

`S0`若得到`u+h>=0.10`，立即关闭10× headline；`u+h>0.05`标为高风险。单bucket的`r_required`超过其
同shape实测roofline/物理上限时STOP。旧版“任意`r_required>10×`默认STOP”不再适用于10×北极星，因为系统
本身就要求约10×平均加速；没有新物理机制时仍用20×作为探索cap，超过即停止该bucket。未达到10×但获得
跨模型、同solver的2—5×且核心机制/消融成立时，可以形成较弱但仍可能可发表的系统结论，不能伪写10×。

### A.5.1 2026-08-27 S0 第一批执行结果

用户已批准按v6开始执行。第一批没有新写kernel，也没有把旧speedup相乘，而是落地了typed 10×预算、两种
claim mode、事务语义覆盖门禁、历史direct ratio隔离ledger和可语义重放artifact。冻结结果为：

- artifact：`artifacts/asplos27-s0-tenx-budget/fsg1-diagnostic-and-history-v1`；
- 10个official B0 fixed-16-iteration-prefix profile中，时间轴/事务拓扑上下文`10/10`闭合；
- MNISTFC 5个run的机制覆盖通过，但ResNet2B 5个run仍有`30.62%—31.51%` inter-call host mechanism
  未解析，因此全矩阵只`5/10`通过`coverage>=97% / unclassified<=3%`；
- 历史CIBC operator `12.7951077×`只作为local hypothesis代入时，各run最乐观projection上限为
  `2.3188568×`；operator无限快的最大系统上限也只有`2.6107780×`；`10/10`均不满足10×预算；
- B3/B0、MR5、MR6和CIBC graph四条direct ratio进入ledger，但因为scope不同，artifact明确不做geomean、
  product或headline；
- 状态=`s0-attribution-not-admitted`，`performance_claimed=false`，`s1_performance_gate_open=false`。

这不是最终判定“10×物理不可达”：当前把全部non-operator host事务按`1×`冻结，只证明**operator-only路线不
可达**。下一刀必须在不改solver语义的前提下给ResNet inter-call host段插入explicit transaction marker，区分
incomplete→BaB handoff、domain prepare/pick、branch score、split/history、bound pre/core/post、domain commit、
termination，并为这些bucket寻找O4—O6的可编译机制。该机制归因通过前，S1性能门禁继续关闭；允许并行实现
marker与paper skeleton，不允许把S1结果预写进论文。

### A.5.2 2026-08-27 S0 explicit transaction 与10×预算关闭结果

第二批没有用outer `verify/general_bab`粗span吞掉空白，而是在固定αβ-CROWN commit上安装33个可恢复
observer target；`compute_bounds`仍作为最深事务，exact target才计入机制覆盖，coarse target只用于界定未解析
范围。formal为两个workload各5个fresh control/profile pair：

- artifact：`artifacts/asplos27-s0-transactions/official-b0-five-pair-v1`；
- ResNet2B最低机制覆盖=`99.632394%`，最大unresolved=`0.367606%`，profile/control中位/最大=
  `0.9959016×/1.0416400×`；
- MNISTFC最低机制覆盖=`99.248363%`，最大unresolved=`0.751637%`，profile/control中位/最大=
  `0.9986577×/1.0653545×`；MNIST r0单对超过1.05，但冻结门禁明确作用于five-pair中位数；
- 两个workload的compute signature均five-fresh exact，10对语义均一致；artifact replay通过，且全重签
  worker-summary与protocol-target篡改均被语义重算拒绝；
- S0 attribution从`NOT ADMITTED`推进为`ADMITTED`，只开放预算重算，不产生BoundFlow性能结果。

按5个profile的`sum(category_ns)/sum(scope_ns)`形成互斥事务桶后，冻结以下**研究目标而非实测结果**：
O1 coarse bound region=`16×`、O2 structured state/batching=`8×`、O3 compiled admission/prepared
runtime=`12×`、O4 memory lifetime/reclamation=`20×`、O5 result/termination=`4×`；全部unresolved按
`1×`保守保留。下表投影还显式假设candidate新增integration overhead `h=0`，尚未计入接入成本；任何
`h>0`都必须从0.10 residual budget中直接扣除。派生artifact为
`artifacts/asplos27-s0-transaction-budget/official-b0-five-pair-v1`，结果为：

| workload | O1 share | O2 share | O3 share | O4 share | unresolved | h | 达10×所需resolved平均 | 条件式目标组合投影 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet2B | 67.9625% | 20.7179% | 7.2578% | 3.7300% | 0.3316% | 0% | 10.3087× | 12.5622× |
| MNISTFC | 20.2428% | 0% | 78.9729% | 0.0527% | 0.7294% | 0% | 10.7081× | 11.6566× |

因此`10×`在当前事务预算下**数学可达但尚未实验验证**。`s1_implementation_open=true`只允许开始S1
canonical CIBC `Primal→Bound→Plan→Relax/TIR→Prepared Runtime`纵向实现；
`s1_performance_gate_open=false`、`performance_claimed=false`继续保持。O1的16×高于现有CIBC local
`12.795×`且远高于B4-B2 local `4.898×`，O2—O4目标也都只是待证伪目标；任何一轴无法达到预算时必须用
新direct cumulative结果重算，不得保留12.56×/11.66×投影。

### A.5.3 2026-08-28 S1 canonical CIBC pipeline closure

S1 已把 standalone ResNet2B IBP 的完整 17-op 图收束到一个 standard Relax function：6 个 Conv 调用
CIBC paired-output TIR，2 个 Linear shape family 由 TVM cuBLAS partition 接管，ReLU/Add/Flatten 保持
standard Relax dataflow；compile 后由 prepared VM + static-address CUDA Graph 作为唯一执行入口。旧
`CIBCIBPCUDAGraphPlanV1`只作为 PyTorch/direct-CIBC oracle，不是 production pipeline。

formal artifact=`artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2`，source=`56c494f`（实现
`aa537ed`）。6 个 fresh
process覆盖`BDP/BPD/DBP/DPB/PBD/PDB`六全排列，每进程30组×每对象200 replay，三侧均包含input copy：

| S1 metric | formal result | gate |
|---|---:|---:|
| direct-CIBC/PyTorch geomean | `2.5023460x` | disclosure |
| canonical pipeline/PyTorch geomean | `2.5028100x` | `>=2.20x` |
| canonical pipeline/PyTorch worst | `2.4600206x` | `>=2.00x` |
| canonical pipeline/direct propagation geomean | `1.0001854x` | `>=0.90x` |
| canonical pipeline/direct propagation worst | `0.9898443x` | disclosure |
| maximum absolute difference | `0.000244140625` | `<=3e-4` + sign exact |

结构门禁同时成立：17-op、6/6 CIBC call_tir、2 cuBLAS partitions、warm DLPack construction=0、
fallback/eager shadow=0、graph-stable output无额外materialization copy；raw→summary replay PASS，8/8 fully
outer-resigned tamper rejected。summary/manifest hash分别为`7c2fe8b0…ff60`/`bd4eaa4a…80cc`。

因此S1=`VALIDATED-S1-CIBC-CANONICAL-PIPELINE`，只说明完整compiler plumbing保住standalone IBP winner；
`same_solver_claimed=false`、`performance_claimed=false`仍保持，不撤销MR1的CIBC full-graph exact-call
`0/51 eligible`。唯一下一工程动作是S2 coarse CROWN/custom VJP进入同一prepared path；在S2/M5之前不得把
本结果写成αβ-CROWN、BaB、query或complete-query speedup。

### A.6 ASPLOS’27 13天submission sprint

ASPLOS’27 September cycle截止为`2026-09-09 AoE`，当前不存在先做完下方M0—M6再写论文的时间。执行顺序
由本节覆盖；下方M阶段保留为技术依赖和长期generalization合同。

| 日期 | 唯一主要产出 | GO条件 |
|---|---|---|
| 8月27—28日 | S0：official B0、两种claim mode、full-stack raw/预算、两页rapid-review草稿、CIBC→BoundFlow changes note骨架 | 核心抽象能在前两页讲清；`u+h<0.10`；已有资产可组成单candidate |
| 8月29—31日 | S1/S2：CIBC mixed graph + coarse CROWN/custom VJP进入同一prepared执行路径 | correctness、fallback=0、累计B0直测出现实质正收益 |
| 9月1—3日 | S3：optimizer/runtime收束；根据residual ledger只开最有价值的O4/O5/O6 | exact-call累计向`>=5×`推进；不存在无法覆盖的>10%硬边界 |
| 9月4—5日 | S4/S5 formal：held-out、消融、cold/amortized、memory、worst pair、solved workload | raw冻结，headline可从raw重算，结果不依赖事后挑样本 |
| 9月6—7日 | 11页论文、图表、limitations、相关工作、匿名artifact/replay | 前两页自包含并明确推进PL/compiler pillar |
| 9月8日 | PDF/匿名/COI/AI disclosure/changes note/format最终检查，内部freeze | 不再改headline或实验协议 |
| 9月9日AoE | 提交 | 只提交polished、完整且所有claim有证据的版本 |

48小时S0未通过时，`ASPLOS’27 submission=STOP`，但10×技术路线继续，转下一合适A会；不得把未实现优化写成
ASPLOS结果，也不得因为赶deadline降低correctness、trajectory或held-out门禁。

ASPLOS’27新增只看前两页的double-blind rapid review，且官方提示多数投稿可能无法进入完整评审。前两页必须
先说明：为什么现有tensor compiler不能安全联合优化stateful verification region；BoundFlow新增了什么可复用
compiler abstraction/pass/runtime机制；实测端到端收益和局限是什么。不能用“神经网络验证很重要”或一个孤立
10×数字代替对PL/compiler intellectual frontier的推进。

投稿操作还必须完成：

- 11页双栏10pt自包含正文，references/appendix不计页数；
- BoundFlow已公开，匿名稿中使用匿名系统名与匿名镜像，PDF/附件/仓库metadata不得泄露身份；
- 准备约1页CIBC/AAAI’24→BoundFlow changes-since-previous-submission说明；
- 建立生成式AI使用清单并按ACM要求在References前完整披露；
- 从2026年8月26日至12月21日结果公布处于宣传静默期：可以开发或上传预印本，但不得在社媒、community
  blog或网页宣传投稿/相关工作，也不得向已知PC/ERC识别或推销；
- 将现有raw-first/replay/tamper/DocOps整理为匿名artifact；若录用，按AE时间线提交可执行脚本、模型、数据、
  expected output与GPU环境说明。

## 0. 直接回答用户的三个问题

### 0.1 有没有 IR 能表示 auto_LiRPA 除算子外的主流程

**从通用表达能力说，有。** MLIR SCF、StableHLO `while` 和 TIR 都有结构化循环/状态携带能力；当前
vendored TVM `6248b5db` 的 Relax function 可表达 DAG、函数调用和 `If`，VM bytecode 有 `Goto/If`，但普通
Relax/TVMScript前端没有一等、可供pass分析的`for/while`。因此默认形态是“compiled tensor step + host
policy gate”；whole-10/9只有在全内容hash replay或全部predicate显式进入受证bounded state时才可展开。TIR
负责闭合tensor region内的loop、buffer、归约和GPU schedule。

**从现成的验证语义说，没有找到一个 IR 能完整表示 production auto_LiRPA + αβ-CROWN：**

- auto_LiRPA `BoundNode` 图表示网络拓扑和逐算子 bound transformer，但不把反向 coefficient wavefront、
  α/β optimizer、best-state、split/history 和 BaB queue 都变成一等程序；
- ConstraintFlow 已经有 certifier DSL、stack-based tensor IR、图遍历和稀疏中间表示，但论文范围未覆盖
  α-CROWN/β-CROWN optimizer、完整 BaB exact-call、GPU TIR lowering 与 solver trajectory；
- Relax/SCF/StableHLO 能装下图或受限控制流，却不知道 lower/upper polarity、dense/Patches/sparse A、soundness、state version、
  saved-state/recompute legality 和 atomic commit；
- TIR 适合闭合 tensor program，不适合动态优先队列、timeout、外部 LP/MIP、Python host object 和事务回滚。

因此，方案不是再造 Plan/Task/Schedule/Transaction 多套执行 IR，也不是把整个求解器硬塞进 TIR。第一版按
四种机制严格分工：

1. **runtime/source trace** 只观察 production 调用、状态、alias、lifetime 与失败边界；它产生证据，不执行优化；
2. **strict `torch.export` + pinned AOTAutograd pair** 作为闭合 tensor region 的正式候选捕获前端；Dynamo与
   non-strict export只作discovery/baseline或全输入状态绑定的单实例replay，不能把裸FX/path witness升级成
   可复用formal语义；
3. **TVM Relax/TIR** 是唯一 production 编译执行 IR：Relax 表达图级 tensor/state SSA，TIR 表达闭合 loop、
   buffer、reduction 与 GPU schedule；
4. **一次性 BFBound/Verification 语义提升 + RVIR typed runtime** 分别补验证事实和动态求解器协议：
   `VerificationOverlayV1`只把事实绑定到最终 Relax stable ID，不拥有算子、边或第二个执行器；RVIR继续拥有
   BaB动态控制和唯一live-state publish。

> **最终结构是“capture/import + verification semantic lifting + Relax/TIR + RVIR”，而不是新造一套
> Verification Flow execution IR。闭合 region 能由 ExportedProgram/AOT 图完整捕获时优先导入；不能捕获的
> host policy 在显式 cut 处保留 host，不能根据一次 trace 手抄成貌似通用的 optimizer/BaB AST。**

对“事务是否也需要 IR”的进一步结论也收紧为：**当前不引入 transaction execution IR，也不预注册
`begin/seal/commit/abort` 编译指令。compiled module只接收readonly/private state，返回staged packet与
pending completion handle；`PreparedExecutionBindingV1`描述final executable入口，只有需要live publish时才增加
`PublishIntentV1 → PreparedPublishBindingV1`，最终由tagged `PreparedArtifactEnvelopeV1`封套。它们都是非执行
ABI/protocol；真实version/alias/validate/commit/compensation仍由RVIR state machine执行。只有multistream arena reuse、
speculative domain 等具体优化证明函数边界与sidecar无法安全表达时，才把最小 readiness/resource-lifetime
事实提升为 compiler-visible token。TIR 始终不得直接修改 solver live state。**

### 0.2 这种主流程表示在哪里产生加速

IR 本身不产生加速；它必须让编译器第一次看见并合法改变以下物理事实：

1. 把 IBP 的逐 op Python DFS 变成静态 dataflow，融合 paired lower/upper、residual 与 epilogue；
2. 把 CROWN 的反向 queue 变成静态 reverse DAG/wavefront，合并 relaxation、β injection、sign select、
   Conv/Linear adjoint、fanout accumulate、bias 与 concretization；
3. 在 dense、Patches、identity、sparse/factorized coefficient 之间按 region 选择，延迟或彻底避免 dense A；
4. 用全图 last-use、escape、alias 和 optimizer epoch 做物理 arena 复用，而不是只删除 Python 引用；
5. 用 custom backward 重算廉价中间量，只保存 compressed α/β、必要 bounds 和版本 token；
6. 把固定 10 evaluation / 9 mutation 中不变的 weight、mapping、bounds、module、view 和 workspace 提出循环，
   预分配 gradient、Adam moments 与 best-state；
7. 把 domain 的 `static intermediate base + sparse split/history delta` 保持为 overlay，避免每轮 clone/repeat
   整套 intermediate bounds；
8. 形成一次 coarse prepared invocation，消除逐 site DLPack、allocation、sync、framework crossing 和 launch。

### 0.3 中间张量会不会带来显著优化

**很可能会，而且仓库已有强证据；但不是自动成立。**

- NRIR-2 在真实 ResNet 的逻辑 lifetime ledger 上把 retain-all `1,860,912 B` 降到 `442,656 B`
  （`-76.213%`），验证了 exact-last-use、aligned first-gap、386 对合法 alias，且85个runtime value在final
  task前释放；旧 runtime
  只删除 Python env 引用，尚未形成物理 CUDA allocator claim；
- R3-1B3 的固定 P-anchor 单 evaluation 物理实测中，candidate/native allocated ratio=`0.0641686x`、
  reserved ratio=`0.166667x`，warm dynamic allocation=`0`、saved dense A=`0`；
- R3-D2B 的同 anchor、不同阶段 wrapper/native geomean=`1.752001x`。两份closure组合支持“正确ownership、
  coarse wrapper与custom backward有望同时改善内存和时间”的假设，但尚需在同一次formal中联合验证；
- 反例同样明确：B4-C2 把 dense A/autograd history 跨层保留后速度只有 `0.337–0.349x`、allocated
  `1.34011x`；R3 早期约 400 个细粒度 launch 时 wrapper 约 `0.133989x`。

所以本计划把中间张量从“后期内存优化”提升为主编译线：**representation + lifetime + rematerialization +
fusion 必须联合规划，并用真实 physical bytes/traffic/launch 验证。**

## 1. production auto_LiRPA / αβ-CROWN 实际主流程

### 1.1 版本与审计边界

仓库 vendored auto_LiRPA 当前固定在
`9d100ec070868440b48d34e2f1dd21b97aab9172`，适合做本仓源码参考；但 RVIR production 实验实际固定
外部 αβ-CROWN `e5c7e17bf0488843acb77b7519f59876717a49f4`、其 auto_LiRPA
`5a098e8f9fb5786a428a024981d833d303921f2d` 与 VNN-COMP
`90419aadcf06cf543ce5c1706cae1059dc9fa6cf`，固定点见
`scripts/run_rvir_v4_production_state_capture.py`。后续 flow capture 必须以 production exact-call 的版本为准，
不得把 vendored 旧路径误当正式运行事实。

M0 admission 还必须冻结 model/property SHA256、完整solver/config与18项optimizer controls、PyTorch/CUDA/
device/dtype/seed、runner及被观察源码blob manifest，并拒绝external repo dirty/imported-blob漂移；仅检查HEAD
不足以形成production truth。

若某个region通过PyTorch捕获进入编译链，identity还必须冻结：capture工具/版本/mode、strict或non-strict、
guards与shape constraints、graph-break/fallback inventory、被常量化的Python/static value、functionalization与
mutation output、alias/view关系、saved-tensor清单、taken-path witness及capture artifact hash。捕获模式或任一
guard变化都必须重新admit，不能把“某次导出成功”外推成一般程序语义。

production αβ-CROWN worker当前使用Torch `2.11.0+cu130`，BoundFlow环境为`2.12.1+cu132`；`.pt2`、pickle或
私有AOT图不得直接跨版本作为formal ABI。第一版只能二选一：在2.11 production worker内完成capture→
normalized Core ATen→Relax import；或在2.11立即输出版本中立的canonical ATen manifest，再由2.12消费。
identity必须绑定Torch wheel/build、ATen overload schema、自定义op registration、decomposition table、
ExportGraphSignature、call spec/pytree和dynamic-shape/range constraints。

### 1.2 IBP：静态 DAG，被 Python DFS 隐式执行

`auto_LiRPA/interval_bound.py::IBP_general` 递归访问输入节点，动态调用每个 op 的
`interval_propagate`，再把 interval/lower/upper 写回 mutable node。Conv 的典型路径是 center/deviation
两次卷积再组成 lower/upper；Linear 同样采用中心/半径和 epilogue。

这段主流程完全适合：

```text
fixed BoundNode DAG
  → Relax dataflow
  → paired interval values
  → existing CIBC BoundConv TIR
  → planned BoundLinear/residual/ReLU/epilogue lowering与fusion
```

当前CIBC只接管6个Conv，Linear仍走`IntervalDomain.affine_transformer`，ReLU/Add/Flatten仍在Python/PyTorch
graph runner；所以它是M2的正确kernel/schedule起点，不是已经完成的全17-op Relax/TIR实现。M2必须逐项接通
其余op并重验整图收益，不能再造IBP execution IR，也不能把Python mixed runner误写成全编译图。

### 1.3 CROWN：degree-driven reverse wavefront，不只是若干算子

`auto_LiRPA/backward_bound.py::backward_general` 的真实结构是：

```text
clear lA/uA
  → calculate reverse fanout degree
  → seed C at start node
  → deque-driven reverse traversal
      → β injection
      → op.bound_backward
      → bias accumulation
      → fanout coefficient accumulation
      → release lA/uA after last consumer
  → root concretization
```

它处理的不只是 Conv/Linear：还包含 A 的 representation、lower/upper polarity、fanout ownership、bias、
intermediate bound recursion 和哪些 A 必须通过 `needed_A_dict` 逃逸。普通 CROWN 已经隐式包含 last-use；
α/β 优化的困难在于 PyTorch autograd 会让 dense A 相关历史跨整次 evaluation 存活到 `loss.backward()`。

### 1.4 α/β-CROWN：固定 tensor 核心 + host stopping policy

`auto_LiRPA/optimized_bounds.py::_get_optimized_bounds` 的 production 形态是：

```text
collect/init α, β, best state, optimizer state
  → for each evaluation
      → full compute_bounds
      → update best lower/alpha/beta/intermediate
      → stop/patience/timeout checks
      → loss.backward
      → optimizer.step
      → clamp/scatter α/β
  → restore best state
```

本工程冻结 workload 中是 10 次 evaluation / 9 次 mutation，其中第10次为9次mutation后的terminal读取；
这不是算法常量，stop/patience/time、pruning、last-iteration policy、cuts/output constraints、scheduler以及
α/β启用状态均可改变控制。这里最值得编译的不是 Python `for` 字样，而是每轮重复的 tensor state
transition：CROWN forward、custom VJP、loss reduction、Adam、clamp、best-state select 和不变量复用。M0先把
它拆成`EvaluationForward`、`AOTForwardBackwardPair`、`FunctionalOptimizerStep`与`HostPolicy`四类wrapper；
前3类闭合tensor transition才有资格导入Relax，stop/prune/timeout/keep-best/restore/loop ownership默认留在
host。第一版formal执行形态是“compiled evaluation/VJP/optimizer step → host policy gate → 下一轮”，不是
无条件10/9 unroll。

whole-10/9展开只允许两种受限用途：冻结全部输入和初始state内容hash的artifact replay；或所有动态predicate
都显式进入state，并能同时屏蔽后续evaluation、Adam、scheduler、best-state与publish的受证bounded control。
否则即使model/config相同，一次“无提前退出”witness也不能证明下一domain/state仍走10/9。stop/patience/
timeout不能假装已被一个TIR kernel自动解决，也不能依据一次observed trace手写成一般optimizer AST。

更严格地说，per-step admission必须冻结optimizer超参数与param-group/step/scheduler状态，并在每轮运行时核对
pruning/preserve/active mask、keep-best owner、dense-β mask、scheduler与last-iteration policy。历史“10/9”只
是观测轨迹与replay oracle，不能当作一般optimizer语义或调用前可判定的静态签名。

### 1.5 BaB：动态 host 搜索，不作为第一阶段 TIR 目标

αβ-CROWN `general_bab` 仍执行：

```text
DomainList pop/preprocess
  → update_bounds_core / CROWN-optimized
  → branching heuristic score/candidate bounds（冻结workload使用KFSB）
  → postprocess
  → DomainList add/sort
  → timeout/domain-limit/termination
```

DomainList、history 和部分 β/intermediate 由 CPU/Python 持有，每轮存在 GPU 往返、rebuild/clone/repeat；
branching heuristic（冻结 workload 使用 KFSB）还会对候选重复执行 bound。RVIR-v4 当前替换的是
`update_bounds_core` exact-call；它在固定 ResNet2B property 0、`max_iterations=1`、1 core/6 domains/depth 1
和固定branching/config下证明了core state/trajectory及成功路径的branch、queue accounting和termination等价，
不能外推到多轮或多workload。atomic compensation只覆盖core末尾12条live α/β与局部host packet；core前
`pick_out()`和core后`add()`没有完整stage/rollback，因此尚未证明整个BaB round的失败原子性或安全retry。
它也没有把整个 BaB 编译进 TIR。

当前production core是lower-CROWN（`bound_upper=False`）。第一版capture/import admission对`bound_upper=True`必须
fail closed；若要支持双极性CROWN，需另加upper-CROWN fixture，而不能从lower-only trace外推。

## 2. 搜索结论：不是“IR或别的”二选一，而是按职责分工

| 机制/表示 | 最适合解决的问题 | 第一版判定 |
|---|---|---|
| runtime/source instrumentation | 发现真实调用、Python/object mutation、alias、sync、allocator、exception与BaB边界 | M0事实权威；只产生evidence/provenance，不作为可重写程序 |
| strict `torch.export` | 捕获闭合forward、ATen use-def、graph signature与shape/range constraints | 正式forward前端候选；原Training IR可含mutation，必须经冻结`run_decompositions`得到functional Core ATen后导入 |
| FX / Dynamo / `torch.compile(fullgraph=True)` | 对现有Python做兼容性探测、自定义backend捕获和PyTorch/Inductor基线 | discovery/baseline；裸FX不含独立guard/mutation ABI，不能获得formal import资格 |
| AOTAutograd / Compiled Autograd | joint forward/backward、partition、saved ABI与rematerialization基线 | 只接受固定API/build/decomposition/partition的forward-backward pair；不自动知道verification-specific VJP cut |
| `BFBoundModule` / `VerificationGraph` | polarity、A representation、soundness、state/effect、legality与tamper identity | 一次性semantic source/certificate；导入后只保留映射hash，不复制ATen DAG或保留第二个production executor |
| TVM Relax IR → `VMExecutable` / `runtime.Module` | 图级tensor/state SSA、函数、`If`、region组合与TIR调用，再生成可执行制品 | Relax是唯一高层production execution IR；VM/runtime module是其制品而不是另一种IR；当前vendored版本无一等structured loop，动态policy保留host |
| TVM TIR | 闭合loop、buffer、reduction、thread/memory schedule与GPU kernel | 唯一kernel execution IR；不承载动态queue、timeout、Python object或live publish |
| RVIR typed host state machine | BaB queue/branch/timeout、版本核对、异常、retry、publish与compensation | 动态solver/runtime owner；不是compiler IR |
| CUDA Graph | pointer/shape/topology稳定后的重复submission | 晚期prepared-runtime优化；不替代图语义、fusion、memory planning或事务 |
| Relax DPL / 局部e-graph | 对纯、闭合BoundRegion枚举pattern或等价表达式候选 | 先用DPL；只有rewrite顺序漏winner和候选爆炸有实证时才试e-graph，不承载effect/commit |
| 新BoundFlow DSL + 独立IR/interpreter | 研究者手写新domain/transformer的潜在作者界面 | 当前拒绝；若未来需要，只做embedded frontend并直接生成现有semantic source/Relax |
| MLIR dialect / IREE Stream | dialect、effect、SCF、async、resource/timepoint与完整progressive lowering | 只借设计；当前迁移会形成第二编译栈，不能自动继承CIBC/B4/R3/TVM ABI、cache与receipt |

外部verification系统仍决定论文边界：

| 系统 | 已覆盖 | 本项目仍需证明的差异 |
|---|---|---|
| ONNX、VNNLIB、DNNV、Marabou InputQuery | 模型、性质、约束或归约 | 不表示 optimized LiRPA / BaB 算法与GPU生命周期 |
| auto_LiRPA BoundNode graph | 网络DAG、逐op bound transformer、mutable bound cache | optimizer/BaB仍是Python循环、autograd和隐式状态 |
| ConstraintFlow | certifier DSL、stack tensor IR、traverse/while、shape分析、领域rewrite、g-BSCR | 已占据certifier IR/稀疏runtime prior art；未覆盖production αβ exact-call、GPU TIR与RVIR publish闭环 |
| Faith / GPUPoly | verification-aware GPU fusion、专用GPU抽象算法与稀疏/内存优化 | 已占据bound fusion/GPU verifier prior art；不是production αβ-CROWN exact-call compiler |
| DiffAI | differentiable abstract interpretation与训练 | 已占据宽泛differentiable abstract interpretation位点 |
| TorchLean / ACT | operator-tagged SSA/共享语义、IBP/CROWN/checking或worklist/BaB结构 | 尚未闭合本计划的αβ optimizer trajectory、TIR GPU lowering、joint lifetime/fusion与RVIR publish边界 |

对production pin源码的AST语法盘点也支持per-region而非whole-solver capture：`IBP_general`约76行、12个
`if`、2个`for`与11个attribute-write target；`backward_general`约269行、43个`if`、deque `while`、6个
`for`、24个attribute/subscript write及2个`.item()`；`_get_optimized_bounds`约572行、92个`if`、16个
`for`、4个`break`与49个attribute/subscript write，并包含timeout、autograd、optimizer与best-state restore。
这是语法复杂度/ownership证据，不是性能数字；它说明原函数不适合被一次trace/export成功就视为一般语义。

结论不是“现有IR表达不了流程”，而是：**捕获工具能发现或导出tensor程序，通用IR能表示和优化它，
verification系统拥有分散的领域语义；BoundFlow缺的是把三者无重复地绑定，并在不改变trajectory与publish
边界的前提下lower到已有TIR资产。**

### 2.1 机制选择规则

只按下面三问决定放在哪里：

1. 编译器是否必须依据该事实做fusion、reorder、rematerialization、representation或buffer reuse？若是，进入
   Relax SSA、TIR或compiler-visible sidecar；
2. 它是否主要描述动态搜索、Python/外部对象、失败恢复或对solver的可见性？若是，留在RVIR/host state machine；
3. 它是否只用于观察、归因、审计或重放？若是，留在trace/artifact，不升格为IR。

**没有compiler pass消费的对象不得命名为新IR；仅“以后可能用到”不是准入理由。**

## 3. v5目标生产架构：capture/import + semantic overlay + Relax/TIR + RVIR

### 3.1 一条 lineage

下图是**目标ownership**，不是“当前已经全部接通”的事实。`[E]`表示已有可复用资产，`[P]`表示本计划新增，
`[M]`表示已有对象需迁移/收束；任何`[P]`对象在对应M阶段closure前都不得写成已实现。

```text
production auto_LiRPA / PyTorch / ONNX
              │
              ├── [P] runtime/source observation ──→ AutoLiRPACaptureEvidenceV1
              │                                  （事实，不执行）
              │
              └── [P] closed tensor-region capture
                    ├─ strict functional ExportedProgram
                    ├─ pinned AOTAutograd forward/backward pair
                    └─ [M] BFBoundModule lowering fallback
                                  │
                                  ▼ one-shot semantic lifting
                 [P] VerificationOverlayV1
                 （Relax stable value/function ID → facts/rules；
                   无op/edge/topology，不执行）
                                  │
                                  ▼
┌──────────────── canonical TVM IRModule lineage ────────────────┐
│ [M] Standard Relax functions                                   │
│   - paired interval dataflow                                   │
│   - reverse coefficient wavefront                              │
│   - exact-signature private optimizer state SSA                │
│                         │                                      │
│                         ▼                                      │
│ [E/M] Standard TIR PrimFuncs                                   │
│   - existing CIBC BoundConv                                   │
│   - planned BoundLinear / residual / ReLU / epilogue          │
│   - relaxation / β injection / sign select                     │
│   - Conv/Linear adjoint / fanout / concretization / custom VJP │
│                                                                │
│ legality → region → representation → fusion → liveness/remat   │
│ → arena → schedule → post-schedule verify                      │
└───────────────────────┬────────────────────────────────────────┘
                        ▼
       [P] PreparedArtifactEnvelope + CompileMemoryWitness + execution binding
           （live publish时另带optional publish extension）
                        ▼
       InvocationFrame → StagedPacket + PendingCompletionHandle
                        ▼
       [E/M] RVIR typed state machine → unique live-state publish
       （BaB queue/branch/timeout仍由provider host拥有）
```

“一条 lineage”允许 imported/source/legalized/fused/scheduled `IRModule` 和 executable 分阶段独立哈希；它只
禁止同一production tensor语义再分叉到手写Flow DAG或Plan/Task/Schedule runtime interpreter。

| 对象 | 当前事实 | v5目标身份 |
|---|---|---|
| `AutoLiRPACaptureEvidenceV1` / `CaptureRegionManifestV1` | 尚未实现 | 观测、admission与provenance artifact；不执行 |
| functional ExportedProgram / pinned AOT pair | PyTorch能力存在，BoundFlow per-region bake-off尚未实现 | 正式导入输入，不是canonical solver IR；Dynamo/裸FX只作discovery |
| `BFBoundModule` / `VerificationGraph` | schema/interpreter/tests存在；BF解释器仍被legacy query调用，VerificationGraph尚未接production | 一次性semantic source/legality certificate；替换closure前保留oracle，目标artifact不保留其解释器或第二份topology |
| `VerificationOverlayV1` | 尚未实现 | 只映射Relax stable value/function ID到facts/rules；无op、edge或可执行拓扑，rewrite后重建并重签 |
| Relax | 已有interval builders、VM/cache/pass原型，尚未承载完整production region | **唯一高层execution IR** |
| TIR | 已有CIBC Conv、B4/R3等局部kernel/runner，覆盖非完整图 | **唯一低层execution IR** |
| `PreparedExecutionBindingV1` / optional publish protocol / `PreparedArtifactEnvelopeV1` | 规划对象；现有RVIR/commit类提供实例语义 | 编译后多入口执行绑定始终存在；live publish时才增加intent/binding；全部为非执行ABI |
| RVIR state machine | state/version/staged publish/compensation资产已存在，scope仍固定 | 动态runtime protocol与唯一publish authority；不是compiler IR |
| Plan/Task/Schedule及Python executors | 当前仍服务interval/typed CROWN真实路径和大量测试 | freeze/no-new-feature；按同scope replacement门禁逐项转为decision/audit/replay oracle |

### 3.2 两类 region、三层控制

只定义两个顶层语义边界：

1. `BoundRegion`：IBP、CROWN、固定 optimizer 的 tensor program，可 lower 到 Relax/TIR；
2. `SolverTransition`：由RVIR/host拥有的动态控制，但不能再把整轮含混地称为一个atomic transaction；第一版
   显式拆成`DomainReservation`（现状仍是destructive pop、尚未实现reservation）、
   `CoreStatePublishTxn`（现有12-path事务）和`QueueAppendTransition`（现状逐项append、仅成功路径已验证）。
   三者的state/effect/receipt必须分别可分析、可重放。

三层控制分别处理：

| 层 | 第一版 owner | 可迁移条件 |
|---|---|---|
| 网络 DAG：IBP forward / CROWN reverse | Relax + TIR | 立即进入主线 |
| 单evaluation/VJP/optimizer tensor step | Relax + TIR；每步后的stop/prune/keep-best等policy默认保留host | M1先闭合纯step；M4才组合轨迹，whole-10/9仅限全内容hash replay或受证bounded control |
| 动态 BaB worklist/timeout/LP/MIP | RVIR/host | 真实profile证明host边界值得且device数据结构/事务均闭合后另立阶段 |

### 3.3 最小 semantic overlay，不新造执行图或 AST

第一版不增加TIR AST node，也不建立新的Python execution graph。这里冻结一条防止“隐藏第三张图”的硬规则：

- `BFBoundModule`只在frontend/oracle阶段存在；Relax import closure后，prepared artifact最多保留它的source hash；
- `VerificationGraph`只生成legality certificate、negative evidence与一次性映射输入，不进入production artifact；
- `VerificationOverlayV1`只能保存`Relax stable value/function ID → facts/rules`，不得定义自己的op、edge、拓扑、
  interpreter或调度顺序；
- rewrite后overlay从新Relax use-def重建并重签；无法唯一映射的事实使candidate fail closed；
- `PreparedArtifactEnvelopeV1`不得持有BFBound/VerificationGraph interpreter、第二份DAG或可独立执行的sidecar。

Relax `Var`不能随意携带attrs，`call_tir` attrs也受op schema约束，因此编码合同必须分层：function/PrimFunc级
使用namespaced DictAttrs；value级role/version/representation由overlay按稳定value-id绑定。纯函数式
state-in→state-out是语义事实；物理in-place只能由liveness/alias pass插入。

value级最小语义如下：

```text
semantic_role:
  immutable_model | immutable_intermediate_bound | solver_readonly_state
  dynamic_alpha_beta | optimizer_state | persistent_best_state
  ephemeral_coefficient | module_private_scratch | staged_output

owner_version:
  query | domain_epoch | optimizer_evaluation | mutation_ordinal

representation:
  interval_pair | dense_A | patches_A | identity_C | onehot_C
  sparse_alpha | sparse_beta | factorized_A | scalar_bias | compact_status

axes:
  domain | spec | sample | node | channel | spatial

memory_effect:
  readonly | private_inplace | staged_write | external_publish | resource_lease

lifetime_contract:
  first_write | last_read | escape_scope | save | rematerialize | alignment
```

第一版使用下面的analysis taxonomy和stable semantic tag：

```text
ibp_propagate
relax
coefficient_propagate
fanout_accumulate
beta_inject
concretize
optimizer_step
select_best
stage_result
```

这些名字**不是九个新Relax opcode或新dialect**。实际计算仍由标准Relax call/dataflow、显式参数/返回值
state SSA与TIR `PrimFunc`表达；tag只帮助semantic lifting、legality和pattern定位。`Composite`是TVM BYOC
保留属性，未实现真正external codegen时禁止用作普通语义标签。只有通过§3.4.1的新IR准入门禁后，才能把
某个tag升级为namespaced Relax Op；TVM核心AST fork默认关闭。

### 3.4 为什么 attrs 不够、为什么也不需要六套 IR

attrs/sidecar 只提供分析 metadata，不能替代 effect：solver live state 必须显式作为 readonly input，candidate
只能写module-private scratch与staged outputs。普通`call_tir`采用DPS并默认产生新输出；warm allocation=0
必须由验证过的arena/memory planning/in-place lowering实现，不会由attrs自动得到。`call_tir_inplace`只有
alias/liveness pass证明module-private buffer安全后才能插入，不能直接对solver input使用。RVIR核对
identity/version/trajectory后唯一commit。

另一方面，PlanTemplate、TaskIR、ScheduleIR 的有用信息不会丢失：candidate、backend、schedule、buffer、
launch 和 receipt 都可从 `IRModule`、pass trace、measurement DB 与 witness 派生；不需要 production runtime
逐层解释这些对象。

### 3.4.1 表达升级梯子与新IR准入门禁

| 等级 | 表达机制 | 何时使用 |
|---|---|---|
| L0 | trace/artifact字段 | 只需观察、审计、归因或重放 |
| L1 | typed sidecar/witness | pass只需查询/验证，程序use-def无需改变 |
| L2 | 标准Relax SSA、参数/返回值、private function与namespaced attrs | 事实影响数据流、state transition或region边界 |
| L3 | Relax DPL/pattern rewrite | 已有真实fusion/lowering pass需要稳定匹配 |
| L4 | 新`boundflow.*` Relax Op | L0—L3无法保真，且同时通过NIR-0—NIR-5 |
| L5 | 新TIR intrinsic/AST | 现有buffer/call/schedule无法表达已证实的硬件机制；默认关闭 |
| 禁止 | standalone execution IR/interpreter | 本计划不开放 |

任何新一等IR概念必须同时通过：

- **NIR-0 concrete blocker**：给出被阻塞的具体合法变换和最小反例，不能用“以后可能需要”申请；
- **NIR-1 insufficiency**：证明标准Relax/TIR、SSA、private function、attrs、analysis map与sidecar均无法安全表达；
- **NIR-2 reuse**：同一需求至少出现在两个production-derived workload，并被至少两个独立compiler pass消费；
- **NIR-3 feasibility**：有可量化物理机会，且同scope Amdahl路由允许继续；
- **NIR-4 ownership**：lowering后该概念消失，不产生第二个runtime owner或production interpreter；
- **NIR-5 closure**：具备canonical hash、verifier、reference lowering、negative tests、replay与退役/兼容计划。

若sidecar足够却仍新增op/dialect，或新对象只被日志、receipt或一个site消费，立即STOP。

### 3.5 先定义“事务”，避免把所有 state 都误叫事务

本计划把执行对象分成四类，只有第一类需要 solver transaction 语义：

1. **外部状态事务**：一组 private candidate 在全部完成、验证和组装之前不得被 solver 观察；发布时必须核对
   版本/身份/alias，且只有一个 terminal outcome（commit、abort或补偿失败后失效）；
2. **函数式 state transition**：例如10/9 optimizer内部的α/β、Adam、best-state。只要它们留在
   `InvocationFrame`内并以SSA state-in/state-out传递，就不是外部事务；失败时丢弃整个frame即可；
3. **资源生命周期协议**：arena lease、CUDA stream/event、frame release。它们需要happens-before和完成token，
   但不等于solver状态commit；
4. **证据/工件事务**：receipt、artifact临时文件与digest/replay。它们服务审计，不进入执行IR。

因此，CROWN的`lA/uA`、TIR scratch、每步optimizer candidate和best-state select本身都不需要“事务IR”。真正
必须闭合的是**compiled region 的 staged result/state 如何唯一地发布到 live α/β、host control packet和后续
solver观察者**。

### 3.6 已有生产事务语义与规模统计

以下数字不是从设计图推测，而是从当前固定ResNet2B formal artifact、fixture与实现独立重数；它们只描述
当前实例，不进入通用schema常量。

已有schema词汇量也已独立枚举：`VerificationGraph`包含17种value role、17种op kind、8种effect resource、
4种effect access和22种fail-closed reason；production state另有19种tensor role与3种ownership。说明本轮不是
从零发明名词，而是补齐现有effect token到真实publish lifecycle之间的结构缺口。

| 边界 | 当前固定实例 | 已证明语义 | 尚未证明/不能外推 |
|---|---:|---|---|
| production core调用树 | 1 core，24个provider `compute_bounds`：initial/α/β=`12/1/11` | 调用拓扑、phase和结果可重放 | 其他模型、property、branching/config |
| optimizer私有轨迹 | 10 evaluation / 9 mutation / 第10次terminal | 逐step lower、α/β与固定Adam/clamp轨迹 | 一般early-stop、timeout、prune、scheduler、cuts/output constraints |
| 每step optimizer state | 24 tensors：6 α、6 β value、6 β location、6 β sign | 12 mutable candidate + 12结构输入 | 完整Adam m/v与所有production动态policy并未都成为通用IR |
| outer production snapshot | 62 tensors | ownership和semantic path被冻结 | “62”不是schema常量 |
| outer readonly | 16：input L/U各1、spec 1、threshold 1、intermediate L/U各6 | exact-call内只读 | 多site/upper-CROWN/aux state |
| outer copy-in | 34：α feature index 16、feature shape 6、β location/sign各6 | 候选投影所需结构状态 | 不代表所有模型都有相同布局 |
| outer mutable publish set | 12：6 α + 6 β value | staged、version guard、commit与content rollback | 并发观察者硬件原子可见性 |
| host publish set | 3个字段：`depths/history/thresholds`；冻结history entry=36 | 与12个device target联合由同一runtime调用更新 | 整个DomainList/queue并未纳入该事务 |
| actual changed set | 12条mutation receipt中7条内容变化 | changed/unchanged逐path可审计 | 不能据此删掉其余5条合同target |
| current device commit工作 | 12 candidate、12 rollback backup、12 live copy、1 host replace | stale-version拒绝、alias/placement检查、故障内容恢复 | 不是单条硬件atomic instruction；rollback不恢复`_version` |
| provider return assembly | 13个必需外部字段，完整assemble/validate后才调用commit | return packet不完整时live state不变 | 一般provider对象/所有配置 |
| BaB round | destructive pop→core→branch/prune→append | 当前固定成功路径的queue/branch/termination等价 | pop/push失败原子性、失败安全retry、完整round rollback |

62个snapshot tensor按ownership精确拆分为：

```text
read_only        = 16
copy_in          = 34
mutable_copy_out = 12
```

optimizer artifact的10个step每步恰有24个state tensor：

```text
6 α              mutable_copy_out
6 β value        mutable_copy_out
6 β location     copy_in
6 β sign         copy_in
```

这些统计揭示了**两层不同的状态边界**：10/9内部可保持private/function-state；exact-call末尾只有12个device
target和3个host字段需要对外publish。把每个optimizer step都做成外部事务，会人为制造9次以上的backup、
version check和commit，正是本计划要避免的错误设计。

### 3.7 当前实现的真实原子性与失败语义

当前最强实现不是旧CPU snapshot路径，而是
`runtime/fsg4_b3_device_atomic_commit.py`和`runtime/fsg4_b3_device_live_return.py`：

```text
static DeviceAtomicCommitPlanV1
  → snapshot 12 target versions + host pre-version
  → construct 12 private GPU candidates
  → assemble complete provider return
  → validate inventory / alias / shape / dtype / device / finite / version
  → clone 12 rollback backups
  → sequentially copy 12 live targets
  → replace host packet
  → emit commit receipt
  → synchronized query后另做content digest audit（不进headline timing）
```

必须使用以下严格措辞：

- **逻辑原子性**：在“同一live-state isolation domain由一个executor独占、外部观察者只能在lease释放后读取”
  的前提下，12个tensor与host packet表现为一次发布；
- **不是硬件原子性**：12次`copy_`与一次host mapping替换之间存在物理中间态；IR、effect token或receipt都
  不能凭空把GPU tensor和Python对象变成一条硬件atomic instruction；
- **pre-commit abort**：在第一次live write前失败，只丢弃private candidate/frame，不需要rollback；
- **compensating rollback**：mid-copy、host replace或post-check失败时，把12个tensor内容和host packet恢复到
  pre-image；
- **不是version rollback**：PyTorch `copy_`会递增`Tensor._version`，恢复backup时再次递增。因此内容可恢复，
  版本不能回到原值；失败transaction、expected-version token和frame必须全部失效，retry必须重新snapshot；
- **异步完成不能省略**：CUDA launch/copy返回不代表device work完成，且event synchronize可能暴露先前的
  asynchronous error。`commit_enqueued`不能直接冒充`committed`；跨stream、host读取或frame释放前必须消费
  completion token；
- **BaB round尚非事务**：`BatchedDomainList.pick_out()`在core前做destructive pop，post阶段的`add()`逐存储
  append，当前没有整个pop→bound→push的stage/rollback。现有RVIR证据证明固定成功路径等价，不证明失败后
  可原地安全retry。

另外，当前headline路径仍有candidate finite的`.all().item()`、commit后的`torch.equal`以及live assembly中的
`.item()/max().item()`等host scalar检查；它们可能形成隐式同步。M0必须将这些同步点与event/completion分开
归因，M1只能在等价device status + fail-closed host gate闭合后前移或合并，不能为了计时直接删除。

### 3.8 现有IR与外部机制能覆盖多少

| 机制 | 可直接借用 | 仍缺什么 |
|---|---|---|
| `VerificationEffectTokenV1` | 8类resource、read/write/read-write/external、版本前后与ordinal | publish group、stage/live绑定、exactly-once runtime terminal、isolation、completion、abort |
| `VerificationRegionV1` | region输入输出、effect边界、saved state、closed-world/postdominator | all-or-none write-set和commit authority |
| GC0 rejection registry | version/order/alias/queue boundary/receipt等fail-closed原因 | missing-await、double terminal等需在M1判断是否能无歧义复用 |
| `DeviceAtomicCommitPlanV1` | 当前12-path placement/alias/version/rollback实例 | 硬编码12、6α/6β与CUDA，不能当通用schema |
| `DeviceAtomicTransactionV1` | dynamic candidate、target version、host version与receipt | 编译pass不可见；缺显式completion/isolation token |
| legacy ScheduleIR | state load/store/invalidate、event、retry/fallback | 不回到production解释栈；只作compatibility/audit oracle |
| MLIR Memory Effects | effect resource、冲突、effect stage，可阻止非法CSE/hoist | 不表示地址级alias、all-or-none、rollback或跨host/device原子性 |
| MLIR SCF | loop-carried SSA与success/failure branch | 不提供事务隔离/提交实现 |
| MLIR Async / StableHLO token | ready/happens-before/error与side-effect顺序 | token不是undo、rollback或atomicity |
| IREE Stream | external/transient/variable resource、symbolic bytes、timepoint与安全复用 | 不拥有αβ-CROWN solver state和publish语义；只借概念，不换backend |
| PyTorch functionalization | 把中间mutation变为函数式值，末尾再fix-up input mutation | 对non-local/global state有限；不能取代RVIR commit owner |
| TVM Relax/TIR | pure dataflow、DPS、buffer/liveness与GPU kernels | Relax无resource-specific transaction；`call_tir_inplace`类型上仍视为pure，不能直接写solver state |

外部机制的共同结论是：**effect解决“哪些操作冲突”，token解决“什么时候ready”，alias/liveness解决“哪些
buffer可复用”，但三者都不自动等于transaction。** BoundFlow只需把这三类事实绑定到verification state和
RVIR publish boundary，不需要复制一个数据库或分布式事务系统。

现有GC0 schema的具体缺口也已确认：fixture里的`COARSE_COMMIT`主要消费compact status并改变抽象
`commit-state`，没有把每个staged α/β value绑定到对应live resource/version；op上的`effect_read_ids/
effect_write_ids`与token自己的`access`也尚未做完整交叉一致性检查。这正是aggregate contract/verifier要补的
地方，而不是再写一套数值执行图。

### 3.9 发布边界不做执行IR：函数边界 + prepared ABI + RVIR

当前single-core/single-stream exact-call不需要`TransactionIRModule → interpreter`。只要compiled Relax/TIR
函数满足以下硬边界，普通函数/SSA边界已经足以阻止compiler跨live publish重排：

- solver live pointer不进入module write set；
- module只读immutable/readonly snapshot，只写private scratch和staged output；
- prepared runtime返回`StagedPacket + PendingCompletionHandle`，不返回“已经ready/commit”的声明；
- publish函数不在同一个可被fusion/CSE/hoist的编译单元内；
- RVIR在函数外执行version/identity/alias/trajectory/assembly核对并保持唯一publish authority。

跨optimizer step的private state fusion/rematerialization、arena和saved-state cut属于Relax/TIR内部数据流问题，
不需要事务IR。multistream frame reuse需要readiness/lifetime依赖，speculative domains需要runtime task/isolation
协议；二者也不自动要求把commit本身编译成IR。只有某个具体compiler transformation证明函数边界、标准SSA
依赖与sidecar均不足时，才能重新通过§3.4.1 NIR门禁申请最小effect/completion token。

因此第一版只增加一组**非执行的prepared publish protocol schema**。它们可被verifier、admission、hash/
replay和RVIR runtime消费，但不被解释为数值程序，也不拥有commit/rollback实现。必须把编译前的语义意图与
最终executable的物理绑定分开，不能让pre-rewrite value ID跨rewrite/schedule后冒充live binding。

### 3.10 最小静态设计：intent、artifact identity、binding与envelope

身份构造顺序固定为：

```text
[optional PublishIntentV1, only for rvir-publish]
  + final selected Relax/TIR module + executable build
  → ExecutableArtifactIdentityV1
  → PreparedExecutionBindingV1
  → [optional PreparedPublishBindingV1]
  → tagged PreparedArtifactEnvelopeV1
```

`PublishIntentV1`是需要live publish时`compile_boundflow`的可选输入；M2 return-only不构造它。它只引用不会随
compiler rewrite漂移的source/semantic/effect identity：

```text
PublishIntentV1
  publish_intent_id
  invocation_signature_id
  semantic_region_hash
  readonly_semantic_ids[]
  private_state_semantic_ids[]
  staged_semantic_output_ids[]
  live_target_effect_ids[]
  publish_boundary_effect_ids[]
  isolation_domain_id
  publish_authority = rvir-host
  validation_policy = identity-version-alias-trajectory-before-write
  publish_policy = validate-all-then-publish-once
  abort_policy = discard-private-stage
  compensation_policy = restore-content-and-invalidate-version
  completion_policy = await-before-publish-and-frame-release
  receipt_schema_id
```

final candidate完成rewrite、TIR lowering、schedule、arena pack与build后，才产生：

```text
ExecutableArtifactIdentityV1
  selected_relax_module_hash
  tir_module_hash
  executable_artifact_hash
  target_build_hash
  compile_memory_witness_hash
  final_entry_symbols[]

PreparedExecutionBindingV1
  execution_binding_id
  executable_artifact_hash
  entries[]:
    entry_role
    final_entry_symbol
    staged_outputs[]:
      final_output_id
      arena_slot_or_storage_binding
      staged_packet_field_id
    status_output_id
    completion_dependency_slot_id

PreparedPublishBindingV1                 # only for exposure_mode=rvir-publish
  binding_id
  executable_artifact_hash
  publish_intent_hash
  execution_binding_hash
  provider_assembly_contract_hash
  compiled_staged_bindings[]:
    producer_entry_role
    staged_packet_field_id
    live_target_effect_id
    publish_group_id
  runtime_assembled_host_bindings[]:
    provider_field_id
    live_host_effect_id
    publish_group_id
    version_owner
    compensation_owner

PreparedArtifactEnvelopeV1
  executable_artifact_identity_hash
  prepared_execution_binding_hash
  verification_overlay_hash
  compile_memory_witness_hash
  exposure_mode = return-only | rvir-publish
  publish_protocol_ref:
    return-only  → none
    rvir-publish → publish_intent_hash + prepared_publish_binding_hash
```

- semantic role/representation/axes来自一次性semantic lifting，execution binding只使用selected executable的
  stable entry/output/storage ID；entry是数组，可同时表达evaluation/VJP/optimizer多个入口；
- resource、access、pre/post version来自live effect identity；alias/lifetime从最终Relax/TIR和arena analysis重算；
- actual semantic path、tensor pointer、PyTorch `_version`、CUDA event和host object只在RVIR adapter/
  `InvocationFrame`动态绑定；`completion_dependency_slot_id`只标识prepared runtime返回槽，实际handle每次调用生成；
- M2的`return-only` envelope不得携带publish hashes；`rvir-publish`必须同时携带intent与binding，且tensor target和
  host packet target（当前实例含3个host字段）位于同一publish group/receipt边界；
- 对`rvir-publish`，`compiled_staged_bindings.live_target_effect_id`与
  `runtime_assembled_host_bindings.live_host_effect_id`必须互斥，二者并集必须恰等于
  `PublishIntentV1.live_target_effect_ids`；缺失、重复或额外target全部fail closed；
- external可观察alias必须exact；compiler内部alias、arena slot与AOT saved set允许变化，但须满足final binding、
  VJP、escape与lifetime门禁；
- schema中禁止出现ResNet2B、固定node ID、`12/6/[6,1]`或B4-C0/C1/C2常数。

每个canonical projection都排除自身hash。intent不知道未来artifact；execution/publish binding单向绑定final
executable；tagged envelope按mode绑定必需字段、overlay与memory witness，杜绝循环hash和stale binding。prepared runtime
启动compiled region后只返回`PENDING_COMPLETION_HANDLE`；它不是“ready”或“commit”证明。只有RVIR await成功
并完成version、trajectory、provider-return、alias/status核对与真实publish后，才能产生commit receipt。

### 3.11 RVIR runtime状态机与唯一可见点

`PreparedArtifactEnvelopeV1`与`InvocationFrame`动态绑定后，RVIR runtime状态机冻结为：

```text
PREPARED
  → OPEN              snapshot versions + acquire exclusive isolation lease
  → LAUNCHED          pure Relax/TIR writes private/staged resources only;
                      pending completion handle registered
  → DEVICE_READY      RVIR awaited handle; async errors surfaced
  → SEALED            staged set complete; no more private mutation
  → VALIDATED         identity/version/alias/numeric/status/trajectory pass
  → ASSEMBLED         complete provider return exists
  → COMMITTING        live writes may begin; lease still exclusive
  → COMMITTED         device completion + host publish + receipt; then release lease
```

终止边：

```text
prewrite stale-version failure → CONFLICT
  no live mutation; old frame invalid; only resnapshot may retry

other OPEN..ASSEMBLED failure → ABORTED
  discard staged resources; await outstanding event before reuse

COMMITTING failure → COMPENSATING

COMPENSATING success → ROLLED_BACK_INVALIDATED
  restore contents/host pre-image; versions remain monotonic; no old token/frame reuse

COMPENSATING failure → POISONED
  isolation domain remains quarantined; no retry/read until explicit rebuild or operator recovery
```

每个frame必须在`COMMITTED / ABORTED / CONFLICT / ROLLED_BACK_INVALIDATED / POISONED`五个terminal中恰好进入
一个；`CONFLICT`不能折叠成普通abort，`POISONED`也不能伪装成已成功rollback。这个状态机是**typed runtime
protocol，不编码成Relax控制流，不由compiler执行，也不预留`bf.txn.*` opcode**。M1只关闭schema与纯verifier；
transition dispatcher必须先分类prewrite stale-version，generic failure分支不得再捕获同一事件。
上述runtime conformance、fault injection和exactly-one动态证明属于M5。未来若异步frame复用只需compiler理解
completion/lifetime，则只提升对应依赖token；commit、abort和compensation仍由RVIR runtime拥有。

### 3.12 lowering与owner分工

```text
Standard Relax + VerificationOverlayV1
  - Relax表达纯bound dataflow、private optimizer SSA、staged packet
  - VerificationOverlayV1约束合法rewrite；rewrite后按Relax use-def重建
                │
                ▼
TIR PrimFunc
  - 只读immutable/solver view
  - 只写private scratch或staged destination
  - 不知道Python host packet，不执行live commit/rollback
                │
                ▼
PreparedArtifactEnvelope / InvocationFrame
  - final binding、arena lease、pointer、stream/event、dynamic versions、isolation lease
                │
                ▼
RVIR host
  - validate/seal/assemble
  - 唯一commit authority
  - compensation、receipt和solver return
```

迁移时：

- `VerificationEffectTokenV1`继续作为单resource版本转换，不另建平行Effect IR；
- `DeviceAtomicCommitPlanV1`是当前backend/signature下prepared publish protocol的runtime binding/oracle，
  不是compiler lowering出的新执行IR；
- `DeviceAtomicTransactionV1`继续作为动态runtime instance，不升级为编译IR；
- `StateLoad/Store/Invalidate`和event Schedule action只保留derived audit view；
- optimizer.py/parametric_optimizer.py保留为迁移oracle，最终归一到Relax显式state SSA；
- receipt只证明execution，不参与compiler rewrite。

### 3.13 publish合同能解决什么、不能解决什么

| 能解决/开放 | 机制 |
|---|---|
| 证明compiled module不写live state、publish boundary不可被跨越 | pure function boundary + staged/live disjoint + external owner |
| private in-place与arena复用 | staged/live disjoint、last-use和completion proof |
| 把静态检查移出warm hot path | prepare时冻结target/alias/shape/dtype/rollback coverage；运行时为`O(N_targets)`遍历、每target做`O(1)` identity/version guard，并以device-reduced status避免内容扫描 |
| 直接生成provider layout | TIR写staged provider layout，减少dense terminal→projection materialization |
| 让runtime commit策略进入合法候选集 | 合同提供escape/alias/isolation事实；策略仍由runtime实现和验证 |
| 让multistream/speculative child进入资格审查 | 合同提供completion HB与独立frame/write-set；不负责调度或执行 |
| 泛化固定12-path实现 | schema按stable value/effect binding，runtime动态解析signature/path |
| 统一PyTorch/TVM候选 | 两者返回同一staged packet并服从同一RVIR publish contract |

| 不能解决 | 原因 |
|---|---|
| 让数学kernel自动变快 | IR只开放合法变换，仍需fusion/schedule/representation实测 |
| 产生跨GPU tensor和Python对象的硬件原子指令 | 必须依赖exclusive isolation、event和runtime publish protocol |
| 让effect token自动等于rollback | token只有依赖/版本语义；compensation仍是runtime动作 |
| 无条件删除12份backup | 只有pointer/epoch indirection且所有reader遵守新owner时才可能 |
| 把动态BaB queue自动编入TIR | queue/timeout/LP/MIP仍是host控制；需另立失败原子性和device data-structure阶段 |
| 用opaque side-effect call获得更多优化 | opaque call通常缩小优化空间，因此纯compute和publish必须分离 |

合同只提供资格、边界和可复核identity，**不选择、不调度、不执行**commit strategy、multistream或speculation；
这些机制仍须各自通过并发、故障和wrapper-inclusive性能门禁。

已有性能事实也说明合同只能看**整体**：B2→B3把atomic commit wall从约`73.295 ms`降至
`22.476 ms`（约`3.26x`），但typed pre-state从`24.412 ms`增至`58.284 ms`；最终core只有
`1.071617x`、query只有`1.006623x`，相对B0仍是`0.910001x`。所以不能只优化/汇报commit子阶段；formal
必须同时覆盖pre-state、stage、compute、assembly、commit、completion和return。

### 3.14 publish协议分阶段准入

`PUB0`（M0内，只读）：冻结真实read/private/staged/live sets、version、alias、isolation、completion、abort、
compensation和BaB queue非原子边界；不加执行IR。

`PUB1`（M1内）：生成`PreparedExecutionBindingV1`、tagged envelope及optional
`PublishIntentV1/PreparedPublishBindingV1` schema与纯verifier；用当前12-device + 3-host fixture验证通用投影和schema负向
规则，但不虚构尚未build的executable hash、final output ID或动态completion handle，也不修改RVIR runtime。

`PUB2`（M2—M4）：每个standalone selected executable产生artifact identity、multi-entry execution binding与
`return-only` envelope；只有明确为M5准备live publish的artifact才额外产生intent/publish binding。重编译/replay
从final module重算并拒绝stale binding；此阶段只运行private/staged输出，不写solver live state。

`PUB3`（M5）：RVIR接入后关闭await、stale conflict、precommit abort、midpublish compensation、compensation
failure/poison、exactly-one terminal和receipt identity的runtime conformance。

`PUB4`（条件开放）：只有具体multistream/async/speculative优化需要compiler-visible completion/lifetime，且
标准SSA与sidecar不足时，重新通过NIR-0—NIR-5申请最小token；不得自动开放transaction opcode或interpreter。

`PUB5`（性能关闭）：wrapper-inclusive formal必须证明由协议/实现变化解锁的优化改善pre-state+stage+commit
整体；否则停在runtime ABI/protocol，不升级“transaction IR”论文claim。

PUB1 schema-negative至少覆盖：

- missing/duplicate target、orphan staged value、unknown effect、write version不变化；
- READ token被放进write set、write readonly state、queue/termination effect被跨越；
- staged/live alias、两个active frame共享writable arena、dense A错误逃逸成staged/publish output；
- return-only携带publish ref、rvir-publish缺intent/binding/provider assembly、entry/field/effect映射不全；
- executable hash、execution binding、publish binding与envelope的hash方向错误或stale final ID；
- terminal枚举缺`CONFLICT/POISONED`、重复terminal ID或非法静态transition表。

PUB3 runtime-negative至少覆盖：

- commit早于stage/await、跨stream无wait、event前release；
- double terminal、abort-then-commit、conflict后写入、无terminal outcome、post-abort use；
- stale version必须只进入`CONFLICT`，rollback后复用旧expected-version/frame必须拒绝；
- final evaluation后多一次mutation、select-best发布错误state；
- 第N个device write、host replace或postcheck失败时的content compensation，以及compensation failure进入poison；
- intent/artifact/execution-binding/publish-binding/envelope/runtime-transaction/receipt全重签篡改仍被语义重算拒绝。

## 4. auto_LiRPA 中间张量与生命周期全景

| 类别 | 代表张量 | 当前隐式行为 | 编译机会 |
|---|---|---|---|
| 模型静态 | weight/bias、topology、spec layout | 跨query重复绑定或检查 | prepare一次、module/cache identity |
| 中间 bounds | 每node lower/upper、reference/intermediate | cache、clone、repeat、按history/refined bounds恢复 | dense或base+COW/override候选、只pin必要escape |
| CROWN coefficient | lA/uA dense/Patches/identity/OneHot | reverse fanout累加，last consumer后删除；autograd时历史存活 | 表示联合选择、late materialize、arena/ping-pong |
| relaxation | slope、bias、mask、unstable index | 按层构造并进入autograd saved state | kernel内重算或compact保存 |
| α/β | α slope、SparseBeta value/location/sign | β可能先scatter成dense contribution | compressed直接消费、density crossover |
| optimizer | grad、Adam m/v、loss、scheduler | 每次optimized call创建/更新 | persistent versioned buffers、fused epilogue |
| best snapshots | best lower/α/β/intermediate | 多轮clone并在末尾恢复 | device-side select、copy-on-improve、compact delta |
| autograd saved | dense A相关graph、ctx-held tensors | 可跨整次evaluation存活 | coarse custom VJP、saved dense A=0 |
| BaB domain | lA/lb/ub/α/β/history/split | CPU store↔GPU rebuild/copy | hot frontier、ragged pack、sparse overlay；host仍拥有队列 |

### 4.1 CompileMemoryWitness 与 ExecutionMemoryReceipt

静态/编译事实与运行时观测必须拆开，避免编译时未知字段和hash自引用。

`CompileMemoryWitness`是**非执行**编译证明，至少绑定：

- value/buffer role、稳定value-id、producer/consumers、version与representation；
- symbolic/concrete use-def、first-write/last-read、escape与选择的retain/remat/materialize策略；
- `SavedStateLedger`：每个entry的role、logical/unique-storage bytes、version、pin interval、frame lease/
  release event，以及saved-state cut、rematerialization recipe/FLOPs；
- arena offset/size/alignment/storage scope、predicted peak-live/high-water和合法alias pairs；
- source/fused/scheduled module、target、pass和schedule identity。

`ExecutionMemoryReceipt`绑定witness hash与invocation/state identity，记录：

- actual explicit-arena high-water；
- CUDA allocated/reserved、profiler-observed HBM、copy/contiguous、CPU↔GPU bytes；
- DLPack、submission、kernel、sync、framework crossing、pointer/stream/status/commit outcome；
- saved dense-A count/bytes、frame lease/release，以及ctx/module/registry动态tensor持有检查；
- attributed/unattributed allocator bytes、kernel time与launch count。

canonical projection排除对象自身hash；runtime receipt单向绑定witness hash。逻辑ledger、显式arena、CUDA
allocated/reserved、capture private pool与profiler HBM必须分开报告，不能互相替代。`bytes × lifetime`的单位在
每个artifact中冻结为byte-op或byte-epoch，不与wall-time积分混用。

### 4.2 物理正确性合同

```text
byte_overlap(u, v) ⇒ lifetime_disjoint(u, v)
```

唯一例外是显式允许的 module-private in-place，且旧值无 live-after-write。单stream可用顺序interval；开放
multistream后必须以event/correlation构造happens-before，不得继续假设线性时间轴。

```text
solver_readonly_state ∩ module_write_set = ∅
staged_output ∩ solver_input_storage = ∅
```

rematerialization legality只要求pure/deterministic、输入version未变、不读host mutable object、数值/sign/
trajectory等价；profitability再比较额外FLOPs、launch、HBM、saved bytes和wrapper，不把性能失败误写成语义
非法。

Autograd 强制：saved dense A bytes/count=`0`；Python ctx普通字段必须tensor-free，但
`ctx.save_for_backward`可保存合同批准的compressed α/β、bounds/checkpoint，全部进入SavedStateLedger；
另一种合法方案是显式pin住InvocationFrame lease直到backward。module handle不得间接持有本次dynamic tensor。

### 4.3 Prepared runtime 必须拆开三种生命周期

```text
PreparedArtifactEnvelope(executable identity, execution binding, optional publish extension,
                         immutable storage, arena pool/layout, cache)
InvocationFrame(exclusive arena slice, borrowed dynamic views, stream/event/version, staged outputs)
AutogradToken(static key, compact saved-state descriptor or pinned-frame lease)
```

static weight/map只能零拷贝借用或单份拥有，unique-storage bytes必须入账，禁止Torch/TVM双份常驻。每个
InvocationFrame取得exclusive arena slice；并发frame不得alias。只有commit/rollback结束且GPU event完成后才可
释放/复用；rejected frame的staged区域必须失效或清理，不能被下一次读取。

这会修正现有部分实验executor把全部输入放在`self.tensors`、再把executor放进`ctx`的危险模式。旧实现继续
作为oracle，不作为新production ownership模板。

## 5. 研究创新应落在哪里

### 5.1 Capture、semantic lifting 与 region formation

runtime/source trace提供dynamic control、state、effect、alias与lifetime事实；`torch.export`/AOTAutograd提供
闭合tensor use-def和forward/backward候选图；它们先按M1门禁导入唯一Relax图，再由BFBoundModule/
VerificationGraph进行one-shot lifting，把fanout、polarity、A representation、soundness与state/version事实映射到
Relax stable ID。trace不得直接充当canonical tensor程序，export/AOT图也不得独占verification semantics，
BF/Verification图也不得保留为第三个production topology。region formation
不只看相邻NN op，而看coefficient producer/consumer、fanout join、concretization sink与VJP cut。

### 5.2 A 表示、lifetime、rematerialization 联合规划

这不是“总用稀疏”：ConstraintFlow 的 g-BSCR 已说明不规则稀疏很重要，也披露了通用表示与运行时固定
开销。BoundFlow 应在每个真实 GPU context 比较：

```text
dense ↔ Patches ↔ identity/OneHot ↔ sparse/factorized
retain ↔ structured-retain ↔ rematerialize ↔ dense-materialize
```

目标成本是 HBM bytes + framework crossing + launch + transfer + remat compute；soundness、tightness、
α/β trajectory、alias、escape和commit是无限代价硬约束。通用 ML compiler 无法从普通 NN graph 推导这些
verification-specific 信息。

### 5.3 Lazy/COW domain overlay（待证伪假设）

候选表示不是简单算术相加，也不预设child变化必然稀疏：

```text
immutable/static intermediate base
  + split/history sparse override or copy-on-write delta
  + unstable/refined per-domain overlay
  + optional reference/aux/cut/clip effect
  + domain/state version
```

M0先统计真实child的changed-node、changed-element和changed-bytes density。只有所有非空来源均被保真表达、
多组真实child通过tightness/correctness，且lookup、transfer、peak与latency都过冻结门禁，compiled kernel才直接
消费overlay；否则回退dense per-domain storage。未支持的reference/aux/cut/clip来源必须fail closed，不能
静默丢掉。它是representation/lifetime贡献中的一个候选机制，不在实现前单独升级claim。

### 5.4 Coarse differentiable CROWN region

融合 ReLU relaxation、compressed β injection、sign select、Conv/Linear adjoint、bias、fanout和concretization；
forward/backward各为一个coarse prepared entry，dense A只作为kernel scratch/recomputed value。已有B4-B2、
R3-1B与R3-D2提供数学、TIR、arena和custom VJP基础，不复活B4-C2 dense-retention。

### 5.5 Optimizer state compilation

默认per-step region中：

- 优先导入M0完整捕获的evaluation、VJP与optimizer-step子图；只有捕获证据证明无法保真时才手工lower，禁止
  同时维护exported graph和手写optimizer DAG两条production owner；
- hoist weight、mapping、fixed intermediate bounds、module/view/cache；
- 预分配α/β grad、Adam m/v、best lower与compact best state；
- fuse loss reduction、best select、Adam update、clamp与compact status；
- 每步后把stop/prune/timeout/keep-best/restore/scheduler/last-iteration policy交还host；18项controls与一次
  “无提前退出”witness不能把未来调用静态化；
- admission绑定α/β、Adam param-group/step/lr、pruning preserve mask、dense-β mask、scheduler和policy input；
  whole-10/9只按§8.5的全内容hash replay或受证bounded predicate准入；
- 比较host policy loop、受限Relax显式展开与compiled VM；compiled mode编译的是VM调度/寄存器操作，kernel calls仍可
  分立，不等于把optimizer自动合成一个GPU kernel；只有设备内约束闭合才实验bounded TIR loop；
- stop/patience/timeout初期仍由host或active predicate处理；predicate必须同时屏蔽state mutation、best-state、
  Adam、scheduler和非法后续tensor计算，不能只把loss乘零。

### 5.6 KFSB batch 与 GPU hot frontier（后期开门）

profile证明值得后，可将top-k候选打成domain batch，GPU上完成candidate score与compact branch packet；再尝试
保留一个device-resident hot domain window，减少DomainList每轮CPU↔GPU搬运。host仍拥有priority semantics、
termination与commit，不能先写并行实现再解释顺序。

### 5.7 Prepared publish 与 RVIR exact-call runtime

所有compiled region只读versioned solver state，写staged result/state packet；RVIR检查version、module、trajectory、
provider return与completion后，对core-owned12-path α/β和局部host packet执行一次逻辑publish。publish前失败
直接abort frame；mid-publish失败做content compensation并使旧version/frame失效。这里直接继承RVIR-v4，
不新造solver runtime，也不把DomainList pop/add误写成已有rollback保障。

## 6. 既有成果如何进入新主线

| 既有成果 | 冻结事实 | 新主线定位 |
|---|---|---|
| `BFBoundModule` | typed value/op/spec/domain/polarity/αβ与hash | one-shot semantic source/oracle，映射完成后prepared artifact只保留source/overlay hash |
| GC0 schema/22 reasons | generic effect/VJP/legality与tamper语料 | legality rules，不形成第二张production graph |
| CIBC Conv | 6 Conv geomean `12.7951x`、worst `9.1423x` | IBP TIR implementation与schedule winner |
| CIBC full IBP graph | geomean `2.45631x`、worst `2.45091x`，含input copy | M2 pipeline传播baseline |
| B2-1—B2-4 | dense/sparse Linear/Conv forward/VJP correctness | CROWN PrimFunc correctness library |
| B4-B2 v2 | P-anchor 1F+1B geomean `4.89834x`、worst `4.68601x` | coarse differentiable region局部winner |
| NRIR-2 storage plan | 逻辑arena `1,860,912→442,656 B`，`-76.213%` | 复用last-use/first-gap/alias算法，lower成physical arena；旧Plan只留oracle |
| R3-1B3 | allocated/reserved ratio `0.06417x/0.16667x`，saved dense A=0 | compile witness/execution receipt/physical arena/custom VJP基线 |
| R3-D2B | P-anchor wrapper/native `1.752001x`、worst `1.724843x` | mixed CROWN module组合oracle |
| B4-B3 | 10/9、terminal/state max diff `3.5762787e-7` | optimizer state-transition oracle |
| MR5/MR6 | generalized Conv正确，但per-site wrapper失败 | 必须coarse region、消除bridge的反面证据 |
| RVIR-v4 | 固定ResNet2B property 0、1 core/6 domains/depth 1、`max_iterations=1`下core state/trajectory与成功路径branch/queue/termination等价；12-path core publish支持content compensation | host exact-call typed ABI和唯一core publish authority；DomainList pop/add失败原子性、多轮/多workload仍待验证 |
| artifact/replay/tamper | raw-first与fail-closed纪律 | 复用纪律、manifest schema与harness模式；M0—M6仍须分别实现本阶段的semantic replay，不能宣称存在一个可直接套用的统一框架 |

关键数字的仓库证据入口固定如下，避免后续只引用本计划的二手汇总：

- CIBC整图：`gemini_doc/BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md`；
- B4-C2与R3：`gemini_doc/BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`、
  `gemini_doc/BOUNDFLOW_R3_1B3_FIVE_FRESH_FORMAL_CLOSURE_2026_08_25.md`、
  `gemini_doc/BOUNDFLOW_R3_D2B_WRAPPER_TIMING_FORMAL_CLOSURE_2026_08_26.md`；
- production bridge：`gemini_doc/BOUNDFLOW_MR3_P_PRODUCTION_BRIDGE_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md`、
  `gemini_doc/BOUNDFLOW_MR5_MULTI_SITE_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md`与
  `gemini_doc/BOUNDFLOW_MR6_GUARD_ATTRIBUTION_FORMAL_NO_GO_CLOSURE_2026_08_26.md`；
- selected-CROWN share：`gemini_doc/BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_CHANGELOG_2026_08_06.md`及
  `artifacts/nrir49a-g1-gpu-attribution/resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/`；
- artifact路径是各closure的冻结raw入口；新阶段不得把历史artifact直接当作新语义replay实现。

### 6.1 必须保留的 NO-GO

| 失败 | 新路线禁止重犯 |
|---|---|
| B4-C0/C1 `0.940/0.948x` | native+candidate双执行、过早materialize |
| B4-C2 `0.337–0.349x`、allocated `1.34011x` | dense A/autograd history跨层保存 |
| R3早期约400个launch、wrapper `0.133989x` | 逐节点解释执行compiled kernel |
| R3-3 active-β `0.668275x`、incremental allocated `10.9375x` | 稀疏语义不等于稀疏物理schedule快 |
| MR3 single-site bridge geomean `0.979727x`、worst `0.916094x` | 单个正确且局部快的site不能证明production bridge有收益；必须算完整outer wrapper |
| MR5/MR6 `0.834407x`、30/27 launch、540个代码计数DLPack/pointer round-trip | per-site framework bridge与热路径guard/crossing累积 |
| MR6 `1.033126x`、相对provider `0.903007x` | 只删guard，不删crossing/materialization/launch |
| NRIR49A selected-CROWN queue/complete share=`7.0986%/7.0523%`，无限加速上限约`1.0764x` | 只优化selected-CROWN却外推到`1.20x` queue；必须重新做full-stack同scope归因 |
| IR-5 regret `1.26160x` | 没有真实crossover时宣传通用planner |
| JIT break-even高于目标Q | 在线JIT作默认；优先AOT/cache |
| CIBC对真实activation-BaB exact-call `0/51` eligible | IBP图收益冒充CROWN/query收益 |

这段时间的工作没有白做：kernel、custom VJP、arena算法、state ownership、same-solver ABI、失败归因和证据
系统全部进入新主线；退出production的只是重复执行容器与逐site旁路。

## 7. 编译 pipeline 与唯一运行入口

### 7.1 pass 顺序

```text
establish runtime/source ground truth
  → run per-region export/Dynamo/AOT capture bake-off
  → select only complete tensor graphs; cut unsupported host policy explicitly
  → import selected graph or BFBoundModule lowering into standard Relax
  → attach BF/Verification semantic overlay
  → semantic legality / soundness
  → if rvir-publish: PublishIntent checks
    else return-only: assert no publish protocol ref
  → form closed regions
  → initial semantic escape + versioned liveness
  → legality-prune bounded representation/fusion/remat candidates
  → for each exact-signature candidate
      → rewrite candidate
      → re-infer use-def/escape/liveness
      → bind/lower TIR concrete buffer signatures
      → schedule + derive workspace/storage scope
      → candidate arena pack + happens-before/alias verify
      → build and collect measurement receipt
  → select by measured wrapper/device/bytes/launch cost
  → persist selected + rejected-candidate ledger
  → final selected-artifact reverify/cache/prepare
  → resolve final entries/output/storage IDs into PreparedExecutionBinding
  → if rvir-publish: resolve tensor + runtime-assembled host fields into PreparedPublishBinding
  → seal tagged PreparedArtifactEnvelope
```

representation、fusion和rematerialization相互耦合，只枚举真实可执行的有界候选，不能伪装成一次不可回看
的线性pass。rewrite会改变use-def，schedule才决定concrete workspace、HBM、launch与latency，所以必须为
exact signature实际lower/schedule/measure后选择；若新物理成本改变排序，只允许预注册次数的闭环重选，并
保留所有拒绝candidate ledger。basic arena不能拖到项目末期，也不能用CUDA Graph掩盖wrapper问题。

### 7.2 API

```text
capture_boundflow(
    production_source,
    invocation_signature,
    capture_policy,
) -> CaptureBundle + CaptureDecision

compile_boundflow(
    tensor_program,
    verification_overlay,
    publish_intent_or_none,
    target,
    runtime_context,
) -> PreparedBoundModule

PreparedBoundModule.run_region(entry_role, readonly_dynamic_state_view)
    -> StagedPacket + PendingCompletionHandle + RegionExecutionReceipt

RVIR.await_validate_publish(
    staged_packet,
    pending_completion_handle,
    prepared_publish_binding,
    expected_versions,
)
    -> BoundResult + CommitReceipt
```

最小 identity 链：

```text
source/config + capture artifact/guard/decision hash
  → model/spec + BF/Verification semantic-overlay hash
  → imported/source/legalized/fused/scheduled Relax IRModule hashes
  → representation + CompileMemoryWitness + TIR schedule/pass hash
  → optional PublishIntent + executable/cache identity
  → PreparedExecutionBinding + optional PreparedPublishBinding + tagged PreparedArtifactEnvelope
  → dynamic state/version + ExecutionMemoryReceipt/execution receipt
```

每个对象的canonical projection都排除其自身hash；runtime receipts只单向绑定compile witness与artifact identity。

旧Plan/Task/Schedule可从这些事实派生为compatibility/audit view，不在热路径执行。

## 8. 技术依赖阶段（受A.6 ASPLOS sprint调度约束）

M0—M6从v6起表示**技术依赖包**，不再表示未来13天必须串行完成的日历。ASPLOS sprint只执行形成最终
cumulative candidate所必需的最小子集：S0先做official B0/full-stack attribution和两页论文骨架；S1—S3
直接复用已有CIBC、B4-B2、R3-D2B、RVIR与GC0资产形成同一执行路径。通用capture bake-off、广泛op coverage
和新schema若不在critical path上，不得抢占投稿时间。

具体映射为：

- S0只执行M0-A中建立同scope B0和速度预算所需的观测，不先完成所有capture工具对比；
- S1优先走已有BFBound/GC0语义与现有Relax/TIR builder的确定性lowering，只有production region确实缺失时才
  启用M0-B/M1-A的局部capture/import；
- S2/S3组合M2—M5已验证部件，先形成一个正确的coarse cumulative candidate，再补generalization；
- S4/S5对应M6 formal、paper和artifact；未进入最终candidate的实验性分支只作为negative result/limitation。

这样既不废弃v5的长期设计，也避免在ASPLOS截止前从头重做一套前端。

```text
M0-A production runtime/source ground truth
  → M0-B per-region capture bake-off
  → M0-C CaptureDecision + region manifest
  ↓
M1-A selected tensor graph import
  → M1-B verification semantic lifting
  → M1-C native/capture/Relax三方语义闭合
  → M1-D prepared publish schema/verifier closure
  ↓
M2 IBP/CIBC mixed Relax-TIR + physical arena
  ↓
M3 P-anchor coarse CROWN forward/custom VJP
  ↓
M4 per-step optimizer + host policy轨迹；受限whole-10/9 + active-β/domain overlay候选
  ↓
M5 RVIR-v4 same-solver exact-call
  ↓
M6 held-out complete-query / queue formal与证据驱动扩展
```

### 8.1 M0：ground truth + per-region capture bake-off，不先优化

M0交付三个非执行artifact：`AutoLiRPACaptureEvidenceV1`、`CaptureBakeoffReportV1`与
`CaptureRegionManifestV1`。admission先核对§1.1的完整commit/model/property/config/environment/
code-revision/capture-tool链。

#### M0-A：runtime/source ground truth

使用source hooks、profiler/NVTX、allocator/autograd hooks捕获：

1. standalone `method=IBP`完整DAG；
2. optimized-CROWN内部由`check_prior_bounds/compute_intermediate_bounds`触发的局部IBP/递归CROWN、sparse C、
   reference/aux bounds与delete-after-use路径；
3. gradient-active的P-anchor与S-anchor evaluation，含loss seed、incoming adjoint、dα/dβ和必要incoming-A梯度；
4. 9次mutation后的第10次terminal no-grad读取；
5. 完整BaB preprocess→solve→branching-consumption→postprocess边界，并在branching heuristic消费/清空前捕获
   六组lA；不能只从return packet倒推；
6. `DomainList.pick_out`每个destructive pop、core private stage、12-path publish、host packet replace、
   postprocess逐项append及其版本/异常边界，明确哪些只是成功路径transition、哪些具有abort/compensation。

#### M0-B：capture bake-off

不预设一个工具覆盖整个solver。先把production路径函数式拆成四类wrapper：

```text
EvaluationForward(explicit tensor state) → lower/loss/VJP-required outputs
AOTForwardBackwardPair(primal state, incoming tangents) → outputs + gradients
FunctionalOptimizerStep(α/β, grad, Adam m/v/step/lr) → new α/β/m/v/step
HostPolicy(stop/prune/timeout/keep-best/restore/loop ownership) → host transition
```

前3类分别在standalone 17-op IBP、P-anchor CROWN、S-anchor active-β上测试；whole-10/9只作trajectory/replay
coverage，`update_bounds_core`与BaB只用于确定host cut。候选固定为：

- formal forward：`torch.export(strict=True)` → `run_decompositions(frozen_table)` → functional Core ATen；
- discovery：`torch.export(strict=False)`；它不能形成可复用formal region，只能帮助缩小wrapper，或在输入/初始
  state内容digest全绑定时形成单实例replay；
- discovery/baseline：Dynamo `fullgraph=True` + BoundFlow capture backend；裸FX graph不进入formal importer；
- formal backward候选：固定Torch build、AOT API、decomposition与partition的AOT forward/backward pair；
- fallback：现有BFBoundModule/BoundNode semantic lowering，与production tensor trace逐项对账。

strict Export artifact必须绑定frozen decomposition table hash、ExportGraphSignature、call spec/pytree、state/
constants digest、range constraints、inline assertions、dynamic-shape spec及每个mutation output kind/target；normalization
后mutable ATen op count必须为0。AOT artifact另绑定joint graph、partition function/config、partitioned forward/
backward graph、primal/tangent/gradient positional mapping、requires-grad、grad/autocast mode、`create_graph=False`、
saved-output→backward-input ABI、`None`/zero gradient规则。

每个候选还冻结captured/expected op/value/output coverage、HOP子图、custom/Python/opaque op、graph break、silent
fallback、alias/view和host-object escape。外部可观察input-input/input-output/publish alias与mutation必须exact；
source internal alias/lifetime只作legality输入，优化后无需storage identity相同；AOT saved ABI绑定partition artifact，
允许相对native改变，但必须保持VJP/escape/lifetime合法并最终满足saved dense A=`0`。re-export determinism按剔除
node name/stack metadata的canonical semantic hash，不要求`.pt2`字节相同。

M0-A instrumentation和M0-B export在独立fresh process运行，并与无观测native oracle对账，避免profiler/hash/
saved-tensor hook的同步改变time-based policy。M0不形成性能claim，但记录capture/build成本供JIT/AOT决策。

#### M0-C：逐region选择

runtime trace是scope discovery和事实权威，不直接变成tensor程序；完整capture artifact提供tensor use-def；
BFBound/GC0提供验证语义。每个region独立选择，不要求同一种工具统治IBP、CROWN、VJP和optimizer：

- 未解释graph break/fallback、mutation/alias/saved-state不完整：缩小region或fail closed；
- importer需要根据一次trace手工复制production control logic：STOP；
- formal reusable region只接受strict functional ExportedProgram、pinned AOT pair或逐项对账的BFBound lowering；
  non-strict/Dynamo只作discovery，或全内容hash绑定的单实例replay；
- 动态BaB、queue、timeout、LP/MIP、Python callbacks与live publish始终形成显式host/RVIR cut。

冻结formal实例的完整嵌套调用树，而不是五个孤立样本：1 core、24个provider `compute_bounds`调用、phase=
`initial 12 / alpha 1 / beta 11`、outer optimized call的10/9；冻结KFSB路径另有3次
`update_bounds(shortcut=True)`，每次产生24个child domains / `[24,1]` lower，合计72个child-lower记录。
两类“24”不得混同。至少记录：

```text
core_id
call_id / parent_call_id
phase / start_node
evaluation_ordinal / mutation_ordinal / branch_candidate_ordinal
bound_lower / bound_upper
return_A / needed_A
root_or_child / domain_depth
solver_boundary_scope / runtime_transition_kind / publish_phase / isolation_domain
read_set / staged_set / live_write_set
expected_version / observed_version / terminal_outcome
producing_stream / completion_event / arena_epoch
```

每个value记录role、shape/dtype/device、representation、producer/consumer/fanout、version、escape、logical
bytes、saved/clone/materialize、CPU/GPU transfer，以及tensor/storage/allocation identity、data_ptr、storage offset、
stride、view base、alias set、PyTorch `_version`、allocation/free、stream/event/correlation和autograd
saved-tensor pack/unpack。publish/runtime transition trace还必须区分`ENQUEUED/DEVICE_READY/PUBLISHED/AUDITED`，记录失败
补偿前后的content与`_version` delta；host序号与CUDA序号通过NVTX/CUPTI correlation及时钟校准绑定。

观察器不得持有tensor强引用或改变allocator/stream/trajectory；只即时落metadata/hash，释放事件用弱引用、
allocator或autograd hook。artifact必须输出`claimed_value_coverage`、allocator bytes、kernel time与launch count的
attributed/unattributed桶；coverage未过预注册门槛时不得形成HBM/Amdahl headline，未捕获不能记作零。

trace与raw output、24-call topology、10/9 trajectory和solver receipt逐项对账；另用failure injection确认
current guarantee只覆盖core state publication，不能把queue pop/add缺少rollback记成零。不得在M0改默认配置
或形成性能结果。M0是一个短instrumentation/capture-decision阶段，不发展成另一套runtime IR；它只冻结
selected tensor graph、semantic overlay输入、PublishIntent输入和真实memory/Amdahl分母。

### 8.2 M1：import-first + semantic lifting + publish ABI闭合

#### M1-A：导入获选tensor graph

- 只实现两个formal importer：`strict ExportedProgram → frozen decompositions → functional Core ATen` forward
  importer，以及pinned AOT partitioned forward/backward-pair importer；裸FX、raw joint graph和Dynamo backend
  output不进入formal；
- forward importer保留op topology、constants、state inputs、graph signature/mutation outputs、shape/dtype/device
  与外部alias；AOT importer另保留primal/tangent/gradient mapping与saved-output→backward-input ABI；
- capture不支持的host policy形成显式external cut，不根据trace重写新的optimizer/BaB AST；
- 若某个region只能从BFBoundModule/BoundNode lowering，则它必须与M0 production tensor trace逐op/value对账；
- 禁止同一region同时维护exported graph与手写Relax DAG两条production owner。

#### M1-B：verification semantic lifting

- BFBoundModule/GC0提供polarity、A representation、soundness、effect/VJP/escape规则；capture graph提供真实
  tensor use-def；runtime evidence提供dynamic role、version、alias、lifetime与control witness；
- 只有影响dataflow/state的事实进入Relax显式SSA；纯分析事实按Relax stable value/function ID进入
  `VerificationOverlayV1`，不复制ATen shape/dtype/use-def，也不拥有op/edge；
- 接入GC0 legality reason，但`VerificationGraph`只生成one-shot certificate与mapping输入；import closure后
  production artifact丢弃其graph/interpreter，只保留source/certificate/overlay hash；
- 第一版仅admit已捕获的lower-CROWN；`bound_upper=True`及非空未支持cut/output/aux effect在build前拒绝。

#### M1-C：三方语义闭合

逐region比较`production native vs captured/exported execution vs imported Relax reference execution`：IBP/CROWN
lower/sign/必要upper，P/S dα/dβ/incoming-A gradient，单evaluation与单optimizer-step的α/β、Adam和mutation
ordinal。whole-10/9只核对历史artifact replay coverage，不在M1宣称可复用unroll closure。外部可观察alias/
mutation必须exact；internal alias和saved set只要求完整归因、VJP等价、escape/lifetime合法及saved-cut门禁，不要求
优化后storage identity与native一致。unsupported op/effect/version/escape必须在build/launch前fail closed；
capture或import任何silent fallback均为失败。

#### M1-D：prepared publish schema/verifier闭合

- 生成§3.10的`ExecutableArtifactIdentityV1/PreparedExecutionBindingV1/PreparedArtifactEnvelopeV1`及optional
  `PublishIntentV1/PreparedPublishBindingV1` schema与纯verifier；用current 12-device + 3-host runtime plan只验证
  通用投影和schema负向规则；
- M1不虚构尚未build的module hash、final output/storage ID或completion handle；M2/M3/M4各自在final selected
  executable之后生成resolved binding/envelope，M5才验证RVIR动态conformance；
- verifier静态闭合read/private/staged/live sets、staged/live disjoint、isolation、binding completeness、hash方向和
  terminal枚举；`COMMITTED/ABORTED/CONFLICT/ROLLED_BACK_INVALIDATED/POISONED`由RVIR产生；
- compiled launch只产生pending completion handle，RVIR await成功后才进入`DEVICE_READY`，不得由module自产
  ready/commit token；
- 该schema closure不阻塞M2 standalone IBP的`return-only` private/staged执行；它只阻止没有final multi-entry
  execution binding或缺optional publish extension的artifact进入M5 live publish；
- 不新增Plan/Task/Schedule schema，不修改TIR AST，不增加transaction interpreter。

M1只证明“capture可导入、验证语义可绑定、publish ABI可闭合”，不计时。

### 8.3 M2：IBP/CIBC mixed module + physical arena

固定路径：

```text
Torch/ONNX/captured IBP → standard Relax + VerificationOverlayV1
  → existing CIBC Conv PrimFunc
  → planned Linear/residual/ReLU/Flatten/epilogue lowering
  → exact-last-use/first-gap physical arena
  → prepared graph invocation
```

复用CIBC **Conv** TIR/schedule、`relax_interval_task_ops.py` graph carrier和NRIR-2 allocator/verifier算法；旧
`CIBCIBPCUDAGraphPlanV1`只作direct oracle。correctness使用独立instrumented/debug module导出冻结intermediate
value/digest，做PyTorch/direct CIBC/mixed module三方lower/upper、sign、residual/fanout与17-op coverage；
formal production module使用独立hash，只导出final/staged result，intermediate escape=`0`。debug module的性能
不得代表formal candidate。

结构门禁：prepare完成且pointer signature不变的warm run内candidate新增allocation=`0`、DLPack view
construction=`0`；是否包含预注册input copy在protocol中固定；显式arena predicted/actual high-water exact；
CUDA allocated/reserved与capture pool另报；copy/layout bytes、submission/internal kernel数如实披露；formal
fallback/eager/native shadow=`0`。

性能资格沿用：mixed/direct geomean `>=0.90x`；mixed/PyTorch geomean `>=2.20x`、worst `>=2.00x`。
不过门只拆dispatch、copy/layout、allocation、HBM、launch与return，不跳去CROWN。

### 8.4 M3：P-anchor coarse CROWN + custom VJP

组装现有R3-1B1/1B2、D1、D2B与B4-B2部件：

```text
compressed α + bounds + weights + spec seed
  → relaxation + coefficient reverse wavefront + concretization
  → compact lower
  ↔ coarse custom VJP
  → compressed dα + compact status
```

一般图slot数由lifetime分析推导；P-anchor必须复现已验证two-slot结果，但不把“2”硬编码进schema。

任何timing前先闭合：state role/version、final escape、physical arena、saved-state cut、saved dense A=`0`、module
handle不持dynamic tensor、CompileMemoryWitness/ExecutionMemoryReceipt、warm allocation/DLPack=`0`。再要求mixed/direct-D2B `>=0.90x`，mixed/native
geomean `>=1.20x`、worst `>=0.98x`。

### 8.5 M4：per-step optimizer、host policy、受限whole-10/9、active β与domain overlay候选

M4-A默认路径是：compiled evaluation/VJP/functional optimizer step返回private candidate与policy inputs，host
执行stop/prune/timeout/keep-best/restore/scheduler/last-iteration判断，再决定是否启动下一step。逐step核对lower、
dα/dβ、α/β、Adam、best state、scheduler、active/preserve mask、stop predicate、mutation ordinal与terminal
output；这才是可复用production路径。

whole-10/9只允许两种受限candidate：全部tensor内容、初始optimizer state与policy inputs均由digest绑定的单实例
artifact replay；或所有动态predicate显式进入bounded state并能正确屏蔽后续evaluation、mutation、scheduler、
best-state与publish。18项controls、shape和一次“无提前退出”witness本身不够。通过后也只能叫
`exact-signature/content-bound specialization`，不能升级成一般optimizer IR；未通过则保留host policy loop。

M4-B：重做active-β物理schedule，直接消费compressed location/sign/value；按density/shape测试
dense/Patches/sparse crossover，不复用R3-3失败的fixed sparse schedule。

M4-C：按M0 density决定是否使用static base + split/history COW + unstable/refined + optional reference/aux/
cut/clip overlay；不能保真或不盈利即回退dense per-domain storage。扩多site前要求representation ledger、
CompileMemoryWitness/ExecutionMemoryReceipt与trajectory closure。

### 8.6 M5：RVIR same-solver exact-call

```text
同一 αβ-CROWN/BaB host
  control: native update_bounds_core
  candidate: RVIR state view → PreparedBoundModule → staged packet
             → await/validate/assemble → CoreStatePublishTxn
```

model/property、branch、termination、timeout、seed、dtype/device、α/β init、optimizer trajectory与queue order
一致；成功路径queue accounting继续由official pre/post核对，但rollback claim只覆盖core publish。必须注入
pre-commit abort、stale conflict、mid-publish compensation和async completion fault，证明旧frame/version不会
复用。历史B3/B0 query=`0.910001x`只作路由：parity至少需`1.09890x`，到`1.15x`至少需
`1.26373x`；必须用新same-solver GPU share、integration overhead和region speedup重算。

### 8.7 M6：formal 与证据驱动扩展

先完成native/direct-region/full compiler三方、complete query和queue formal。只有新profile开门，才依次考虑：

1. KFSB候选batch；
2. GPU hot frontier；
3. domain/spec/node联合tiling；
4. pointer/shape/module稳定后的CUDA Graph；
5. overlap bucket `>=10%`后的multistream；
6. reuse覆盖compile cost后的JIT/autotuning；
7. 动态BaB更多device化；只有failure/retry或speculative queue确有收益时，才先把destructive pop改为
   reservation/lease、把逐项add改为完整queue delta后单次publish，再讨论whole-round transaction。

## 9. 统一门禁

### 9.0 Capture/import与新IR门禁

正式region必须满足：

- capture expected/captured op、value、output、mutation与backward coverage逐项闭合；
- 可复用formal forward只接受strict export经冻结decomposition转成的functional Core ATen，mutable op count=`0`；
  formal backward只接受绑定Torch build/API/decomposition/partition及完整saved/gradient ABI的AOT pair；
- non-strict export与Dynamo/裸FX只作discovery/baseline，除非全部输入和初始state内容hash绑定为单实例replay；
- graph break=`0`、silent fallback=`0`；unsupported host effect在capture前拒绝或形成显式host cut；
- guards、shape constraints、constantized Python controls、source/config/taken-path进入artifact identity；
- external observable alias与functionalized mutation output必须和ground truth exact；internal alias与saved set须
  完整归因并通过VJP、escape、lifetime和saved-cut legality，不要求与native storage identity或saved inventory相同；
- capture→Relax importer不复制production control logic，native/capture/Relax三方通过冻结容差与discrete exact门禁。

任何新Relax Op、dialect、TIR intrinsic/AST或execution interpreter还必须通过§3.4.1 NIR-0—NIR-5；未通过时
只能使用trace字段、sidecar、标准Relax SSA/DPL或runtime protocol。

### 9.1 语义与publish runtime

- lower/upper、sign、verdict、bound tightness在冻结容差内；
- α/β optimizer每step state、best-state、preserve/active mask、stop policy、mutation ordinal与terminal等价；
- 成功路径branch、termination与queue accounting等价；不能据此声称queue失败rollback；
- core publish的staged/live set、version、isolation、completion与content compensation等价；每个frame在
  `COMMITTED/ABORTED/CONFLICT/ROLLED_BACK_INVALIDATED/POISONED`中exactly one；
- rollback后content可恢复但version必须单调并使旧frame/token失效；
- unsupported representation/effect/alias/version在launch前拒绝；
- formal candidate fallback/eager/native shadow=`0`。

### 9.2 物理结构

- dense A materialization/escape/saved count与bytes；
- peak-live、allocated、reserved、HBM read/write、CPU↔GPU transfer分别报告；
- warm dynamic allocation/DLPack construction的起止边界、input-copy政策与pointer signature；
- submission/kernel/sync/framework crossing；
- kernel/device time、achieved bandwidth、occupancy/compute utilization等可归因硬件指标；
- predicted explicit-arena high-water与actual exact；不把该exact要求外推到CUDA allocator/capture pool/HBM；
- attributed/unattributed coverage与single-stream序列或multistream happens-before模型；
- fixed-batch latency与等显存最大batch吞吐并报，不能用大batch掩盖单query退化。

若IR变化没有改善至少一个可归因物理指标，就不开放性能claim；改善既可来自bytes/launch/transfer下降，也可
来自locality、occupancy、vectorization/tensor-core利用或kernel/device time下降，不能把“计数不变”误判成
没有物理机制。

memory-path admission还要求：真实workload native peak达到预算约80%或发生真实OOM；或者candidate在同一预算
下支持更大合法domain/spec batch并提升等显存吞吐。否则只能形成arena机制结果，不能形成solver memory
claim。若声称HBM带来加速，还需先证明目标region memory-bound并实测traffic下降。

### 9.3 Amdahl 路由

```text
S = 1 / ((1 - s) + s / r + h)
h = candidate_extra_integration_time / native_same_scope_time
r_required = s / (1/T - (1-s) - h)
```

- `s`、`r`、`h`与目标`T`必须同scope；
- `r_required`分母`<=0`立即STOP；
- 对原`T=1.15/1.20`的单bucket边际优化，`required_region_speedup >10x`仍默认STOP，除非已有同shape
  roofline、schedule或结构机制证据；
- 对v6 `T=10x`全栈目标，不再使用“`r_required>10x`自动STOP”，而使用
  `sum_i(s_i/r_i)+u+h<=0.10`残差预算；`u+h>=0.10`直接关闭10×，`u+h>0.05`高风险；
- 任一bucket所需速度超过同shape实测/roofline上限，或在没有新物理机制时超过20×探索cap，停止该bucket并
  重新做scope/region设计，禁止靠乘局部数字继续；
- `T=1.15x`时，即使region无限快，same-scope share也必须大于`13.043%`；
- 独立IBP graph share不得代入CROWN/query目标。

### 9.4 最终研究门槛

- **正确性门禁**：complete-query/queue/trajectory、bound tightness与验证强度不降；
- **最低资格线**：complete-query geomean `>=1.15x`、queue geomean `>=1.20x`；它只表示系统开始传播收益；
- **强系统结果线**：direct B0→cumulative candidate在至少两个held-out family达到跨模型2—5×，并有完整
  representation/fusion/runtime消融；未到10×时只能按实测数字投稿；
- **v6 headline stretch target**：same-solver complete-query geomean约`10×`、建议worst `>=5×`，并在
  fixed-trajectory systems mode与solved-query TTV mode分别成立或明确区分；
- 至少两个held-out model family；
- 至少一个baseline/candidate均能在timeout内solve的公开property；
- peak memory不恶化；memory claim要求真实pressure workload `<=0.75x native`；
- compile/setup/cache/fallback、`u/h`、cold/amortized和worst pair全部披露，held-out不得事后调参；
- final headline必须来自同一`BoundFlow-final` cumulative executable的direct pair，历史局部数字只作机制与消融。

## 10. 论文新颖性边界

### 10.1 不能再宣称

- 不加限定的“首个bound/certifier IR”或“首个verification compiler”；
- 首个verification sparse intermediate runtime；
- 首个bound-aware GPU fusion或首个GPU verifier；
- 首个通用计算图LiRPA、可微bound propagation或BaB GPU acceleration；
- 仅凭CIBC或单anchor microbenchmark即ASPLOS-ready。

正式发表的auto_LiRPA、GPUPoly、β-CROWN、Faith等已经覆盖通用图LiRPA、GPU verification、稀疏表示、
bound-aware graph transform/kernel/fusion等宽泛位点；Faith尤其已有double-bound、weight-pairing与
bound-aware cross-layer fusion，因此lower/upper横向融合本身不能作为BoundFlow的新贡献。DiffAI也已覆盖
differentiable abstract interpretation这一宽泛位点。

ConstraintFlow compiler、TorchLean长文和TP/FSDP auto_LiRPA目前主要是arXiv/preprint或workshop状态，不能
当成已正式主会发表来否定BoundFlow的正式首发资格；但它们仍是公开并发工作，论文必须准确引用和逐项比较。
CIBC按用户提供的研究沿革视为2023年AAAI’24既有稿件，不在本计划中重复审计其构思时间。

### 10.2 可检验的候选贡献

在完整实现和相关工作压力测试前，只把以下内容作为**候选贡献**：

1. 面向 production optimized-LiRPA exact-call 的per-region capture与verification semantic lifting，把闭合
   reverse coefficient DAG、exact-signature α/β tensor state transition导入Relax/TIR，同时显式保持trajectory
   与RVIR publish boundary；
2. verification-aware coefficient representation + versioned liveness + sound rematerialization编译，在保持bound
   tightness与α/β trajectory时避免dense A和full intermediate materialization；lazy/COW domain overlay只是
   其中待门禁证明的候选子机制；
3. RVIR completion-aware prepared-publish runtime，在完整BaB host内做same-solver exact-call替换、staged
   publish与core-state逻辑原子提交；不含whole-round queue transaction claim。

只有S5 formal确实通过后，才允许增加第四条结果型贡献：

4. 首个在保持production αβ-CROWN/BaB solver trajectory时，由verification compiler获得约10×
   same-solver complete-query提升的实证；若实测不是10×，本条必须改成真实数字或删除。

推荐论文定位：

> **面向 production αβ-CROWN/BaB exact-call 的 verification-aware compiler/runtime，在保持求解器
> trajectory 的前提下，联合规划抽象表示、张量生命周期、rematerialization 与 GPU region fusion。**

### 10.3 ASPLOS前两页与11页叙事预算

Rapid review只读前两页，因此前两页固定回答四问：

1. production αβ-CROWN为何被逐op framework、dense coefficient lifetime、autograd saved state和host/runtime
   边界限制，而普通tensor IR不知道哪些跨层变换仍sound；
2. BoundFlow的最小verification semantic lifting如何使标准Relax/TIR能够合法联合做representation、fusion、
   rematerialization和custom VJP；
3. CIBC→coarse CROWN→optimizer/runtime→RVIR的一个cumulative executable如何保持trajectory；
4. direct B0→final的complete-query/TTV、memory、cold cost、worst case与局限到底是多少。

11页建议预算：前两页problem/result/contribution；1页motivation/characterization；1.5页semantic lifting与
legality；2页joint compiler optimization；1页prepared runtime/RVIR；0.5页implementation；2.5页evaluation/
ablation；0.5页limitations/related-work positioning。正文必须自包含，appendix和artifact不能替代关键实验。

## 11. 代码迁移与退役

| 路径 | 动作 |
|---|---|
| production capture adapter（新增） | 负责trace/export/Dynamo/AOT bake-off与`CaptureDecision`，不成为runtime executor |
| `boundflow/ir/primal.py` + Torch/ONNX frontends | 保留统一import/normalize/source mapping，尽早lower到Relax，不作为热路径IR |
| `boundflow/ir/bound.py` | 保留backend-neutral semantic source/interpreter/oracle；one-shot lifting后production artifact只保留source/overlay hash |
| `boundflow/ir/verification_graph.py` | 保留schema/rules/tests，输出legality certificate与overlay输入；不得把graph/interpreter或第二份topology带入prepared artifact |
| `boundflow/ir/plan.py` | 当前仍被typed query链消费；freeze/no-new-feature，保留decision/alias/evidence，待同scope替换后转oracle |
| `boundflow/ir/task_v1.py`、`schedule.py` | 当前仍有真实lowering/dispatch路径；先保留执行兼容，M3/M5同scope closure后才转compatibility/replay |
| `boundflow/ir/task.py`、`planner/core.py`、`planner/pipeline.py`、`runtime/scheduler.py` | interval-v0 legacy API/大量测试的兼容层；新API closure前不删除，production不再扩展 |
| `runtime/task_ir_executor.py`、`schedule_ir_executor.py` | Python reference/correctness/replay executor，不升级为production runtime |
| `planner/storage_plan_variants.py` | 复用exact-last-use/first-gap算法到Relax/TIR lifetime pass |
| `runtime/storage_plan_runtime.py` | 保留logical oracle；physical runtime由PreparedArtifact arena替代 |
| `planner/materialization*.py` | 复用candidate/cost输入，不成为并列production planner |
| `runtime/materialization.py` | 复用观测schema，扩physical bytes/lifetime/escape |
| `runtime/tvm_executor.py` | 复用build/VM/cache，formal GPU路径移除CPU NumPy和静默fallback |
| `runtime/compiler_query_runtime.py` | 只复用API/validation经验；其legacy Task+Bound+Plan payload改为Relax/TIR artifact + prepared runtime façade |
| CIBC Conv PrimFunc/schedule | 提取为M2 Relax-callable backend；现有Python mixed/CUDA Graph runner继续作direct oracle |
| B4-B2/R3 backend/runtime | 只直接复用PrimFunc builder、schedule、DLPack/stream ABI、receipt和correctness oracle；exact-workload runner不得整体注册 |
| 专用optimizer/structured/differentiable/R3 arena IR | 保留frozen artifact、correctness oracle与迁移fixture；提取通用legality/lifetime规则，不新增专用production interpreter |
| RVIR-v4 | 保留core state/effect/publish唯一owner；DomainList queue仍由provider host拥有且不得虚构rollback |

退役按对象和同scope门禁执行：interval-v0 planner/task/scheduler至少等M2 IBP mixed-module correctness、replay、
identity与性能资格；typed CROWN Plan/Task/Schedule/dispatch至少等M3 region与M5 exact-call的语义、failure、replay、
artifact identity parity；RVIR publish owner不在本计划退役。B4/R3实现进入registry前还必须有通用capability
predicate、Relax-callable wrapper、dynamic-tensor-free prepared handle与formal no-fallback gate。不为目录整齐
提前删除任何oracle或兼容路径。

## 12. 建议提交序列

用户批准且S0投稿可行性门禁通过后，提交序列改为围绕一个cumulative candidate：

1. `docs: freeze asplos27 thesis b0 scope and ten-x budget`
2. `bench: attribute fixed-trajectory and solved-query full stack`
3. `feat(compiler): route complete ibp graph through cibc relax tir path`
4. `feat(compiler): assemble coarse crown forward and custom vjp region`
5. `feat(memory): select coefficient representation remat and physical arena`
6. `feat(compiler): compile optimizer transition with explicit host policy cut`
7. `feat(runtime): collapse site bridges into one prepared execution`
8. `feat(adapter): route rvir exact call through cumulative boundflow module`
9. `perf(solver): batch branching and residual runtime buckets by evidence`
10. `bench: close direct-b0 cumulative complete-query formal`
11. `artifact: package anonymous replay tamper and expected outputs`
12. `paper: freeze asplos27 submission and cibc resubmission note`

每个correctness closure保留raw/replay/tamper；不要求每个小提交外审，性能或claim升级才形成正式artifact。

## 13. 第一刀

如果用户批准，第一刀改为**S0的48小时投稿可行性与10×预算门禁**，不是先完成通用capture bake-off：

> **冻结official B0完整solved-query与fixed-trajectory两种scope；用低扰动raw把全部时间归到IBP、CROWN、
> VJP/optimizer、branching、queue、bridge/launch、allocation/transfer/sync和unclassified，覆盖至少97%、
> unclassified至多3%；计算每个bucket的`u/h/r_required`和10×残差预算。同时把现有CIBC mixed path、
> R3-D2B coarse wrapper与RVIR exact-call放入一个最小cumulative smoke，直接测B0→candidate方向，并写出
> ASPLOS rapid-review前两页与CIBC changes note骨架。**

S0只观察和组合已有资产，不新增IR、schema或新kernel。它在48小时后必须给出三选一结论：

1. `GO-ASPLOS27-TEN-X`：10×数学可达、单candidate集成路径明确、前两页贡献成立；
2. `GO-ASPLOS27-LOWER-HONEST-RESULT`：10×高风险但已有强端到端结果，按真实2—5×目标继续；
3. `STOP-ASPLOS27-CONTINUE-RESEARCH`：13天无法形成polished论文，保留10×路线转下一A会。

只有S0选择GO，才按A.6执行S1/S2；通用capture/import bake-off作为具体region确实缺失时的工具，而不再成为
论文前的串行总前置。

## 14. 外部资料入口

- [ASPLOS 2027 CFP](https://www.asplos-conference.org/asplos2027/cfp/)：September deadline、rapid review、
  pillars、11页格式、匿名、resubmission note、AI披露与宣传静默期的权威来源；
- [ASPLOS 2027 Artifact Evaluation](https://www.asplos-conference.org/asplos2027/artifact-evaluation/)与
  [AE Author Guide](https://www.asplos-conference.org/asplos2027/ae-for-authors/)：artifact时间线、badge与打包要求；
- [auto_LiRPA paper](https://arxiv.org/abs/2002.12920)、
  [official repository](https://github.com/Verified-Intelligence/auto_LiRPA)与
  [production pinned source](https://github.com/Verified-Intelligence/auto_LiRPA/tree/5a098e8f9fb5786a428a024981d833d303921f2d)：general computational graph LiRPA；
- [production αβ-CROWN pin](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/e5c7e17bf0488843acb77b7519f59876717a49f4)：本计划exact-call源码范围；
- [β-CROWN](https://arxiv.org/abs/2103.06624)：split constraints与optimized bounds背景；
- [TVM Relax abstraction](https://tvm.apache.org/docs/deep_dive/relax/learning.html)：图级dataflow、`call_tir`与DPS；
- [TVM Relax op API](https://tvm.apache.org/docs/reference/api/python/relax/op.html)：`call_tir_inplace`在类型
  系统中仍视为pure，官方要求只能由已证明alias/liveness安全的pass插入；
- [TVM Relax VM](https://tvm.apache.org/docs/arch/relax_vm.html)：Call/Ret/Goto/If与compiled mode；
- [TVMScript](https://tvm.apache.org/docs/arch/tvmscript.html)：当前Relax structured-loop前端边界；
- [TVM TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html)：loop、buffer、block和schedule；
- [TVM fusion](https://tvm.apache.org/docs/arch/fusion.html)：graph grouping、pattern fusion和TIR fusion；
- [TVM Relax DPL](https://tvm.apache.org/docs/deep_dive/relax/dpl.html)：pattern matching、project-specific
  rewrite与backend dispatch；第一版优先于引入e-graph或新dialect；
- [vendored TVM pin](https://github.com/leezear2022/tvm/tree/6248b5db43505fbcfb13cc289d11877d5d2649e8)：能力判定以该commit为准，官方滚动文档只作交叉校准；
- [MLIR SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/)：structured control与loop-carried SSA；
- [MLIR Side Effects and Speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)：effect
  resource/stage、冲突与speculation边界，不等于transaction/rollback；
- [MLIR Async dialect](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)：completion token/value与await；
- [MLIR One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/)：use-def、alias与in-place分析参照；
- [MLIR dialect definition](https://mlir.llvm.org/docs/DefiningDialects/)与
  [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)：说明新dialect、legality与lowering的完整维护
  成本；本计划只借机制，不迁移第二编译栈；
- [StableHLO specification](https://openxla.org/stablehlo/spec)：`while`的loop-carried state，以及side-effect
  token/`after_all`的顺序语义；
- [IREE Stream dialect](https://iree.dev/reference/mlir-dialects/Stream/)：external/transient/variable resource、
  symbolic bytes与timepoint后安全复用的设计参照；不引入IREE backend；
- [PyTorch functionalization](https://docs.pytorch.org/docs/stable/generated/torch.func.functionalize.html)：把
  intermediate mutation函数式化、末尾fix-up输入mutation及non-local state边界；
- [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
  与[event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)：launch返回不代表完成，
  event synchronize还可能返回先前异步错误；
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)：固定workflow
  的definition/instantiation/repeated execution；只作晚期submission优化，不是solver IR；
- [PyTorch 2 paper](https://docs.pytorch.org/assets/pytorch2-2.pdf)：AOTAutograd joint graph与min-cut rematerialization；
- [torch.export programming model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/programming_model.html)
  与[Export IR specification](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/ir_spec.html)：
  example-path、static/dynamic值、guards、functionalized mutation与capture artifact边界；
- [PyTorch compiler graph breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)：Python/data-dependent控制边界；
- [PyTorch custom backends](https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html)：Dynamo
  fullgraph capture backend与Inductor对照接口；
- [PyTorch higher-order operators](https://docs.pytorch.org/docs/stable/higher_order_ops/index.html)：受限tensor control capture能力；
- [ConstraintFlow compiler](https://arxiv.org/abs/2507.20055)及
  [repository](https://github.com/uiuc-focal-lab/constraintflow)、
  [active compiler fork](https://github.com/ADAPT-uiuc/constraintflow-gpu)：certifier DSL、stack tensor IR与g-BSCR prior art；
- [Faith](https://arxiv.org/abs/2209.12708)：verification-aware graph transform、fusion与GPU kernels；
- [GPUPoly](https://arxiv.org/abs/2007.10868)：专用GPU抽象算法、structured sparsity与内存处理；
- [DiffAI](https://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf)：differentiable abstract interpretation prior art；
- [TorchLean](https://arxiv.org/abs/2602.22631)与 [ACT](https://github.com/SVF-tools/ACT)：新近verification IR/共享语义实现边界；
- [multi-GPU auto_LiRPA](https://arxiv.org/abs/2606.09377)：α tensor与memory scaling的新近证据；
- [egg equality saturation](https://arxiv.org/abs/2004.03082)：e-graph用于纯等价候选与domain analysis；只有
  Relax DPL顺序式rewrite出现实测漏解后才作为局部候选搜索器；
- `docs/CIBC_for_DAC.pdf`：本工程已有BC graph→fused tensor expression→target code设计。

## 15. 与 GC0-1 和外审的关系

- 用户已明确“先不用外审”，本修订不处理GC0-1异步审计findings，不执行respond/close；
- GC0-1 analysis算法作为M1一次性semantic lifting的legality输入，GC0-0 schema/negative reasons继续复用；
- GC0不是tensor capture机制，也不再决定当前主节奏或扩张为execution graph；ASPLOS sprint中只复用已完成
  schema/规则，不等待新外审；
- S0—S3已经按本稿顺序形成standalone IBP、single-evaluation CROWN和本地10/9 wrapper的内部关闭点，但
  不把10x、ASPLOS-ready或局部数字升级为same-solver claim；
- 本稿当前为`execution-authority=true/code-change-open=false-pending-s3-external-audit`；下一动作是完成
  S3独立外审并审阅S4 all-mutable-state预注册。批准后才允许S4-0 coverage代码；S4正式timing、未归因
  O6/O7、complete-query与10x claim仍关闭。
