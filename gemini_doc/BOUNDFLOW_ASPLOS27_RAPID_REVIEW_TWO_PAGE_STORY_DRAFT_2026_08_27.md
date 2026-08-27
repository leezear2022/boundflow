# BoundFlow ASPLOS’27 rapid-review 前两页故事骨架（S0草稿）

date: 2026-08-27
status: evidence-aware-draft-not-submission-ready
performance-claimed: false

## Page 1：问题、反例与核心洞见

神经网络验证器已经大量使用GPU tensor算子，但通用tensor compiler看不到验证语义：lower/upper必须成对
传播，CROWN coefficient可能是dense/Patches/sparse，α/β和split/history跨多次evaluation变异，失败不能
部分publish，branch/termination又必须保持原solver轨迹。因此“把一个Conv或Gemm换成快kernel”不会自动
成为快验证器。

BoundFlow的反例证据很直接：CIBC BoundConv局部达到`12.795×`、IBP图达到`2.456×`，CROWN局部TIR达到
`4.898×`；但三site production bridge只有`0.834×`，删多数同步guard后相对native仍只有`0.903×`，B3
same-solver prefix相对B0也只有`0.910×`。局部kernel是有效的，收益在表示materialization、framework
crossing、optimizer transition和solver runtime中消失。

核心洞见是：不发明一套替代TIR的solver IR，而把production tensor regions导入标准Relax/TIR；BoundFlow
仅提升验证合法性事实和publish/state合同，使compiler能联合做horizontal/vertical fusion、representation、
lifetime/rematerialization、custom VJP与prepared runtime。branch、termination和外层BaB策略继续由原
αβ-CROWN host拥有。

S0还发现operator-only的数学上限：official B0 fixed-prefix里operator约61%、non-operator约39%；即使全部
operator按CIBC `12.795×`运行，最乐观只有`2.3189×`，operator无限快也不超过`2.6108×`。因此10×若可达，
必须同时把optimizer、domain/branch、state transition和runtime submission变成coarse compiled transactions。

## Page 2：系统、贡献与评估

系统形成一个cumulative candidate：production capture/import保留exact-call identity；verification semantic
lifting证明哪些lower/upper、coefficient、α/β、split/history和effect可以融合或重算；Relax/TIR生成IBP和
CROWN coarse regions；custom VJP只保存compressed state；prepared runtime持有module、arena、views与cache；
RVIR在一次原子publish处把结果交还原solver。

候选贡献只保留三项：

1. verification-semantic lifting与fail-closed legality，使标准tensor compiler可以安全改变verification
   representation、lifetime和VJP；
2. 从CIBC BoundConv扩展出的joint region optimization，跨IBP/CROWN/optimizer联合fusion、rematerialization
   和physical memory planning；
3. trajectory-preserving exact-call integration，把局部GPU收益传播到同solver fixed-work与complete-query。

评估必须直接比较official B0与单一BoundFlow-final，覆盖至少两个模型族和一个双方可solve公开property；报告
warm/cold/amortized、fixed trajectory、TTV/solved、peak memory、worst pair及O1—O7累计消融。`10×`只是
stretch target：只有同scope complete-query raw达到才写headline；否则报告真实2—5×或NO-GO。

S0 explicit transaction现已把ResNet/MNISTFC fixed-prefix机制覆盖分别提高到最低`99.632%/99.248%`，
observer扰动中位数为`0.996×/0.999×`（最大单对`1.042×/1.065×`）。按互斥事务桶冻结的研究目标组合在
`h=0`、不计接入成本时条件式投影为`12.562×/11.657×`，但这只是可证伪预算：O1—O5 target尚未direct
验证，S1 performance claim仍关闭。下一实验是把已有CIBC winner
接入唯一`Primal→Bound→Plan→Relax/TIR→Prepared Runtime`路径并直接测O1/O3，而不是继续堆独立kernel。
