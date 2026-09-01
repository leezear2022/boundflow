# 修改记录：MR3 P-anchor Production Bridge Implementation

> 日期：2026-08-26
> 状态：implementation preflight完成，尚无formal correctness结论

## 第一阶段实现

- 新增真实provider P-anchor bridge，将`/49`下`/input-24 → /input-20` lower path路由到现有
  CIBC dense Conv TIR forward/custom VJP；
- provider float value继续作为exact value，first-order gradient只流向candidate，避免native/candidate
  双重α梯度；
- provider继续拥有loss、Adam、scheduler、clamp、split/history、termination与final state；
- P β只接受一个`[6,0]`empty tensor；ReLU→Conv按content接续，不要求pointer相同；
- executor新增只读backward completion callback，使bridge可证明10/9 launch而不跨evaluation保存
  executor/dense A；
- 新增真实provider control/bridge worker与receipt负向测试；不记录timing。

## 待验证

- 先运行一个fresh bridge worker，确认真实10/9 autograd与provider ABI；
- 再补逐mutation α/gradient/Adam moment 与atomic rollback ledger；
- 未完成5 pair formal前不得claim bridge correctness。

## 首次 live admission 修正

首个真实bridge worker在evaluation-0、optimizer mutation前fail closed：provider sparse-feature α
重建必须保留spec维，输入/输出为`[1,6,86] → [1,6,16,8,8]`。实现已改为按provider原布局
重建后再squeeze spec，未放宽ABI或启用fallback。

第二个live worker推进到首个candidate forward launch后，在返回结构重建处fail closed：ReLU仅有1个
coefficient input，Conv必须保留input/weight/bias共3个slot。重建器现显式要求`1/3`并只替换首个
lower-A slot；其余provider slot原样保留。失败仍发生在首次backward/mutation前。

第三个live worker进入首次`loss.backward()`后，旧executor只接受stride=`(0,0)`的broadcast bias
adjoint，而真实provider传入contiguous `[6,1]`全1 tensor。门禁现按语义接受shape/device/dtype正确且
所有元素逐位相等的scalar seed，继续拒绝非均匀per-domain seed；失败发生在optimizer step前。

第四个live worker证明上述修正已跨过9次真实backward与Adam mutation，随后在第10次final
evaluation fail closed。原因是旧bridge把full α的`requires_grad=True`错误地要求为10/10；provider
合同实际是evaluation 0–8 grad-enabled并各执行一次backward/mutation，evaluation 9在`no_grad`
下只做最终值评估。门禁现显式要求这两个阶段分别为9/9与1/1，禁止任意ordinal的grad模式漂移。

第五个live worker首次闭合10/9并与一个fresh provider对照通过（inner/outer/final-state最大差分别
`1.55e-6 / 5.96e-7 / 3.04e-6`），但实现复查发现该版本保留native forward float value、只把VJP
路由给candidate。这只能作为custom-backward接通预检，不能作为完整region replacement formal。
因此在生成formal raw前主动收紧：目标ReLU/Conv调用provider时令lower-A为`None`，只保留upper与
返回结构；bridge handoff是原incoming A，最终lower A/bias直接来自candidate TIR。native lower
region不再作为value owner，也不得参与候选计时。

第六个live worker在evaluation-0、mutation前发现provider ReLU不支持仅为返回骨架而传入
`last_lA=None,last_uA=None`：其sparse-α重建会解引用空的两侧。进一步核对真实ABI后，lower-only
ReLU返回固定为1个input coefficient slot，Conv在weight/bias未扰动时固定为3个slot且后两者均为
`(None,None)`。bridge现直接构造这两个lower-only三元返回，不再执行目标region的native lower
函数；非目标start node仍原样调用provider。

第七个live worker继续在evaluation-0、mutation前fail closed，证明provider lower-only返回ABI的
inactive upper bias不是`None`而是标量`0`；`backward_general`会无条件把它累加到upper accumulator。
bridge返回已修正为lower bias tensor + upper bias scalar zero，未引入upper tensor或额外计算。

第八个live bridge完整闭合10/9；新增trajectory观测后发现不能把provider Conv单段operator bias与
candidate fused bias直接比较。provider在ReLU返回intercept bias、在Conv返回operator bias，TIR按
合同一次返回两者之和。control ledger现显式暂存ReLU bias并只记录`relu_bias + conv_bias`，每个
evaluation后必须清空；这修正观测scope，不改变solver或candidate路径。

首次在evaluation-5 dispatch后注入异常时，rollback receipt自身fail closed：pre-hash错误地从
`detach().clone()`生成，因而把原owner tensor的`requires_grad=True`误记为false。clone仍只用于
冻结内容，但pre semantic hash现从原tensor生成；随后必须在同一注入点重跑，不能把这次失败计作
atomicity通过。

第二次同点注入证明owner内容hash已exact恢复、version均前进，但pointer hash未恢复。原因是provider
clamp/β路径以`.data = ...`替换同一tensor对象的storage。snapshot现同时保留原storage view与冻结
content；rollback先恢复原`.data`指向，再`copy_`冻结内容，从而分别验证content/pointer/version。

第三次同点注入通过：在evaluation-5 candidate dispatch后（已有5次Adam mutation）捕获预期异常，
12个owner tensor的content/pointer hash前后逐位一致，version delta范围`1..6`，且
`launch/emit/commit/rollback=1/0/0/1`。正常preflight pair中，P-region A 61,440元素最大差
`3.35e-8`、combined bias 60元素最大差`2.24e-7`；9-step gradient/Adam/α轨迹最大差分别为
`2.33e-8 / 3.73e-9 / 4.77e-7`（`exp_avg_sq=1.02e-12`），lr与clamp mask exact。

这些数字仅为implementation preflight，不是5-pair formal artifact；下一提交冻结source后必须从
pair-0全新运行，禁止复用`/tmp`预检raw，也没有performance claim。
