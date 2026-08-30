# ASPLOS'27 S4-1A ordered buffer formal 变更记录

date: 2026-08-30  
status: formal-candidate-pass-pending-external-audit  
performance-claimed: false

## 1. 本轮完成

- `8834aa5`：新增S4-1A ordered mutable buffer ABI与唯一resource owner；
- `29bcaa0`：新增正向/负向ownership门禁；
- `f03eb56`、`bce26f0`：新增12-process formal、stdlib replay与tamper，并修正一项同值攻击；
- `00893c5`：冻结正式artifact与tamper report。

实现从S4-0 one-shot lease ticket建立：

- 6个compressed lower-α leaf parameter；
- 1个active β leaf parameter与5个empty β typed token；
- 7个persistent gradient、lower `[6]`、fixed upstream `[6,1]`；
- 16个互异CUDA storage与16个完整identity-keyed base DLPack view；
- tensor/pointer/error-free canonical receipt；
- view→output→gradient→parameter→ticket的幂等close。

没有加入CROWN evaluator、TIR launch、Adam、trajectory、terminal handoff、fallback、计时或性能路径。

## 2. 施工中发现并关闭的问题

formal worker首次正向probe显示close后`allocated_delta=19,968 B`。生产owner已经正确close；泄漏来自worker
局部`parameters` tuple仍持有candidate。清空临时tuple与upstream oracle后，正向5 fresh与adoption fault均回到
`allocated_delta=0`。该修复属于证据worker生命周期，不修改生产owner语义。

首次tamper probe又发现`fault-detail`攻击把parameter fault改成原本相同的detail，因而被接受。攻击改为不同detail、
重新提交clean source并从空目录重跑12个worker后，10/10均在stdlib derived-semantic重算处拒绝。旧候选已移动到
`/tmp/s4-1a-buffer-invalid-fault-detail-20260830`，未覆盖正式artifact。

## 3. 冻结数字

- positive/fault/total fresh process=`5/7/12`；
- source/candidate indexed binary pair=`40/40 exact`；
- parameter/gradient=`7/7`、`4,254/4,254` elements、`17,016/17,016 B`；
- candidate storage/view=`16/16`，logical bytes=`34,080 B`；
- empty β token/physical=`5/0`；
- S4-1A D2H=`32/85,056 B`，累计S4-0+1A=`56/153,072 B`；
- parameter D2D=`7/17,016 B`；
- isolated fault clean=`7/7`；
- negative registry=`77`（门槛68）；
- tamper=`10/10 rejected`；
- unit=`80 passed`；S4-0+S4-1A related=`151 passed`；artifact+unit=`84 passed`；
- 最终全量回归=`2050 passed, 3 skipped, 6 warnings`（`724.52s`）；3个skip均为既有环境边界。

## 4. Claim边界

当前只允许：

`FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1A`

不得写`VALIDATED-S4-1A`。本轮没有证明provider mapping在整个query稳定、process-global exclusivity、CROWN
数值语义、optimizer trajectory、same-solver性能或complete-query收益。10/10 tamper只证明已注册的
derived-semantics-inconsistent全链重签攻击被拒绝，不把E0自洽artifact冒充物理执行真实性；下一轮外审需控制
fresh进程并独立核对source。

## 5. 下一动作

只执行S4-1A独立外审。外审批准前，S4-1B0 implementation/correctness、S4-1D evaluator、S4-2 optimizer、
S4 timing/performance全部关闭。
