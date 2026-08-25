# BoundFlow MR3 P-anchor Production Bridge 正式关闭

> 日期：2026-08-26
> 结论：`VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS`
> 性能声明：`performance_claimed=false`

## 1. 结论

冻结ResNet2B property-0的真实αβ-CROWN beta-split optimized exact call中，P-anchor
`/49: /input-24 → /input-20` lower region已由BoundFlow CIBC dense Conv TIR forward/custom
backward接管。provider继续拥有split/history、其余site、loss、Adam、scheduler、clamp、termination与
最终提交。MR3 correctness门禁全部通过。

本轮没有记录可采信timing。它只开放另行预注册的single-site bridge timing；S-anchor、multi-site、
same-solver complete-query与queue性能仍关闭。

## 2. 冻结来源

- implementation source=`baddf7c9fb7d6881a4429de484847bcfe2b52368`；
- formal protocol commit=`6d8d43b`；
- αβ-CROWN/auto_LiRPA/vnncomp commits分别为`e5c7e17`/`5a098e8`/`90419aa`；
- artifact=`artifacts/measurement-recovery/mr3-p-production-bridge-correctness-v1`；
- summary hash=`1ae9d2cb7a17cc373dbaaf8a1f13fc04d4dee7fd153ca4bd58154d0f7e77d40c`；
- manifest hash=`6eb22a7ae1d778d293b24b8694330226fc2a888a5b63d84a0a114b4f495371c1`。

## 3. Five-pair correctness

固定顺序`PB/BP/PB/BP/PB`，共5 pair/10 fresh独立GPU进程。每个bridge worker均满足：

- outer exact call launch/emit/atomic commit=`1/1/1`；
- candidate forward/backward=`10/9`；
- empty β=`10`个`[6,0]`tensor、总numel=`0`；
- fallback/eager/native shadow/persistent dense A=`0/0/0/0`；
- solver result=`verified`、visited domains=`6`；
- timing/performance claim=`false/false`。

每pair机械比较一般状态72,202元素、optimizer trajectory 46,440元素；5 pair全部allclose且sign
exact。全局一般与optimizer最坏absolute diff均为`3.159046173095703e-06`，远低于冻结的
`2e-4/2e-5`两级门禁。

一般状态覆盖10-step P-region lower A/combined bias、aggregate loss、inner/outer result、final α、
完整module owner state与final clip。optimizer状态逐步覆盖9个compressed dα、α pre/post clamp、Adam
`exp_avg/exp_avg_sq`、lr、step与clamp mask。

## 4. Atomic rollback

独立第11个进程在evaluation-5 candidate dispatch后注入异常，即此前已有5次Adam mutation。12个
provider owner tensor均恢复为pre-call content与原storage pointer；version delta为`1..6`，证明恢复
写入发生。receipt=`launch/emit/commit/rollback=1/0/0/1`，没有partial commit。

## 5. Replay与篡改

- manifest内全部文件SHA256通过；replay逐字重建summary hash；
- 18/18 fully re-signed tamper拒绝；覆盖source/run order/mode/model/verdict/timing/count/fallback、
  region value/loss/gradient/Adam moment/lr/clamp/final α、atomic commit与rollback pointer；
- artifact检索无`/home/lee`、`/tmp`或`file://`本机路径。

工程验证：MR3 targeted=`26 passed`；全量=`1721 passed,3 skipped,6 warnings`；Black clean、
mypy clean、pylint=`10.00/10`、`git diff --check`通过。3个skip均为既有环境边界：1个避免重复TVM
编译成本，2个冻结VNN-COMP checkout不可用。

## 6. Claim边界与下一步

允许claim：单个P-anchor已在真实provider exact call中完成typed structured lower-region replacement，
10/9 optimizer trajectory与failure atomicity等价。

不得claim：single-site speedup、B0/B3 parity、multi-site/S-anchor coverage、same-solver query/queue收益或
ASPLOS-ready。

唯一下一动作：先预注册single-site production bridge timing，明确去除formal观测、warm compile/cache、
provider与bridge对称顺序、host-wall/CUDA-event/memory口径与kill gate；预注册提交前不得运行计时。
