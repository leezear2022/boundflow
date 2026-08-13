# RVIR-v4 V4-3C Native KFSB Candidate Evaluation 修改记录

日期：2026-08-13

## 修改

- 新增typed `NativeKfsbEvaluationV4`，显式拥有六层unstable mask、三组top-3候选、三组
  `[24,1]` child lower、候选reduction值和最终六域decision；
- unstable mask只由shared intermediate bounds与terminal native split state推导，六层逐元素对照V4-3A
  truth均exact，共4200个unstable neuron；
- 复刻production KFSB的BaBSR alpha/intercept score、固定`min` reduction、阈值惩罚与tie-break；bias从
  BoundFlow primal/task graph和参数绑定推导，不读取provider branching对象；
- 每个候选由BoundFlow构造12个decision的active/inactive两侧，共执行24个child domain；三候选合计
  72个child evaluation，不调用`LiRPANet.update_bounds`、provider `compute_bounds`或core；
- 新增独立artifact runner，重新执行V4-2 optimizer和V4-3B backward export后再运行native KFSB；
  V4-3A truth只进入独立comparator。

## Capture-ready诊断

- 三组candidate split共36项与production truth逐项exact；
- 72个child lower全部sign exact，最大绝对差`3.0994415283203125e-06`；
- 最终decision exact：`[[5,27],[5,32],[5,90],[5,90],[5,32],[5,90]]`；
- provider core/`compute_bounds`/`update_bounds` callback与fallback=`0/0/0/0`；
- related focused tests=`15 passed`，mypy三个source clean，Pylint=`10.00/10`；
- 当前状态仅为`IMPLEMENTED-NATIVE-KFSB / FORMAL-ARTIFACT-PENDING`，V4-3C、V4-3、B2和性能claim
  尚未关闭。

## 下一动作

完成full-resign tamper probe，提交clean source基线；随后从该commit生成正式artifact，运行semantic replay、
targeted/full/Black/mypy/Pylint/DocOps门禁。全部通过后才允许将V4-3C升级为
`VALIDATED-NATIVE-KFSB`并准入V4-3D live return assembly。
