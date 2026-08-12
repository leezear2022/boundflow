# RVIR-v4 V4-2 Optimizer Mutation 预注册修改记录

日期：2026-08-13

## 修改

- 新增V4-2独立预注册计划；
- 从正式V4-0 capture和provider源码冻结10 evaluation/9 update、双学习率、fixed intermediate、
  lower-only、batch-any stop与12-path atomic copy-out合同；
- 将工作拆为policy/iteration、step trace、pre-state mapper、10-step mutation、atomic copy-out五个切片；
- 明确逐step比较和重签名tamper门禁，禁止只比较最终lower；
- 当前GPU probe为driver/library mismatch，formal step-trace artifact显式阻塞，但CPU可验证的policy合同
  可以继续实现。

## 事实诊断

- production policy：iteration=`10`、lr alpha/beta=`0.01/0.05`；
- provider语义：10次evaluation、9次optimizer update；
- capture：1 core、6 domains、12 mutable receipts、7 changed；
- native差距：统一lr、`steps+1` evaluation、自建IBP intermediate、无step tensor artifact；
- 本轮只预注册，不产生optimizer parity或性能claim。

## 下一步

V4-2A已开始实现：native optimizer新增可选beta learning rate参数组，旧统一lr payload保持兼容；
新增`ProductionMutationPolicyV4`，把production `iteration=10`显式映射为10次evaluation/9次update，
并对lower-only、fixed intermediate和batch-any stop fail closed。完成测试前仍不宣称V4-2A关闭；
V4-2、V4-3和B2仍未关闭。

为使本轮触及的optimizer模块可以完整执行mypy，同时清理了该文件既有的Optional narrowing、局部变量
重名与`DenseLinearOperator`协议注解问题；只改变类型表达，不改变证书或bound计算。

## V4-2A 验证

- distinct alpha/beta Adam groups=`0.01/0.05`；
- production iteration/evaluation/update=`10/10/9`；
- 旧统一lr payload兼容，非法策略与beta LR fail closed；
- focused=`17 passed`；full=`1100 passed, 39 skipped`；
- mypy 4个相关文件clean；新增/typed policy模块Pylint=`10.00/10`；
- V4-2A关闭为`VALIDATED-POLICY-CONTRACT`，V4-2总体、V4-3和B2仍未关闭。

下一动作是V4-2B step-trace schema/capture runner；当前GPU driver/library mismatch阻塞formal run，
不阻塞schema、replay与CPU负向测试。
