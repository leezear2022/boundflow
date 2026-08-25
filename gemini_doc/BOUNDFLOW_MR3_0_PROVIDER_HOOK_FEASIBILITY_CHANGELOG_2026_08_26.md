# 修改记录：MR3-0 Provider Hook Feasibility

> 日期：2026-08-26  
> 状态：计划冻结，待实现与实测

## 修改

- 将 MR3 中“先确认真实 provider hook”拆成可重放 MR3-0 preflight；
- 锁定真实目标为 beta-split 外层 optimized call，而非 5-step alpha 初始化；
- 冻结 `/49` 下 `/input-24 → /input-20` 的 node-level pass-through hook 合同；
- 冻结两组 fresh control/probe、10 evaluation 邻接与 ABI 门禁；
- 明确 MR3-0 不替换数值、不计时、不形成性能 claim。

## 下一步

- 实现 pass-through hook worker、semantic replay 与负向测试；
- 从 clean source 生成 MR3-0 artifact 并执行全重签 tamper；
- 通过后直接实现 MR3 candidate bridge，不再等待手工评审。

