# 修改记录：MR3-0 Provider Hook Feasibility

> 日期：2026-08-26  
> 状态：计划冻结，待实现与实测

## 修改

- 将 MR3 中“先确认真实 provider hook”拆成可重放 MR3-0 preflight；
- 锁定真实目标为 beta-split 外层 optimized call，而非 5-step alpha 初始化；
- 冻结 `/49` 下 `/input-24 → /input-20` 的 node-level pass-through hook 合同；
- 冻结两组 fresh control/probe、10 evaluation 邻接与 ABI 门禁；
- 明确 MR3-0 不替换数值、不计时、不形成性能 claim。

## Exploratory correction（formal 前）

- 实测 P β 为一个 `[6,0]` empty tensor（对象数 1、`numel=0`），修正“对象数 0”假设；
- 实测 ReLU→Conv coefficient-map handoff 内容逐位相同但 storage pointer 改变，门禁改为
  shape/version/content 语义邻接并披露 pointer receipt；
- 实测 fresh GPU process 的连续 float hash 不稳定，pair 等价改用冻结 `2e-4` 容差与 sign exact；
- 上述修正发生在 formal artifact 与 candidate replacement 前，不含事后调参。

## 下一步

- 实现 pass-through hook worker、semantic replay 与负向测试；
- 从 clean source 生成 MR3-0 artifact 并执行全重签 tamper；
- 通过后直接实现 MR3 candidate bridge，不再等待手工评审。
