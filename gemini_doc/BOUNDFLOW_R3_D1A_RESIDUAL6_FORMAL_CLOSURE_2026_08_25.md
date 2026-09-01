---
status: validated-r3-d1a-residual6-correctness
updated: 2026-08-25T17:18:00+08:00
type: closure
topic: boundflow
slug: r3-d1a-residual6-formal-closure
stage: s01
---

# R3-D1-A Residual6 Staged Correctness 正式关闭

## 结论

R3-D1-A 的第二个生产热点 residual6 staged factorization 以
`VALIDATED-R3-D1A-RESIDUAL6-CORRECTNESS` 关闭。它证明两阶段 TIR 在冻结 production state
上与原 v1 以及独立 CPU float64 闭合公式等价，并满足 2 launch、1 scratch、无跨层 dense A
持有的 ownership 合同。

本轮没有计时，不能形成 speedup、wrapper、query 或 ASPLOS performance claim。D1-A 两个热点块
正确性均已关闭，因此只开放 D1-B 固定 schedule qualification；D1-C cumulative wrapper、R3-3、
same-solver 仍关闭。

## 冻结来源与证据

- source：`52fc62c3f1dfa563d8fde7e770a4e771d0ce5622`；
- artifact：`artifacts/r3-structured-owner/r3-d1a-residual6-staged-v1`；
- protocol hash：`026c79b51b906a09cb9db743458597c281e823ded865a1cc75a48eff799e07c6`；
- summary hash：`18acaf3419a5fedc33c4fa455cff5d88ed7aa0c7c7c5846959b9ec3ec631fa0c`；
- manifest hash：`b158ac85d12098bed1d0e0bbdd178511b1c927cf3eb035812fab9604c808d062`；
- 5 个 fresh worker，`122,940` 个逐元素比较量；
- 三方最大绝对误差 `1.916179825922626e-06 ≤ 2e-4`，sign exact；
- 10/10 fully re-signed tamper rejected；
- targeted：`6 passed`。

## 三方数值上界

- candidate-v1：output `2.9802322387695312e-08`，bias `9.5367431640625e-07`；
- candidate-float64 oracle：output `5.20636509460104e-08`，bias
  `1.6777612468210634e-06`；
- v1-float64 oracle：output `4.192402774938486e-08`，bias
  `1.916179825922626e-06`。

## 实现与 ownership 合同

1. stage-1：`incoming --conv4^T--> scratch[6,16,8,8]`；
2. stage-2：在 scratch 上应用 ReLU19 slope/intercept，执行 stride-2 `conv2^T`，同时加入
   `conv5^T` 1×1 shortcut 与 bias；
3. runtime 使用 non-default current stream、15/15 DLPack pointer exact；
4. 每次执行恰 2 launch、1 个 caller-owned scratch；
5. `persistent_dense_a=false`，fallback/eager 为零；
6. artifact 与 replay 强制 `timing_recorded=false`、`performance_claimed=false`。

## D1-B admission

D1-B 只允许对 residual6/residual11 已关闭的等价实现做固定、预注册 schedule 比较。首轮只改变
threads/reduction 等合法 schedule 参数，不能修改数学表达式、加入 persistent dense A、扩展到新 site，
也不能用 isolated timing 冒充 cumulative wrapper speedup。D1-C 只有在 D1-B correctness、receipt、
five-fresh timing 与 opportunity gate 完整通过后才可能开放。
