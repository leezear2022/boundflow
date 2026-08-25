# 修改记录：MR3-0 Provider Hook Feasibility Implementation

> 日期：2026-08-26  
> 状态：implementation 完成，formal artifact 待 clean source 生成

## 实现

- 新增真实 αβ-CROWN `BoundedModule.compute_bounds` 外层 beta-optimized call tracker；
- 在目标 exact call 生命周期内对 `/input-24` ReLU 与 `/input-20` Conv 安装可恢复的 instance-level
  pass-through hook，外部仓库零修改；
- 记录 10 次 evaluation 的 start-node、sparse α/index、empty β、bounds、weight/bias、ReLU→Conv
  coefficient handoff、CUDA device/stream；
- 记录 outer/inner result、target α 与完整 alpha/beta module state 的 raw float payload；
- replay 使用 discrete exact + `atol=2e-4,rtol=2e-4` + sign exact 独立重算，不以跨进程 hash
  相同代替数值比较；
- 新增 synthetic fail-closed tests，覆盖 adjacency、α ABI、stream drift、provider numeric state 与
  未重签 worker tamper。

## Exploratory evidence（非 formal）

- exact call=`1`，inner evaluation=`10`，P ReLU/Conv=`10/10`；
- P β=`1` 个 `[6,0]` empty tensor、总 `numel=0`；
- control/probe 最大差：outer `4.76837158203125e-07`、inner `1.430511474609375e-06`、
  target α/full module state `9.5367431640625e-07`；
- solver status=`verified`，visited domains=`[6]`，两侧相同；
- 本节仅用于 implementation 调试，不能替代 clean-source formal artifact。

## 验证

- `pytest tests/test_mr3_provider_hook_feasibility.py`:6 passed；
- mypy clean；pylint 目标 10.00/10；
- 两个实际 GPU worker（control/probe）完成，未记录 timing claim。

