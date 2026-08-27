# BoundFlow ASPLOS’27 S0 外审响应与提交关闭记录

date: 2026-08-28
status: audit-approved-minors-closed-ready-to-commit
audit: `external_audit_asplos27_s0_2026_08_27.md`
performance-claimed: false

## 1. 外审结论

外审 verdict=`approve-with-minor`，0 blocker、0 major、2 minor、3 info。审计方使用不import BoundFlow的
stdlib-only脚本从raw独立重算33个marker、20个worker、10对formal、覆盖率、扰动和10×预算，所有headline
数字一致；双artifact replay、审计方自建全重签tamper、专项28与全量1860/3均通过。

## 2. Finding响应

### F1 integration overhead隐含为0

接受并关闭：

- `derive_transaction_budgets`新增显式`integration_overhead_share`，默认及formal固定为`0.0`；
- Amdahl residual现在计算`h + u + Σs_i/r_i`，required resolved speedup分母显式扣除`h`；
- protocol、artifact summary、README和权威文档均写明`12.5622×/11.6566×`只在`h=0`、不计接入成本时成立；
- 新增`h=0.03`负向预算测试，必须使两个workload都跌破10×并关闭S1 implementation route。

### F2 中位扰动门禁未披露单对最大值

接受并关闭：

- formal workload summary新增`maximum_perturbation_ratio`；
- ResNet median/max=`0.9959016×/1.0416400×`；
- MNISTFC median/max=`0.9986577×/1.0653545×`；
- 文档明确冻结门禁作用于five-pair median，MNIST r0单对超过1.05不违反原协议，但不再隐藏。

## 3. Info处置

- 本批S0代码、测试、计划、外审报告与DocOps记录将在同一提交中锚定；GC0 exchange与用户PDF排除；
- mypy clean限定为本批文件口径，不宣称仓库级mypy clean；
- 全量pytest复核命令明确要求conda `boundflow`环境及`env.sh`。

## 4. Claim边界与下一步

F1/F2只提高披露与预算fail-closed程度，不改变raw、不升级性能、不改变S0 verdict：

- S0 attribution admitted；
- S1 implementation open；
- S1 performance gate closed；
- `performance_claimed=false`。

提交后下一唯一动作仍为S1 canonical CIBC vertical path与PyTorch/direct-CIBC/pipeline三方直接计时。

## 5. 修复后验证

- 派生刷新前后GPU raw `worker_runs.jsonl` SHA256均为
  `90df685eb0b45b3dac503684162c0a06c56e15836a308716b6242916c8201bc0`；
- S0/10×专项=`29 passed`；
- transaction replay hash=`293e31c1db697a701660dbc4e6f8f85671086f0cd556b3c30384477f6a6c1435`；
- budget replay hash=`880f89cd2e765a2c519e898d5944851165b9c294a3c904666b7d31bb5317d0a7`；
- Black/mypy/pylint/diff/DocOps在提交前复核。
