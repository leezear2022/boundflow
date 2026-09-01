# BAB4 same-solver five-fresh formal candidate 收官记录

status: validated-formal-candidate-awaiting-external-audit
date: 2026-09-01
source-commit: 7bd28bdae4ac4f0093089d66510806ef09cf9028
performance-claimed: false
external-audit-requested: false

## 1. 结论

`B4-A` control 与四段 TIR `BAB4` candidate 在同一 αβ-CROWN host solver、同一
ResNet2B/property、同一 branch/termination/state transaction 下完成 5 组交替 fresh pairs。

完整查询研究门槛通过：

- complete-query geomean：`1.3001806128790074x`；
- complete-query worst pair：`1.2564622742158507x`；
- query parity `>=1.00x`：PASS；
- query research `>=1.15x`：PASS。

core geomean 为 `1.1771837870387067x`，没有达到独立的 `1.20x` core gate，因此不得写成
“core research gate 已关闭”。但五组 core 都为正收益，worst pair 为
`1.1283364243741671x`。

## 2. 冻结 artifact

路径：

`artifacts/bab4-same-solver-five-fresh/resnet2b-prop0-v1`

身份：

- source HEAD：`7bd28bdae4ac4f0093089d66510806ef09cf9028`；
- code revision：19/19 当前 blob hash 一致；
- candidate assets hash：
  `996a4fcc619786116a07149e04ddf9cb13505b242ff36183e582bd583df5f439`；
- production plan hash：
  `abb1b169fd5bc2ed1ceda970a83f9a10427d0c152590e8bdc5f001791f6fb2f5`；
- protocol hash：
  `86b2357d1d4beb7158627561edd8d61e02a7e9939a9b2ed1c1550a4c69eff9ad`；
- summary hash：
  `62b22b94fc091057c8dfb9bf4ccb72811111cc408819672200fd5b81a639e0a3`；
- manifest hash：
  `fda8be19e606b374b8bcbebe3b0aeaf91191be61201ac0377045948afa265dbf`；
- 文件数：79；体积约 15 MiB。

## 3. 五组 raw 结果

| pair | order | B4-A core ns | BAB4 core ns | core speedup | B4-A query ns | BAB4 query ns | query speedup | lower max diff |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | B4-A→BAB4 | 286792456 | 211737688 | `1.3544705x` | 1584723870 | 1133623424 | `1.3979280x` | `7.74860e-7` |
| 1 | BAB4→B4-A | 250002451 | 217139712 | `1.1513438x` | 1436005756 | 1124041576 | `1.2775379x` | `1.60933e-6` |
| 2 | B4-A→BAB4 | 242820315 | 215202053 | `1.1283364x` | 1419043243 | 1129395822 | `1.2564623x` | `1.22190e-6` |
| 3 | BAB4→B4-A | 243387644 | 213846326 | `1.1381427x` | 1417083390 | 1112625902 | `1.2736387x` | `1.96695e-6` |
| 4 | B4-A→BAB4 | 243373890 | 215606838 | `1.1287856x` | 1457198124 | 1120867979 | `1.3000622x` | `7.45058e-7` |

汇总：

- 5/5 pair lower sign exact；
- 5/5 pair discrete semantics exact；
- lower maximum absolute difference：`1.9669532775878906e-6`；
- 10/10 selected worker 环境 admitted；
- mean static prepare：`9.4219087412 s`；
- mean query saved：`0.338699936 s`；
- cold break-even：约 `27.82` queries。

## 4. 环境选择记录

pair 0--3 的 control/candidate 都在 attempt 0 准入。pair 4 的 control 与 candidate 均在
attempt 1 准入；attempt 0 保留在 raw 目录，没有删除或覆盖。selection 文件明确记录
`all_attempts_preserved=true`。

最终 headline 只使用 selected admitted workers。环境不准入的 attempt 不进入 summary，但仍受
manifest 绑定。

## 5. Replay 与篡改

最终 replay：PASS。

- summary hash 重算一致；
- manifest hash 重算一致；
- worker raw 重新形成 5-pair semantics/performance summary；
- BAB4 receipt 的 10/9、4 segments、76/36 launches、fallback 0 与 warmup 合同重新验证；
- 五个 candidate compiled identity 唯一。

outer-resigned tamper：`10/10 rejected`，包括 latency、lower、discrete semantics、environment、
compiled identity、launch count、fallback、warmup dependency 和 claim flag。

边界：探针模型是“改 raw worker + 重签最外层 manifest”，不是同时伪造 raw、summary、所有身份
来源的 coherent full resign；`coherent-full-resign_claimed=false`。

## 6. 相对上一状态的意义

此前四段 TIR 只有 local region `~1.37x` 和 3-pair 开发诊断。现在已证明：

1. terminal/residual/projection/input-domain 四段能够经过真实 RVIR exact-call；
2. 10 evaluation / 9 Adam mutation 的 terminal state 与六份 lA 能进入原有 handoff/commit；
3. kernel/region 收益没有被 host solver integration 吞掉；
4. 完整 query 的 1.15x 研究线在 5 组环境准入 fresh pairs 中通过。

这仍不是“BoundFlow 总体比 vanilla auto_LiRPA 快 10x”的证明。对照是当前工程的 B4-A
same-solver control，workload 仍只有一个 ResNet2B property，且 core 1.20x 门禁未过。

## 7. 验证与下一步

新增冻结 artifact 测试：

- source/code blob identity；
- replay；
- query research gate；
- 10 类 outer-resigned tamper。

下一工程动作不再是继续重复外审，而是二选一的实测归因：

1. profile BAB4 的 4 segments + PyTorch Adam orchestration，解释 core 为何停在 `1.177x`；
2. 优先消除跨 segment host/autograd/optimizer 提交开销，目标让 core geomean 也达到 `1.20x`，
   同时保持 query worst pair `>=1.00x`。

待下一轮用户统一交审时，再把本 artifact、replay、tamper 与本收官记录一起交给外部模型。
