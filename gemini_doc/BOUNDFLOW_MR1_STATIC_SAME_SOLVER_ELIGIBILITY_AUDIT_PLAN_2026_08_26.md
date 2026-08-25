# BoundFlow MR1 Same-Solver 静态可替换性审计预注册

> 日期：2026-08-26  
> 性质：只读、无计时、无新 solver 执行的 admission 审计  
> 前置：MR0 explicit-event budget=`VALIDATED-NO-GO`  
> 性能声明：`performance_claimed=false`

## 1. 问题与边界

MR0 证明 CIBC production 17-op CUDA Graph 上的逐 op event 记录不可作为低扰动 share
测量手段。MR1 不再尝试从扰动计时反推机会，而先回答更基础的问题：现有真实
same-solver raw 中，是否存在可由当前 CIBC **整图 IBP executor**直接替换的调用。

本轮只读取冻结 artifact，不运行 αβ-CROWN、不运行 GPU、不记录时间，也不实现优化。
它不能形成 CIBC query speedup、op share、complete-query 或 ASPLOS-ready claim。

## 2. 冻结输入

审计必须绑定下列输入及 SHA256；任何不一致均 fail closed：

1. RVIR v2 activation raw：
   `artifacts/rvir/rvir-cpu-correctness-v2-20260803/activation_calls.jsonl`，
   SHA256=`b8dc6652d487dbe3fd2a00933443a1f20221221babc22ae2f4f4f32a58462c4d`；
2. RVIR v2 manifest：SHA256=
   `0f8927c5b1909b7a0b671f1c2cda28835956ca259ff088c358fd0121f96979f6`；
3. RVIR-v3 production-state inventory：`inventory.json` SHA256=
   `ab0595bb002b79d80be8b78abd7a795a8aa634b17c9a8524df5a1b9b5fe19e06`，
   manifest SHA256=`a4bc22f52163b4b668a4753587c50a25e3d45e4fe484f07b226f714b6f31fde3`；
4. B3 same-solver formal manifest：SHA256=
   `d88eeecafcd6a7a9394cdf9654962a36497b1c7afd15d7862048b1c3ccd7db4a`；
5. CIBC 17-op formal manifest：SHA256=
   `b260fa6a49e77e3b8b1ff9502e6cc6bc27c6ddfcd5c01ba3bcd73b249d6dd807`。

目标 model hash 冻结为 ResNet2B
`onnx:791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`；
CIBC topology 冻结为 `6 Conv + 2 Linear + 6 ReLU + 2 Add + 1 Flatten`。

## 3. 逐调用 admission 规则

所有 394 条 activation raw 都必须计数；ResNet2B 子集还必须逐条输出 admission ledger。
当前 CIBC 整图替换只有在以下条件**全部**成立时 eligible：

1. model hash 与冻结 ResNet2B 相同；
2. solver phase 是无 split 的 initial/IBP graph evaluation，而不是
   `activation_bab_bound`/`bab_node_eval`；
3. `bound_method=IBP`，不得把 CROWN、α-CROWN 或 αβ-CROWN 伪装成 IBP；
4. `requires_grad=false`，无 α optimizer/autograd owner；
5. `split_state_present=false`，且 split signature/state identity 完整无未解析项；
6. requested output 是完整 interval graph 输出，不是 CROWN spec bound；
7. backend/semantics owner 允许 CIBC full-graph executor，不是
   `external_abcrown_exact_call` 的 provider-owned call；
8. shape/dtype/device/layout 与 CIBC runtime admission 合同一致；
9. 无 cuts/history/parent-lineage 或动态 branch mutation；
10. CIBC compile key 与17-op topology receipt可构造，fallback/eager/native shadow均为0。

任一 raw 缺字段、identity limitation、输入 hash 不一致或未知枚举都必须拒绝，不能跳过。
拒绝原因按优先级固定为上面 1→10；同时保留完整 reason set，避免首因掩盖多重边界。

## 4. 工件与防篡改

正式工件目录固定为
`artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1`，至少包含：

- `protocol.json`：输入 hash、规则版本、目标 topology/model；
- `ledger.jsonl`：ResNet2B 每条调用的 eligibility、首因与完整原因；
- `coverage.json`：394 总量、workload/model/method/phase/grad/split 分布；
- `summary.json`：eligible 数、拒绝计数、机械 route；
- `manifest.json`、`replay_stdout.txt`、`tamper_results.json`。

replay 必须重算输入 SHA256、逐行 ledger、coverage、summary 与 manifest。篡改至少覆盖：
input digest、删除/重复调用、model hash、phase、method、grad、split、semantics owner、ledger
eligibility、拒绝原因、summary count、route，并允许“重签外层 digest”后仍因语义重算拒绝。

## 5. 机械 verdict

- 若 `eligible_resnet_calls > 0`：结论只允许为
  `VALIDATED-MR1-STATIC-ELIGIBILITY`，只开放新的 direct end-to-end
  `B0/B3/candidate` A/B 预注册；仍不得直接计时或实现 R2。
- 若 `eligible_resnet_calls == 0`：结论为
  `VALIDATED-NO-GO-MR1-CIBC-FULL-GRAPH-SAME-SOLVER`，关闭当前“CIBC 17-op 整图直接
  替换 same-solver call”假设。

NO-GO 只针对当前 full-graph executor，不否定 CIBC operator/subgraph、重新设计的 CROWN
structured owner 或未来显式 IBP call capture。任何后继路线必须有新的 production call contract，
不能把历史独立 graph `2.45631x` 代入 query。

