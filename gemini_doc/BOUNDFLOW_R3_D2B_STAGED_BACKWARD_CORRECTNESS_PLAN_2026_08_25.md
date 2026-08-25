---
status: preregistered-correctness-open
updated: 2026-08-25T23:20:00+08:00
type: plan
topic: boundflow
slug: r3-d2b-staged-backward-correctness
stage: s01
---

# R3-D2-B Staged Backward Correctness 预注册

## 1. 冻结输入与假设

D2-A source=`a6eaac4` 的 five-fresh 归因已证明 coefficient-sign minimum wrapper share=`0.870614`，
达到 D1-C/native 1.20x 所需 worst region speedup=`11.8762x`；五次 dominant symbols 均先后为
raw residual6、raw residual11，且 D1-B 相同 production signature 的 staged schedule worst isolated
speedup=`56.8625x`。D2-B 只验证该既有 schedule 能否合法接入 backward，不把 isolated 数字传播为
wrapper claim。

## 2. 唯一允许的改动

新增 `PreparedR3D2BStagedBackwardCandidateV1`，继承 D1-C cumulative candidate：

- forward 完全沿用 D1-C 的 residual11/residual6 staged replacement；
- backward 只在 `_coefficient_sign_pass` 内拦截两个 raw B1 symbol；
- residual11 使用 `s1[6144:12288]` 作 staged scratch，输出仍为 `s1[:6144]`；
- residual6 使用 `s0[12288:18432]` 作 staged scratch，输出仍为 `s0[:12288]`；
- bias 只允许 caller-owned accumulator 原位 alias；
- effective-value、recompute-a26、compressed gradient、optimizer/scheduler 不改。

禁止新增 persistent dense A、autograd history、global workspace、fallback、eager candidate、native shadow
或动态 tuning。D1-C forward 与 D2-B backward 计数器必须分离，不能把 backward launch 伪计入 forward
receipt。

## 3. Correctness 门禁

先实现且只运行 correctness：

1. 单次完整 10/9 wrapper 对 D1-C terminal lower/α：`atol=rtol=2e-4/2e-5`、lower sign exact；
2. 五个 fresh pair 逐 evaluation 比 lower、compressed dα、α、Adam exp_avg/exp_avg_sq；
3. execution 固定 10 evaluation、9 optimizer/scheduler mutation、10 custom forward/backward；
4. 每次 evaluation：D1-C forward staged launch=`4`，D2-B backward staged launch=`4`，raw B1 backward
   launch=`13`，bias alias=`2`；fallback/eager/native-shadow=`0`；
5. scratch pointer 必须等于两个 caller arena 的冻结 tail，互异且不与对应 live input/output 重叠；
6. 负路径至少覆盖 receipt claim、persistent dense A、launch、scratch pointer、fallback 与 ABI drift。

五个 fresh raw 必须由独立进程生成并由 replay 从 tensor 重算差值与 hash；全重签 tamper 必须拒绝。

## 4. 阶段边界

D2-B correctness artifact 关闭前不得记录 timing。通过后只开放 D2-B five-fresh wrapper-inclusive timing，
直接基线是 D1-C，并同时带回同 pair 的冻结 native latency。`11.8762x` 是 coefficient-sign region 的
required speedup，不能误写成 whole-wrapper 比值；timing 必须分别报告 staged/raw region 与 whole wrapper。
只有每个 fresh 的 staged/raw region 达到其同 pair Amdahl required，且 whole candidate/native geomean 与
worst 均 `≥1.20x`，才通过 research gate。否则按预注册结果 reduced 或 NO-GO，不放宽门槛。R3-3、
same-solver、query/queue 与 ASPLOS claim 保持关闭。
