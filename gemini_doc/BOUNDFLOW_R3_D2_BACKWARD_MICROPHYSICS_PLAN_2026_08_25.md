---
status: preregistered-d2a-readiness-gated
updated: 2026-08-25T22:40:00+08:00
type: plan
topic: boundflow
slug: r3-d2-backward-microphysics
stage: s01
---

# R3-D2 Backward-First Microphysics 预注册

## 1. 起因与目标

D1-C 已把 forward residual 热点从 v1 raw TIR 降到约 `5.44 ms/10 evaluations`，但完整 wrapper 仍约
`394 ms`；一次热态归因显示 custom backward 约 `369 ms`。D2 不假设 backward 一定可解，只验证：

1. 哪个 backward phase/symbol 形成真实 GPU wall 主导；
2. 在 native parity 与 1.20x wrapper 目标下，所需区域加速是否物理可达；
3. 是否存在保持 compressed α ownership、10/9 optimizer 轨迹和零 dense-A retention 的合法替代。

## 2. Scope 与禁止项

- 冻结基线为 D1-C source/artifact；不修改 native、D1-C forward、optimizer、α/β/split/history；
- D2-A 只读，不改 kernel/schedule/default，不做 tuning；
- 禁止恢复跨 evaluation dense A、autograd history 或 native shadow；
- 禁止把单 symbol/kernel-sum 直接写成 wrapper/query claim；
- D2-B 只有在 D2-A 量化 admission 后开放，且必须先 correctness、后 timing。

## 3. D2-A five-fresh 归因

每个 fresh process 先执行测量前 readiness：至少 3、最多 10 次不插桩完整 D1-C 10/9 wrapper；最近 3 次
必须全部落在对应 D1-C formal latency 的 `±10%`，且 `max/min≤1.05`。通过后立即执行一次不插桩 anchor，
anchor 仍须在 formal `±10%`；随后执行 phase-only wrapper，必须在 anchor `±10%`。任何条件失败均 fail
closed，不生成部分 artifact，也不得通过反复重抽样或放宽阈值形成 headline。symbol-only ledger 固定 3 warmup，
不参与 headline。

phase-only wrapper 在同一 non-default stream 上用 CUDA event
覆盖并分层记录：

- whole forward；
- whole custom backward；
- `_coefficient_sign_pass`；
- `_effective_value_pass`；
- `_recompute_a26`；
- compressed-gradient terminal；
- optimizer/host uncovered。

同时冻结每个 B1/B2 symbol 的 launch count 与 CUDA duration。event nesting 必须满足 child sum 不超过
parent tolerance；单 stream 禁止 overlap-adjustment。5 fresh 逐项重算 readiness、anchor、share、稳定性与
required speedup。

## 4. 量化路由

对每个候选区域使用同 scope 公式：

`r_required = s / (1/T - (1-s))`

其中 `s` 来自 D2-A 的 D1-C wrapper host-equivalent share：

- qualification parity：`T = native_median / d1c_median` 对应达到 `1.00x native`；
- research：目标 D1-C latency=`native/1.20`；
- 若分母 `≤0`，该区域即使无限加速也不能达标；
- 通用未知区域若 worst fresh `r_required >10x`，不进入 D2-B；
- **预 formal 修订**：若 dominant symbol 精确映射到 D1-B 已正式验证的 residual6/residual11 signature，
  则 admission 上限改为 `15.50x`；同时要求 D1-B formal worst `56.8625x` 作为既有物理上界，且映射
  必须由 symbol/shape/ABI receipt 逐项证明。超过 `15.50x` 仍关闭；
- 只有 share 跨 5 fresh 均 `≥20%`、required 不超过其适用上限、且存在合法 structured-owner
  schedule，才开放。

按当前非正式 attribution 粗算：去除 backward 后的其余成本约 `24.75 ms`；要达到 native parity，
backward 约需 `5.0x`，要达到 native/1.20 约需 `6.5x`。这只是 feasibility estimate，D2-A 必须从
formal raw 重算。

预 formal phase-only smoke 进一步定位：coefficient-sign约`342.90 ms`，其中symbol-only扰动账本显示
旧raw residual6/residual11分别约`257.38/105.02 ms`；它们与D1-B staged signature同构。该 smoke
只用于冻结上述15.50x已验证路径例外，不形成share或性能claim。

## 5. D2-B 候选顺序

只有 D2-A admission 后按 dominant symbol 选择一条：

1. 重算/符号扫描 fusion：合并连续 arena traversal，保持 compressed sign bitmap owner；
2. reduction schedule：固定 64/128/256 与 serial/shared/warp 候选，先 calibration 后 winner；
3. recompute sharing：仅在 evaluation 内保留最小 structured scratch，不跨 evaluation；
4. custom VJP epilogue fusion：compressed gradient 与 terminal reduction 合并，禁止 dense gradient materialize。

一次只实现 dominant path；若 first legal candidate worst isolated speedup 不足 required，则关闭该 path，
不扩到多 site。

## 6. 当前动作

只开放 D2-A five-fresh backward attribution。D2-B、R3-3、same-solver、query/queue 与 ASPLOS performance
claim 全部关闭。
